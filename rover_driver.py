"""
rover_driver.py
"""

import pyrealsense2.pyrealsense2 as rs
import time
import os
import numpy as np
import cv2
import tensorflow as tf
import utilities.drone_lib as dl

# Path to the trained model weights
MODEL_NAME = "models/rover_model_02_ver01_final.h5"

# Rover driving command limits
MIN_STEERING, MAX_STEERING = 1000, 2000
MIN_THROTTLE, MAX_THROTTLE = 1500, 2000
SAFE_MIN_THROTTLE, SAFE_MAX_THROTTLE = 1300, 1700
MAX_THROTTLE_STEP = 8
WARMUP_FRAMES = 25

"""
HINT:  Get values to the above by querying your own rover...
rover.parameters['RC3_MAX']
rover.parameters['RC3_MIN']
rover.parameters['RC1_MAX']
rover.parameters['RC1_MIN']
"""

# Image processing parameters
white_L, white_H = 200, 255  # White color range
resize_W, resize_H = 160, 120  # Resized image dimensions
crop_W, crop_B, crop_T = 160, 120, 40  # Crop box dimensions
FRAME_TIMEOUT_MS = 1000
SAVE_DEBUG_FRAMES = True
DEBUG_FRAME_COUNT = 40
DEBUG_FRAME_ROOT = "runtime_debug_frames"

def get_model(filename):
    """Load the model, with a compatibility fallback for older Keras runtimes."""
    try:
        model = tf.keras.models.load_model(filename, compile=False)
        model.compile()
        print("Loaded full model.")
        return model
    except TypeError as exc:
        print(f"Full-model load failed ({exc}). Falling back to architecture+weights.")
        model = define_inference_model(input_shape=(80, 160))
        model.load_weights(filename)
        print("Loaded model weights with local architecture.")
        return model


def define_inference_model(input_shape=(80, 160)):
    """Architecture must match model_training.py exactly."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Rescaling(1.0 / 255.0),
            tf.keras.layers.Reshape((input_shape[0], input_shape[1], 1)),
            tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(48, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(2, activation="linear"),
        ]
    )
    model.compile()
    return model

def min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val - v_min) / (v_max - v_min)


def invert_min_max_norm(val, v_min=1000.0, v_max=2000.0):
    return (val * (v_max - v_min)) + v_min


def denormalize(steering, throttle):
    """Denormalize steering and throttle values to the rover's command range."""
    steering = invert_min_max_norm(steering, MIN_STEERING, MAX_STEERING)
    throttle = invert_min_max_norm(throttle, MIN_THROTTLE, MAX_THROTTLE)
    return steering, throttle

def initialize_pipeline(brg=False):
    """Initialize the RealSense pipeline for video capture."""
    pipeline = rs.pipeline()
    config = rs.config()

    if brg:
        config.enable_stream(rs.stream.color, 
                             640, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480,
                             rs.format.rgb8, 30)
    pipeline.start(config)
    return pipeline

def get_video_data(pipeline):
    """Capture a video frame, preprocess it, and prepare it for model prediction."""
    deadline = time.monotonic() + (FRAME_TIMEOUT_MS / 1000.0)
    frame = None
    while time.monotonic() < deadline:
        frame = pipeline.poll_for_frames()
        if frame:
            break
        time.sleep(0.005)
    if not frame:
        print(f"Camera timeout waiting for frame ({FRAME_TIMEOUT_MS} ms).")
        return None

    color_frame = frame.get_color_frame()
    if not color_frame:
        print("Camera returned no color frame.")
        return None

    image = np.asanyarray(color_frame.get_data())

    # Match training preprocessing in rover_data_processor.py:
    # RGB -> Gray -> Resize(160x120) -> Crop[40:120, 0:160] -> Binary threshold (inRange)
    gray_frame = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_frame = cv2.resize(gray_frame, (resize_W, resize_H), interpolation=cv2.INTER_AREA)
    cropped_frame = resized_frame[crop_T:crop_B, :crop_W]
    bw_frame = cv2.inRange(cropped_frame, white_L, white_H)

    # Model expects batches; training used (80, 160) grayscale inputs.
    model_input = np.expand_dims(bw_frame, axis=0)
    return model_input

def set_rover_data(rover, steering, throttle):
    """Set rover control commands, ensuring they're within the valid range."""
    
    # May uncomment below to force a specific range, if your model is 
    # sometimes outputting weird ranges (should not be needed)
    #steering, throttle = check_inputs(int(steering), int(throttle))
    
    rover.channels.overrides = {"1": steering, "3": throttle}
    print(f"Steering: {steering}, Throttle: {throttle}")


def check_inputs(steering, throttle):
    """Check and clamp the steering and throttle inputs to their allowed ranges."""
    steering = np.clip(steering, MIN_STEERING, MAX_STEERING)
    throttle = np.clip(throttle, SAFE_MIN_THROTTLE, SAFE_MAX_THROTTLE)
    return steering, throttle

def main():

    """Main function to drive the rover based on model predictions."""
   
    # Setup and connect to the rover
    rover = dl.connect_device("/dev/ttyACM0")

    # Load the trained model
    model = get_model(MODEL_NAME)

    if model is None:
        print("Unable to load CNN model!")
        rover.close()
        print("Terminating program...")
        exit()
        
    while True:
        print("Arm vehicle to start mission.")
        print("(CTRL-C to stop process)")
        while not rover.armed:
            print("armed:", rover.armed, "mode:", rover.mode.name if rover.mode else None)
            time.sleep(1)

        print("Armed detected, starting camera pipeline...")
        
        # Initialize video capture
        pipeline = initialize_pipeline()

        print("Camera pipeline started. Entering drive loop.")
        frame_count = 0
        last_throttle = SAFE_MIN_THROTTLE
        debug_saved = 0
        debug_dir = None
        if SAVE_DEBUG_FRAMES:
            debug_dir = os.path.join(DEBUG_FRAME_ROOT, time.strftime("session_%Y%m%d-%H%M%S"))
            os.makedirs(debug_dir, exist_ok=True)
            print(f"Saving first {DEBUG_FRAME_COUNT} preprocessed frames to: {debug_dir}")
        
        while rover.armed:
            processed_image = get_video_data(pipeline)
            if processed_image is None:
                print("No image from camera.")
                continue

            if SAVE_DEBUG_FRAMES and debug_saved < DEBUG_FRAME_COUNT:
                debug_file = os.path.join(debug_dir, f"frame_{debug_saved:04d}_bw.png")
                cv2.imwrite(debug_file, processed_image[0])
                debug_saved += 1
                if debug_saved == DEBUG_FRAME_COUNT:
                    print(f"Saved {DEBUG_FRAME_COUNT} debug frames to: {debug_dir}")

            # Predict steering and throttle from the processed image
            output = model.predict(processed_image, verbose=0)

            # Model outputs normalized labels [steering, throttle]
            steering, throttle = denormalize(output[0][0], output[0][1])
            steering, throttle = check_inputs(int(steering), int(throttle))

            # Keep throttle neutral briefly after arming while camera/model stabilize.
            if frame_count < WARMUP_FRAMES:
                throttle = SAFE_MIN_THROTTLE
            else:
                # Rate-limit throttle increase/decrease to avoid sudden acceleration.
                throttle_delta = int(np.clip(throttle - last_throttle, -MAX_THROTTLE_STEP, MAX_THROTTLE_STEP))
                throttle = int(last_throttle + throttle_delta)

            frame_count += 1
            last_throttle = throttle

            # Send predicted values to rover
            set_rover_data(rover, steering, throttle)

        # stop recording
        pipeline.stop()
        time.sleep(1)
        pipeline = None
        rover.close()
        print("Done.")

if __name__ == "__main__":
    main()
