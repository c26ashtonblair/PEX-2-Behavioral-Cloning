"""
rover_data_processor.py

"""

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import time
import csv
import os
from imutils.video import FPS

# Paths for source and destination data
SOURCE_PATH = "/home/usafa/Documents/PEX02/rover_data"
DEST_PATH = "/home/usafa/Documents/PEX02/rover_data_processed"

# Parameters for image processing
# Define the range of white values to be considered for binary conversion
white_L, white_H = 200, 255
# Resize dimensions (quarter of 640x480)
resize_W, resize_H = 160, 120
# Crop dimensions to focus on relevant parts of the image
crop_W, crop_B, crop_T = 160, 120, 40  # Crop from top third down
PNG_COMPRESSION = 1  # 0-9 (lower is faster, larger files)

def load_telem_file(path):
    """
    Loads telemetry data from a CSV file to associate video frames with sensor data like throttle and steering.
    """
    with open(path, "r") as f:
        dict_reader = csv.DictReader(f)
        rows = list(dict_reader)
        # Build O(1) lookup by logged index to avoid repeated linear scans.
        lookup = {int(row["index"]): row for row in rows}
        return lookup, rows

def process_bag_file(source_file, dest_folder=None, skip_if_exists=True):
    """
    Processes a single .bag file, extracting frames and converting them to grayscale and binary images.
    Saves these images to a specified destination directory.
    """
    fps = None
    playback = None
    pipeline = None

    try:
        print(f"Processing {source_file}...")
        file_name = os.path.basename(source_file.replace(".bag", ""))
        dest_path = os.path.join(dest_folder or DEST_PATH, file_name)
        
        if skip_if_exists and os.path.isdir(dest_path):
            existing_files = [
                name for name in os.listdir(dest_path)
                if os.path.isfile(os.path.join(dest_path, name))
            ]
            if existing_files:
                print(f"{file_name} previously processed; skipping.")
                return
            print(f"{file_name} has empty output folder; reprocessing.")

        os.makedirs(dest_path, exist_ok=True)
        frm_lookup, frm_rows = load_telem_file(source_file.replace(".bag", ".csv"))
        processed_count = 0
        skipped_count = 0
        seq_idx = 0

        # Setup RealSense pipeline
        config, pipeline = rs.config(), rs.pipeline()
        rs.config.enable_device_from_file(config, source_file, False)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        pipeline.start(config)
        
        # Allow time for pipeline to stabilize
        time.sleep(1)
        playback = pipeline.get_active_profile().get_device().as_playback()
        playback.set_real_time(False)
        fps = FPS().start()

        # Processing loop
        while playback.current_status() == rs.playback_status.playing:
            try:
                
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Skip if no telemetry data for frame
                frm_num = color_frame.frame_number
                entry = frm_lookup.get(frm_num)
                # Some recordings may not preserve frame_number values exactly.
                # Fallback to sequential index mapping if needed.
                if entry is None and seq_idx < len(frm_rows):
                    entry = frm_lookup.get(seq_idx)
                seq_idx += 1
                if entry is None:
                    skipped_count += 1
                    continue
                
                # Extract throttle, steering, and heading data
                throttle = entry["throttle"]
                steering = entry["steering"]
                heading = entry["heading"]
                color_frame = np.asanyarray(color_frame.get_data())

                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
                resized_frame = cv2.resize(gray_frame, (resize_W, resize_H), interpolation=cv2.INTER_AREA)
                cropped_frame = resized_frame[crop_T:crop_B, :crop_W]
                Img_frame_placeholder = cv2.inRange(cropped_frame, white_L, white_H)
                # Save processed images WITH LABELS in the name
                bw_frm_name = f"{int(frm_num):09d}_{throttle}_{steering}_{heading}_bw.png"
                cv2.imwrite(
                    os.path.join(dest_path, bw_frm_name),
                    Img_frame_placeholder,
                    [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
                )
                processed_count += 1
                fps.update()

            except Exception as e:
                print(e)
                continue
    except Exception as e:
        print(e)
    finally:
        # Cleanup and stats
        if fps:
            fps.stop()
        if playback and playback.current_status() == rs.playback_status.playing:
            playback.pause()
        if pipeline:
            pipeline.stop()
        print(
            f"Finished {source_file}. FPS: {fps.fps() if fps else 'N/A'} | "
            f"saved={processed_count if 'processed_count' in locals() else 0}, "
            f"skipped={skipped_count if 'skipped_count' in locals() else 0}"
        )

def main():
    """
    Main function to process all .bag files in the source directory.
    """
    for filename in filter(lambda f: f.endswith(".bag"), os.listdir(SOURCE_PATH)):
        process_bag_file(os.path.join(SOURCE_PATH, filename))

if __name__ == "__main__":
    main()
