"""
model_training.py

"""

import os
import data_gen
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Reshape,
    Rescaling,
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from matplotlib import pyplot

# Configuration parameters
DEVICE = "/GPU:0"  # Device to use for computation. Change to "/GPU:0" if GPU is available
DATA_PATH = "/home/usafa/Documents/PEX02/rover_data_processed"  # Path to processed data
MODEL_NUM = 2 # Model number for naming
TRAINING_VER = 1  # Training version for naming
NUM_EPOCHS = 5  # Number of epochs to train
BATCH_SIZE = 13  # Batch size for training
TRAIN_VAL_SPLIT = 0.8  # Train/validation split ratio


# Define the CNN model structure for steering/throttle regression
def define_model(input_shape=(80, 160)):
    model = Sequential(
        [
            Input(shape=input_shape),
            Rescaling(1.0 / 255.0),
            Reshape((input_shape[0], input_shape[1], 1)),
            Conv2D(24, (5, 5), strides=(2, 2), activation="relu"),
            BatchNormalization(),
            Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
            BatchNormalization(),
            Conv2D(48, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(2, activation="linear"),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model

# Train the model with data from a generator, using checkpoints and a specified device
def train_model(amt_data=1.0):
    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"Data path not found: {DATA_PATH}. "
            "Set the ROVER_DATA_PATH environment variable to your processed dataset root."
        )
    
    # Load samples (i.e. preprocessed frames for training).
    # Note that we are using sequences consisting of 13 frames.
    samples = data_gen.get_sequence_samples(DATA_PATH, sequence_size=13)
    if not samples:
        raise ValueError(f"No training samples found under {DATA_PATH}")
    
    # You may wish to do simple testing using only 
    # a fraction of your training data...
    if amt_data < 1.0:
        # Use only a portion of the entire dataset
        samples, _ = data_gen.split_samples(samples, fraction=amt_data)

    # Now, split our samples into training and validation sets
    # Note that train_samples will contain a flat list of sequenced 
    # image file paths.
    train_samples, val_samples = data_gen.split_samples(samples, fraction=TRAIN_VAL_SPLIT)

    train_steps = max(1, len(train_samples) // BATCH_SIZE)
    val_steps = max(1, len(val_samples) // BATCH_SIZE)

    # Create data generators that will supply both the training and validation data during training.
    train_gen = data_gen.batch_generator(train_samples, batch_size=BATCH_SIZE)
    val_gen = data_gen.batch_generator(val_samples, batch_size=BATCH_SIZE)
    
    with tf.device(DEVICE):

        # Input shape must match preprocessed image size (80x160 grayscale).
        model = define_model(input_shape=(80, 160))
        model.summary()  # Print a summary of the model architecture
        
        # Path for saving the best model checkpoints
        os.makedirs("models", exist_ok=True)
        filePath = "models/rover_model_" + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}" + "_epoch{epoch:04d}_val_loss{val_loss:.4f}.h5"
        
        # Save only the best (i.e. min validation loss) epochs
        checkpoint_best = ModelCheckpoint(filePath, monitor="val_loss", 
                                          verbose=1, save_best_only=True, 
                                          mode="min")
        
        # Train your model here.
        history = model.fit(
            train_gen,
            steps_per_epoch=train_steps,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=NUM_EPOCHS,
            callbacks=[checkpoint_best],
            verbose=1,
        )

        final_model_path = (
            "models/rover_model_"
            + f"{MODEL_NUM:02d}_ver{TRAINING_VER:02d}_final.h5"
        )
        model.save(final_model_path)
        print(f"Final model saved to: {final_model_path}")

        #print(history.history.keys())
    return history


# Plot training and validation loss over epochs
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(len(histories),1,1)
        pyplot.title('Training Loss Curves')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')

    pyplot.show()

# Run the training process and display training diagnostics
def main():
    history = train_model()
    summarize_diagnostics([history])


# Entry point to start the training process
if __name__ == "__main__":
    main()
