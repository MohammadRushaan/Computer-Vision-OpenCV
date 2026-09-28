# ==============================================================================
# simple_neural_network.py
# Cleaned CNN pipeline for Dogs vs. Cats
# ==============================================================================

import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Parse Command Line Arguments
ap = argparse.ArgumentParser(
    description="Train a CNN classifier on Dogs vs. Cats dataset."
)
ap.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="Path to dataset root folder containing Cat and Dog subdirectories",
)
ap.add_argument(
    "-m",
    "--model",
    required=True,
    help="Path to save the trained model (.keras)",
)
args = vars(ap.parse_args())

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 15

# 2. Stream Data and Enforce 3 Channels (RGB)
print("[INFO] Loading and streaming dataset from disk...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    args["dataset"],
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",  # Forces all images to standard 3-channel RGB
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    args["dataset"],
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    label_mode="categorical",
)

class_names = train_ds.class_names
print(f"[INFO] Classes detected: {class_names}")

# Optimize pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Build Convolutional Neural Network (using Input layer to avoid warning)
print("[INFO] Constructing CNN architecture...")
model = models.Sequential(
    [
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255),
        # Conv Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Conv Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Conv Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax"),
    ]
)

# 4. Compile Model
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()

# 5. Train
print("[INFO] Training CNN...")
H = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

# 6. Save
output_dir = os.path.dirname(args["model"])
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"[INFO] Saving model to {args['model']}...")
model.save(args["model"])
print("[INFO] Training completed successfully!")