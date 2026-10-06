# train_keras.py

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION & IMPORT STATEMENTS
# ==============================================================================
import os

# Suppress verbose C++ oneDNN and TensorFlow logs for clean terminal output
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import paths

# Machine learning utilities
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow / Keras deep learning modules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ==============================================================================
# 2. HYPERPARAMETERS & SETUP
# ==============================================================================
# 25 epochs is ideal for fine-tuning pre-trained MobileNetV2 on small datasets
EPOCHS = 25

# A gentle learning rate ensures pre-trained weights adapt without exploding
INIT_LR = 1e-4

# Batch size of 16 ensures frequent gradient updates
BS = 16

# MobileNetV2 expects standard 224x224 input images
IMAGE_DIMS = (224, 224, 3)

data = []
labels = []

# ==============================================================================
# 3. LOAD AND PREPROCESS DATASET IMAGES
# ==============================================================================
print("[INFO] Loading and preprocessing dataset images...")
imagePaths = sorted(list(paths.list_images("dataset")))

for imagePath in imagePaths:
    # Extract the class folder name (e.g. 'dataset/pikachu/001.jpg' -> 'pikachu')
    label = imagePath.split(os.path.sep)[-2]

    # Skip negative/background directory if you had created one previously
    if label == "background":
        continue

    # Load image including alpha transparency channel if present
    raw_img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if raw_img is None:
        continue

    # If the image is a 4-channel PNG (BGRA), composite it over a clean white background
    if raw_img.ndim == 3 and raw_img.shape[2] == 4:
        alpha = raw_img[:, :, 3] / 255.0
        bgr = raw_img[:, :, :3]
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        img = (bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
    elif raw_img.ndim == 3 and raw_img.shape[2] == 3:
        img = raw_img
    else:
        continue

    # Convert OpenCV BGR format to RGB format expected by modern neural networks
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize image to MobileNetV2 standard dimensions (224, 224)
    img = cv2.resize(img, (IMAGE_DIMS[1], IMAGE_DIMS[0]))

    data.append(img)
    labels.append(label)

# Convert image list to float32 NumPy array in range [0, 255]
data = np.array(data, dtype="float32")

# Apply official MobileNetV2 scaling (scales pixel range directly to [-1.0, 1.0])
data = preprocess_input(data)
labels = np.array(labels)

print(f"[INFO] Total valid images loaded: {len(data)}")
unique_classes, counts = np.unique(labels, return_counts=True)
for cls, count in zip(unique_classes, counts):
    print(f"  - {cls}: {count} images")

# ==============================================================================
# 4. LABEL ENCODING & STRATIFIED TRAIN / TEST SPLIT
# ==============================================================================
# Convert string labels into one-hot encoded binary matrices
lb = LabelBinarizer()
labels_encoded = lb.fit_transform(labels)

# Handle binary class edge case where LabelBinarizer outputs a single column
if len(lb.classes_) == 2:
    labels_encoded = np.hstack((1 - labels_encoded, labels_encoded))

# Stratify guarantees that train and validation sets have proportional class representations
(trainX, testX, trainY, testY) = train_test_split(
    data,
    labels_encoded,
    test_size=0.2,
    stratify=labels_encoded,
    random_state=42
)

# ==============================================================================
# 5. COMPUTE CLASS WEIGHTS (FIXES CHARMANDER UNDER-REPRESENTATION)
# ==============================================================================
# Compute balanced penalty weights so smaller classes (Charmander) are not drowned out by Pikachu
y_integers = np.argmax(trainY, axis=1)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_integers),
    y=y_integers
)
class_weight_dict = dict(enumerate(class_weights))
print("[INFO] Balancing weights applied to classes:")
for idx, weight in class_weight_dict.items():
    print(f"  - {lb.classes_[idx]}: weight multiplier = {weight:.2f}")

# Data augmentation to create diverse poses, scales, and positions
aug = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ==============================================================================
# 6. MODEL ARCHITECTURE & TOP-LAYER FINE-TUNING
# ==============================================================================
print("[INFO] Loading pre-trained MobileNetV2 backbone...")
baseModel = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=IMAGE_DIMS)
)

# Freeze early low-level layers (preserve general edge/texture detection)
# Unfreeze the top 30 layers so the network adapts to cartoon outlines and shapes
for layer in baseModel.layers[:-30]:
    layer.trainable = False
for layer in baseModel.layers[-30:]:
    layer.trainable = True

# Build custom classification head on top of the backbone
head = baseModel.output
head = GlobalAveragePooling2D()(head)
head = Dense(128, activation="relu")(head)
head = Dropout(0.4)(head)
head = Dense(len(lb.classes_), activation="softmax")(head)

model = Model(inputs=baseModel.input, outputs=head)

# Compile model using categorical crossentropy loss
opt = Adam(learning_rate=INIT_LR)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# ==============================================================================
# 7. MODEL TRAINING WITH BALANCING
# ==============================================================================
print("[INFO] Starting training with balanced class weighting...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=max(1, len(trainX) // BS),
    epochs=EPOCHS,
    class_weight=class_weight_dict,  # Applies error penalty multiplier to minority classes
    verbose=1
)

# ==============================================================================
# 8. SAVE ARTIFACTS AND PLOT ACCURACY CURVES
# ==============================================================================
print("[INFO] Saving trained model and label binarizer to disk...")
model.save("pokedex.model.keras")

with open("lb.pickle", "wb") as f:
    f.write(pickle.dumps(lb))

# Generate and save diagnostic training visualization
plt.style.use("ggplot")
plt.figure(figsize=(10, 6))
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("MobileNetV2 Pokédex Training Loss & Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss / Accuracy")
plt.legend(loc="upper right")
plt.savefig("plot.png")
print("[INFO] Training finished successfully! Diagnostic curve saved to 'plot.png'.")