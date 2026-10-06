# classify.py

# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION & IMPORT STATEMENTS
# ==============================================================================
import os

# Suppress verbose C++ oneDNN and TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import pickle
import cv2
import numpy as np
import imutils
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================================================================
# 2. CONSTRUCT AND PARSE COMMAND LINE ARGUMENTS
# ==============================================================================
ap = argparse.ArgumentParser(description="Classify an image with custom Keras Pokédex model")
ap.add_argument("-i", "--image", required=True, 
                help="Path to the test image file")
ap.add_argument("-m", "--model", default="pokedex.model.keras", 
                help="Path to the trained Keras model file")
ap.add_argument("-l", "--labelbin", default="lb.pickle", 
                help="Path to the serialized LabelBinarizer file")
args = vars(ap.parse_args())

# ==============================================================================
# 3. LOAD AND PREPROCESS TEST IMAGE
# ==============================================================================
raw_image = cv2.imread(args["image"], cv2.IMREAD_UNCHANGED)
if raw_image is None:
    raise SystemExit(f"[ERROR] Could not load image from: {args['image']}")

# Handle transparent 4-channel PNGs by blending onto a clean white background
if raw_image.ndim == 3 and raw_image.shape[2] == 4:
    alpha = raw_image[:, :, 3] / 255.0
    bgr = raw_image[:, :, :3]
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    processed = (bgr * alpha[:, :, np.newaxis] + white_bg * (1 - alpha[:, :, np.newaxis])).astype(np.uint8)
else:
    processed = raw_image[:, :, :3].copy()

# Keep a display copy
output = processed.copy()

# Convert OpenCV BGR format to RGB
image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

# Resize to 224x224 to match the trained MobileNetV2 input shape
image = cv2.resize(image, (224, 224))

# Convert to float32 in range [0, 255] and apply official MobileNetV2 preprocessing
image = image.astype("float32")
image = preprocess_input(image)

# Expand dimensions to create batch of size 1: shape becomes (1, 224, 224, 3)
image = np.expand_dims(image, axis=0)

# ==============================================================================
# 4. LOAD MODEL & COMPUTE PREDICTIONS
# ==============================================================================
print("[INFO] Loading trained model and label dictionary...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# Run forward inference pass
proba = model.predict(image)[0]

# Display full probability breakdown in console for debugging
print("\n===============================")
print("     CLASS PROBABILITIES       ")
print("===============================")
for i, class_name in enumerate(lb.classes_):
    print(f"  {class_name:<15}: {proba[i] * 100:.2f}%")
print("===============================\n")

# Find class with highest probability score
idx = np.argmax(proba)
label = lb.classes_[idx]
confidence = proba[idx] * 100
label_text = f"{label}: {confidence:.2f}%"

# ==============================================================================
# 5. VISUALIZE PREDICTION
# ==============================================================================
output = imutils.resize(output, width=600)
cv2.putText(
    output,
    label_text,
    (15, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (0, 255, 0),
    2
)

print(f"[FINAL PREDICTION] {label_text}")
cv2.imshow("Custom Keras Pokédex Prediction", output)
print("[INFO] Press any key on the image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()