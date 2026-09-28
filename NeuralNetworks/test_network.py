# ==============================================================================
# test_network.py
# Runs predictions on sample images using the trained CNN
# ==============================================================================

import argparse
import cv2
from imutils import paths
import numpy as np
from tensorflow.keras.models import load_model

# 1. Parse Command Line Arguments
ap = argparse.ArgumentParser(description="Test CNN on sample images.")
ap.add_argument(
    "-m",
    "--model",
    required=True,
    help="Path to trained model (.keras)",
)
ap.add_argument(
    "-t",
    "--test_images",
    required=True,
    help="Directory containing images to test",
)
args = vars(ap.parse_args())

# Class mapping matching folder structure: Cat -> 0, Dog -> 1
CLASSES = ["Cat", "Dog"]

# 2. Load Model
print(f"[INFO] Loading CNN from {args['model']}...")
model = load_model(args["model"])

# 3. Predict and Display
image_paths = list(paths.list_images(args["test_images"]))

for image_path in image_paths:
    # Load original image for visualization
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        continue

    # Preprocess image to match CNN inputs: (64, 64, 3)
    # Note: Model contains built-in Rescaling layer (1./255)
    image_resized = cv2.resize(orig_image, (64, 64))
    image_input = np.expand_dims(image_resized, axis=0)

    # Predict class probabilities
    probs = model.predict(image_input, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = f"{CLASSES[idx]}: {probs[idx] * 100:.2f}%"

    # Display label text on image
    color = (0, 255, 0) if CLASSES[idx] in image_path else (0, 0, 255)
    cv2.putText(
        orig_image,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
    )

    # Show window
    cv2.imshow("CNN Prediction (Press any key for next, ESC to exit)", orig_image)
    key = cv2.waitKey(0)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()