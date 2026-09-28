# ==============================================================================
# clean_dataset.py
# Scans an image dataset folder and purges corrupted, zero-byte, or malformed
# files that crash OpenCV or TensorFlow internal C++ decoders.
# ==============================================================================

import os
import cv2
import tensorflow as tf

# Define the root directory containing the subfolders ('Cat', 'Dog')
DATASET_DIR = "kaggle_dogs_vs_cats"

# Allowed file extensions to inspect
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Counters to track progress and reporting
removed_count = 0
total_checked = 0

print(f"[INFO] Scanning '{DATASET_DIR}' for corrupted and malformed images...")

# os.walk traverses the directory tree recursively
# root: current directory path
# _: subdirectories (unused here)
# files: list of filenames in the current directory
for root, _, files in os.walk(DATASET_DIR):
    for filename in files:
        # Filter out non-image files (such as .txt, .DS_Store, or hidden files)
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        # Construct the absolute/relative system path to the image
        file_path = os.path.join(root, filename)
        total_checked += 1
        is_corrupt = False

        # ----------------------------------------------------------------------
        # Check 1: Binary Header Validation (Magic Bytes)
        # ----------------------------------------------------------------------
        # Valid JPEG files must begin with the byte signature 0xFFD8 (SOI marker).
        # Files missing this header are truncated, 0-byte empty files, or corrupted.
        try:
            with open(file_path, "rb") as f:
                header = f.read(2)
                if header != b"\xff\xd8":
                    is_corrupt = True
        except Exception:
            # If the OS cannot open/read the file stream, mark as corrupt
            is_corrupt = True

        # ----------------------------------------------------------------------
        # Check 2: OpenCV Decoding Check
        # ----------------------------------------------------------------------
        # If the file passed the header check, verify that OpenCV's LibJPEG
        # can decompress and parse the pixel matrix into a valid NumPy array.
        if not is_corrupt:
            img = cv2.imread(file_path)
            # cv2.imread returns None if the image data is damaged or unreadable
            if img is None:
                is_corrupt = True

        # ----------------------------------------------------------------------
        # Check 3: TensorFlow Native C++ Decoder Check
        # ----------------------------------------------------------------------
        # TensorFlow uses its own internal image parser. Some images pass OpenCV
        # but fail in TF due to invalid color spaces (e.g., 2-channel grayscale+alpha).
        # We explicitly enforce 3-channel RGB decoding to simulate training behavior.
        if not is_corrupt:
            try:
                # Read the file as a raw binary tensor
                raw_bytes = tf.io.read_file(file_path)
                # Attempt decoding with strict channel layout (3 channels)
                # expand_animations=False prevents errors on animated formats (e.g., APNG/GIF)
                _ = tf.io.decode_image(raw_bytes, channels=3, expand_animations=False)
            except Exception:
                # Catches TensorFlow's InvalidArgumentError on malformed EXIF or bad channels
                is_corrupt = True

        # ----------------------------------------------------------------------
        # Deletion Stage
        # ----------------------------------------------------------------------
        # If any of the three validation gates failed, remove the file from disk
        if is_corrupt:
            try:
                os.remove(file_path)
                removed_count += 1
                print(f"[REMOVED] Corrupted image: {file_path}")
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path}: {e}")

# ------------------------------------------------------------------------------
# Summary Output
# ------------------------------------------------------------------------------
print("=" * 60)
print(f"[INFO] Scan complete across {total_checked} images.")
print(f"[INFO] Total corrupted files removed: {removed_count}")
print("=" * 60)