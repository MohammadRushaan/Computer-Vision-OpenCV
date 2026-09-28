# ==============================================================================
# deep_clean.py
# Scans dataset and checks the actual internal channel dimension of every image.
# Removes 2-channel files (RGBA palette/grayscale alpha) that crash Keras pipelines.
# ==============================================================================

import os
import tensorflow as tf

# Target directory containing the 'Cat' and 'Dog' folders
DATASET_DIR = "kaggle_dogs_vs_cats"
VALID_EXTS = (".jpg", ".jpeg", ".png")

removed_count = 0
total_checked = 0

print(f"[INFO] Running deep channel inspection on '{DATASET_DIR}'...")

# Traverse all subfolders
for root, _, files in os.walk(DATASET_DIR):
    for filename in files:
        if not filename.lower().endswith(VALID_EXTS):
            continue

        file_path = os.path.join(root, filename)
        total_checked += 1
        should_delete = False

        try:
            # 1. Read binary tensor directly from storage
            raw_bytes = tf.io.read_file(file_path)

            # 2. Decode using raw channel detection (channels=0 tells TF to inspect actual inherent channels)
            image_tensor = tf.io.decode_image(raw_bytes, channels=0, expand_animations=False)

            # 3. Read shape: [height, width, channels]
            # Standard images have 1 (grayscale), 3 (RGB), or 4 (RGBA) channels.
            # Files with 2 channels cause Keras's C++ iterator to crash.
            num_channels = image_tensor.shape[-1]
            if num_channels not in (1, 3, 4):
                print(f"[CORRUPT] Invalid channel count ({num_channels}): {file_path}")
                should_delete = True

        except Exception as e:
            # Catches corrupted headers, truncated bitstreams, or decoding failures
            print(f"[CORRUPT] Failed to decode ({e}): {file_path}")
            should_delete = True

        # Purge the problematic file from disk
        if should_delete:
            try:
                os.remove(file_path)
                removed_count += 1
                print(f"[DELETED] Successfully removed: {file_path}")
            except Exception as delete_error:
                print(f"[ERROR] Could not delete {file_path}: {delete_error}")

        # Progress tracker for large datasets
        if total_checked % 5000 == 0:
            print(f"[PROGRESS] Verified {total_checked} images...")

print("=" * 60)
print(f"[SUMMARY] Total images inspected: {total_checked}")
print(f"[SUMMARY] Problematic files purged: {removed_count}")
print("=" * 60)