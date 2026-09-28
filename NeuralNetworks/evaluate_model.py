# ==============================================================================
# evaluate_model.py
# Computes metrics and exports a 4-panel visual performance dashboard.
# ==============================================================================

import argparse
import os
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorflow.keras.models import load_model

# ------------------------------------------------------------------------------
# 1. Parse Arguments & Load Pretrained Model
# ------------------------------------------------------------------------------
ap = argparse.ArgumentParser(description="Evaluate model and generate visual dashboard.")
ap.add_argument("-m", "--model", required=True, help="Path to trained model (.keras)")
ap.add_argument("-d", "--dataset", required=True, help="Path to dataset directory")
ap.add_argument("-o", "--output", default="output/evaluation_dashboard.png", help="Path to save output plot")
args = vars(ap.parse_args())

print(f"[INFO] Loading model from {args['model']}...")
model = load_model(args["model"])

# Inspect whether the model expects 1D vector (3072,) or 2D image (64, 64, 3)
input_shape = model.input_shape
is_flattened = len(input_shape) == 2 and input_shape[1] == 3072

# ------------------------------------------------------------------------------
# 2. Gather Data and Run Inference in Batches
# ------------------------------------------------------------------------------
image_paths = list(paths.list_images(args["dataset"]))
print(f"[INFO] Evaluating across {len(image_paths)} images...")

CLASSES = ["cat", "dog"]
data = []
y_true = []
y_probs = []  # Stores probability scores for Class 1 (Dog)

for i, img_path in enumerate(image_paths):
    raw_label = img_path.split(os.path.sep)[-2].lower()
    if raw_label not in CLASSES:
        raw_label = img_path.split(os.path.sep)[-1].split(".")[0].lower()

    if raw_label not in CLASSES:
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    # Match preprocessing applied during training
    if is_flattened:
        feature = cv2.resize(img, (32, 32)).flatten() / 255.0
    else:
        feature = cv2.resize(img, (64, 64))

    data.append(feature)
    y_true.append(CLASSES.index(raw_label))

    # Evaluate in chunks of 256 to conserve RAM
    if len(data) == 256:
        preds = model.predict(np.array(data, dtype="float32"), verbose=0)
        y_probs.extend(preds)
        data = []

    if i > 0 and (i + 1) % 5000 == 0:
        print(f"[INFO] Evaluated {i + 1}/{len(image_paths)} images...")

if len(data) > 0:
    preds = model.predict(np.array(data, dtype="float32"), verbose=0)
    y_probs.extend(preds)

y_probs = np.array(y_probs)
y_pred = np.argmax(y_probs, axis=1)
y_true = np.array(y_true)

# ------------------------------------------------------------------------------
# 3. Print Terminal Classification Report
# ------------------------------------------------------------------------------
print("\n" + "=" * 55)
print("            CLASSIFICATION SUMMARY REPORT")
print("=" * 55)
print(classification_report(y_true, y_pred, target_names=["Cat", "Dog"], digits=4))

# ------------------------------------------------------------------------------
# 4. Generate Visual Performance Dashboard
# ------------------------------------------------------------------------------
print(f"[INFO] Generating performance visual map -> {args['output']}...")
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
plt.subplots_adjust(hspace=0.35, wspace=0.3)

# --- Subplot 1: Annotated Confusion Matrix Heatmap ---
cm = confusion_matrix(y_true, y_pred)
im = axes[0, 0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
axes[0, 0].figure.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
axes[0, 0].set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=["Cat", "Dog"],
    yticklabels=["Cat", "Dog"],
    title="Confusion Matrix Heatmap",
    ylabel="Actual Label",
    xlabel="Predicted Label",
)

# Overlay numerical values and percentage in each cell
thresh = cm.max() / 2.0
for r in range(cm.shape[0]):
    for c in range(cm.shape[1]):
        count = cm[r, c]
        pct = (count / np.sum(cm[r, :])) * 100
        axes[0, 0].text(
            c,
            r,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="center",
            color="white" if count > thresh else "black",
            fontweight="bold",
        )

# --- Subplot 2: Per-Class Precision, Recall, and F1 Comparison ---
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
x = np.arange(2)
width = 0.25

axes[0, 1].bar(x - width, precision * 100, width, label="Precision", color="#4C72B0")
axes[0, 1].bar(x, recall * 100, width, label="Recall", color="#55A868")
axes[0, 1].bar(x + width, f1 * 100, width, label="F1-Score", color="#C44E52")

axes[0, 1].set_ylabel("Score (%)")
axes[0, 1].set_title("Per-Class Metric Comparison")
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(["Cat", "Dog"])
axes[0, 1].set_ylim(0, 110)
axes[0, 1].legend(loc="lower right")
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

# --- Subplot 3: Confidence Distribution (Correct vs. Incorrect) ---
max_confs = np.max(y_probs, axis=1) * 100
correct_mask = y_pred == y_true

axes[1, 0].hist(
    max_confs[correct_mask],
    bins=25,
    alpha=0.6,
    color="green",
    label=f"Correct ({np.sum(correct_mask):,})",
    edgecolor="black",
)
axes[1, 0].hist(
    max_confs[~correct_mask],
    bins=25,
    alpha=0.6,
    color="red",
    label=f"Misclassified ({np.sum(~correct_mask):,})",
    edgecolor="black",
)
axes[1, 0].set_title("Prediction Confidence Distribution")
axes[1, 0].set_xlabel("Predicted Probability / Confidence (%)")
axes[1, 0].set_ylabel("Number of Samples")
axes[1, 0].legend()
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

# --- Subplot 4: ROC Curve & Area Under the Curve (AUC) ---
fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
roc_auc = auc(fpr, tpr)

axes[1, 1].plot(fpr, tpr, color="#E377C2", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
axes[1, 1].plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Chance (AUC = 0.50)")
axes[1, 1].set_xlim([0.0, 1.0])
axes[1, 1].set_ylim([0.0, 1.05])
axes[1, 1].set_xlabel("False Positive Rate (1 - Specificity)")
axes[1, 1].set_ylabel("True Positive Rate (Recall)")
axes[1, 1].set_title("Receiver Operating Characteristic (ROC)")
axes[1, 1].legend(loc="lower right")
axes[1, 1].grid(True, linestyle="--", alpha=0.6)

# Save figure
os.makedirs(os.path.dirname(args["output"]), exist_ok=True)
plt.savefig(args["output"], dpi=300, bbox_inches="tight")
print(f"[INFO] Plot successfully generated and saved to {args['output']}")
plt.show()