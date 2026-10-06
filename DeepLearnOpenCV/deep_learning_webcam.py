# deep_learning_webcam.py

# ==============================================================================
# 1. IMPORT REQUIRED PACKAGES
# ==============================================================================
import collections       # Deque for maintaining a rolling history of predictions
import argparse          # CLI argument parser
import time              # FPS and latency measurement
import numpy as np       # Array math, softmax, and rolling average
import cv2               # OpenCV for video capture, DNN inference, and GUI rendering

# ==============================================================================
# 2. PARSE COMMAND-LINE ARGUMENTS
# ==============================================================================
ap = argparse.ArgumentParser(description="Stabilized Real-Time Deep Learning Classification")
ap.add_argument("-p", "--prototxt", default="squeezenet.prototxt",
                help="Path to Caffe deploy prototxt file")
ap.add_argument("-m", "--model", default="squeezenet.caffemodel",
                help="Path to pre-trained Caffe model weights")
ap.add_argument("-l", "--labels", default="synset_words.txt",
                help="Path to ImageNet synset labels file")
ap.add_argument("-c", "--camera", type=int, default=0,
                help="Webcam camera index (default: 0)")
ap.add_argument("-t", "--threshold", type=float, default=30.0,
                help="Minimum confidence percentage threshold to display a label (default: 30.0%)")
args = vars(ap.parse_args())

# ==============================================================================
# 3. LOAD LABELS AND CAFFE MODEL
# ==============================================================================
# Parse ImageNet synsets into clean human-readable names
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

print("[INFO] Loading deep learning model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# ==============================================================================
# 4. INITIALIZE CAMERA & SMOOTHING BUFFERS
# ==============================================================================
print("[INFO] Starting webcam stream. Press 'q' to exit.")
cap = cv2.VideoCapture(args["camera"], cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(args["camera"])

if not cap.isOpened():
    raise SystemExit(f"[ERROR] Could not access webcam index {args['camera']}")

# Smoothing queue: keeps probability vectors from the last 10 frames
# This eliminates rapid frame-to-frame flickering
ROLLING_WINDOW_SIZE = 10
prob_history = collections.deque(maxlen=ROLLING_WINDOW_SIZE)

prev_time = time.time()
fps = 0.0

# ==============================================================================
# 5. REAL-TIME INFERENCE LOOP
# ==============================================================================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # FPS Calculation
    curr_time = time.time()
    time_diff = curr_time - prev_time
    prev_time = curr_time
    if time_diff > 0:
        fps = 1.0 / time_diff

    # Preprocess frame: resize to (224, 224) and subtract ImageNet mean channels
    blob = cv2.dnn.blobFromImage(
        frame, 
        scalefactor=1.0, 
        size=(224, 224), 
        mean=(104.0, 117.0, 123.0)
    )

    # Forward pass
    net.setInput(blob)
    preds = net.forward().flatten()

    # Apply softmax if outputs are raw logits
    if preds.max() > 1.0 or preds.min() < 0.0:
        exp_preds = np.exp(preds - np.max(preds))
        preds = exp_preds / np.sum(exp_preds)

    # Append current frame's probability distribution to rolling buffer
    prob_history.append(preds)

    # Compute temporally averaged probability vector across the rolling buffer
    smooth_preds = np.mean(prob_history, axis=0)

    # Extract top prediction
    top_idx = np.argsort(smooth_preds)[::-1][0]
    confidence = smooth_preds[top_idx] * 100
    label = classes[top_idx]

    # ==============================================================================
    # 6. ANNOTATE VIDEO FRAME
    # ==============================================================================
    # Black top bar for legible text
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0, 0, 0), -1)

    # Filter out low-confidence random guesses
    if confidence >= args["threshold"]:
        label_text = f"{label}: {confidence:.1f}%"
        color = (0, 255, 0)   # Green for confident predictions
    else:
        label_text = f"Scanning / Uncertain ({label}: {confidence:.1f}%)"
        color = (0, 165, 255) # Orange for uncertain classifications

    # Overlay prediction
    cv2.putText(
        frame, 
        label_text, 
        (10, 36), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.75, 
        color, 
        2
    )

    # Overlay FPS
    cv2.putText(
        frame, 
        f"FPS: {fps:.1f}", 
        (frame.shape[1] - 140, 36), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (255, 255, 0), 
        2
    )

    cv2.imshow("Live Deep Learning Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()