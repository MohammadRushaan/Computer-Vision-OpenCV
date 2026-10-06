# realtime_object_detection.py

# ==============================================================================
# 1. IMPORT REQUIRED PACKAGES
# ==============================================================================
import numpy as np       # Used for array manipulations and coordinate re-scaling
import argparse          # Parses command-line flags and parameters
import time              # Calculates frame rates (FPS) and timing
import cv2               # OpenCV library for video capture, DNN inference, and drawing

# ==============================================================================
# 2. CONSTRUCT AND PARSE ARGUMENTS
# ==============================================================================
ap = argparse.ArgumentParser(description="Real-Time Deep Learning Multi-Object Detection")

# Model architecture text file
ap.add_argument("-p", "--prototxt", default="ssd_deploy.prototxt",
                help="Path to Caffe deploy prototxt file")

# Serialized weights binary file
ap.add_argument("-m", "--model", default="ssd_weights.caffemodel",
                help="Path to pre-trained Caffe model weights")

# Minimum detection probability threshold
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="Minimum confidence threshold to display a detected object")

# Camera index (default: 0)
ap.add_argument("--camera", type=int, default=0,
                help="Webcam index (0 for default, 1 for external USB)")

args = vars(ap.parse_args())

# ==============================================================================
# 3. DEFINE DETECTABLE CLASSES AND COLOR PALETTE
# ==============================================================================
# The 20 everyday object classes MobileNet-SSD was trained on (Pascal VOC dataset)
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
]

# Generate distinct RGB colors for drawing unique bounding boxes per class
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ==============================================================================
# 4. LOAD DEEP LEARNING MODEL
# ==============================================================================
print("[INFO] Loading MobileNet-SSD Deep Learning network...")
# Read the network architecture and pre-trained weights into OpenCV DNN
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# ==============================================================================
# 5. INITIALIZE WEBCAM (WITH ROBUST WINDOWS DIRECTSHOW BACKEND)
# ==============================================================================
print("[INFO] Opening webcam...")
# cv2.CAP_DSHOW forces Windows DirectShow API to prevent stream dropouts
cap = cv2.VideoCapture(args["camera"], cv2.CAP_DSHOW)

# Fallback: if DirectShow index fails, attempt default system index
if not cap.isOpened():
    print(f"[WARN] DirectShow failed on camera {args['camera']}. Attempting default backend...")
    cap = cv2.VideoCapture(args["camera"])

# If still not open, abort with error message
if not cap.isOpened():
    raise SystemExit(f"[ERROR] Could not open webcam index {args['camera']}. Please check your camera connection.")

# Small pause to allow camera sensor exposure to stabilize
time.sleep(1.0)
print("[INFO] Webcam active! Press 'q' while focused on the window to exit.")

prev_time = time.time()
fps = 0.0

# ==============================================================================
# 6. REAL-TIME OBJECT DETECTION LOOP
# ==============================================================================
while True:
    # Read a single frame from the camera stream
    ret, frame = cap.read()
    if not ret or frame is None:
        # If a single frame dropped, try reading again
        continue

    # Calculate real-time Frames Per Second (FPS)
    curr_time = time.time()
    time_diff = curr_time - prev_time
    prev_time = curr_time
    if time_diff > 0:
        fps = 1.0 / time_diff

    # Obtain original image dimensions (height, width)
    (h, w) = frame.shape[:2]

    # Preprocess image into a 4D tensor (blob):
    # - Resizes to fixed spatial resolution (300, 300) required by MobileNet-SSD
    # - Scales pixels by factor 0.007843 (which is 2 / 255)
    # - Performs mean subtraction centering around 127.5
    blob = cv2.dnn.blobFromImage(
        frame, 
        scalefactor=0.007843, 
        size=(300, 300), 
        mean=127.5
    )

    # Set blob as input to the deep neural network
    net.setInput(blob)

    # Run forward-pass inference to generate all detected candidates
    # detections output tensor shape: (1, 1, num_detections, 7)
    detections = net.forward()

    # ==============================================================================
    # 7. PARSE MULTIPLE DETECTIONS & DRAW BOUNDING BOXES
    # ==============================================================================
    # Loop over every detected candidate object found in the frame
    for i in range(detections.shape[2]):
        # Extract the confidence score (probability that the object is real)
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections below our minimum threshold
        if confidence > args["confidence"]:
            # Extract class index
            idx = int(detections[0, 0, i, 1])

            # Safety check: avoid out-of-range index
            if idx >= len(CLASSES):
                continue

            # Scale normalized bounding-box coordinates [0.0 to 1.0] back to full frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Clip bounding box coordinates so they stay within visible screen boundaries
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w - 1, endX), min(h - 1, endY)

            # Format label string and color
            label = f"{CLASSES[idx]}: {confidence * 100:.1f}%"
            color = [int(c) for c in COLORS[idx]]

            # Draw the rectangular bounding box around the detected entity
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            # Determine text vertical position (draw above box if space permits)
            y = startY - 10 if startY - 10 > 15 else startY + 15

            # Draw solid pill behind text for easy readability
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (startX, y - text_h - 4), (startX + text_w, y + 4), color, -1)

            # Draw text label in white
            cv2.putText(
                frame, 
                label, 
                (startX, y), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )

    # Display running FPS counter in top-left corner
    cv2.putText(
        frame, 
        f"FPS: {fps:.1f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 255, 255), 
        2
    )

    # Render frame in window
    cv2.imshow("Multi-Object Detection (Deep Learning)", frame)

    # Listen for key press: wait 1ms; quit if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================================================================
# 8. CLEANUP RESOURCES
# ==============================================================================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Application closed successfully.")