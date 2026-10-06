# yolo_webcam_detection.py

# ==============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================================================================
import cv2                        # OpenCV for video stream capture and rendering
import time                       # Benchmark inference latency and FPS calculation
from ultralytics import YOLO      # Ultralytics modern object detection framework

# ==============================================================================
# 2. INITIALIZE PRE-TRAINED YOLO MODEL
# ==============================================================================
# "yolov8n.pt" (Nano) will automatically download on the first run (~6 MB).
# It is trained on the COCO dataset covering 80 everyday categories:
# (person, bottle, chair, laptop, cell phone, backpack, cup, etc.)
print("[INFO] Loading YOLOv8 nano model...")
model = YOLO("yolov8n.pt")

# ==============================================================================
# 3. INITIALIZE WEBCAM WITH DIRECTSHOW
# ==============================================================================
print("[INFO] Initializing webcam...")
# cv2.CAP_DSHOW prevents camera startup lag and frozen frame captures on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Fallback to default index if DirectShow fails
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("[ERROR] Could not open webcam index 0. Check camera connections.")

# Let camera exposure stabilize
time.sleep(1.0)
print("[INFO] Camera stream live! Press 'q' while focused on the window to exit.")

# Metrics for running FPS calculation
prev_time = time.time()
fps = 0.0

# ==============================================================================
# 4. REAL-TIME OBJECT DETECTION LOOP
# ==============================================================================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    # Calculate real-time FPS
    curr_time = time.time()
    time_diff = curr_time - prev_time
    prev_time = curr_time
    if time_diff > 0:
        fps = 1.0 / time_diff

    # Run deep learning object detection on the current frame:
    # - conf=0.5: Discards detections with lower than 50% probability
    # - verbose=False: Suppresses repetitive console output per frame
    results = model.track(frame, conf=0.5, persist=True, verbose=False)

    # Plot bounding boxes, class labels, and percentages directly onto the image
    annotated_frame = results[0].plot()

    # Draw real-time FPS counter in the top-left corner
    cv2.putText(
        annotated_frame, 
        f"FPS: {fps:.1f}", 
        (15, 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 255, 255), 
        2
    )

    # Display window
    cv2.imshow("Real-Time YOLOv8 Object Detection", annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================================================================
# 5. CLEANUP RESOURCES
# ==============================================================================
cap.release()
cv2.destroyAllWindows()
print("[INFO] Stream terminated. Clean exit.")