import os
import cv2

# =====================================================================
# 1. MODEL SETUP & WEBCAM INITIALIZATION
# =====================================================================
MODEL_PATH = "face_detection_yunet_2023mar.onnx"

if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Initialize OpenCV video capture
# 0 targets the primary webcam/integrated camera on your PC
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  raise RuntimeError("Could not access the webcam. Check camera permissions.")

# Read the native resolution of your webcam stream
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[INFO] Webcam initialized at resolution: {frame_w}x{frame_h}")

# =====================================================================
# 2. CONFIGURE THE YUNET DETECTOR
# =====================================================================
detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(
        frame_w,
        frame_h,
    ),  # Must match the webcam frame dimensions exactly
    score_threshold=0.6,  # Ignore detections with confidence < 60%
    nms_threshold=0.3,  # Suppress duplicate overlapping boxes
    top_k=5000,
)

print("[INFO] Starting real-time face detection... Press 'q' to quit.")

# =====================================================================
# 3. REAL-TIME VIDEO PROCESSING LOOP
# =====================================================================
while True:
  # Grab a single frame from the camera buffer
  # ret is a boolean indicating success; frame is the image array
  ret, frame = cap.read()
  if not ret:
    print("[WARN] Failed to grab frame from camera stream.")
    break

  # Forward pass through the ONNX detector for this specific frame
  _, faces = detector.detect(frame)

  # Annotate detections if any faces are visible
  if faces is not None:
    for face in faces:
      # Extract bounding box [x, y, w, h] and confidence score
      box = face[0:4].astype(int)
      confidence = face[-1]
      x, y, w, h = box

      # Draw bounding box (Green)
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

      # Draw confidence label
      label = f"{confidence * 100:.1f}%"
      label_y = y - 8 if y - 8 > 8 else y + 15
      cv2.putText(
          frame,
          label,
          (x, label_y),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

      # Extract and draw 5 facial landmarks (Red circles)
      landmarks = face[4:14].reshape((5, 2)).astype(int)
      for px, py in landmarks:
        cv2.circle(frame, (px, py), 2, (0, 0, 255), 2)

  # Display the annotated frame in real-time
  cv2.imshow("Webcam Face Detection (YuNet ONNX)", frame)

  # cv2.waitKey(1) waits 1 millisecond for key press
  # If the user presses 'q', break out of the loop and terminate
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# =====================================================================
# 4. RESOURCE CLEANUP
# =====================================================================
# Release the camera hardware lock so other applications can use it
cap.release()

# Close all OpenCV display windows
cv2.destroyAllWindows()
print("[INFO] Webcam stream stopped successfully.")