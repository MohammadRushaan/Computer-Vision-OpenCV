import os
import cv2

# =====================================================================
# 1. FILE PATHS & SETUP
# =====================================================================
# Path to the input image file you want to test
IMAGE_PATH = "sample.jpg"

# Path to OpenCV's official pre-trained ONNX face detection model (YuNet)
# YuNet is a lightweight, high-performance face detection model designed for edge devices.
MODEL_PATH = "face_detection_yunet_2023mar.onnx"

# Verify that both required files exist before continuing
if not os.path.exists(IMAGE_PATH):
  raise FileNotFoundError(f"Input image not found: {IMAGE_PATH}")
if not os.path.exists(MODEL_PATH):
  raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# =====================================================================
# 2. IMAGE LOADING & PREPARATION
# =====================================================================
# cv2.imread loads the image from disk into a NumPy array in BGR format
image = cv2.imread(IMAGE_PATH)

# Extract spatial dimensions: height, width, and color channels
# Unlike older Caffe models that require rigid 300x300 input resizing,
# YuNet dynamically takes the exact dimensions of your input image.
h, w, _ = image.shape

# =====================================================================
# 3. INITIALIZE OPENCV'S FaceDetectorYN
# =====================================================================
# cv2.FaceDetectorYN.create initializes the ONNX-backed face detection object.
# Parameters:
# - model: Path to the .onnx model weights file.
# - config: Optional text configuration file (empty string "" for ONNX).
# - input_size: A tuple of (width, height) matching your input image dimensions.
# - score_threshold: Minimum confidence score (0.0 to 1.0) to filter false positives.
# - nms_threshold: Non-Maximum Suppression threshold (0.0 to 1.0) to remove overlapping duplicate boxes.
# - top_k: Maximum number of bounding boxes to keep before Non-Maximum Suppression.
detector = cv2.FaceDetectorYN.create(
    model=MODEL_PATH,
    config="",
    input_size=(w, h),  # Note: input_size takes (width, height)
    score_threshold=0.6,  # 60% confidence filter (raise to reduce false positives)
    nms_threshold=0.3,  # Overlap suppression threshold
    top_k=5000,
)

# =====================================================================
# 4. RUN INFERENCE (DETECT FACES)
# =====================================================================
# detector.detect(image) performs the forward pass through the ONNX network.
# It returns:
# - retval: Boolean status (or 0/1 indicator).
# - faces: A 2D NumPy array where each row represents one detected face.
#   If no faces are found, faces is None.
_, faces = detector.detect(image)

# =====================================================================
# 5. PARSE DETECTIONS & ANNOTATE THE IMAGE
# =====================================================================
# Only loop if at least one face was detected
if faces is not None:
  print(f"[INFO] Found {len(faces)} face(s).")

  for face in faces:
    # -----------------------------------------------------------------
    # A. Bounding Box Extraction
    # -----------------------------------------------------------------
    # Indices [0:4] contain the bounding box: [x, y, width, height]
    box = face[0:4].astype(int)
    x, y, box_w, box_h = box

    # Index [-1] (last column, index 14) holds the confidence score (0.0 to 1.0)
    confidence = face[-1]

    # Draw the bounding rectangle:
    # - Top-left corner: (x, y)
    # - Bottom-right corner: (x + box_w, y + box_h)
    # - Color: (0, 255, 0) in BGR -> Bright Green
    # - Thickness: 2 pixels
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

    # -----------------------------------------------------------------
    # B. Confidence Label Rendering
    # -----------------------------------------------------------------
    label_text = f"{confidence * 100:.1f}%"
    label_y = y - 10 if y - 10 > 10 else y + 15

    # Render text label directly above the bounding box
    cv2.putText(
        image,
        label_text,
        (x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=2,
    )

    # -----------------------------------------------------------------
    # C. Facial Landmark Extraction (5 Key Points)
    # -----------------------------------------------------------------
    # Indices [4:14] contain (x, y) coordinates for 5 facial landmarks:
    # 1. Right eye, 2. Left eye, 3. Nose tip, 4. Right mouth corner, 5. Left mouth corner
    landmarks = face[4:14].reshape((5, 2)).astype(int)

    # Draw a small red circle for each facial landmark
    for px, py in landmarks:
      cv2.circle(
          img=image,
          center=(px, py),
          radius=2,  # Circle radius in pixels
          color=(0, 0, 255),  # (0, 0, 255) in BGR -> Bright Red
          thickness=2,
      )
else:
  print("[INFO] No faces detected in the image.")

# =====================================================================
# 6. DISPLAY RESULT ON SCREEN
# =====================================================================
# Create a desktop GUI window and display the annotated image
cv2.imshow("YuNet Face Detection (ONNX)", image)

# cv2.waitKey(0) pauses execution indefinitely until any keyboard key is pressed
cv2.waitKey(0)

# Release GUI window memory
cv2.destroyAllWindows()