import cv2

# 1. Load image and extract dimensions
image_path = "sample.jpg"
image = cv2.imread(image_path)
h, w, _ = image.shape

# 2. Initialize OpenCV's FaceDetectorYN with the ONNX model
model_path = "face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(w, h),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000,
)

# 3. Detect faces
_, faces = detector.detect(image)

# 4. Draw bounding boxes and landmarks
if faces is not None:
  for face in faces:
    # Bounding box coordinates: [x, y, w, h]
    box = face[0:4].astype(int)
    confidence = face[-1]

    # Draw face box
    cv2.rectangle(
        image, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2
    )

    # Label with confidence
    text = f"{confidence * 100:.1f}%"
    cv2.putText(
        image,
        text,
        (box[0], box[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # Landmarks: 5 points (right eye, left eye, nose tip, right mouth corner, left mouth corner)
    landmarks = face[4:14].reshape((5, 2)).astype(int)
    for px, py in landmarks:
      cv2.circle(image, (px, py), 2, (0, 0, 255), 2)

# 5. Show result
cv2.imshow("YuNet Face Detection (ONNX)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()