import cv2

model_path = "face_detection_yunet_2023mar.onnx"
cap = cv2.VideoCapture(0)

# Read camera frame dimensions
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detector = cv2.FaceDetectorYN.create(
    model=model_path,
    config="",
    input_size=(frame_w, frame_h),
    score_threshold=0.6,
    nms_threshold=0.3,
)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
  ret, frame = cap.read()
  if not ret:
    break

  # Detect
  _, faces = detector.detect(frame)

  if faces is not None:
    for face in faces:
      box = face[0:4].astype(int)
      confidence = face[-1]

      cv2.rectangle(
          frame,
          (box[0], box[1]),
          (box[0] + box[2], box[1] + box[3]),
          (0, 255, 0),
          2,
      )
      cv2.putText(
          frame,
          f"{confidence * 100:.1f}%",
          (box[0], box[1] - 8),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 255, 0),
          2,
      )

  cv2.imshow("Webcam Face Detection (YuNet ONNX)", frame)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()