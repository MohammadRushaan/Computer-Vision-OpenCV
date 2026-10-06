# classify_webcam.py

import cv2
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Load classes
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, len(class_names))
)
model.load_state_dict(torch.load("custom_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

prev_time = time.time()
print("[INFO] Target Box Active. Hold objects inside the green square. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
    prev_time = curr_time

    h, w = frame.shape[:2]

    # Define a 300x300 center Region of Interest (ROI)
    box_size = 300
    x1 = int((w - box_size) / 2)
    y1 = int((h - box_size) / 2)
    x2 = x1 + box_size
    y2 = y1 + box_size

    # Crop ONLY the center box for model inference
    roi = frame[y1:y2, x1:x2]

    # Convert ROI to PIL image and run forward pass
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_roi)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)

    label = class_names[predicted_idx.item()]
    conf_score = confidence.item() * 100

    # Draw target box guidelines
    box_color = (0, 255, 0) if (label != "background" and conf_score > 60) else (180, 180, 180)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Status banner
    cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)

    display_text = f"{label}: {conf_score:.1f}%" if label != "background" else "Place object inside box"
    text_color = (0, 255, 0) if (label != "background" and conf_score > 60) else (200, 200, 200)

    cv2.putText(frame, display_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Custom Classifier (Target ROI)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()