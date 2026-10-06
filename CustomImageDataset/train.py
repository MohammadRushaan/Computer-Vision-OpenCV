# train.py

# ==============================================================================
# 1. IMPORT REQUIRED LIBRARIES
# ==============================================================================
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==============================================================================
# 2. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DATA_DIR = "dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[INFO] Training on hardware device: {DEVICE}")

# ==============================================================================
# 3. DATA PREPROCESSING & AUGMENTATION PIPELINE
# ==============================================================================
# Data augmentation helps prevent overfitting on small scraped datasets
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load images from folder structure (class per subfolder)
full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=data_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"[INFO] Classes detected: {class_names} (Total: {num_classes})")
print(f"[INFO] Total images found: {len(full_dataset)}")

# Train / Validation split (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Save class labels list to a text file for inference
with open("classes.txt", "w") as f:
    for cls in class_names:
        f.write(f"{cls}\n")

# ==============================================================================
# 4. INITIALIZE PRE-TRAINED MODEL (TRANSFER LEARNING)
# ==============================================================================
print("[INFO] Initializing MobileNetV2 backbone...")
# Replace the model setup in train.py with this:
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze lower layers, but keep the top feature block trainable
for param in model.features[:-4].parameters():
    param.requires_grad = False
for param in model.features[-4:].parameters():
    param.requires_grad = True

# Custom classifier head with dropout to prevent overfitting
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, num_classes)
)
model = model.to(DEVICE)

# Train with a lower learning rate across trainable layers


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=0.0003
)

# ==============================================================================
# 5. TRAINING LOOP
# ==============================================================================
print("[INFO] Starting training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data).item()
        total_train += labels.size(0)

    epoch_loss = running_loss / train_size
    epoch_acc = (correct_train / total_train) * 100

    # Validation pass
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct_val += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)

    val_acc = (correct_val / total_val) * 100 if total_val > 0 else 0.0

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%")

total_duration = time.time() - start_time
print(f"[INFO] Training finished in {total_duration:.1f}s")

# ==============================================================================
# 6. SAVE MODEL WEIGHTS
# ==============================================================================
torch.save(model.state_dict(), "custom_model.pth")
print("[INFO] Trained weights saved successfully to 'custom_model.pth'")