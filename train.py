import os
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# =========================
# Device & paths
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

root_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(root_dir, "dataset", "train")
val_dir = os.path.join(root_dir, "dataset", "val")
model_dir = os.path.join(root_dir, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "alzheimer_model.pth")

# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(12),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

print("Loading datasets...")
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# Class weights — helps when ModerateDemented has fewer images
label_counts = Counter(label for _, label in train_dataset.samples)
num_samples = len(train_dataset)
class_weights = torch.tensor(
    [num_samples / label_counts[i] for i in range(len(train_dataset.classes))],
    dtype=torch.float32,
).to(device)
print("Class weights:", {train_dataset.classes[i]: round(class_weights[i].item(), 3) for i in range(4)})

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# =========================
# Model
# =========================
print("Loading ResNet18 (pretrained)...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 4),
)

# Unfreeze last ResNet block for better accuracy (fine-tuning)
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# Lower LR for backbone, higher for new head
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# =========================
# Train / validate helpers
# =========================
def run_epoch(loader, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train_mode:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


# =========================
# Training loop
# =========================
epochs = 25
patience = 5
epochs_without_improve = 0
best_accuracy = 0.0

print("Training started...")
for epoch in range(epochs):
    train_loss, train_acc = run_epoch(train_loader, train_mode=True)
    val_loss, val_acc = run_epoch(val_loader, train_mode=False)
    scheduler.step(val_acc)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train loss: {train_loss:.4f} acc: {train_acc:.2f}% | "
        f"Val loss: {val_loss:.4f} acc: {val_acc:.2f}%"
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        epochs_without_improve = 0
        torch.save(model.state_dict(), model_path)
        print(f"  Best model saved ({val_acc:.2f}%)")
    else:
        epochs_without_improve += 1
        if epochs_without_improve >= patience:
            print(f"Early stopping — no improvement for {patience} epochs.")
            break

print(f"Training complete. Best validation accuracy: {best_accuracy:.2f}%")
print(f"Model file: {model_path}")
