"""Faster training on a larger subset — better than old quick run, still faster than train.py."""
import os
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

root_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(root_dir, "dataset", "train")
val_dir = os.path.join(root_dir, "dataset", "val")
model_dir = os.path.join(root_dir, "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "alzheimer_model.pth")

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_full = datasets.ImageFolder(train_dir, transform=train_transform)
val_full = datasets.ImageFolder(val_dir, transform=val_transform)

per_class_train = 1500
per_class_val = 400


def subset_indices(dataset, per_class):
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    indices = []
    for label, idxs in by_class.items():
        random.seed(42 + label)
        indices.extend(random.sample(idxs, min(per_class, len(idxs))))
    return indices


train_dataset = Subset(train_full, subset_indices(train_full, per_class_train))
val_dataset = Subset(val_full, subset_indices(val_full, per_class_val))
print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

label_counts = Counter(train_full.samples[i][1] for i in train_dataset.indices)
num_samples = len(train_dataset)
class_weights = torch.tensor(
    [num_samples / label_counts[i] for i in range(len(train_full.classes))],
    dtype=torch.float32,
).to(device)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 4),
)
for param in model.layer4.parameters():
    param.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam([
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

epochs = 12
best_accuracy = 0.0
print("Training started...")

for epoch in range(epochs):
    model.train()
    correct = total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    train_acc = 100 * correct / total

    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch + 1}/{epochs}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), model_path)
        print("  Best model saved")

print(f"Done. Best validation accuracy: {best_accuracy:.2f}%")
