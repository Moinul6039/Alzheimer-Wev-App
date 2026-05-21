import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
root_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(root_dir, "dataset", "train")
val_dir = os.path.join(root_dir, "dataset", "val")
model_dir = os.path.join(root_dir, "model")
os.makedirs(model_dir, exist_ok=True)

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Load ResNet18
model = models.resnet18(pretrained=True)

# Change Final Layer
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss}")

# Save Model
torch.save(model.state_dict(), os.path.join(model_dir, "alzheimer_model.pth"))
print("Model Saved")
