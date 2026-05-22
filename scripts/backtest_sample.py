import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from collections import defaultdict

# Parameters
MAX_SAMPLES = 200

# Paths
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root, 'model', 'alzheimer_model.pth')
test_dir = os.path.join(root, 'dataset', 'test')

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]
folder_names = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
folder_to_idx = {fn: i for i, fn in enumerate(folder_names)}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

if not os.path.isfile(model_path):
    raise SystemExit(f"Model not found at {model_path}. Run train.py first.")

state_dict = torch.load(model_path, map_location=torch.device('cpu'))
hidden = state_dict['fc.0.weight'].shape[0]
dropout = 0.5 if hidden == 128 else 0.4
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, hidden),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden, 4),
)
model.load_state_dict(state_dict)
model.eval()

# Run sample
rows = []
conf_mat = [[0]*len(classes) for _ in classes]
counts = defaultdict(int)
correct = 0
total = 0

for folder in sorted(os.listdir(test_dir)):
    folder_path = os.path.join(test_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    if folder not in folder_to_idx:
        continue
    true_idx = folder_to_idx[folder]
    for fname in sorted(os.listdir(folder_path)):
        if total >= MAX_SAMPLES:
            break
        fpath = os.path.join(folder_path, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            img = Image.open(fpath).convert('RGB')
            inp = transform(img).unsqueeze(0)
            with torch.no_grad():
                out = model(inp)
                probs = torch.nn.functional.softmax(out[0], dim=0)
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())
        except Exception:
            continue
        total += 1
        if pred_idx == true_idx:
            correct += 1
        conf_mat[true_idx][pred_idx] += 1

    if total >= MAX_SAMPLES:
        break

accuracy = 100 * correct / total if total else 0.0
print(f'Sample size: {total} | Accuracy: {accuracy:.2f}%')
print('Confusion matrix (rows=true):')
for row in conf_mat:
    print(row)
