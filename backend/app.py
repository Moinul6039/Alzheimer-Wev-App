import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

base_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.normpath(os.path.join(base_dir, "..", "frontend"))
app = Flask(__name__, static_folder=frontend_dir, static_url_path="/static")
CORS(app)

classes = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]

model_path = os.path.normpath(os.path.join(base_dir, "..", "model", "alzheimer_model.pth"))

if not os.path.isfile(model_path):
    raise FileNotFoundError(
        f"Model weights not found: {model_path}. Run train.py first and wait until it saves the model."
    )

def build_model(state_dict):
    """Support both old (128) and new (256) saved checkpoints."""
    hidden = state_dict["fc.0.weight"].shape[0]
    dropout = 0.5 if hidden == 128 else 0.4
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 4),
    )
    model.load_state_dict(state_dict)
    return model


state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model = build_model(state_dict)
model.eval()

# Must match train.py preprocessing (including ImageNet normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.route("/")
def home():
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file part named 'file' in request"}), 400

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

    return jsonify({
        "prediction": classes[predicted.item()],
        "confidence": round(confidence.item() * 100, 2),
    })


if __name__ == "__main__":
    app.run(debug=True)
