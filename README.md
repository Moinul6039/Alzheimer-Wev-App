# Alzheimer Web App

MRI brain scan classifier that predicts dementia stage using a ResNet-18 model and a simple web interface.

## Classes

| Label | Meaning |
|-------|---------|
| Non Demented | No dementia |
| Very Mild Demented | Very mild stage |
| Mild Demented | Mild stage |
| Moderate Demented | Moderate stage |

## Project structure

```
Alzheimer-Wev-App/
├── backend/          Flask API + serves frontend
├── frontend/         React UI (Tailwind CSS)
├── model/            Trained weights (alzheimer_model.pth)
├── train.py          Train ResNet-18 on dataset
├── split_dataset.py  Split images into train/val/test
└── dataset/          MRI images (not in repo — too large)
```

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Dataset (local only)

Place class folders under `dataset/` with names:

- `MildDemented`
- `ModerateDemented`
- `NonDemented`
- `VeryMildDemented`

Then split and train:

```bash
python split_dataset.py
python train.py
```

The repo includes a pre-trained `model/alzheimer_model.pth`, so you can run the app without retraining.

### 3. Run the app

```bash
cd backend
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

## API

`POST /predict` — multipart form field `file` (image)

Response:

```json
{
  "prediction": "Mild Demented",
  "confidence": 87.42
}
```

## Tech stack

- **ML:** PyTorch, torchvision (ResNet-18)
- **Backend:** Flask
- **Frontend:** React 18, Tailwind CSS (CDN)

## Disclaimer

For research and educational use only — not a substitute for professional medical diagnosis.
