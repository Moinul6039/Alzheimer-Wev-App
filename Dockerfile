FROM python:3.11-slim

WORKDIR /app

# PyTorch CPU only (smaller image, faster deploy)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt gunicorn

COPY backend/ backend/
COPY frontend/ frontend/
COPY model/alzheimer_model.pth model/alzheimer_model.pth

WORKDIR /app/backend
ENV PORT=8080
EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 120 app:app
