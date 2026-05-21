# Deploy Alzheimer Web App

## Before deploy

1. Train and save `model/alzheimer_model.pth`
2. Push code + model to GitHub (dataset is not required for the live app)

```bash
git add .
git commit -m "Add deployment config"
git push
```

---

## Option A: Render.com (recommended, free tier)

1. Sign up at [render.com](https://render.com)
2. **New → Web Service** → connect `Moinul6039/Alzheimer-Wev-App`
3. Settings:
   - **Runtime:** Docker
   - **Branch:** main
   - **Plan:** Free
4. Deploy — first build may take 10–15 minutes (PyTorch image)
5. Open the URL Render gives you (e.g. `https://alzheimer-web-app.onrender.com`)

Free tier sleeps after inactivity; first request may be slow (~30s).

---

## Option B: Railway

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select this repo; Railway detects `Dockerfile`
3. Deploy and copy the public URL

---

## Option C: Any VPS (Ubuntu)

```bash
git clone https://github.com/Moinul6039/Alzheimer-Wev-App.git
cd Alzheimer-Wev-App
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
cd backend
gunicorn --bind 0.0.0.0:8080 --workers 1 --timeout 120 app:app
```

Use Nginx as reverse proxy + HTTPS (Certbot) for production.

---

## Local production test

```bash
docker build -t alzheimer-app .
docker run -p 8080:8080 alzheimer-app
```

Open http://localhost:8080

---

## Notes

- **Do not deploy** the `dataset/` folder — too large; only `model/alzheimer_model.pth` is needed
- PyTorch makes the image ~1–2 GB; free hosts may have memory limits
- This is a demo app, not certified medical software
