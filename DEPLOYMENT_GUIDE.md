# 🚀 Deployment Guide

## Option 1: Streamlit Cloud (Recommended - 5 min)

```bash
git add .
git commit -m "v1.2: production ready"
git push origin main
```

Go to https://share.streamlit.app → New app → Select repo → Deploy

Live at: `https://[username]-jobguard-ai.streamlit.app/`

## Option 2: Docker Compose (Local)

```bash
docker-compose build
docker-compose up -d

# Access:
# UI: http://localhost:8501
# API: http://localhost:8000/docs
```

## Option 3: Railway (API Only)

1. Connect GitHub repo
2. Set environment: `API_PORT=8000`
3. Run command: `uvicorn src.fastapi_backend:app --host 0.0.0.0`
4. Deploy

## Environment Variables

```bash
# .env file
STREAMLIT_SERVER_PORT=8501
API_PORT=8000
LOG_LEVEL=info
```

## Troubleshooting

**Model files not found:**
```bash
# Ensure PKL files in src/ directory
ls -la src/*.pkl
```

**OCR not working:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

**Port already in use:**
```bash
# Change port
streamlit run src/app.py --server.port 8502
```

## Testing

```bash
pytest tests/ -v --cov=src
```

## Production Checklist

- [ ] All tests passing
- [ ] Docker image builds
- [ ] Environment variables set
- [ ] Model files present
- [ ] Documentation updated
- [ ] Pushed to GitHub
- [ ] Deployed to cloud
