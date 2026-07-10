# 🛡️ JobGuard AI — Explainable Job Fraud Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-brightgreen)](https://jobguard-ai.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v1.2-green.svg)](https://fastapi.tiangolo.com/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Explainable ML system detecting job posting scams targeting freshers in India**

## 🎯 Problem

Every year, lakhs of freshers lose ₹5K-50K to fake job scams, plus personal data theft:
- Suspicious job postings flood LinkedIn/job portals
- Binary detection is insufficient — candidates need **explainable risk scoring**
- No transparent system exists to catch these scams

## ✨ Solution

**JobGuard AI** combines:
- **ML Pipeline:** Logistic Regression (LR) with TF-IDF (5000 features) + behavioral signals
- **Explainability:** Exact SHAP values showing which words/features triggered fraud detection
- **Multi-Modal:** Combines NLP + email validation + behavioral analysis
- **Production:** Streamlit UI + FastAPI REST API + Docker containerization

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.9800 |
| **F1 (Fraud Class)** | 0.88 |
| **Precision** | 92% |
| **Recall** | 88% |
| **CV AUC** | 0.96 ± 0.01 |

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/AkashMs24/explainable-job-scam-risk-detection-system.git
cd explainable-job-scam-risk-detection-system
```

### 2. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# For OCR (optional):
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS: brew install tesseract
```

### 3. Run Streamlit App
```bash
streamlit run src/app.py
```
**Access at:** http://localhost:8501

### 4. Run FastAPI Backend (Optional)
```bash
python src/fastapi_backend.py
```
**Swagger UI at:** http://localhost:8000/docs

### 5. Docker Deployment
```bash
docker-compose build
docker-compose up -d
```

## 📁 Project Structure
