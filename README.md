# 🛡️ JobGuard AI — Explainable Job Fraud Detection

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live-brightgreen)](https://scamguardai.streamlit.app/)
[![API Live](https://img.shields.io/badge/API-Live-brightgreen)](https://explainable-job-scam-risk-detection.vercel.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v1.3-green.svg)](https://fastapi.tiangolo.com/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Explainable ML system detecting job posting scams targeting freshers in India**

## 🎯 Problem

Every year, lakhs of freshers lose ₹5K–50K to fake job scams, plus personal data theft:
- Suspicious job postings flood LinkedIn/job portals
- Binary detection is insufficient — candidates need **explainable risk scoring**
- No transparent system exists to catch these scams

## ✨ Solution

**JobGuard AI** combines:
- **ML Pipeline:** Logistic Regression with TF-IDF (5,000 features) + behavioral signals (urgency language, free-email domains, suspicious salary patterns)
- **Explainability:** Exact SHAP values (`φᵢ = coefᵢ × featureᵢ`, closed-form for linear models) showing which words/features triggered each fraud score
- **Multi-Modal:** Combines NLP + email validation + behavioral analysis
- **Production:** Streamlit UI + FastAPI REST API, deployed serverlessly on Vercel

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Test AUC** | 0.9800 |
| **F1 (Fraud Class)** | 0.88 |
| **Precision** | 92% |
| **Recall** | 88% |
| **CV AUC** | 0.96 ± 0.01 |

## 🏗️ Model Serving Architecture

The trained model doesn't ship as a `.pkl`/pickle file. Instead, the fitted TF-IDF vocabulary, IDF weights, and logistic regression coefficients are exported to a plain-text `model_weights.json`, and inference is reimplemented in dependency-free Python (`src/lite_model.py`) that reproduces sklearn's TF-IDF + logistic regression math exactly — verified against the original model with **zero prediction difference** across test cases.

This was a deliberate serving-layer decision, not a shortcut:
- **No `scikit-learn` at runtime** → smaller, faster serverless cold starts (Vercel's function-size limit was a hard constraint here)
- **No pickle deserialization** → no version-mismatch risk if sklearn versions drift between training and serving, and no binary-file corruption risk from web-based version control workflows
- **Fully auditable weights** → `model_weights.json` is human-readable, diffable in PRs, and easy to reason about for the "explainable" part of the project

The training pipeline (feature engineering, cross-validation, model selection) still uses the full scikit-learn stack — only the **serving path** is pickle-free.

## 🏢 Company / Email Verification

Beyond the ML risk score, two independent due-diligence checks are wired into the API and UI:

- **`POST /verify/company`** — validates a GSTIN or CIN's **format + official checksum**. This runs entirely offline and instantly catches obviously fabricated or malformed numbers, a common signal in fake postings.
- **`POST /verify/email`** — flags free/disposable email providers instantly, plus a live SPF/DMARC DNS lookup on the sending domain.

**On live GSTIN/CIN lookup:** there is no free public government API for either. Real-time verification against the GST or MCA databases requires a paid, KYC-gated API key from a licensed provider (a "GSP" for GSTIN — e.g. Masters India, Cygnet, ClearTax — or a compliance-API vendor like Signzy/Probe42 for MCA/CIN data). This repo ships the **adapter**, not a key: set `GST_LIVE_API_URL` / `GST_LIVE_API_KEY` (or `MCA_LIVE_API_URL` / `MCA_LIVE_API_KEY`) as environment variables and `/verify/company` will call your provider automatically; with no key set, it transparently falls back to the structural check above, so the feature is a genuine drop-in upgrade whenever you have credentials, not a config change that breaks anything today. See `src/company_verification.py` → `verify_company_live()`.

For a placement/interview project, the structural-only version is already a legitimate, defensible signal on its own — say so plainly if asked ("checksum validation, not a live registry lookup") rather than implying more than it does.

## 🎲 Uncertainty & 🧪 A/B Testing

- `include_uncertainty: true` on `/predict` runs a Monte Carlo feature-dropout pass (`src/uncertainty.py`) and returns a 95% credible interval — opt-in, since it adds latency.
- `experiment_name` on `/predict` buckets the request into an A/B arm (`src/ab_testing.py`); `GET /experiments/{name}/summary` and `POST /experiments/{name}/significance` report results. Note: on Vercel's serverless runtime, in-memory experiment logs don't persist across separate invocations — this works best on a long-running process (Docker/local).



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

### 3. Run FastAPI Backend
```bash
cd src
uvicorn fastapi_backend:app --reload --port 8000
```
**Swagger UI at:** http://localhost:8000/docs

*(Optional — only if you have a paid GSP/MCA provider key for live company lookup; skip otherwise, structural validation still works fully without these)*
```bash
export GST_LIVE_API_URL=https://your-gsp-provider.example.com/v1/gstin
export GST_LIVE_API_KEY=your_key_here
```

### 4. Run Streamlit App
```bash
cd ..
export BACKEND_URL=http://localhost:8000   # or your deployed API URL
streamlit run streamlit_app/app.py
```
**Access at:** http://localhost:8501

### 5. Live Deployment
- **API:** https://explainable-job-scam-risk-detection.vercel.app/ (FastAPI on Vercel, serverless)
- **UI:** https://jobguard-ai.streamlit.app/ (Streamlit Cloud, calls the API above via `BACKEND_URL`)

## 📁 Project Structure

```
├── src/
│   ├── fastapi_backend.py     # FastAPI app — /predict, /verify/*, /experiments/*
│   ├── lite_model.py          # Pure-Python TF-IDF + logistic regression inference
│   ├── model_weights.json     # Trained model weights (vocab, idf, coefficients)
│   ├── utils.py                # Feature engineering, SHAP, risk scoring
│   ├── company_verification.py # GSTIN/CIN structural check + pluggable live lookup
│   ├── email_reputation.py     # Free/disposable + SPF/DMARC checks
│   ├── uncertainty.py          # Monte Carlo confidence-interval estimation
│   ├── ab_testing.py           # Experiment bucketing + significance testing
│   └── requirements.txt        # Minimal deps for the deployed API (Vercel picks this up)
├── streamlit_app/
│   ├── app.py                  # Streamlit UI
│   └── requirements.txt
├── requirements.txt            # Full deps (training + Streamlit + OCR)
├── vercel.json                 # Vercel serverless function config
├── docker-compose.yml
└── Dockerfile
```

## 🧪 API Example

```bash
curl -X POST https://explainable-job-scam-risk-detection.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "job_title": "Part Time Typing Work",
    "job_description": "Easy money! Copy paste data, unlimited earnings, registration fee applies, act now!",
    "company_profile": "Contact us at jobs@yahoo.com",
    "salary_range": "Rs 5000-40000 per day"
  }'
```

Returns fraud probability, a 0–100 risk score, top SHAP-driven contributing features, matched scam phrases, and a human-readable recommendation.

## 📄 License

MIT
