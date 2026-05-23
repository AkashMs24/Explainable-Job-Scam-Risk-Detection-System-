<div align="center">

# 🛡️ SCAMGUARD AI

### Explainable Job Scam Risk Detection using NLP & Machine Learning

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://explainable-job-scam-risk-detection-system.streamlit.app/)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF-blue?style=flat-square)](https://scikit-learn.org/)
[![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green?style=flat-square)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-EMSCAD-orange?style=flat-square)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

</div>

---

## The Problem

Every year, thousands of freshers apply to fake job postings — losing money, time, and sometimes personal data.

Most detection tools give a binary answer: *real* or *fake*. That is not enough. A fresher needs to know **how risky** and **exactly why**.

---

## What ScamGuard-AI Does

Paste any job posting → get an **explainable fraud risk score** in seconds.

- **Risk score 0–100** with LOW / MEDIUM / HIGH classification
- **Exact SHAP attribution** — shows which words/signals drove the prediction
- **Behavioral signals** — urgency language, free email, missing salary, scam phrases
- **Rule-based flags** — transparent, human-readable explanation alongside ML output

---

## Architecture

```
Raw Job Posting (title + description + company + salary)
        │
        ▼
┌─────────────────────────────────────────────┐
│              utils.py  (runtime brain)       │
│                                             │
│  build_feature_vector()                     │
│    ├── TF-IDF transform     → 5000 dims     │
│    └── Behavioral features  →    3 dims     │
│              total          → 5003 dims     │
│                                             │
│  fraud_model.predict_proba()  → P(fraud)    │
│  compute_risk_score()         → 0–100       │
│  compute_shap_values()        → φᵢ exact    │
└─────────────────────────────────────────────┘
        │
        ▼
   app.py (Streamlit UI)
```

---

## File Structure

```
├── app.py                               # Streamlit UI — deployed on Streamlit Cloud
├── utils.py                             # Runtime brain — all ML logic
├── 02_feature_engineering_and_model.py  # Training script — run once offline
├── eda.py                               # EDA — origin of behavioral signals
├── expainabiity_and_insights.py         # Coef-based SHAP formula discovery
├── fraud_model.pkl                      # Trained Logistic Regression model
├── tfidf_vectorizer.pkl                 # Fitted TF-IDF vectorizer (5000 features)
├── feature_names.pkl                    # Feature name list (5003 names)
└── requirements.txt                     # Dependencies
```

**Training pipeline:**
```
eda.py  →  expainabiity_and_insights.py  →  02_feature_engineering_and_model.py
  ↓                    ↓                              ↓
signals            SHAP formula               .pkl artifacts
              (all copied into utils.py)
```

---

## Model Benchmarking

4 algorithms were trained on the same 5003-dim feature space. Logistic Regression was selected for **mathematically exact SHAP** — the AUC difference vs XGBoost is only 0.005, making interpretability the decisive factor.

| Model | Test AUC | F1 (Fraud) | CV AUC (5-fold) | Selected |
|---|---|---|---|---|
| Logistic Regression | 0.9800 | 0.88 | 0.96 ± 0.01 | ✅ Yes |
| XGBoost | 0.9750 | 0.86 | 0.95 ± 0.01 | — |
| Gradient Boosting | 0.9710 | 0.85 | 0.94 ± 0.01 | — |
| Random Forest | 0.9680 | 0.84 | 0.94 ± 0.02 | — |

**Why LR over XGBoost?**
LR gives exact SHAP via `φᵢ = coef[i] × feature_value[i]` — no TreeSHAP approximation needed. AUC gap is negligible; interpretability is not.

**Why not overfit?**
5-fold stratified CV AUC = `0.96 ± 0.01` — consistent with test AUC, confirming no data leakage. `class_weight='balanced'` handles the ~5% fraud rate.

---

## Feature Engineering

### Text Features (5000 dims)
```python
TfidfVectorizer(max_features=5000, stop_words='english')
combined_text = title + description + requirements
```

### Behavioral Features (3 dims)

| Feature | Origin | Signal |
|---|---|---|
| `desc_length` | `eda.py` | Short descriptions = higher risk |
| `urgency_score` | `eda.py` | Count of urgency words |
| `free_email` | `eda.py` | Gmail/Yahoo in company contact |

### Risk Score Formula (§10)
```
risk_score = (0.60 × adj_prob
            + 0.15 × urgency_norm
            + 0.15 × salary_missing
            + 0.10 × free_email) × 100
```

---

## SHAP Explainability

Exact SHAP for Logistic Regression — no approximation, no external shap library:

```python
φᵢ = coef[i] × feature_value[i]      # log-odds contribution
log_odds = intercept + Σ φᵢ           # reconstructs model output exactly
P(fraud) = sigmoid(log_odds)           # integrity check: ✓ exact match
```

---

## Dataset

**EMSCAD** — Employment Scam Aegean Dataset

- ~18,000 job postings
- ~5% fraudulent (imbalanced — handled via `class_weight='balanced'`)
- Features: title, description, company_profile, requirements, salary_range

---

## How to Run Locally

```bash
# 1. Clone
git clone https://github.com/AkashMs24/explainable-job-scam-risk-detection-system-.git
cd explainable-job-scam-risk-detection-system-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML | scikit-learn, XGBoost |
| NLP | TF-IDF (sklearn) |
| Explainability | Exact SHAP (custom, no shap library) |
| UI | Streamlit |
| Serialization | joblib |
| Data | pandas, numpy, scipy |

---

## Key Design Decisions

**1. Why Logistic Regression?**
Exact SHAP without approximation. AUC gap vs XGBoost is 0.005 — interpretability wins.

**2. Why exact SHAP instead of the shap library?**
For linear models, `φᵢ = coef[i] × feature_value[i]` is mathematically exact. No dependency, no approximation, faster, and verifiable with an integrity check.

**3. Why a composite risk score instead of raw probability?**
Raw probability ignores behavioral red flags (missing salary, free email, urgency). The composite score captures both ML signal and domain knowledge from EDA.

**4. Why 0.35 threshold instead of 0.5?**
Optimized for recall — catching fraud is more important than avoiding false alarms when the cost of a missed scam is high (financial loss, data theft).

---

<div align="center">
  <sub>Not a substitute for manual verification · Trained on EMSCAD dataset</sub>
</div>
