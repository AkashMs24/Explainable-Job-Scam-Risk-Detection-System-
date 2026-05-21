<div align="center">

# SCAMGUARD AI

### Explainable Job Scam Risk Detection using NLP & Machine Learning

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://explainable-job-scam-risk-detection-system.streamlit.app/)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)

</div>

---

## The problem

Every year, thousands of freshers apply to fake job postings — losing money, time,
and sometimes personal data.

Most detection tools give a binary answer: *real* or *fake*.  
That is not enough. A fresher needs to know **how risky**, and **exactly why**.

---

## What SCAMGUARD AI does differently

This is not a classifier. It is a **risk scoring and explanation engine.**

- Assigns a **0–100 scam risk score** — not just fake/real
- Combines **NLP text analysis** with **behavioral fraud indicators**
- Explains every decision in plain English — no black box
- Prioritizes **recall over precision** — missing a scam is far more dangerous than flagging a real job

---

## Example

**Input:**Job title   : Data Entry Intern
Description : Urgent hiring! Work from home. Limited slots. Apply immediately.
Contact     : gmail.com address
Salary      : Not mentioned

**Output:**Scam Risk Score: 83 / 100  ⚠️ HIGH RISKWhy:
→ Urgency-driven language detected         (+31 risk)
→ Salary information missing               (+24 risk)
→ Free email domain (Gmail) used           (+18 risk)
→ Vague job description                    (+10 risk)

---

## How it works

**1. NLP feature extraction**
- TF-IDF vectorization of job title + description
- Urgency language detection ("urgent", "limited slots", "apply immediately")
- Free email domain flagging (Gmail, Yahoo, Hotmail)
- Description length and structure analysis

**2. Model**
- Logistic Regression with class-weight balancing
- Trained on ~18,000 job postings (Kaggle — Real or Fake Job Posting dataset)
- Optimized for recall — catching scams matters more than avoiding false alarms

**3. Risk scoring engine**
- Combines ML probability with rule-based behavioral indicators
- Outputs a 0–100 score with per-feature contribution breakdown

**4. Explainability layer**
- Feature importance mapped to human-readable explanations
- Every score is justified, not just produced

---

## Dataset

- Source: Kaggle — Real or Fake Job Posting Prediction
- 18,000 job postings · binary label (fraudulent: 0/1)
- Features: job title, description, company profile, requirements, metadata
- Dataset not included in this repo

---

## Key design decisions

**Risk score over binary label** — a score of 83 is actionable; "fake" is not. Users can set their own risk tolerance.

**Recall-first optimization** — the cost of missing a scam (financial loss, data theft) far outweighs the cost of flagging a real job (minor inconvenience).

**Behavioral signals + NLP** — text alone misses patterns. Combining urgency language, email domains, and missing salary fields catches scams that slip past pure text models.

**Decision-support, not decision-maker** — the system flags and explains. The user verifies and decides.

---

## Stack

`Python` `scikit-learn` `TF-IDF` `Pandas` `NumPy` `Streamlit` `Joblib`

---

## Project structureExplainable-Job-Scam-Risk-Detection-System/
├── app.py                              # Streamlit application
├── fraud_model.pkl                     # Trained model
├── tfidf_vectorizer.pkl                # TF-IDF vectorizer
├── feature_names.pkl                   # Feature names for explainability
├── 02_feature_engineering_and_model.py # Training pipeline
├── eda.py                              # Exploratory analysis
├── explainability_and_insights.py      # SHAP + explanation layer
├── requirements.txt
└── data/
└── fake_job_postings.csv

---

## Run locally

---

## Run locally

```bash
git clone https://github.com/AkashMs24/Explainable-Job-Scam-Risk-Detection-System-.git
cd Explainable-Job-Scam-Risk-Detection-System-
pip install -r requirements.txt
streamlit run app.py
```

---

## What's next

- BERT-based text embeddings for richer NLP features
- Company registration cross-verification via API
- Browser extension for real-time job board scanning
- Multilingual support (Hindi, Kannada)

---

## Related projects

- [Fraud Detection System](https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System) — cost-sensitive financial fraud detection
- [Employee Attrition XAI](https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning) — SHAP-powered HR risk scoring
- [Decision Intelligence System](https://github.com/AkashMs24/Decisioniq-ai-business-intelligence) — ML + LLM business platform

---

<div align="center">

Built by **Akash M S** · Presidency University, Bengaluru  
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com

</div>

<div align="center">

Built by **Akash M S** · Presidency University, Bengaluru  
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com

</div>
