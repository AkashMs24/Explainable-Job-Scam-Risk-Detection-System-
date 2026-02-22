import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(
    page_title="SCAMGUARD-AI",
    layout="centered"
)

# =====================================================
# PATHS (MATCH YOUR PROJECT STRUCTURE)
# =====================================================
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

fraud_model = joblib.load(BASE_DIR / "fraud_model.pkl")
tfidf_vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
feature_names = joblib.load(BASE_DIR / "feature_names.pkl")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🛡️ SCAMGUARD-AI")
st.sidebar.caption("Explainable Job Scam Risk Intelligence System")

st.sidebar.markdown("---")
st.sidebar.write("• NLP-based fraud detection")
st.sidebar.write("• Behavioral scam indicators")
st.sidebar.write("• Risk scoring (0–100)")
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ Decision-support system.\n"
    "Always verify job offers manually."
)

# =====================================================
# MAIN UI
# =====================================================
st.title("🛡️ SCAMGUARD-AI")
st.caption("Protecting Freshers from Fraudulent Job & Internship Scams")

st.markdown("### 📥 Job Posting Details")

job_title = st.text_input("Job Title")
job_description = st.text_area("Job Description")
company_profile = st.text_area("Company Profile / Contact Info")
salary_range = st.text_input("Salary Range (optional)")

# =====================================================
# FEATURE ENGINEERING (MATCH TRAINING)
# =====================================================
urgency_words = [
    "urgent", "immediate", "limited",
    "apply fast", "hurry", "few slots", "act now"
]

free_domains = [
    "gmail.com", "yahoo.com",
    "outlook.com", "hotmail.com"
]

def urgency_score(text):
    text = str(text).lower()
    return sum(word in text for word in urgency_words)

def free_email_flag(text):
    text = str(text).lower()
    return int(any(domain in text for domain in free_domains))

# =====================================================
# RISK ANALYSIS
# =====================================================
if st.button("🔍 Analyze Scam Risk"):

    if job_title.strip() == "" and job_description.strip() == "":
        st.warning("Please enter at least a job title or description.")
        st.stop()

    # ---------- TEXT FEATURES ----------
    combined_text = job_title + " " + job_description
    X_text = tfidf_vectorizer.transform([combined_text])

    # ---------- BEHAVIORAL FEATURES (ORDER MATTERS!) ----------
    desc_length = len(job_description)
    urgency = urgency_score(job_description)
    free_email = free_email_flag(company_profile)

    X_behavior = np.array([[desc_length, urgency, free_email]])

    # ---------- FINAL MODEL INPUT ----------
    X_final = hstack([X_text, X_behavior])

    # ---------- MODEL PROBABILITY ----------
    fraud_prob = fraud_model.predict_proba(X_final)[0][1]

    # =================================================
    # RISK SCORING ENGINE (BUSINESS LOGIC)
    # =================================================
    salary_missing = int(salary_range.strip() == "")

    risk_score = (
        0.60 * fraud_prob +
        0.15 * min(urgency / 5, 1) +
        0.15 * salary_missing +
        0.10 * free_email
    ) * 100

    risk_score = round(min(risk_score, 100), 2)

    if risk_score < 30:
        level = "LOW"
        color = "green"
        advice = "Job appears relatively safe. Still verify company details."
    elif risk_score < 60:
        level = "MEDIUM"
        color = "orange"
        advice = "Proceed with caution. Avoid sharing personal information."
    else:
        level = "HIGH"
        color = "red"
        advice = "High scam risk detected. Strongly avoid applying."

    # =================================================
    # OUTPUT
    # =================================================
    st.markdown("### 📊 Scam Risk Assessment")

    st.metric("Composite Scam Risk Score", f"{risk_score} / 100")
    st.progress(int(risk_score))

    st.markdown(f"**Risk Category:** :{color}[{level}]")
    st.markdown(f"**Recommended Action:** {advice}")

    with st.expander("🔍 Why was this job flagged?"):
        if urgency > 0:
            st.write("• Urgency-driven language detected")
        if salary_missing:
            st.write("• Salary information missing")
        if free_email:
            st.write("• Free email domain detected in company details")
        if urgency == 0 and salary_missing == 0 and free_email == 0:
            st.write("• No strong scam indicators detected")

    st.markdown("---")
    st.caption(
        "SCAMGUARD-AI provides ML-based decision support, "
        "not a definitive judgment."

    )
