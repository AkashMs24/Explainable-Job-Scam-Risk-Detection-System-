import streamlit as st
import numpy as np
import joblib
from pathlib import Path
from scipy.sparse import hstack
import re

# =====================================================
# APP CONFIG
# =====================================================
st.set_page_config(
    page_title="SCAMGUARD-AI",
    layout="centered"
)

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
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
st.sidebar.write("• Decision-support risk scoring")
st.sidebar.markdown("---")
st.sidebar.caption(
    "⚠️ This system provides guidance, not final judgment.\n"
    "Manual verification is always recommended."
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
# FEATURE ENGINEERING
# =====================================================

URGENCY_WORDS = [
    "urgent", "immediate", "limited",
    "apply fast", "hurry", "few slots", "act now",
    "last chance", "closing soon", "today only", "don't miss"
]

FREE_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com",
    "hotmail.com", "ymail.com", "rediffmail.com"
]

# Phrases that commonly appear in scam postings
SCAM_PHRASES = [
    "no experience required", "work from home", "earn up to",
    "be your own boss", "unlimited earnings", "weekly payout",
    "processing fee", "registration fee", "refundable deposit",
    "part time", "home based", "per day earning",
    "data entry", "typing work", "copy paste"
]

# Legitimate signals that reduce scam likelihood
LEGIT_SIGNALS = [
    "interview process", "background check", "employee benefits",
    "annual leave", "health insurance", "provident fund",
    "job description", "qualifications required", "minimum degree"
]


def urgency_score(text: str) -> int:
    """Count urgency keyword hits in text."""
    text = str(text).lower()
    return sum(1 for word in URGENCY_WORDS if word in text)


def free_email_flag(text: str) -> int:
    """1 if a free email domain is found in company contact info."""
    text = str(text).lower()
    return int(any(domain in text for domain in FREE_DOMAINS))


def scam_phrase_score(text: str) -> int:
    """Count scam-indicative phrase hits."""
    text = str(text).lower()
    return sum(1 for phrase in SCAM_PHRASES if phrase in text)


def legit_signal_score(text: str) -> int:
    """Count legitimate job-posting signals."""
    text = str(text).lower()
    return sum(1 for phrase in LEGIT_SIGNALS if phrase in text)


def caps_ratio(text: str) -> float:
    """
    Ratio of UPPERCASE words to total words.
    Scam posts often use excessive caps for hype (e.g. 'EARN BIG NOW').
    """
    words = str(text).split()
    if len(words) == 0:
        return 0.0
    caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
    return caps_words / len(words)


def suspicious_salary_flag(salary_text: str) -> int:
    """
    Detect unrealistically high salaries often used as bait.
    Extracts numbers and flags if max value > 5 lakhs/month or
    contains vague phrases like 'unlimited'.
    """
    if not salary_text.strip():
        return 0

    salary_lower = salary_text.lower()

    # Vague / too-good-to-be-true language
    if any(word in salary_lower for word in ["unlimited", "upto", "up to", "per day"]):
        return 1

    # Extract all numbers from salary string
    numbers = re.findall(r'\d[\d,]*', salary_text.replace(",", ""))
    if numbers:
        try:
            max_val = max(int(n.replace(",", "")) for n in numbers)
            # Flag if monthly salary claim > 5,00,000 (unrealistic for fresher)
            if max_val > 500000:
                return 1
        except ValueError:
            pass

    return 0


def has_contact_info(text: str) -> int:
    """1 if a phone number or email is present in company profile."""
    phone_pattern = r'\b[\+]?[\d][\d\s\-]{8,14}\d\b'
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    has_phone = bool(re.search(phone_pattern, str(text)))
    has_email = bool(re.search(email_pattern, str(text)))
    return int(has_phone or has_email)


def model_confidence_label(prob: float) -> str:
    """
    Confidence is highest when probability is far from the decision boundary (0.5).
    Distance from 0.5 → [0, 0.5]; normalize to [0, 1].
    """
    distance = abs(prob - 0.5)  # 0 = most uncertain, 0.5 = most certain
    if distance >= 0.35:
        return "High"
    elif distance >= 0.15:
        return "Moderate"
    else:
        return "Low"


# =====================================================
# RISK ANALYSIS
# =====================================================
if st.button("🔍 Analyze Scam Risk"):

    if job_title.strip() == "" and job_description.strip() == "":
        st.warning("Please enter at least a job title or description.")
        st.stop()

    # ---------- TEXT FEATURES ----------
    # Include all text fields for richer NLP signal
    combined_text = " ".join([
        job_title,
        job_description,
        company_profile,
        salary_range
    ])
    X_text = tfidf_vectorizer.transform([combined_text])

    # ---------- BEHAVIORAL FEATURES ----------
    desc_length = len(job_description)
    urgency        = urgency_score(job_description + " " + job_title)
    free_email     = free_email_flag(company_profile)
    scam_phrases   = scam_phrase_score(job_description + " " + job_title)
    legit_signals  = legit_signal_score(job_description + " " + company_profile)
    caps            = caps_ratio(job_title + " " + job_description)
    sus_salary      = suspicious_salary_flag(salary_range)
    has_contact     = has_contact_info(company_profile)
    salary_missing  = int(salary_range.strip() == "")

    X_behavior = np.array([[
        desc_length,
        urgency,
        free_email,
        scam_phrases,
        legit_signals,
        caps,
        sus_salary,
        has_contact,
        salary_missing
    ]])

    # ---------- FINAL MODEL INPUT ----------
    # NOTE: X_behavior columns must match what was used during training.
    # If your model was trained with only [desc_length, urgency, free_email],
    # use only those 3 columns here. The extra features above are available
    # for rule-based scoring even if not used by the model directly.
    X_final = hstack([X_text, X_behavior[:, :3]])  # Keep first 3 for model compatibility

    # ---------- MODEL PROBABILITY ----------
    fraud_prob = fraud_model.predict_proba(X_final)[0][1]

    # ---------- MODEL CONFIDENCE (FIXED) ----------
    model_confidence = model_confidence_label(fraud_prob)

    # =================================================
    # RISK SCORING ENGINE
    # =================================================
    # Normalize component scores to [0, 1]
    urgency_norm     = min(urgency / max(len(URGENCY_WORDS), 1), 1.0)
    scam_norm        = min(scam_phrases / 5, 1.0)
    legit_norm       = min(legit_signals / 5, 1.0)        # reduces risk
    caps_norm        = min(caps / 0.5, 1.0)               # >50% caps = fully flagged

    risk_score = (
        0.45 * fraud_prob           +   # Primary: ML model
        0.15 * urgency_norm         +   # Urgency language
        0.10 * salary_missing       +   # No salary info
        0.10 * free_email           +   # Free email domain
        0.08 * scam_norm            +   # Scam phrase density
        0.07 * sus_salary           +   # Suspicious salary claim
        0.05 * caps_norm            -   # Caps hype
        0.10 * legit_norm * (1 - fraud_prob)  # Legit signals reduce risk (context-sensitive)
    ) * 100

    risk_score = round(min(max(risk_score, 0), 100), 2)

    # ---------- RISK BUCKET ----------
    if risk_score < 30:
        level  = "LOW"
        color  = "green"
        advice = "Job appears relatively safe. Still verify company details independently."
    elif risk_score < 60:
        level  = "MEDIUM"
        color  = "orange"
        advice = "Proceed with caution. Avoid sharing personal documents or paying any fee."
    else:
        level  = "HIGH"
        color  = "red"
        advice = "High scam risk detected. Strongly avoid applying or sharing information."

    # ---------- PRIMARY RISK DRIVER (ranked by impact) ----------
    drivers = {
        "ML model fraud signal":           fraud_prob * 0.45 * 100,
        "Urgency-driven language":         urgency_norm * 0.15 * 100,
        "Scam phrase density":             scam_norm * 0.08 * 100,
        "Free/unverified email domain":    free_email * 0.10 * 100,
        "Suspicious salary claim":         sus_salary * 0.07 * 100,
        "Lack of salary transparency":     salary_missing * 0.10 * 100,
        "Excessive capitalization (hype)": caps_norm * 0.05 * 100,
    }
    # Filter out zero-contribution drivers, pick the top contributor
    active_drivers = {k: v for k, v in drivers.items() if v > 0}
    if active_drivers:
        primary_driver = max(active_drivers, key=active_drivers.get)
    else:
        primary_driver = "No dominant risk driver identified"

    # =================================================
    # OUTPUT
    # =================================================
    st.markdown("### 📊 Scam Risk Assessment")

    st.metric("Composite Scam Risk Score", f"{risk_score} / 100")
    st.progress(int(risk_score))

    st.markdown(f"**Risk Category:** :{color}[{level}]")
    st.markdown(f"**Model Confidence:** {model_confidence}")
    st.markdown(f"**Primary Risk Driver:** {primary_driver}")
    st.markdown(f"**Recommended Action:** {advice}")

    # =================================================
    # INTELLIGENCE LAYER — context-aware pattern matching
    # =================================================
    if free_email and (urgency > 1 or scam_phrases > 2):
        context = "Pattern matches mass scam campaigns: free email + urgency/scam phrases together are a strong combined signal."
    elif sus_salary and salary_missing == 0:
        context = "Unrealistically high salary claim detected — a common bait tactic in fresher-targeted scams."
    elif scam_phrases >= 3:
        context = f"High density of scam-associated phrases ({scam_phrases} detected). This language pattern closely mirrors known fraudulent postings."
    elif legit_signals >= 3 and fraud_prob < 0.4:
        context = "Multiple legitimate job-posting signals found (benefits, interview process, qualifications). Risk is likely low."
    elif urgency > 0:
        context = "Urgency-based language detected — pressure tactics are frequently used to prevent candidates from doing due diligence."
    elif caps > 0.2:
        context = "Excessive capitalization detected — a hype/attention-grab pattern common in low-credibility postings."
    elif not has_contact and company_profile.strip():
        context = "Company profile provided but lacks verifiable contact information (phone/email)."
    else:
        context = "No dominant scam pattern detected based on known behavioral signals."

    st.info(f"🧠 **Risk Context Insight:** {context}")

    # ---------- BORDERLINE WARNING ----------
    if 40 <= risk_score <= 60:
        st.warning(
            "⚠️ **Borderline Risk Detected**: "
            "The system is uncertain. Manual review and independent verification is strongly recommended."
        )

    # =================================================
    # EXPLAINABILITY — full signal breakdown
    # =================================================
    with st.expander("🔍 Why was this job flagged?"):
        st.markdown(f"**ML Model Fraud Probability:** `{fraud_prob:.2%}`")
        if urgency > 0:
            st.write(f"• Urgency-driven language detected ({urgency} keyword(s))")
        if scam_phrases > 0:
            st.write(f"• Scam-associated phrases found ({scam_phrases} phrase(s))")
        if free_email:
            st.write("• Free/personal email domain detected in company details")
        if sus_salary:
            st.write("• Suspicious or unrealistically high salary claim")
        if salary_missing:
            st.write("• Salary information not provided")
        if caps > 0.2:
            st.write(f"• Excessive capitalization in text ({caps:.0%} of words in caps)")
        if not has_contact and company_profile.strip():
            st.write("• No verifiable contact details (phone/email) found in company profile")
        if legit_signals > 0:
            st.write(f"✅ Legitimate signals found: {legit_signals} indicator(s) reduced risk score")
        if urgency == 0 and scam_phrases == 0 and free_email == 0 and salary_missing == 0:
            st.write("• No strong scam indicators detected in rule-based analysis")

    # =================================================
    # SIGNAL SUMMARY TABLE
    # =================================================
    with st.expander("📋 Feature Signal Summary"):
        signal_data = {
            "Signal": [
                "ML Fraud Probability",
                "Urgency Keywords",
                "Scam Phrases",
                "Legit Signals",
                "Free Email Domain",
                "Suspicious Salary",
                "Salary Provided",
                "Contact Info Present",
                "Description Length (chars)",
                "Caps Ratio",
            ],
            "Value": [
                f"{fraud_prob:.2%}",
                urgency,
                scam_phrases,
                legit_signals,
                "Yes" if free_email else "No",
                "Yes" if sus_salary else "No",
                "No" if salary_missing else "Yes",
                "Yes" if has_contact else "No",
                desc_length,
                f"{caps:.0%}",
            ]
        }
        st.table(signal_data)

    # =================================================
    # ADVISORY LAYER
    # =================================================
    with st.expander("✅ How to reduce scam risk"):
        st.write("• Verify company website and LinkedIn presence independently")
        st.write("• Never pay registration, processing, or security deposit fees")
        st.write("• Cross-check salary claims with market standards (Glassdoor, Naukri)")
        st.write("• Do not share Aadhaar, PAN, or bank details before a formal offer letter")
        st.write("• Prefer companies with official domain emails (not Gmail/Yahoo)")
        st.write("• Check for company registration on MCA21 (India) or official registries")

    # =================================================
    # DISCLAIMER
    # =================================================
    st.markdown("---")
    st.caption(
        "SCAMGUARD-AI is an ML-based decision-support system. "
        "Predictions depend on historical training data and may not capture emerging scam strategies. "
        "Always perform manual verification before applying."
    )
