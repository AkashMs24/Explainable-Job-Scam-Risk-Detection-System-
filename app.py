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
    page_icon="🛡️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS — DARK MODERN UI
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&family=Bebas+Neue&display=swap');

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0a0b0f;
    color: #e8e8f0;
}
.stApp { background: #0a0b0f; }

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1000px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0e0f14 !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

/* ── INPUTS ── */
input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] {
    background-color: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
}
input:focus, textarea:focus {
    border-color: rgba(220,38,38,0.6) !important;
    box-shadow: 0 0 0 2px rgba(220,38,38,0.1) !important;
}
[data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
    background-color: transparent !important;
}
label, .stTextInput label, .stTextArea label {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: rgba(255,255,255,0.4) !important;
    text-transform: uppercase !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #dc2626, #b91c1c) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-family: 'Bebas Neue', cursive !important;
    font-size: 16px !important;
    letter-spacing: 3px !important;
    width: 100% !important;
    box-shadow: 0 4px 24px rgba(220,38,38,0.3) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 32px rgba(220,38,38,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── PROGRESS BAR ── */
[data-testid="stProgress"] > div > div {
    border-radius: 4px !important;
    height: 8px !important;
    background: rgba(255,255,255,0.07) !important;
}
[data-testid="stProgress"] > div > div > div {
    border-radius: 4px !important;
}

/* ── METRIC ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    color: rgba(255,255,255,0.4) !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', cursive !important;
    font-size: 40px !important;
    letter-spacing: 2px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1px !important;
    color: rgba(255,255,255,0.7) !important;
}

/* ── INFO / WARNING / SUCCESS BOXES ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
}

/* ── DIVIDER ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── CAPTION ── */
.stCaption, small {
    color: rgba(255,255,255,0.3) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}

/* ── CUSTOM CARD ── */
.sg-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.sg-card-red   { border-color: rgba(220,38,38,0.35); box-shadow: 0 0 20px rgba(220,38,38,0.1); }
.sg-card-orange{ border-color: rgba(255,170,0,0.35);  box-shadow: 0 0 20px rgba(255,170,0,0.1); }
.sg-card-green { border-color: rgba(0,230,118,0.35);  box-shadow: 0 0 20px rgba(0,230,118,0.1); }

.sg-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.sg-badge-red    { background: rgba(220,38,38,0.15);  border: 1px solid rgba(220,38,38,0.4);  color: #f87171; }
.sg-badge-orange { background: rgba(255,170,0,0.15);  border: 1px solid rgba(255,170,0,0.4);  color: #fbbf24; }
.sg-badge-green  { background: rgba(0,230,118,0.15);  border: 1px solid rgba(0,230,118,0.4);  color: #4ade80; }

.sg-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.sg-label { font-family: 'Space Mono', monospace; font-size: 10px; letter-spacing: 2px; color: rgba(255,255,255,0.4); text-transform: uppercase; min-width: 160px; }
.sg-value { font-size: 14px; color: #e8e8f0; font-weight: 500; }

.sg-flag-item {
    display: flex; align-items: center; gap: 8px;
    padding: 8px 12px; border-radius: 6px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 6px;
    font-size: 13px; color: rgba(255,255,255,0.7);
}
.sg-dot-red    { width:8px; height:8px; border-radius:50%; background:#f87171; flex-shrink:0; }
.sg-dot-green  { width:8px; height:8px; border-radius:50%; background:#4ade80; flex-shrink:0; }

.sg-title {
    font-family: 'Bebas Neue', cursive;
    font-size: 42px;
    letter-spacing: 4px;
    background: linear-gradient(135deg, #fff 60%, #dc2626);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.sg-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: rgba(255,255,255,0.35);
    margin-top: 4px;
}
.sg-section-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    color: #dc2626;
    text-transform: uppercase;
    margin-bottom: 14px;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
BASE_DIR = Path(__file__).resolve().parent

fraud_model      = joblib.load(BASE_DIR / "fraud_model.pkl")
tfidf_vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
feature_names    = joblib.load(BASE_DIR / "feature_names.pkl")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
            <div style='width:36px;height:36px;background:linear-gradient(135deg,#dc2626,#7f1d1d);border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:18px;'>🛡️</div>
            <div>
                <div style='font-family:Bebas Neue,cursive;font-size:20px;letter-spacing:3px;'>SCAMGUARD<span style="color:#dc2626">-AI</span></div>
                <div style='font-family:Space Mono,monospace;font-size:9px;color:rgba(255,255,255,0.35);letter-spacing:2px;'>INTELLIGENCE SYSTEM</div>
            </div>
        </div>
        <div style='font-size:12px;color:rgba(255,255,255,0.4);line-height:1.7;margin-top:12px;'>
            Explainable Job Scam Risk Intelligence System
        </div>
    </div>
    <hr style='border-color:rgba(255,255,255,0.07);margin:16px 0;'/>
    <div style='font-family:Space Mono,monospace;font-size:11px;color:rgba(255,255,255,0.5);line-height:2;'>
        ▶ NLP-based fraud detection<br>
        ▶ Behavioral scam indicators<br>
        ▶ Decision-support risk scoring
    </div>
    <hr style='border-color:rgba(255,255,255,0.07);margin:16px 0;'/>
    <div style='font-size:11px;color:rgba(255,255,255,0.3);line-height:1.7;'>
        ⚠️ This system provides guidance, not final judgment. Manual verification is always recommended.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div style='margin-bottom:32px;'>
    <div class='sg-title'>SCAMGUARD-AI</div>
    <div class='sg-subtitle'>// PROTECTING FRESHERS FROM FRAUDULENT JOB & INTERNSHIP SCAMS</div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT SECTION
# =====================================================
st.markdown("<div class='sg-section-label'>// JOB POSTING INPUT</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    job_title    = st.text_input("Job Title", placeholder="e.g. Software Engineer Intern")
    salary_range = st.text_input("Salary Range (optional)", placeholder="e.g. $20/hr or ₹15,000/month")
with col2:
    company_profile = st.text_area("Company Profile / Contact Info", placeholder="e.g. contact@company.com, LinkedIn URL...", height=106)

job_description = st.text_area("Job Description", placeholder="Paste the full job description here...", height=160)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍  ANALYZE SCAM RISK")

# =====================================================
# FEATURE ENGINEERING
# =====================================================
urgency_words = ["urgent","immediate","limited","apply fast","hurry","few slots","act now"]
free_domains  = ["gmail.com","yahoo.com","outlook.com","hotmail.com"]

def urgency_score(text):
    text = str(text).lower()
    return sum(word in text for word in urgency_words)

def free_email_flag(text):
    text = str(text).lower()
    return int(any(domain in text for domain in free_domains))

# =====================================================
# ANALYSIS
# =====================================================
if analyze_btn:
    if job_title.strip() == "" and job_description.strip() == "":
        st.warning("⚠️ Please enter at least a job title or description.")
        st.stop()

    with st.spinner("Scanning job posting..."):

        # Text features
        combined_text = job_title + " " + job_description
        X_text = tfidf_vectorizer.transform([combined_text])

        # Behavioral features
        desc_length = len(job_description)
        urgency     = urgency_score(job_description)
        free_email  = free_email_flag(company_profile)

        X_behavior = np.array([[desc_length, urgency, free_email]])
        X_final    = hstack([X_text, X_behavior])

        # Model
        fraud_prob = fraud_model.predict_proba(X_final)[0][1]

        # Confidence
        model_confidence = "Moderate" if 0.4 <= fraud_prob <= 0.6 else "High"

        # Risk score
        salary_missing = int(salary_range.strip() == "")
        risk_score = (
            0.60 * fraud_prob +
            0.15 * min(urgency / 5, 1) +
            0.15 * salary_missing +
            0.10 * free_email
        ) * 100
        risk_score = round(min(risk_score, 100), 2)

        # Risk level
        if risk_score < 30:
            level, color_key, verdict_icon, advice = "LOW", "green", "✅", "Job appears relatively safe. Still verify company details."
        elif risk_score < 60:
            level, color_key, verdict_icon, advice = "MEDIUM", "orange", "⚠️", "Proceed with caution. Avoid sharing personal information."
        else:
            level, color_key, verdict_icon, advice = "HIGH", "red", "⛔", "High scam risk detected. Strongly avoid applying."

        # Primary driver
        if urgency > 2:
            primary_driver = "Urgency-driven language"
        elif free_email:
            primary_driver = "Use of free email domain"
        elif salary_missing:
            primary_driver = "Lack of salary transparency"
        else:
            primary_driver = "No dominant risk driver"

        # Context insight
        if urgency > 2 and free_email:
            context = "This pattern strongly resembles mass internship scam campaigns."
        elif salary_missing and desc_length < 300:
            context = "Short descriptions with missing salary often indicate low-effort scams."
        elif urgency > 0:
            context = "Urgency-based language suggests pressure tactics commonly used in scams."
        else:
            context = "No dominant scam pattern detected based on known behavior."

        # Color map
        score_color = {"red": "#f87171", "orange": "#fbbf24", "green": "#4ade80"}[color_key]
        progress_color = {"red": "#dc2626", "orange": "#f59e0b", "green": "#22c55e"}[color_key]

    # =====================================================
    # RESULTS
    # =====================================================
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='sg-section-label'>// SCAM RISK ASSESSMENT</div>", unsafe_allow_html=True)

    # Score + verdict card
    st.markdown(f"""
    <div class='sg-card sg-card-{color_key}'>
        <div style='display:flex; align-items:center; gap:24px; flex-wrap:wrap;'>
            <div style='text-align:center; min-width:100px;'>
                <div style='font-family:Bebas Neue,cursive; font-size:52px; color:{score_color}; line-height:1; letter-spacing:2px;'>{risk_score}</div>
                <div style='font-family:Space Mono,monospace; font-size:10px; color:rgba(255,255,255,0.35); letter-spacing:2px;'>/ 100</div>
            </div>
            <div>
                <div style='font-family:Bebas Neue,cursive; font-size:26px; letter-spacing:3px; color:{score_color}; margin-bottom:8px;'>{verdict_icon} {level} RISK</div>
                <span class='sg-badge sg-badge-{color_key}'>{level} RISK</span>
                &nbsp;
                <span class='sg-badge' style='background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);color:rgba(255,255,255,0.5);'>CONFIDENCE: {model_confidence.upper()}</span>
            </div>
        </div>
        <div style='margin-top:16px; background:rgba(255,255,255,0.04); border-radius:4px; height:8px; overflow:hidden;'>
            <div style='width:{risk_score}%; height:100%; background:{progress_color}; border-radius:4px; box-shadow:0 0 10px {progress_color};'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Details row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='sg-card' style='text-align:center;'>
            <div class='sg-label' style='margin-bottom:6px;'>Primary Risk Driver</div>
            <div style='font-size:13px; color:#e8e8f0; font-weight:500;'>{primary_driver}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='sg-card' style='text-align:center;'>
            <div class='sg-label' style='margin-bottom:6px;'>Model Confidence</div>
            <div style='font-size:13px; color:#e8e8f0; font-weight:500;'>{model_confidence}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='sg-card' style='text-align:center;'>
            <div class='sg-label' style='margin-bottom:6px;'>Fraud Probability</div>
            <div style='font-size:13px; color:#e8e8f0; font-weight:500;'>{round(fraud_prob*100, 1)}%</div>
        </div>""", unsafe_allow_html=True)

    # AI Insight
    st.markdown(f"""
    <div style='padding:14px 18px; background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.25); border-radius:10px; margin-bottom:16px;'>
        <div style='font-family:Space Mono,monospace; font-size:10px; color:#818cf8; letter-spacing:2px; margin-bottom:6px;'>🧠 RISK CONTEXT INSIGHT</div>
        <div style='font-size:13px; color:rgba(255,255,255,0.75); line-height:1.6;'>{context}</div>
    </div>
    """, unsafe_allow_html=True)

    # Borderline warning
    if 45 <= risk_score <= 55:
        st.warning("⚠️ **Borderline Risk Detected** — The system is uncertain. Manual review is strongly recommended.")

    # Recommended action
    st.markdown(f"""
    <div style='padding:14px 18px; background:rgba({{"red":"220,38,38","orange":"255,170,0","green":"0,230,118"}}[color_key], 0.08); border:1px solid {score_color}44; border-radius:10px; margin-bottom:16px;'>
        <div style='font-family:Space Mono,monospace; font-size:10px; color:rgba(255,255,255,0.4); letter-spacing:2px; margin-bottom:6px;'>📋 RECOMMENDED ACTION</div>
        <div style='font-size:13px; color:{score_color}; font-weight:600;'>{advice}</div>
    </div>
    """, unsafe_allow_html=True)

    # Why flagged expander
    with st.expander("🔍 Why was this job flagged?"):
        flags = []
        if urgency > 0:   flags.append("Urgency-driven language detected")
        if salary_missing: flags.append("Salary information missing")
        if free_email:     flags.append("Free email domain detected in company details")

        if flags:
            for f in flags:
                st.markdown(f"<div class='sg-flag-item'><div class='sg-dot-red'></div>{f}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='sg-flag-item'><div class='sg-dot-green'></div>No strong scam indicators detected</div>", unsafe_allow_html=True)

    # How to reduce risk expander
    with st.expander("✅ How to reduce scam risk"):
        tips = [
            "Verify company website and LinkedIn presence",
            "Avoid paying registration or processing fees",
            "Cross-check salary with market standards",
            "Do not share documents before formal interviews",
        ]
        for tip in tips:
            st.markdown(f"<div class='sg-flag-item'><div class='sg-dot-green'></div>{tip}</div>", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("SCAMGUARD-AI is an ML-based decision-support system. Predictions depend on historical patterns and may not capture new scam strategies.")
