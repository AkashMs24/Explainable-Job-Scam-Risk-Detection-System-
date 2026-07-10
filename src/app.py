# ==============================
# app.py — JobGuard AI Streamlit UI
# Version: 1.2 (Production Ready)
# Last Updated: 2026-07-09
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from pathlib import Path
import io
import re
from datetime import datetime
import uuid
import logging

# ======== LOCAL IMPORTS ========
from src.utils import (
    build_feature_vector,
    compute_risk_score,
    get_risk_level,
    compute_shap_values,
    top_shap_features,
    top_driver,
    matched_scam_phrases,
    validate_inputs,
    get_model_info,
    model_confidence,
    preprocess_email_text,
    extract_text_from_file,
)

# ======== LOGGING SETUP ========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======== PAGE CONFIG ========
st.set_page_config(
    page_title="JobGuard AI — Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======== CONSTANTS ========
MAX_FILE_SIZE_MB = 10
MAX_TEXT_LENGTH = 50000
MIN_DESCRIPTION_LENGTH = 50
FRAUD_THRESHOLD = 0.35

# ======== THEME & STYLING ========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    :root {
        --bg:        #080808;
        --surface:   #111111;
        --surface2:  #1a1a1a;
        --border:    #2a2a2a;
        --accent:    #c8ff00;
        --accent2:   #ff4d4d;
        --accent3:   #4d9fff;
        --text:      #f5f5f5;
        --muted:     #888888;
        --radius:    12px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    .main .block-container {
        padding: clamp(1rem, 4vw, 3rem);
        max-width: 1400px;
        background: var(--bg);
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .hero-wrap {
        text-align: center;
        padding: clamp(2rem, 6vw, 5rem) 1rem clamp(1.5rem, 4vw, 3rem);
        background: linear-gradient(135deg, #0f0f0f 0%, #111 60%, #0a0a0a 100%);
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(200,255,0,0.1);
        border: 1px solid rgba(200,255,0,0.3);
        color: var(--accent);
        font-size: clamp(0.65rem, 1.5vw, 0.75rem);
        font-weight: 500;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.35rem 1rem;
        border-radius: 100px;
        margin-bottom: 1.2rem;
    }
    
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: clamp(2rem, 6vw, 4.5rem);
        font-weight: 800;
        line-height: 1.05;
        color: var(--text);
        margin: 0 0 0.6rem;
        letter-spacing: -0.02em;
    }
    
    .hero-title span {
        color: var(--accent);
    }

    .hero-subtitle {
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        color: #aaa;
        margin: 0;
        font-weight: 300;
    }

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }

    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem 1.4rem !important;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    
    [data-testid="stMetric"]:hover {
        border-color: var(--accent) !important;
        transform: translateY(-2px);
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 1.8rem);
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    
    .card-accent { border-left: 3px solid var(--accent); }
    
    .card-danger {
        border-left: 3px solid var(--accent2);
        background: rgba(255, 77, 77, 0.05);
    }
    
    .card-info {
        border-left: 3px solid var(--accent3);
        background: rgba(77, 159, 255, 0.05);
    }

    .risk-high   { 
        background: rgba(255,77,77,0.15);
        color: #ff6b6b;
        border: 1px solid rgba(255,77,77,0.3);
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-medium { 
        background: rgba(255,196,0,0.15);
        color: #ffc400;
        border: 1px solid rgba(255,196,0,0.3);
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-low    { 
        background: rgba(0,230,118,0.15);
        color: #00e676;
        border: 1px solid rgba(0,230,118,0.3);
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    .metric-box {
        background: var(--surface2);
        border: 1px solid var(--border);
        padding: 1.2rem;
        border-radius: var(--radius);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin-bottom: 0.25rem;
    }

    .metric-sublabel {
        font-size: 0.8rem;
        color: #666;
    }

    hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ======== LOAD MODELS (CACHED) ========
@st.cache_resource
def load_models():
    """Load pre-trained model artifacts."""
    base_dir = Path(__file__).parent
    try:
        model = joblib.load(base_dir / "fraud_model.pkl")
        tfidf = joblib.load(base_dir / "tfidf_vectorizer.pkl")
        feature_names = joblib.load(base_dir / "feature_names.pkl")
        logger.info("✅ Models loaded successfully")
        return model, tfidf, feature_names
    except FileNotFoundError as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise

try:
    model, tfidf, feature_names = load_models()
except FileNotFoundError:
    st.error("❌ Model files not found in src/ directory")
    st.stop()

# ======== HERO SECTION ========
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>🛡️ EXPLAINABLE AI</div>
    <h1 class='hero-title'>JobGuard<span>AI</span></h1>
    <p class='hero-subtitle'>Detect Job Scams Before They Cost You</p>
</div>
""", unsafe_allow_html=True)

# ======== SIDEBAR NAVIGATION ========
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["📊 Overview", "🔎 Predict Job", "📈 Statistics", "💬 Feedback"]
    )

# ======== PAGE: OVERVIEW ========
if page == "📊 Overview":
    st.markdown("""
    <div class='card card-accent'>
    <h2>🛡️ JobGuard AI — How It Works</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-label'>Model</div>
            <div class='metric-value' style='color: #c8ff00; font-size: 1.2rem;'>LR</div>
            <div class='metric-sublabel'>Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-label'>Test AUC</div>
            <div class='metric-value' style='font-size: 1.8rem;'>0.98</div>
            <div class='metric-sublabel'>Top 1% Performance</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-label'>Precision</div>
            <div class='metric-value' style='font-size: 1.8rem;'>92%</div>
            <div class='metric-sublabel'>Fraud Detection</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-box'>
            <div class='metric-label'>Explainability</div>
            <div class='metric-value' style='font-size: 1.2rem;'>SHAP</div>
            <div class='metric-sublabel'>Exact Values</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🚀 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Feature Engineering**
        - TF-IDF text features (5000)
        - Behavioral signals (3)
        - Total: 5003 dimensions
        
        **Step 2: ML Prediction**
        - Logistic Regression model
        - Trained on 18K+ job postings
        - Fraud probability output
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Risk Scoring**
        - Composite risk 0-100
        - Multiple signals combined
        - Actionable risk level
        
        **Step 4: Explainability**
        - Exact SHAP values
        - Top influential features
        - Transparent decision-making
        """)

    st.markdown("---")
    st.subheader("🚩 Red Flags Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ⚠️ **Scam Indicators:**
        - Urgent language ("apply now")
        - Free email domain (gmail, yahoo)
        - Missing salary info
        - Suspicious salary keywords
        - Vague job description
        - Scam phrases ("no exp required")
        """)
    
    with col2:
        st.markdown("""
        ✅ **Legitimate Signals:**
        - Detailed description (500+ chars)
        - Professional domain
        - Clear salary range
        - Specific location/role
        - Company verification
        - Formal tone
        """)

# ======== PAGE: PREDICT JOB ========
elif page == "🔎 Predict Job":
    st.subheader("Predict a Job Posting")
    
    tab_upload, tab_manual = st.tabs(["📎 Upload Email/PDF", "✏️ Type Manually"])
    
    # TAB 1: FILE UPLOAD
    with tab_upload:
        st.markdown("""
        <div class='card card-accent'>
        Upload a screenshot or PDF of a suspicious job offer.
        The app will extract text and scan for scam indicators.
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload email screenshot or PDF",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            help=f"Max: {MAX_FILE_SIZE_MB}MB | PDF: ≤20 pages | Image: ≤5000×5000",
            key="upload_file"
        )
        
        if uploaded_file is not None:
            with st.spinner("📖 Reading your file..."):
                extracted_text, success, method_or_error = extract_text_from_file(uploaded_file, MAX_FILE_SIZE_MB, MAX_TEXT_LENGTH)
            
            if not success:
                st.error(f"❌ {method_or_error}")
                st.info("💡 Try uploading a clearer image or use the 'Type Manually' tab.")
            else:
                st.success(f"✅ Text extracted via {method_or_error} ({len(extracted_text)} chars)")
                
                cleaned_text = preprocess_email_text(extracted_text)
                
                with st.expander("👁️ View extracted text"):
                    st.text_area(
                        "Extracted text",
                        value=cleaned_text[:1000] + ("..." if len(cleaned_text) > 1000 else ""),
                        height=200,
                        disabled=True
                    )
                
                upload_title = st.text_input(
                    "Job Title (optional)",
                    placeholder="e.g., Data Entry — Work From Home",
                    key="upload_title"
                )
                
                if st.button("🔍 Analyze This File", use_container_width=True, key="upload_analyze"):
                    errors = validate_inputs(upload_title, cleaned_text, "")
                    if errors:
                        for err in errors:
                            st.error(f"❌ {err}")
                    else:
                        render_prediction_results(model, tfidf, feature_names, upload_title, cleaned_text, "", key_prefix="upload")
    
    # TAB 2: MANUAL INPUT
    with tab_manual:
        st.markdown("""
        <div class='card card-accent'>
        Type or paste job posting details below for instant analysis.
        </div>
        """, unsafe_allow_html=True)
        
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Senior Data Analyst — Remote",
            key="manual_title"
        )
        
        col_l, col_r = st.columns(2)
        with col_l:
            company = st.text_input(
                "Company/Contact Email",
                placeholder="Company name or email",
                key="manual_company"
            )
        with col_r:
            salary = st.text_input(
                "Salary Range (optional)",
                placeholder="e.g., 50000-60000 or 5L-6L",
                key="manual_salary"
            )
        
        description = st.text_area(
            "Job Description",
            height=250,
            placeholder="Full job description text...",
            key="manual_desc"
        )
        
        if st.button("🔍 Analyze Posting", use_container_width=True, key="manual_analyze"):
            errors = validate_inputs(job_title, description, company)
            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                render_prediction_results(model, tfidf, feature_names, job_title, description, company, key_prefix="manual")

# ======== PAGE: STATISTICS ========
elif page == "📈 Statistics":
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "18K+ jobs")
    col2.metric("Fraud Rate", "~5%")
    col3.metric("CV Folds", "5 (Stratified)")
    col4.metric("Features", "5003")
    
    st.markdown("---")
    st.subheader("Model Benchmarking")
    
    bench_data = {
        "Model": ["Logistic Regression ✅", "XGBoost", "Gradient Boosting", "Random Forest"],
        "Test AUC": [0.9800, 0.9750, 0.9710, 0.9680],
        "F1 (Fraud)": [0.88, 0.86, 0.85, 0.84],
        "CV AUC": ["0.96±0.01", "0.95±0.01", "0.94±0.01", "0.94±0.02"]
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Why Logistic Regression?")
    st.markdown("""
    - **Exact SHAP values:** φᵢ = coef[i] × feature[i] (no approximation)
    - **Performance:** AUC only 0.005 behind XGBoost (0.98 vs 0.975)
    - **Transparency:** Fully interpretable model
    - **Trust:** Critical for fraud detection compliance
    - **Production:** Minimal latency, maximum explainability
    """)

# ======== PAGE: FEEDBACK ========
elif page == "💬 Feedback":
    st.subheader("Help Improve JobGuard AI")
    
    st.markdown("""
    Your feedback helps us improve fraud detection.
    Report predictions that were incorrect or suggest missing features.
    """)
    
    with st.form("feedback_form"):
        feedback_type = st.selectbox(
            "Type of Feedback",
            ["❌ False Positive (marked fraud, but legitimate)",
             "⚠️ False Negative (marked safe, but scam)",
             "💡 Feature Request",
             "🐛 Bug Report",
             "📝 General Feedback"]
        )
        
        description = st.text_area("Describe your feedback", height=150)
        contact = st.text_input("Email (optional, for follow-up)")
        
        submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
        
        if submitted:
            if not description.strip():
                st.error("Please provide feedback details")
            else:
                feedback_id = str(uuid.uuid4())[:8]
                st.success(f"✅ Thank you! Feedback ID: {feedback_id}")
                st.info("We'll review this and improve our model.")
                logger.info(f"Feedback {feedback_id}: {feedback_type} | {description[:50]}")

# ======== FOOTER ========
st.markdown("""
<hr>
<div style='text-align:center; padding:1.5rem 0; color:#444; font-size:0.75rem; font-family:DM Sans,sans-serif; letter-spacing:0.05em;'>
    JOBGUARD AI v1.2 &nbsp;·&nbsp; Explainable ML Fraud Detection &nbsp;·&nbsp; 2026
    <br>
    Always verify independently · Not a substitute for manual verification
</div>
""", unsafe_allow_html=True)


# ======== PREDICTION RESULTS RENDERER ========
def render_prediction_results(model, tfidf, feature_names, job_title, description, company, key_prefix):
    """Render complete prediction results with SHAP and risk breakdown."""
    
    if not description.strip():
        st.warning("Please provide a job description.")
        return
    
    try:
        # Build features
        X_input, fd = build_feature_vector(tfidf, job_title, description, company, "")
        
        # Predict
        prob = model.predict_proba(X_input)[0][1]
        risk_score = compute_risk_score(prob, fd)
        level, _, _, level_color, advice = get_risk_level(risk_score)
        confidence, _ = model_confidence(prob)
        
        # SHAP values
        shap_vals, _, _ = compute_shap_values(model, X_input, feature_names)
        top_feats = top_shap_features(shap_vals, feature_names, n=8)
        
        # Risk drivers
        adj = (0.5 + (prob - FRAUD_THRESHOLD) / (1 - FRAUD_THRESHOLD) * 0.5) if prob >= FRAUD_THRESHOLD else (prob / FRAUD_THRESHOLD * 0.5)
        _, contribs = top_driver(adj * 100, fd)
        
        # Scam phrases
        scam_hits = matched_scam_phrases(job_title, description)
        
        # ====== RENDER RESULTS ======
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Risk Score Card
        level_class = {"HIGH": "card-danger", "MEDIUM": "card", "LOW": "card-accent"}.get(level, "card")
        st.markdown(
            f"""<div class='card {level_class}' style='text-align:center;padding:2rem;'>
            <div style='font-family:Syne,sans-serif;font-size:0.8rem;color:#666;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;'>Risk Assessment</div>
            <div style='font-family:Syne,sans-serif;font-size:3rem;font-weight:800;color:{level_color};line-height:1;'>{risk_score:.1f}</div>
            <div style='font-size:0.85rem;color:#666;margin:0.4rem 0 1rem;'>/ 100</div>
            <div class='risk-{level.lower()}'>{level} Risk</div>
            <div style='margin-top:1rem;font-size:0.85rem;color:#aaa;'>
                Fraud Probability: <strong style='color:{level_color};'>{prob:.1%}</strong>
            </div>
            <div style='margin-top:0.5rem;font-size:0.82rem;color:#888;'>{advice}</div>
            </div>""",
            unsafe_allow_html=True
        )
        
        # Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fraud Probability", f"{prob:.1%}")
        c2.metric("Risk Score", f"{risk_score:.1f}/100")
        c3.metric("Confidence", confidence)
        c4.metric("Urgency Signals", str(fd.get("urgency", 0)))
        
        # Risk Drivers
        st.markdown("<br>**📊 Risk Score Breakdown**", unsafe_allow_html=True)
        driver_df = pd.DataFrame(list(contribs.items()), columns=["Driver", "Points"])
        driver_df = driver_df[driver_df["Points"] > 0].sort_values("Points", ascending=True)
        
        fig_d = go.Figure(go.Bar(
            y=driver_df["Driver"],
            x=driver_df["Points"],
            orientation="h",
            marker=dict(color="#c8ff00", line=dict(color="#111", width=1)),
        ))
        fig_d.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            height=250,
            xaxis=dict(gridcolor="#2a2a2a"),
            margin=dict(l=200, r=20, t=20, b=20),
            font=dict(size=11)
        )
        st.plotly_chart(fig_d, use_container_width=True, key=f"{key_prefix}_driver")
        
        # Top Features
        st.markdown("**🔍 Top Influencing Words**")
        shap_df = pd.DataFrame(top_feats, columns=["Feature", "SHAP Value"])
        shap_colors = ["#ff4d4d" if v > 0 else "#c8ff00" for v in shap_df["SHAP Value"]]
        
        fig_s = go.Figure(go.Bar(
            y=shap_df["Feature"],
            x=shap_df["SHAP Value"],
            orientation="h",
            marker=dict(color=shap_colors, line=dict(color="#111", width=1)),
        ))
        fig_s.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            height=300,
            xaxis=dict(gridcolor="#2a2a2a"),
            yaxis=dict(categoryorder="total ascending"),
            margin=dict(l=150, r=20, t=20, b=20),
            font=dict(size=11)
        )
        st.plotly_chart(fig_s, use_container_width=True, key=f"{key_prefix}_shap")
        
        # Scam Phrases
        if scam_hits:
            st.warning(f"⚠️ Scam phrases detected: **{', '.join(scam_hits)}**")
        
        # Feedback Section
        st.markdown("---")
        st.markdown("**Was this prediction helpful?**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Correct", key=f"{key_prefix}_correct"):
                st.success("Thanks! This helps us improve.")
        with col2:
            if st.button("👎 Incorrect", key=f"{key_prefix}_incorrect"):
                st.info("Please submit details in Feedback tab so we can learn.")
        with col3:
            if st.button("❓ Uncertain", key=f"{key_prefix}_uncertain"):
                st.info("Always verify independently with the company directly.")
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        logger.error(f"Prediction error: {repr(e)}")
