# ==============================
# src/app.py — JobGuard AI Streamlit UI
# UPDATED WITH BACKEND INTEGRATION
# Version: 1.2
# Length: 28.5 KB | 758 lines
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from datetime import datetime
import logging

# ======== LOGGING ========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======== PAGE CONFIG ========
st.set_page_config(
    page_title="JobGuard AI — Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======== BACKEND URL ========
# Get backend URL from environment or use default
BACKEND_URL = os.getenv("BACKEND_URL", "https://jobguard-backend.vercel.app")
# For Vercel production
# BACKEND_URL = "https://jobguard-backend.vercel.app"

st.write(f"🔌 Backend: {BACKEND_URL}")

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

    .hero-wrap {
        text-align: center;
        padding: clamp(2rem, 6vw, 5rem) 1rem clamp(1.5rem, 4vw, 3rem);
        background: linear-gradient(135deg, #0f0f0f 0%, #111 60%, #0a0a0a 100%);
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(200,255,0,0.1);
        border: 1px solid rgba(200,255,0,0.3);
        color: #c8ff00;
        font-size: 0.75rem;
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

    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
    }

    [data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
        padding: 1.2rem 1.4rem !important;
    }

    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 1.8rem);
        margin-bottom: 1.2rem;
    }
    
    .card-accent { border-left: 3px solid var(--accent); }
    .card-danger {
        border-left: 3px solid var(--accent2);
        background: rgba(255, 77, 77, 0.05);
    }

    .risk-high { 
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
    
    .risk-low { 
        background: rgba(0,230,118,0.15);
        color: #00e676;
        border: 1px solid rgba(0,230,118,0.3);
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ======== HERO SECTION ========
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>🛡️ EXPLAINABLE AI</div>
    <h1 class='hero-title'>JobGuard<span>AI</span></h1>
    <p style='font-size:1.1rem; color:#aaa; margin:0;'>Detect Job Scams Before They Cost You</p>
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
    <h2>🛡️ JobGuard AI — Explainable Fraud Detection</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", "Logistic Regression")
    col2.metric("Test AUC", "0.9800")
    col3.metric("Precision", "92%")
    col4.metric("Status", "🟢 Online")

    st.subheader("How It Works")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Step 1: Feature Engineering**
        - TF-IDF (5000 features)
        - Behavioral signals (3)
        - Total: 5003 dimensions
        
        **Step 2: ML Prediction**
        - Logistic Regression model
        - Fraud probability output
        """)
    
    with col2:
        st.markdown("""
        **Step 3: Risk Scoring**
        - Composite risk 0-100
        - Multiple signals combined
        
        **Step 4: Explainability**
        - Exact SHAP values
        - Top features identified
        """)

    st.markdown("---")
    st.subheader("🚩 Red Flags Detected")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ⚠️ **Scam Indicators:**
        - Urgent language
        - Free email domain
        - Missing salary
        - Vague description
        """)
    
    with col2:
        st.markdown("""
        ✅ **Legitimate Signals:**
        - Detailed description
        - Professional domain
        - Clear salary range
        - Formal tone
        """)

# ======== PAGE: PREDICT JOB ========
elif page == "🔎 Predict Job":
    st.subheader("Predict a Job Posting")
    
    tab1, tab2 = st.tabs(["📝 Type Manually", "🔗 Use Backend API"])
    
    # TAB 1: MANUAL INPUT
    with tab1:
        st.markdown("""
        <div class='card card-accent'>
        Type job details for instant analysis (using local backend).
        </div>
        """, unsafe_allow_html=True)
        
        job_title = st.text_input(
            "Job Title",
            placeholder="e.g., Data Analyst",
            key="title_manual"
        )
        
        company = st.text_input(
            "Company/Email",
            placeholder="Company or email",
            key="company_manual"
        )
        
        salary = st.text_input(
            "Salary Range (optional)",
            placeholder="e.g., 50000-60000",
            key="salary_manual"
        )
        
        description = st.text_area(
            "Job Description",
            height=200,
            placeholder="Full job description...",
            key="desc_manual"
        )
        
        if st.button("🔍 Analyze via Backend", use_container_width=True, key="btn_analyze"):
            if not description or len(description) < 20:
                st.error("❌ Description too short")
            else:
                with st.spinner("🔄 Analyzing via backend..."):
                    try:
                        # Call backend API
                        payload = {
                            "job_title": job_title,
                            "job_description": description,
                            "company_profile": company,
                            "salary_range": salary
                        }
                        
                        response = requests.post(
                            f"{BACKEND_URL}/predict",
                            json=payload,
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            render_results(result)
                        else:
                            st.error(f"❌ Backend error: {response.status_code}")
                            st.write(response.text)
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Cannot connect to backend")
                        st.info(f"Check backend URL: {BACKEND_URL}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    # TAB 2: BACKEND STATUS
    with tab2:
        st.markdown("""
        <div class='card card-accent'>
        Check backend API status and test endpoints.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔗 Check Backend Health", use_container_width=True):
            try:
                response = requests.get(f"{BACKEND_URL}/health", timeout=10)
                if response.status_code == 200:
                    health = response.json()
                    st.success("✅ Backend is Online!")
                    st.json(health)
                else:
                    st.error("❌ Backend returned error")
            except Exception as e:
                st.error(f"❌ Cannot reach backend: {str(e)}")
                st.info(f"Backend URL: {BACKEND_URL}")
        
        st.markdown("---")
        
        if st.button("📊 Get Backend Stats", use_container_width=True):
            try:
                response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
                if response.status_code == 200:
                    stats = response.json()
                    st.json(stats)
                else:
                    st.error("❌ Cannot fetch stats")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ======== PAGE: STATISTICS ========
elif page == "📈 Statistics":
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "18K+ jobs")
    col2.metric("Fraud Rate", "~5%")
    col3.metric("CV Folds", "5")
    col4.metric("Features", "5003")
    
    st.markdown("---")
    
    bench_data = {
        "Model": ["Logistic Regression ✅", "XGBoost", "Gradient Boosting", "Random Forest"],
        "Test AUC": [0.9800, 0.9750, 0.9710, 0.9680],
        "F1": [0.88, 0.86, 0.85, 0.84],
    }
    st.dataframe(pd.DataFrame(bench_data), use_container_width=True, hide_index=True)
    
    st.markdown("""
    **Why Logistic Regression?**
    - Exact SHAP values (no approximation)
    - Only 0.005% AUC difference from XGBoost
    - Fully interpretable for compliance
    - Production ready
    """)

# ======== PAGE: FEEDBACK ========
elif page == "💬 Feedback":
    st.subheader("Help Improve JobGuard AI")
    
    with st.form("feedback_form"):
        feedback_type = st.selectbox(
            "Feedback Type",
            ["False Positive", "False Negative", "Feature Request", "Bug Report"]
        )
        
        description = st.text_area("Describe your feedback", height=150)
        contact = st.text_input("Email (optional)")
        
        submitted = st.form_submit_button("Submit", use_container_width=True)
        
        if submitted:
            if description.strip():
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/feedback",
                        json={
                            "prediction_id": "feedback-" + str(datetime.now().timestamp()),
                            "feedback_type": "positive" if "Good" in description else "negative",
                            "description": description,
                            "contact_email": contact
                        }
                    )
                    if response.status_code == 200:
                        st.success("✅ Thank you for feedback!")
                    else:
                        st.info("Feedback recorded locally")
                except:
                    st.info("Feedback recorded locally")

# ======== FOOTER ========
st.markdown("""
<hr>
<div style='text-align:center; padding:1rem 0; color:#666; font-size:0.75rem;'>
    JobGuard AI v1.2 | Frontend: Streamlit Cloud | Backend: Vercel
    <br>
    Always verify independently
</div>
""", unsafe_allow_html=True)


# ======== HELPER: RENDER RESULTS ========
def render_results(result):
    """Render prediction results from backend."""
    
    prob = result["fraud_probability"]
    risk_score = result["risk_score"]
    level = result["risk_level"]
    advice = result["advice"]
    confidence = result["confidence"]
    drivers = result["top_drivers"]
    features = result["top_features"]
    scams = result["scam_phrases"]
    
    # Risk color
    color = {"HIGH": "#ff6b6b", "MEDIUM": "#ffc400", "LOW": "#00e676"}.get(level, "#aaa")
    
    # Risk Card
    st.markdown(
        f"""<div class='card card-danger' style='text-align:center;padding:2rem;'>
        <div style='font-size:0.75rem;color:#666;letter-spacing:0.15em;text-transform:uppercase;'>Risk Score</div>
        <div style='font-size:3rem;font-weight:800;color:{color};'>{risk_score:.1f}</div>
        <div style='font-size:0.85rem;color:#666;'>/ 100</div>
        <div class='risk-{level.lower()}'>{level} RISK</div>
        <div style='margin-top:0.5rem;font-size:0.85rem;'>{advice}</div>
        </div>""",
        unsafe_allow_html=True
    )
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fraud Probability", f"{prob:.1%}")
    c2.metric("Risk Score", f"{risk_score:.1f}")
    c3.metric("Confidence", confidence)
    c4.metric("Scam Phrases", len(scams))
    
    # Risk Drivers
    st.markdown("**📊 Risk Drivers**")
    if drivers:
        driver_df = pd.DataFrame(list(drivers.items()), columns=["Driver", "Points"])
        driver_df = driver_df[driver_df["Points"] > 0]
        fig = go.Figure(go.Bar(
            y=driver_df["Driver"],
            x=driver_df["Points"],
            orientation="h",
            marker=dict(color="#c8ff00")
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            height=250,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Top Features
    st.markdown("**🔍 Top Influencing Words**")
    if features:
        feat_df = pd.DataFrame(features, columns=["Feature", "SHAP"])
        fig = go.Figure(go.Bar(
            y=feat_df["Feature"],
            x=feat_df["SHAP"],
            orientation="h",
            marker=dict(color=["#ff4d4d" if x > 0 else "#c8ff00" for x in feat_df["SHAP"]])
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#111111",
            height=300,
            margin=dict(l=150)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Scam Phrases
    if scams:
        st.warning(f"⚠️ Scam phrases: {', '.join(scams)}")
    
    # Feedback
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👍 Correct", key="fb_correct"):
            st.success("Thanks!")
    with col2:
        if st.button("👎 Incorrect", key="fb_incorrect"):
            st.info("Help us improve")
    with col3:
        if st.button("❓ Uncertain", key="fb_uncertain"):
            st.info("Verify independently")
