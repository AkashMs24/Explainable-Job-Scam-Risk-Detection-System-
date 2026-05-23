"""
SCAMGUARD-AI  |  app.py  (v5 — No Sidebar, utils.py imports)
=============================================================
All logic imported from utils.py.
utils.py is extracted directly from:
  - 02_feature_engineering_and_mo...py  (training pipeline)
  - expainabiity_and_insights.py        (feature importance, confidence)
  - eda.py                              (behavioral signals, EDA features)
"""

import streamlit as st
import numpy as np
import joblib
import time
from pathlib import Path

# ── Import all logic from utils.py ───────────────────────────────────────────
from utils import (
    URGENCY_WORDS,
    FREE_DOMAINS,
    FRAUD_THRESHOLD,
    build_feature_vector,
    compute_risk_score,
    get_risk_level,
    model_confidence,
    get_feature_importance,
    caps_ratio,
    suspicious_salary,
    top_driver,
    matched_scam_phrases,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import pandas as pd
    PD_AVAILABLE = True
except ImportError:
    PD_AVAILABLE = False

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SCAMGUARD-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: #0b0c0f !important;
    color: #e8e5de !important;
}
.stApp { background-color: #0b0c0f !important; }
.block-container { padding-top: 1.2rem !important; max-width: 1400px !important; }

/* ── HIDE SIDEBAR TOGGLE ── */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

/* ── TOPBAR ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1.5rem;
    background: #0f1014;
    border: 1px solid #1c1e24;
    border-radius: 12px;
    margin-bottom: 0.25rem;
}
.topbar-brand { display: flex; align-items: center; gap: 10px; }
.topbar-icon {
    width: 32px; height: 32px; border-radius: 8px;
    background: #1c1e24;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; line-height: 1;
}
.topbar-name {
    font-size: 1rem; font-weight: 800; letter-spacing: 0.06em;
    color: #f5f3ec; text-transform: uppercase;
}
.topbar-sub {
    font-size: 0.65rem; color: #40405a;
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 1px;
}
.topbar-status {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.68rem; font-weight: 600; color: #9090a8;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #c0bdb0; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── HERO ── */
.hero { text-align: center; padding: 2rem 1rem 1.5rem; }
.hero-kicker {
    font-size: 0.63rem; font-weight: 700; letter-spacing: 0.2em;
    text-transform: uppercase; color: #50506a; margin-bottom: 0.7rem;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.1;
    color: #f5f3ec; letter-spacing: -0.02em; margin: 0 0 0.6rem;
}
.hero-title em {
    font-style: normal; color: #ffffff;
    text-decoration: underline;
    text-decoration-color: #3a3a52;
    text-underline-offset: 6px;
}
.hero-desc {
    font-size: 0.88rem; color: #6a6a84;
    max-width: 520px; margin: 0 auto; line-height: 1.65;
}
.hero-rule {
    width: 40px; height: 2px; background: #2a2c32;
    margin: 1.2rem auto 0; border-radius: 2px;
}

/* ── STAT STRIP ── */
.stat-strip { display: flex; gap: 10px; margin: 1rem 0 1.5rem; }
.stat-box {
    flex: 1; background: #0f1014; border: 1px solid #1c1e24;
    border-radius: 10px; padding: 0.8rem 1rem; text-align: center;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem; font-weight: 500; color: #e8e5de; line-height: 1;
}
.stat-lbl {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #40405a; margin-top: 5px;
}

/* ── SECTION LABEL ── */
.slabel {
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.18em;
    text-transform: uppercase; color: #3a3a52;
    display: flex; align-items: center; gap: 10px;
    margin: 1.4rem 0 0.7rem;
}
.slabel::before { content:''; width: 20px; height: 1px; background: #3a3a52; }
.slabel::after  { content:''; flex: 1; height: 1px; background: #1c1e24; }

/* ── INPUT PANEL ── */
.input-panel {
    background: #0f1014; border: 1px solid #1c1e24;
    border-radius: 12px; padding: 1.4rem 1.4rem 1rem;
    margin-bottom: 1rem;
}

/* ── METRIC CARDS ── */
.mc {
    background: #0f1014; border: 1px solid #1c1e24;
    border-radius: 10px; padding: 0.9rem 1rem;
    border-top: 2px solid #2a2c32;
}
.mc.danger { border-top-color: #ffffff; }
.mc.warn   { border-top-color: #888898; }
.mc.good   { border-top-color: #505060; }
.mc.neutral{ border-top-color: #2a2c32; }
.mc-lbl {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #40405a; margin-bottom: 6px;
}
.mc-val {
    font-size: 1.5rem; font-weight: 700; color: #f5f3ec;
    font-family: 'IBM Plex Mono', monospace; line-height: 1.1;
}
.mc-val.danger { color: #ffffff; }
.mc-val.warn   { color: #c0bdb0; }
.mc-val.good   { color: #9090a8; }

/* ── PILLS ── */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 5px;
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.pill-red   { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.2); color: #f5f3ec; }
.pill-amber { background: rgba(160,160,160,0.08); border: 1px solid rgba(160,160,160,0.2); color: #c0bdb0; }
.pill-green { background: rgba(90,90,100,0.12);  border: 1px solid rgba(90,90,100,0.25);  color: #9090a8; }
.pill-gray  { background: rgba(60,60,70,0.12);   border: 1px solid rgba(60,60,70,0.25);   color: #60607a; }

/* ── FEATURE ROW ── */
.feat-row {
    background: #0b0c0f; border: 1px solid #1c1e24;
    border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.feat-label { font-size: 0.8rem; font-weight: 600; color: #c0bdb0; }
.feat-sub   { font-size: 0.65rem; color: #40405a; margin-top: 2px; }
.feat-val   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; color: #9090a8; font-weight: 500;
}

/* ── MINI BAR ── */
.mini-bar-bg   { background: #1a1c22; border-radius: 3px; height: 3px; margin-top: 6px; }
.mini-bar-fill { height: 3px; border-radius: 3px; background: #505060; }

/* ── INSIGHT / ADVISORY / DS-NOTE BOXES ── */
.insight {
    background: #111214; border: 1px solid #1c1e24;
    border-left: 3px solid #505060;
    border-radius: 8px; padding: 0.9rem 1.1rem;
    font-size: 0.82rem; color: #9090a8; line-height: 1.6; margin: 0.6rem 0;
}
.insight b { color: #c0bdb0; }

.ds-note {
    background: #0d0e12; border: 1px solid #1a1c24;
    border-left: 3px solid #2a3048;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.78rem; color: #60708a; line-height: 1.6; margin: 0.5rem 0;
}
.ds-note b    { color: #8090b0; }
.ds-note code {
    background: #1a1c22 !important; color: #8090b0 !important;
    padding: 1px 4px !important; border-radius: 3px !important;
    font-size: 0.72rem !important;
}

.advisory {
    background: #0f1014; border: 1px solid #1c1e24;
    border-left: 3px solid #3a3a52;
    border-radius: 8px; padding: 0.8rem 1rem;
    font-size: 0.8rem; color: #9090a8; line-height: 1.6; margin: 0.5rem 0;
}
.advisory b { color: #c0bdb0; }

/* ── FLAG ROW ── */
.flag-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 0.55rem 0; border-bottom: 1px solid #141619;
    font-size: 0.82rem; color: #c0bdb0;
}
.flag-icon { font-size: 0.75rem; margin-top: 2px; min-width: 16px; }

/* ── TRACE BOX ── */
.trace-box {
    background: #0b0c0f; border: 1px solid #1c1e24;
    border-radius: 8px; padding: 1rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; color: #8888a8; line-height: 1.9;
}
.trace-box span.key  { color: #40405a; }
.trace-box span.val  { color: #c0bdb0; }
.trace-box span.ok   { color: #9090a8; }
.trace-box span.warn { color: #888898; }

/* ── CHECKLIST ── */
.check-item {
    background: #0f1014; border: 1px solid #1c1e24;
    border-radius: 8px; padding: 0.7rem 1rem; margin-bottom: 0.4rem;
    display: flex; gap: 10px; align-items: flex-start;
}
.check-num {
    width: 22px; height: 22px; border-radius: 5px;
    background: rgba(255,255,255,0.03); border: 1px solid #2a2c32;
    color: #9090a8; font-size: 0.7rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.check-title  { font-size: 0.82rem; font-weight: 600; color: #d0cdc6; }
.check-detail { font-size: 0.7rem; color: #50506a; margin-top: 2px; }

/* ── SCORE GAUGE ── */
.gauge-wrap { text-align: center; padding: 1rem 0; }
.gauge-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3.5rem; font-weight: 500; line-height: 1;
}
.gauge-label {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #40405a; margin-top: 8px;
}
.gauge-bar-bg  { background: #1a1c22; border-radius: 4px; height: 6px; margin: 12px 0 6px; }
.gauge-bar-fill{ height: 6px; border-radius: 4px; transition: width 0.5s; }

/* ── STREAMLIT BUTTON ── */
div.stButton > button {
    background: #e8e5de !important;
    color: #0b0c0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
div.stButton > button:hover { opacity: 0.75 !important; transform: translateY(-1px) !important; }

/* ── INPUTS ── */
.stTextInput input, .stTextArea textarea {
    background: #0b0c0f !important;
    border: 1px solid #1c1e24 !important;
    border-radius: 8px !important;
    color: #e8e5de !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #50506a !important;
    box-shadow: 0 0 0 2px rgba(80,80,106,0.2) !important;
}

/* ── EXPANDERS ── */
.stExpander {
    background: #0f1014 !important;
    border: 1px solid #1c1e24 !important;
    border-radius: 10px !important;
}
.stExpander summary {
    color: #c0bdb0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
.stExpander summary p  { color: #c0bdb0 !important; font-size: 0.85rem !important; font-weight: 600 !important; }
.stExpander summary svg{ fill: #50506a !important; }

/* ── MISC ── */
.stAlert      { background: #0f1014 !important; border-radius: 8px !important; border: 1px solid #1c1e24 !important; }
.stDataFrame  { background: #0b0c0f !important; border-radius: 8px !important; }
hr, .stDivider{ border-color: #1c1e24 !important; }
.stSpinner > div { color: #9090a8 !important; }
label {
    color: #72728a !important; font-size: 0.8rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
}

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 1.2rem 0 0.5rem;
    font-size: 0.68rem; color: #28283a;
    border-top: 1px solid #141619;
    margin-top: 2rem; letter-spacing: 0.06em;
}
.footer b { color: #50506a; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# LOAD MODEL ARTIFACTS
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    model      = joblib.load(BASE_DIR / "fraud_model.pkl")
    vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    feat_names = joblib.load(BASE_DIR / "feature_names.pkl")
    return model, vectorizer, feat_names

fraud_model, tfidf_vectorizer, feature_names = load_artifacts()

# =====================================================================
# TOPBAR
# =====================================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-brand">
        <div class="topbar-icon">🛡️</div>
        <div>
            <div class="topbar-name">ScamGuard-AI</div>
            <div class="topbar-sub">Explainable Fraud Detection</div>
        </div>
    </div>
    <div class="topbar-status">
        <div class="status-dot"></div>
        Model Online
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# HERO
# =====================================================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">◈ Logistic Regression · TF-IDF · SHAP · Behavioral Signals</div>
    <h1 class="hero-title">Job Scam <em>Risk Analyzer</em></h1>
    <p class="hero-desc">
        Paste any job posting. Get an explainable fraud risk score powered by
        a trained ML model, behavioral heuristics, and SHAP attribution.
    </p>
    <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# STAT STRIP
# =====================================================================
st.markdown("""
<div class="stat-strip">
    <div class="stat-box"><div class="stat-val">0.35</div><div class="stat-lbl">Decision Threshold</div></div>
    <div class="stat-box"><div class="stat-val">5003</div><div class="stat-lbl">Feature Dims</div></div>
    <div class="stat-box"><div class="stat-val">0.98</div><div class="stat-lbl">AUC-ROC</div></div>
    <div class="stat-box"><div class="stat-val">~92%</div><div class="stat-lbl">Fraud Recall</div></div>
    <div class="stat-box"><div class="stat-val">LR</div><div class="stat-lbl">Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# INPUT FORM
# =====================================================================
st.markdown('<div class="slabel">Job Posting Input</div>', unsafe_allow_html=True)
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    job_title       = st.text_input("Job Title", placeholder="e.g. Data Entry Executive")
    company_profile = st.text_area(
        "Company Profile / Contact Info",
        placeholder="Company details, email address, phone…",
        height=110
    )
with c2:
    salary_range    = st.text_input("Salary Range", placeholder="e.g. ₹15,000/month  or leave blank")
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste full job description here…",
        height=110
    )

st.markdown('</div>', unsafe_allow_html=True)

_, mid, _ = st.columns([3, 1, 3])
with mid:
    run = st.button("🔍 Analyze Risk", use_container_width=True)

# =====================================================================
# ANALYSIS
# =====================================================================
if run:
    if not job_title.strip() and not job_description.strip():
        st.warning("Please enter at least a job title or description.")
        st.stop()

    with st.spinner("Running ML analysis…"):
        time.sleep(0.35)

    # ── Build features & predict (utils.py → build_feature_vector) ───────
    X_final, fd  = build_feature_vector(
        tfidf_vectorizer, job_title, job_description, company_profile, salary_range
    )
    fraud_prob   = fraud_model.predict_proba(X_final)[0][1]

    # ── Scoring & levels (utils.py → compute_risk_score, get_risk_level) ─
    risk_score   = compute_risk_score(fraud_prob, fd)
    lvl, lvl_pill, card_cls, accent_color, advice = get_risk_level(risk_score)

    # ── Confidence & verdict (utils.py → model_confidence) ───────────────
    conf, conf_pill = model_confidence(fraud_prob)
    model_decision  = "FRAUD" if fraud_prob >= FRAUD_THRESHOLD else "LEGITIMATE"

    # ── Adjusted probability points ───────────────────────────────────────
    if fraud_prob >= FRAUD_THRESHOLD:
        adj = 0.5 + (fraud_prob - FRAUD_THRESHOLD) / (1 - FRAUD_THRESHOLD) * 0.5
    else:
        adj = fraud_prob / FRAUD_THRESHOLD * 0.5
    adj = min(adj, 1.0) * 100

    # ── Top driver & extra signals (utils.py → top_driver, caps_ratio…) ──
    driver, contributions = top_driver(adj, fd)
    caps                  = caps_ratio(job_title + " " + job_description)
    sus_sal               = suspicious_salary(salary_range)
    scam_matches          = matched_scam_phrases(job_title, job_description)

    # ── SECTION 1: SUMMARY ───────────────────────────────────────────────
    st.markdown('<div class="slabel">① Risk Assessment Summary</div>', unsafe_allow_html=True)

    g_col, m_col = st.columns([1, 2.5], gap="large")

    with g_col:
        st.markdown(f"""
<div class="gauge-wrap">
    <div class="gauge-num" style="color:{accent_color};">{risk_score}</div>
    <div class="gauge-label">Risk Score / 100</div>
    <div class="gauge-bar-bg">
        <div class="gauge-bar-fill" style="width:{int(risk_score)}%;background:{accent_color};"></div>
    </div>
    <span class="pill {lvl_pill}" style="margin-top:8px;display:inline-block;">{lvl} RISK</span>
</div>
        """, unsafe_allow_html=True)

    with m_col:
        col_a, col_b, col_c, col_d = st.columns(4, gap="small")
        with col_a:
            st.markdown(f"""
<div class="mc {card_cls}">
    <div class="mc-lbl">ML Fraud Prob</div>
    <div class="mc-val {card_cls}">{fraud_prob:.1%}</div>
</div>""", unsafe_allow_html=True)
        with col_b:
            dc = "danger" if model_decision == "FRAUD" else "good"
            st.markdown(f"""
<div class="mc {dc}">
    <div class="mc-lbl">Model Verdict</div>
    <div class="mc-val {dc}" style="font-size:1rem;">{model_decision}</div>
    <div style="font-size:0.6rem;color:#40405a;margin-top:4px;">@ threshold {FRAUD_THRESHOLD}</div>
</div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
<div class="mc neutral">
    <div class="mc-lbl">Confidence</div>
    <div style="margin-top:6px;"><span class="pill {conf_pill}">{conf}</span></div>
</div>""", unsafe_allow_html=True)
        with col_d:
            st.markdown(f"""
<div class="mc neutral">
    <div class="mc-lbl">Top Driver</div>
    <div style="font-size:0.72rem;color:#9090a8;margin-top:6px;line-height:1.4;">{driver[:28]}…</div>
</div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="advisory" style="margin-top:0.75rem;"><b>Recommended Action:</b> {advice}</div>',
            unsafe_allow_html=True
        )
        if abs(fraud_prob - FRAUD_THRESHOLD) < 0.08:
            st.markdown(
                f'<div class="insight"><b>⚡ Borderline:</b> Model probability ({fraud_prob:.1%}) is very close to decision threshold ({FRAUD_THRESHOLD}). Manual review strongly recommended.</div>',
                unsafe_allow_html=True
            )

    # ── SECTION 2: FEATURE BREAKDOWN ─────────────────────────────────────
    st.markdown('<div class="slabel">② Feature Signal Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-note"><b>Source:</b> Trained model features from <code>02_feature_engineering_and_mo...py</code>. '
        'Rule-based signals from <code>eda.py</code> — additive post-model indicators.</div>',
        unsafe_allow_html=True
    )

    f1, f2 = st.columns(2, gap="large")

    with f1:
        st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#40405a;margin-bottom:0.6rem;">Trained Model Features</p>', unsafe_allow_html=True)

        email_status = "⬛ DETECTED" if fd["free_email"] else "⬜ CLEAR"
        st.markdown(f"""
<div class="feat-row">
    <div>
        <div class="feat-label">Free Email Domain</div>
        <div class="feat-sub"><code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">free_email_flag()</code> · utils.py</div>
    </div>
    <div class="feat-val">{email_status}</div>
</div>""", unsafe_allow_html=True)

        urg_pct = min(fd["urgency"] / len(URGENCY_WORDS), 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
    <div style="display:flex;justify-content:space-between;width:100%;">
        <div>
            <div class="feat-label">Urgency Keywords</div>
            <div class="feat-sub"><code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">urgency_score()</code> · utils.py</div>
        </div>
        <div class="feat-val">{fd['urgency']} / {len(URGENCY_WORDS)}</div>
    </div>
    <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(urg_pct*100)}%;"></div></div>
</div>""", unsafe_allow_html=True)

        desc_pct = min(fd["desc_length"] / 1000, 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
    <div style="display:flex;justify-content:space-between;width:100%;">
        <div>
            <div class="feat-label">Description Length</div>
            <div class="feat-sub"><code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">desc_length()</code> · utils.py · Short &lt;200 = higher risk</div>
        </div>
        <div class="feat-val">{fd['desc_length']} chars</div>
    </div>
    <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(desc_pct*100)}%;background:#3a3a52;"></div></div>
</div>""", unsafe_allow_html=True)

        if PD_AVAILABLE and contributions:
            st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#40405a;margin:0.8rem 0 0.4rem;">Score Contributions</p>', unsafe_allow_html=True)
            rows = {k: round(v, 2) for k, v in contributions.items() if v > 0}
            df_c = pd.DataFrame.from_dict(rows, orient="index", columns=["Points"])
            df_c = df_c.sort_values("Points", ascending=False)
            st.dataframe(df_c, use_container_width=True, height=170)

    with f2:
        st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#40405a;margin-bottom:0.6rem;">Additional Behavioral Signals (eda.py)</p>', unsafe_allow_html=True)

        sal_status = "⬜ PROVIDED" if not fd["salary_missing"] else "⬛ MISSING"
        st.markdown(f"""
<div class="feat-row">
    <div>
        <div class="feat-label">Salary Info</div>
        <div class="feat-sub">Opacity signal · <code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">salary_risk</code> from training script §10</div>
    </div>
    <div class="feat-val">{sal_status}</div>
</div>""", unsafe_allow_html=True)

        scam_pct = min(len(scam_matches) / 10, 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
    <div style="display:flex;justify-content:space-between;width:100%;">
        <div>
            <div class="feat-label">Scam Phrases Detected</div>
            <div class="feat-sub">{(', '.join(scam_matches[:4])) if scam_matches else 'None matched'} · <code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">matched_scam_phrases()</code></div>
        </div>
        <div class="feat-val">{len(scam_matches)}</div>
    </div>
    <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(scam_pct*100)}%;background:#888898;"></div></div>
</div>""", unsafe_allow_html=True)

        caps_pct = min(caps / 0.5, 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:6px;">
    <div style="display:flex;justify-content:space-between;width:100%;">
        <div>
            <div class="feat-label">Caps / Hype Ratio</div>
            <div class="feat-sub"><code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">caps_ratio()</code> · utils.py</div>
        </div>
        <div class="feat-val">{caps:.0%}</div>
    </div>
    <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(caps_pct*100)}%;background:#888898;"></div></div>
</div>""", unsafe_allow_html=True)

        sus_text = "⬛ FLAGGED" if sus_sal else "⬜ NORMAL"
        st.markdown(f"""
<div class="feat-row">
    <div>
        <div class="feat-label">Suspicious Salary</div>
        <div class="feat-sub"><code style="background:#1a1c22;color:#9090a8;padding:1px 4px;border-radius:3px;">suspicious_salary()</code> · utils.py</div>
    </div>
    <div class="feat-val">{sus_text}</div>
</div>""", unsafe_allow_html=True)

    # ── SECTION 3: SHAP ──────────────────────────────────────────────────
    st.markdown('<div class="slabel">③ SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-note"><b>Source:</b> <code>expainabiity_and_insights.py</code> · '
        'For LR, SHAP = <code>coef[i] × feature_value[i]</code> in log-odds space — mathematically exact. '
        '<b style="color:#c0bdb0;">Light → FRAUD</b> &nbsp;·&nbsp; <b style="color:#505060;">Dark → LEGIT</b></div>',
        unsafe_allow_html=True
    )

    if SHAP_AVAILABLE and MPL_AVAILABLE:
        with st.spinner("Computing SHAP values…"):
            try:
                explainer   = shap.LinearExplainer(
                    fraud_model, shap.sample(X_final, 1),
                    feature_perturbation="interventional"
                )
                shap_values = explainer.shap_values(X_final)
                sv = np.array(
                    shap_values[0] if isinstance(shap_values, list) else shap_values
                ).flatten()
                n  = min(len(sv), len(feature_names))
                sv, fn    = sv[:n], feature_names[:n]
                top_idx   = np.argsort(np.abs(sv))[-15:][::-1]
                top_names = [fn[i] for i in top_idx]
                top_vals  = sv[top_idx]

                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(9, 5))
                fig.patch.set_facecolor("#0f1014")
                ax.set_facecolor("#0b0c0f")
                colors = ["#e8e5de" if v > 0 else "#505060" for v in top_vals]
                ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1], height=0.6)
                ax.set_yticks(range(len(top_names)))
                ax.set_yticklabels(top_names[::-1], fontsize=8.5, color="#9090a8")
                ax.axvline(0, color="#2a2c32", linewidth=1, linestyle="--")
                ax.set_xlabel("SHAP Value  ( + → FRAUD  ·  − → LEGIT )", fontsize=8.5, color="#50506a")
                ax.set_title(
                    "Top 15 Feature Contributions (SHAP — Logistic Regression)",
                    fontweight="bold", color="#c0bdb0", fontsize=10, pad=12
                )
                ax.tick_params(colors="#50506a")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#1c1e24")
                light_p = mpatches.Patch(color="#e8e5de", label="→ FRAUD")
                dark_p  = mpatches.Patch(color="#505060", label="→ LEGIT")
                ax.legend(
                    handles=[light_p, dark_p], fontsize=8,
                    facecolor="#0f1014", edgecolor="#1c1e24", labelcolor="#9090a8"
                )
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP error: {e}")
    else:
        missing = []
        if not SHAP_AVAILABLE: missing.append("`pip install shap`")
        if not MPL_AVAILABLE:  missing.append("`pip install matplotlib`")
        st.info(f"Install to enable SHAP: {', '.join(missing)}")

    # ── SECTION 4: CONTEXT INTELLIGENCE ──────────────────────────────────
    st.markdown('<div class="slabel">④ Risk Context Intelligence</div>', unsafe_allow_html=True)

    if fd["free_email"] and fd["urgency"] > 1:
        ctx = "<b>Pattern match:</b> Free email + urgency language — matches known mass scam campaign fingerprint."
    elif sus_sal:
        ctx = "<b>Salary anomaly:</b> Vague or unrealistically high salary — classic bait-and-switch."
    elif len(scam_matches) >= 3:
        ctx = f"<b>High scam phrase density:</b> {len(scam_matches)} known scam phrases detected. Lexical fingerprint matches fraudulent postings."
    elif fd["urgency"] > 0:
        ctx = "<b>Pressure tactic:</b> Urgency language reduces applicant due-diligence — a documented manipulation technique."
    elif fraud_prob >= FRAUD_THRESHOLD:
        ctx = f"<b>ML model flagged:</b> Probability ({fraud_prob:.1%}) ≥ threshold ({FRAUD_THRESHOLD}). TF-IDF patterns match training fraud examples."
    else:
        ctx = "<b>No dominant pattern:</b> Behavioral features are clean. ML probability is the primary (weak) signal."

    st.markdown(f'<div class="insight">🧠 {ctx}</div>', unsafe_allow_html=True)

    # ── EXPANDERS ────────────────────────────────────────────────────────
    with st.expander("🔍 Full Rule-Based Explainability Report"):
        any_flag   = False
        flags_html = ""
        if fd["urgency"] > 0:
            hit = [w for w in URGENCY_WORDS if w in job_description.lower()]
            flags_html += f'<div class="flag-row"><div class="flag-icon">▪</div><div>Urgency keywords: <span style="color:#c0bdb0;">{fd["urgency"]} hit(s)</span> — {", ".join(hit)}</div></div>'
            any_flag = True
        if fd["free_email"]:
            hit_d = [d for d in FREE_DOMAINS if d in company_profile.lower()]
            flags_html += f'<div class="flag-row"><div class="flag-icon">▪</div><div>Free email domain: <span style="color:#c0bdb0;">{"".join(hit_d)}</span></div></div>'
            any_flag = True
        if fd["salary_missing"]:
            flags_html += '<div class="flag-row"><div class="flag-icon">◦</div><div>Salary not provided</div></div>'
            any_flag = True
        if scam_matches:
            flags_html += f'<div class="flag-row"><div class="flag-icon">◦</div><div>Scam phrases: {", ".join(scam_matches[:5])}</div></div>'
            any_flag = True
        if caps > 0.15:
            flags_html += f'<div class="flag-row"><div class="flag-icon">◦</div><div>High caps ratio: {caps:.0%}</div></div>'
            any_flag = True
        if sus_sal:
            flags_html += f'<div class="flag-row"><div class="flag-icon">◦</div><div>Suspicious salary: {salary_range}</div></div>'
            any_flag = True
        if not any_flag:
            flags_html = '<div style="color:#9090a8;font-size:0.82rem;">✓ No rule-based flags raised</div>'

        st.markdown(f'<div class="trace-box">{flags_html}</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="trace-box" style="margin-top:0.75rem;">
<span class="key">raw_probability   </span><span class="val">{fraud_prob:.4f}</span>
<span class="key">decision_threshold</span><span class="val">{FRAUD_THRESHOLD}</span>
<span class="key">model_verdict     </span><span class="{'val' if model_decision=='FRAUD' else 'ok'}">{model_decision}</span>
<span class="key">adjusted_prob     </span><span class="val">{adj:.1f} pts</span>
<span class="key">composite_score   </span><span class="{'val' if risk_score>=60 else 'warn' if risk_score>=30 else 'ok'}">{risk_score} / 100</span>
<span class="key">risk_level        </span><span class="{'val' if lvl=='HIGH' else 'warn' if lvl=='MEDIUM' else 'ok'}">{lvl}</span>
</div>""", unsafe_allow_html=True)

    with st.expander("📊 Feature Importance (from expainabiity_and_insights.py)"):
        # get_feature_importance mirrors expainabiity_and_insights.py §Feature importance
        importance_pairs = get_feature_importance(fraud_model, feature_names)
        top15 = importance_pairs[:15]
        if PD_AVAILABLE:
            df_imp = pd.DataFrame(top15, columns=["Feature", "Coefficient"])
            df_imp["Direction"] = df_imp["Coefficient"].apply(lambda x: "→ FRAUD" if x > 0 else "→ LEGIT")
            st.dataframe(df_imp, use_container_width=True, height=400)
        else:
            for feat, coef in top15:
                st.text(f"{feat:<40} {coef:+.4f}")

    with st.expander("✅ Defensive Checklist"):
        items = [
            ("Verify company on MCA21 / LinkedIn",      "Official registration eliminates ghost companies"),
            ("Confirm official domain email",            "Legit companies don't use Gmail/Yahoo for hiring"),
            ("Cross-check salary on Glassdoor/Naukri",  "Compare claimed CTC with market rates"),
            ("Never pay upfront fees",                   "No legitimate employer charges registration fees"),
            ("Don't share Aadhaar/PAN early",            "Only after receiving a formal offer letter"),
            ("Google: [company] + scam/fraud",           "Many scam companies have reported complaints"),
            ("Verify recruiter on LinkedIn",             "Check profile age, connections, endorsements"),
        ]
        for i, (title, detail) in enumerate(items, 1):
            st.markdown(f"""
<div class="check-item">
    <div class="check-num">{i}</div>
    <div>
        <div class="check-title">{title}</div>
        <div class="check-detail">{detail}</div>
    </div>
</div>""", unsafe_allow_html=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown(f"""
<div class="footer">
    SCAMGUARD-AI &nbsp;·&nbsp; Logistic Regression + TF-IDF &nbsp;·&nbsp;
    Decision threshold: <b>{FRAUD_THRESHOLD}</b> &nbsp;·&nbsp;
    Trained on EMSCAD dataset &nbsp;·&nbsp; Not a substitute for manual verification
</div>
""", unsafe_allow_html=True)
