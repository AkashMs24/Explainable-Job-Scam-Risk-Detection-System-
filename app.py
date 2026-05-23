"""
SCAMGUARD-AI  |  app.py  (v7 — Fixed + Model Comparison UI)
============================================================
Fixes applied:
  - ZeroDivisionError in SHAP force plot (max_abs = 0 when all features zero)
  - Model comparison section added (LR vs XGBoost vs RF vs GB)
  - Justified LR choice for exact SHAP
  - Stratified CV results displayed
  - FastAPI backend note added to UI
"""

import streamlit as st
import numpy as np
import joblib
import time
from pathlib import Path

from utils import (
    URGENCY_WORDS,
    FREE_DOMAINS,
    FRAUD_THRESHOLD,
    build_feature_vector,
    compute_risk_score,
    get_risk_level,
    model_confidence,
    get_feature_importance,
    compute_shap_values,
    top_shap_features,
    caps_ratio,
    suspicious_salary,
    top_driver,
    matched_scam_phrases,
)

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
    background-color: #080809 !important;
    color: #f0ede6 !important;
}
.stApp            { background-color: #080809 !important; }
.block-container  { padding-top: 1rem !important; max-width: 1380px !important; }

[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

/* ── TOPBAR ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 1.6rem;
    background: #0e0f12;
    border: 1px solid #222330;
    border-radius: 14px;
    margin-bottom: 0.5rem;
}
.topbar-brand   { display: flex; align-items: center; gap: 12px; }
.topbar-icon    {
    width: 36px; height: 36px; border-radius: 10px;
    background: #1a1b22; display: flex; align-items: center;
    justify-content: center; font-size: 18px;
}
.topbar-name    {
    font-size: 1.15rem; font-weight: 800; letter-spacing: 0.07em;
    color: #ffffff; text-transform: uppercase;
}
.topbar-sub     {
    font-size: 0.72rem; color: #555568;
    letter-spacing: 0.1em; text-transform: uppercase; margin-top: 2px;
}
.status-row     { display: flex; align-items: center; gap: 8px;
                  font-size: 0.75rem; font-weight: 700; color: #aaaabc;
                  letter-spacing: 0.08em; text-transform: uppercase; }
.status-dot     { width: 7px; height: 7px; border-radius: 50%;
                  background: #ddddee; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

/* ── HERO ── */
.hero           { text-align: center; padding: 2.2rem 1rem 1.8rem; }
.hero-kicker    {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.22em;
    text-transform: uppercase; color: #555568; margin-bottom: 0.8rem;
}
.hero-title     {
    font-size: 2.8rem; font-weight: 800; line-height: 1.1;
    color: #ffffff; letter-spacing: -0.02em; margin: 0 0 0.8rem;
}
.hero-title em  { font-style: normal; text-decoration: underline;
                  text-decoration-color: #333344; text-underline-offset: 7px; }
.hero-desc      {
    font-size: 1rem; color: #777788;
    max-width: 540px; margin: 0 auto; line-height: 1.7;
}
.hero-rule      { width: 40px; height: 2px; background: #222230;
                  margin: 1.4rem auto 0; border-radius: 2px; }

/* ── STAT STRIP ── */
.stat-strip     { display: flex; gap: 10px; margin: 1.2rem 0 1.8rem; }
.stat-box       {
    flex: 1; background: #0e0f12; border: 1px solid #1e1f28;
    border-radius: 12px; padding: 1rem 1.1rem; text-align: center;
}
.stat-val       {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem; font-weight: 500; color: #ffffff; line-height: 1;
}
.stat-lbl       {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; color: #444456; margin-top: 6px;
}

/* ── SECTION LABELS ── */
.slabel {
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.2em;
    text-transform: uppercase; color: #444456;
    display: flex; align-items: center; gap: 10px;
    margin: 1.6rem 0 0.9rem;
}
.slabel::before { content:''; width: 20px; height: 1px; background: #333340; }
.slabel::after  { content:''; flex: 1; height: 1px; background: #1a1b22; }

/* ── INPUT PANEL ── */
.input-panel {
    background: #0e0f12; border: 1px solid #1e1f28;
    border-radius: 14px; padding: 1.6rem 1.6rem 1.2rem;
    margin-bottom: 1.2rem;
}

/* ── METRIC CARDS ── */
.mc {
    background: #0e0f12; border: 1px solid #1e1f28;
    border-radius: 12px; padding: 1rem 1.1rem;
    border-top: 3px solid #2a2b38;
}
.mc.danger { border-top-color: #ffffff; }
.mc.warn   { border-top-color: #aaaabc; }
.mc.good   { border-top-color: #555568; }
.mc.neutral{ border-top-color: #2a2b38; }
.mc-lbl {
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #444456; margin-bottom: 8px;
}
.mc-val {
    font-size: 1.7rem; font-weight: 700; color: #ffffff;
    font-family: 'IBM Plex Mono', monospace; line-height: 1.1;
}
.mc-val.danger { color: #ffffff; }
.mc-val.warn   { color: #ccccde; }
.mc-val.good   { color: #aaaabc; }

/* ── PILLS ── */
.pill {
    display: inline-block; padding: 3px 12px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 800;
    letter-spacing: 0.1em; text-transform: uppercase;
}
.pill-red   { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.25); color: #ffffff; }
.pill-amber { background: rgba(200,200,200,0.08); border: 1px solid rgba(200,200,200,0.22); color: #ddddee; }
.pill-green { background: rgba(100,100,120,0.12); border: 1px solid rgba(100,100,120,0.28); color: #aaaabc; }
.pill-gray  { background: rgba(60,60,80,0.12);   border: 1px solid rgba(60,60,80,0.28);   color: #666678; }

/* ── GAUGE ── */
.gauge-wrap  { text-align: center; padding: 1.2rem 0; }
.gauge-num   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 4rem; font-weight: 500; line-height: 1;
}
.gauge-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.16em;
    text-transform: uppercase; color: #444456; margin-top: 10px;
}
.gauge-bar-bg   { background: #1a1b22; border-radius: 5px; height: 7px; margin: 14px 0 8px; }
.gauge-bar-fill { height: 7px; border-radius: 5px; transition: width 0.5s; }

/* ── FEATURE ROWS ── */
.feat-row {
    background: #0b0c0f; border: 1px solid #1a1b22;
    border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; justify-content: space-between;
}
.feat-label { font-size: 0.9rem; font-weight: 700; color: #ddddee; }
.feat-sub   { font-size: 0.68rem; color: #444456; margin-top: 3px; }
.feat-val   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem; color: #aaaabc; font-weight: 500;
}

/* ── MINI BAR ── */
.mini-bar-bg   { background: #1a1b22; border-radius: 3px; height: 4px; margin-top: 8px; }
.mini-bar-fill { height: 4px; border-radius: 3px; background: #555568; }

/* ── SHAP BARS ── */
.shap-row {
    display: flex; align-items: center; gap: 10px;
    padding: 5px 0; border-bottom: 1px solid #111118;
}
.shap-name {
    font-size: 0.82rem; color: #ccccde; min-width: 200px;
    flex-shrink: 0; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
    font-family: 'IBM Plex Mono', monospace;
}
.shap-track { flex: 1; height: 20px; position: relative; }
.shap-bar   { position: absolute; top: 4px; height: 12px; border-radius: 2px; }
.shap-mid   { position: absolute; left: 50%; top: 0; width: 1px; height: 20px;
              background: #2a2b38; }
.shap-num   {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; min-width: 60px; text-align: right;
    flex-shrink: 0; font-weight: 500;
}

/* ── MODEL COMPARISON TABLE ── */
.model-row {
    background: #0b0c0f; border: 1px solid #1a1b22;
    border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 0.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.model-name { font-size: 0.9rem; font-weight: 700; color: #ddddee; min-width: 180px; }
.model-badge {
    display: inline-block; padding: 2px 10px; border-radius: 5px;
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-selected { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.25); color: #ffffff; }
.badge-other    { background: rgba(60,60,80,0.12);   border: 1px solid rgba(60,60,80,0.28);   color: #666678; }

/* ── INSIGHT / DS-NOTE / ADVISORY ── */
.insight {
    background: #111218; border: 1px solid #1e1f28;
    border-left: 3px solid #555568;
    border-radius: 10px; padding: 1rem 1.2rem;
    font-size: 0.88rem; color: #aaaabc; line-height: 1.65; margin: 0.7rem 0;
}
.insight b { color: #ffffff; }

.ds-note {
    background: #0d0e14; border: 1px solid #1a1b22;
    border-left: 3px solid #2a3050;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.82rem; color: #666680; line-height: 1.6; margin: 0.6rem 0;
}
.ds-note b    { color: #9090b0; }
.ds-note code {
    background: #1a1c28 !important; color: #9090b0 !important;
    padding: 1px 5px !important; border-radius: 3px !important;
    font-size: 0.76rem !important;
}

.advisory {
    background: #0e0f12; border: 1px solid #1e1f28;
    border-left: 3px solid #444456;
    border-radius: 10px; padding: 0.9rem 1.1rem;
    font-size: 0.9rem; color: #aaaabc; line-height: 1.6; margin: 0.6rem 0;
}
.advisory b { color: #ffffff; }

/* ── TRACE BOX ── */
.trace-box {
    background: #0b0c0f; border: 1px solid #1a1b22;
    border-radius: 10px; padding: 1.1rem 1.3rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem; color: #888898; line-height: 2.1;
}
.trace-box span.key  { color: #444456; }
.trace-box span.val  { color: #ddddee; }
.trace-box span.ok   { color: #aaaabc; }
.trace-box span.warn { color: #888898; }

/* ── FLAG ROW ── */
.flag-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 0.6rem 0; border-bottom: 1px solid #141618;
    font-size: 0.86rem; color: #ccccde;
}
.flag-icon { font-size: 0.8rem; margin-top: 2px; min-width: 18px; }

/* ── CHECKLIST ── */
.check-item {
    background: #0e0f12; border: 1px solid #1e1f28;
    border-radius: 10px; padding: 0.85rem 1.1rem; margin-bottom: 0.5rem;
    display: flex; gap: 12px; align-items: flex-start;
}
.check-num {
    width: 26px; height: 26px; border-radius: 6px;
    background: rgba(255,255,255,0.04); border: 1px solid #2a2b38;
    color: #aaaabc; font-size: 0.78rem; font-weight: 800;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.check-title  { font-size: 0.9rem; font-weight: 700; color: #eeeeee; }
.check-detail { font-size: 0.75rem; color: #555568; margin-top: 3px; }

/* ── STREAMLIT OVERRIDES ── */
div.stButton > button {
    background: #f0ede6 !important; color: #080809 !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
    font-size: 0.95rem !important; letter-spacing: 0.04em !important;
    padding: 0.55rem 1.4rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
}
div.stButton > button:hover { opacity: 0.8 !important; transform: translateY(-1px) !important; }

.stTextInput input, .stTextArea textarea {
    background: #0b0c0f !important; border: 1px solid #1e1f28 !important;
    border-radius: 10px !important; color: #f0ede6 !important;
    font-family: 'Syne', sans-serif !important; font-size: 0.95rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #555568 !important;
    box-shadow: 0 0 0 2px rgba(85,85,104,0.25) !important;
}

label {
    color: #888898 !important; font-size: 0.85rem !important;
    font-weight: 700 !important; letter-spacing: 0.05em !important;
}

.stExpander {
    background: #0e0f12 !important; border: 1px solid #1e1f28 !important;
    border-radius: 12px !important;
}
.stExpander summary p   { color: #ccccde !important; font-size: 0.92rem !important; font-weight: 700 !important; }
.stExpander summary svg { fill: #555568 !important; }

.stAlert { background: #0e0f12 !important; border-radius: 10px !important; border: 1px solid #1e1f28 !important; }
hr, .stDivider { border-color: #1a1b22 !important; }
.stSpinner > div { color: #aaaabc !important; }

/* ── FOOTER ── */
.footer {
    text-align: center; padding: 1.4rem 0 0.6rem;
    font-size: 0.72rem; color: #2a2b38;
    border-top: 1px solid #151620; margin-top: 2.5rem;
    letter-spacing: 0.07em;
}
.footer b { color: #555568; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    model       = joblib.load(BASE_DIR / "fraud_model.pkl")
    vectorizer  = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    feat_names  = joblib.load(BASE_DIR / "feature_names.pkl")
    return model, vectorizer, feat_names

fraud_model, tfidf_vectorizer, feature_names = load_artifacts()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON DATA
# (populated by 02_feature_engineering_and_mo...py benchmarking section)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_COMPARISON = [
    {"name": "Logistic Regression", "auc": 0.9800, "f1": 0.88, "cv_auc": "0.96 ± 0.01", "selected": True,
     "reason": "Chosen — exact SHAP without approximation"},
    {"name": "XGBoost",             "auc": 0.9750, "f1": 0.86, "cv_auc": "0.95 ± 0.01", "selected": False,
     "reason": "Good AUC but requires TreeSHAP approximation"},
    {"name": "Random Forest",       "auc": 0.9680, "f1": 0.84, "cv_auc": "0.94 ± 0.02", "selected": False,
     "reason": "Ensemble but slower + approx SHAP"},
    {"name": "Gradient Boosting",   "auc": 0.9710, "f1": 0.85, "cv_auc": "0.94 ± 0.01", "selected": False,
     "reason": "Competitive but no explainability advantage"},
]


# ═══════════════════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">🛡️</div>
    <div>
      <div class="topbar-name">ScamGuard-AI</div>
      <div class="topbar-sub">Explainable Fraud Detection · FastAPI + Streamlit</div>
    </div>
  </div>
  <div class="status-row">
    <div class="status-dot"></div> Model Online
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-kicker">◈ Logistic Regression · TF-IDF · Exact SHAP · Behavioral Signals · FastAPI</div>
  <h1 class="hero-title">Job Scam <em>Risk Analyzer</em></h1>
  <p class="hero-desc">
    Paste any job posting. Get an explainable fraud risk score powered by
    a trained ML model, behavioral heuristics, and exact SHAP attribution.
    4 algorithms benchmarked — LR chosen for mathematically exact SHAP.
  </p>
  <div class="hero-rule"></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAT STRIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="stat-strip">
  <div class="stat-box"><div class="stat-val">0.35</div><div class="stat-lbl">Decision Threshold</div></div>
  <div class="stat-box"><div class="stat-val">5003</div><div class="stat-lbl">Feature Dims</div></div>
  <div class="stat-box"><div class="stat-val">0.98</div><div class="stat-lbl">AUC-ROC</div></div>
  <div class="stat-box"><div class="stat-val">0.96±0.01</div><div class="stat-lbl">5-Fold CV AUC</div></div>
  <div class="stat-box"><div class="stat-val">~92%</div><div class="stat-lbl">Fraud Recall</div></div>
  <div class="stat-box"><div class="stat-val">4</div><div class="stat-lbl">Models Benchmarked</div></div>
  <div class="stat-box"><div class="stat-val">LR</div><div class="stat-lbl">Final Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON SECTION (always visible — interview-ready)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="slabel">⓪ Model Benchmarking — Why Logistic Regression?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ds-note">'
    '<b>Design decision:</b> 4 algorithms were trained on the same 5003-dim feature space '
    '(TF-IDF 5000 + 3 behavioral). LR was selected because it yields '
    '<b style="color:#9090b0;">mathematically exact SHAP values</b> '
    'via <code>φᵢ = coef[i] × feature_value[i]</code> — no TreeSHAP approximation needed. '
    'AUC difference vs XGBoost is only 0.005, making interpretability the decisive factor. '
    '5-fold stratified CV confirms results are not due to data leakage.'
    '</div>',
    unsafe_allow_html=True
)

mc1, mc2 = st.columns([2, 1], gap="large")

with mc1:
    for m in MODEL_COMPARISON:
        badge_cls  = "badge-selected" if m["selected"] else "badge-other"
        badge_txt  = "✓ SELECTED" if m["selected"] else "BENCHMARKED"
        row_border = "border: 1px solid #2a2b38;" if m["selected"] else ""
        st.markdown(f"""
<div class="model-row" style="{row_border}">
  <div class="model-name">{m['name']}</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#aaaabc;min-width:80px;">
    AUC {m['auc']:.4f}
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:#888898;min-width:80px;">
    F1 {m['f1']:.2f}
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#666678;min-width:110px;">
    CV {m['cv_auc']}
  </div>
  <div style="font-size:0.75rem;color:#555568;flex:1;text-align:right;">{m['reason']}</div>
  <div style="margin-left:14px;"><span class="model-badge {badge_cls}">{badge_txt}</span></div>
</div>""", unsafe_allow_html=True)

with mc2:
    st.markdown("""
<div class="mc neutral" style="height:100%;">
  <div class="mc-lbl">CV Methodology</div>
  <div style="font-size:0.82rem;color:#aaaabc;line-height:1.8;margin-top:10px;">
    ▪ 5-Fold Stratified KFold<br>
    ▪ class_weight = 'balanced'<br>
    ▪ stratify=y in train_test_split<br>
    ▪ 80/20 train-test split<br>
    ▪ EMSCAD dataset (~18K rows)<br>
    ▪ ~5% fraud rate handled
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="slabel">Job Posting Input</div>', unsafe_allow_html=True)
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")
with c1:
    job_title       = st.text_input("Job Title", placeholder="e.g. Data Entry Executive")
    company_profile = st.text_area(
        "Company Profile / Contact Info",
        placeholder="Company details, email address, phone…",
        height=120
    )
with c2:
    salary_range    = st.text_input("Salary Range", placeholder="e.g. ₹15,000/month  or leave blank")
    job_description = st.text_area(
        "Job Description",
        placeholder="Paste full job description here…",
        height=120
    )

st.markdown('</div>', unsafe_allow_html=True)

_, mid, _ = st.columns([3, 1, 3])
with mid:
    run = st.button("🔍  Analyze Risk", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
if run:
    if not job_title.strip() and not job_description.strip():
        st.warning("Please enter at least a job title or description.")
        st.stop()

    with st.spinner("Running ML analysis…"):
        time.sleep(0.25)

    # ── build_feature_vector ──────────────────────────────────────────────────
    X_final, fd = build_feature_vector(
        tfidf_vectorizer, job_title, job_description, company_profile, salary_range
    )

    # ── Model prediction ──────────────────────────────────────────────────────
    fraud_prob     = fraud_model.predict_proba(X_final)[0][1]
    model_decision = "FRAUD" if fraud_prob >= FRAUD_THRESHOLD else "LEGITIMATE"

    # ── Risk scoring ──────────────────────────────────────────────────────────
    risk_score = compute_risk_score(fraud_prob, fd)
    lvl, lvl_pill, card_cls, accent_color, advice = get_risk_level(risk_score)

    # ── Confidence ────────────────────────────────────────────────────────────
    conf, conf_pill = model_confidence(fraud_prob)

    # ── Adjusted probability ──────────────────────────────────────────────────
    if fraud_prob >= FRAUD_THRESHOLD:
        adj = 0.5 + (fraud_prob - FRAUD_THRESHOLD) / (1 - FRAUD_THRESHOLD) * 0.5
    else:
        adj = fraud_prob / FRAUD_THRESHOLD * 0.5
    adj = min(adj, 1.0) * 100

    # ── EXACT SHAP ────────────────────────────────────────────────────────────
    shap_vals, shap_intercept, shap_log_odds = compute_shap_values(
        fraud_model, X_final, feature_names
    )
    shap_top = top_shap_features(shap_vals, feature_names, n=15)

    # ── Behavioral signals ────────────────────────────────────────────────────
    driver, contributions = top_driver(adj, fd)
    caps                  = caps_ratio(job_title + " " + job_description)
    sus_sal               = suspicious_salary(salary_range)
    scam_matches          = matched_scam_phrases(job_title, job_description)


    # ═══════════════════════════════════════════════════════════════════════════
    # ① RISK ASSESSMENT SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="slabel">① Risk Assessment Summary</div>', unsafe_allow_html=True)

    g_col, m_col = st.columns([1, 2.8], gap="large")

    with g_col:
        st.markdown(f"""
<div class="gauge-wrap">
  <div class="gauge-num" style="color:{accent_color};">{risk_score}</div>
  <div class="gauge-label">Risk Score / 100</div>
  <div class="gauge-bar-bg">
    <div class="gauge-bar-fill" style="width:{int(risk_score)}%;background:{accent_color};"></div>
  </div>
  <span class="pill {lvl_pill}" style="margin-top:10px;display:inline-block;">{lvl} RISK</span>
</div>
        """, unsafe_allow_html=True)

    with m_col:
        ca, cb, cc, cd = st.columns(4, gap="small")
        with ca:
            st.markdown(f"""
<div class="mc {card_cls}">
  <div class="mc-lbl">ML Fraud Prob</div>
  <div class="mc-val {card_cls}">{fraud_prob:.1%}</div>
</div>""", unsafe_allow_html=True)
        with cb:
            dc = "danger" if model_decision == "FRAUD" else "good"
            st.markdown(f"""
<div class="mc {dc}">
  <div class="mc-lbl">Model Verdict</div>
  <div class="mc-val {dc}" style="font-size:1.1rem;">{model_decision}</div>
  <div style="font-size:0.65rem;color:#444456;margin-top:5px;">@ threshold {FRAUD_THRESHOLD}</div>
</div>""", unsafe_allow_html=True)
        with cc:
            st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">Confidence</div>
  <div style="margin-top:8px;"><span class="pill {conf_pill}">{conf}</span></div>
</div>""", unsafe_allow_html=True)
        with cd:
            st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">Top Driver</div>
  <div style="font-size:0.78rem;color:#aaaabc;margin-top:8px;line-height:1.5;">{driver}</div>
</div>""", unsafe_allow_html=True)

        st.markdown(
            f'<div class="advisory" style="margin-top:0.9rem;"><b>Recommended Action:</b> {advice}</div>',
            unsafe_allow_html=True
        )
        if abs(fraud_prob - FRAUD_THRESHOLD) < 0.08:
            st.markdown(
                f'<div class="insight"><b>⚡ Borderline case:</b> '
                f'P(fraud) = {fraud_prob:.1%} is within 8 pts of threshold {FRAUD_THRESHOLD}. '
                f'Manual review strongly recommended.</div>',
                unsafe_allow_html=True
            )


    # ═══════════════════════════════════════════════════════════════════════════
    # ② FEATURE SIGNAL BREAKDOWN
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="slabel">② Feature Signal Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-note"><b>Source:</b> Trained model features from '
        '<code>02_feature_engineering_and_mo...py</code> §3–§5. '
        'Behavioral signals from <code>eda.py</code> → consolidated in <code>utils.py</code>. '
        'All 5003 dimensions feed into the LR model; the 3 behavioral features '
        'also contribute additively to the composite risk score via §10.</div>',
        unsafe_allow_html=True
    )

    f1, f2 = st.columns(2, gap="large")

    with f1:
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:800;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#555568;margin-bottom:0.8rem;">'
            'Trained Model Features (5003-dim input)</p>',
            unsafe_allow_html=True
        )

        email_status = "⬛ DETECTED" if fd["free_email"] else "⬜ CLEAR"
        st.markdown(f"""
<div class="feat-row">
  <div>
    <div class="feat-label">Free Email Domain</div>
    <div class="feat-sub"><code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">free_email_flag()</code> · utils.py → eda.py origin</div>
  </div>
  <div class="feat-val">{email_status}</div>
</div>""", unsafe_allow_html=True)

        urg_pct = min(fd["urgency"] / max(len(URGENCY_WORDS), 1), 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:8px;">
  <div style="display:flex;justify-content:space-between;width:100%;">
    <div>
      <div class="feat-label">Urgency Keywords</div>
      <div class="feat-sub"><code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">urgency_score()</code> · utils.py → eda.py origin</div>
    </div>
    <div class="feat-val">{fd['urgency']} / {len(URGENCY_WORDS)}</div>
  </div>
  <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(urg_pct*100)}%;"></div></div>
</div>""", unsafe_allow_html=True)

        desc_pct = min(fd["desc_length"] / 1000, 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:8px;">
  <div style="display:flex;justify-content:space-between;width:100%;">
    <div>
      <div class="feat-label">Description Length</div>
      <div class="feat-sub"><code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">desc_length()</code> · utils.py  ·  Short &lt;200 = higher risk</div>
    </div>
    <div class="feat-val">{fd['desc_length']} chars</div>
  </div>
  <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(desc_pct*100)}%;background:#444456;"></div></div>
</div>""", unsafe_allow_html=True)

        if PD_AVAILABLE and contributions:
            st.markdown(
                '<p style="font-size:0.75rem;font-weight:800;letter-spacing:0.12em;'
                'text-transform:uppercase;color:#555568;margin:1rem 0 0.5rem;">'
                'Score Contributions (§10 formula)</p>',
                unsafe_allow_html=True
            )
            rows = {k: round(v, 2) for k, v in contributions.items() if v >= 0}
            df_c = pd.DataFrame.from_dict(rows, orient="index", columns=["Points"])
            df_c = df_c.sort_values("Points", ascending=False)
            st.dataframe(df_c, use_container_width=True, height=175)

    with f2:
        st.markdown(
            '<p style="font-size:0.75rem;font-weight:800;letter-spacing:0.12em;'
            'text-transform:uppercase;color:#555568;margin-bottom:0.8rem;">'
            'Additional Behavioral Signals (eda.py)</p>',
            unsafe_allow_html=True
        )

        sal_status = "⬜ PROVIDED" if not fd["salary_missing"] else "⬛ MISSING"
        st.markdown(f"""
<div class="feat-row">
  <div>
    <div class="feat-label">Salary Info</div>
    <div class="feat-sub">Opacity signal · <code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">salary_missing</code> in §10 risk formula</div>
  </div>
  <div class="feat-val">{sal_status}</div>
</div>""", unsafe_allow_html=True)

        scam_pct  = min(len(scam_matches) / 10, 1.0)
        scam_text = ", ".join(scam_matches[:4]) if scam_matches else "None matched"
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:8px;">
  <div style="display:flex;justify-content:space-between;width:100%;">
    <div>
      <div class="feat-label">Scam Phrases</div>
      <div class="feat-sub">{scam_text} · <code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">matched_scam_phrases()</code></div>
    </div>
    <div class="feat-val">{len(scam_matches)}</div>
  </div>
  <div class="mini-bar-bg" style="width:100%;"><div class="mini-bar-fill" style="width:{int(scam_pct*100)}%;background:#888898;"></div></div>
</div>""", unsafe_allow_html=True)

        caps_pct = min(caps / 0.5, 1.0)
        st.markdown(f"""
<div class="feat-row" style="flex-direction:column;align-items:flex-start;gap:8px;">
  <div style="display:flex;justify-content:space-between;width:100%;">
    <div>
      <div class="feat-label">Caps / Hype Ratio</div>
      <div class="feat-sub"><code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">caps_ratio()</code> · utils.py  ·  eda.py origin</div>
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
    <div class="feat-sub"><code style="background:#1a1b22;color:#888898;padding:2px 5px;border-radius:3px;">suspicious_salary()</code> · utils.py  ·  eda.py origin</div>
  </div>
  <div class="feat-val">{sus_text}</div>
</div>""", unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # ③ SHAP EXPLAINABILITY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="slabel">③ SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ds-note">'
        '<b>Source:</b> <code>utils.compute_shap_values()</code> — called from <code>app.py</code>. '
        'Formula: <code>φᵢ = coef[i] × feature_value[i]</code> in log-odds space. '
        'Mathematically exact for LR — no external shap package required. '
        '<b style="color:#aaaabc;">White bars → push toward FRAUD</b> &nbsp;·&nbsp; '
        '<b style="color:#555568;">Gray bars → push toward LEGIT</b>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── SHAP summary metrics ──────────────────────────────────────────────────
    sm1, sm2, sm3, sm4 = st.columns(4, gap="small")
    with sm1:
        st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">Intercept (bias)</div>
  <div class="mc-val" style="font-size:1.3rem;">{shap_intercept:+.3f}</div>
  <div style="font-size:0.65rem;color:#444456;margin-top:4px;">log-odds base</div>
</div>""", unsafe_allow_html=True)
    with sm2:
        st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">SHAP Sum Σφᵢ</div>
  <div class="mc-val" style="font-size:1.3rem;">{float(shap_vals.sum()):+.3f}</div>
  <div style="font-size:0.65rem;color:#444456;margin-top:4px;">feature contributions</div>
</div>""", unsafe_allow_html=True)
    with sm3:
        st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">Log-odds</div>
  <div class="mc-val" style="font-size:1.3rem;">{shap_log_odds:+.3f}</div>
  <div style="font-size:0.65rem;color:#444456;margin-top:4px;">intercept + Σφᵢ</div>
</div>""", unsafe_allow_html=True)
    with sm4:
        check_lo  = np.log(fraud_prob / (1 - fraud_prob + 1e-10))
        match     = abs(shap_log_odds - check_lo) < 0.01
        check_col = "#aaaabc" if match else "#ffffff"
        check_lbl = "✓ exact match" if match else "⚠ mismatch"
        st.markdown(f"""
<div class="mc neutral">
  <div class="mc-lbl">Integrity Check</div>
  <div class="mc-val" style="font-size:1.1rem;color:{check_col};">{check_lbl}</div>
  <div style="font-size:0.65rem;color:#444456;margin-top:4px;">logit(P) = log-odds</div>
</div>""", unsafe_allow_html=True)

    # ── SHAP matplotlib bar chart ─────────────────────────────────────────────
    if MPL_AVAILABLE and shap_top:
        names_shap  = [p[0] for p in shap_top][::-1]
        values_shap = [p[1] for p in shap_top][::-1]
        colors_shap = ["#e8e5de" if v > 0 else "#3a3b4a" for v in values_shap]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor("#0e0f12")
        ax.set_facecolor("#0b0c0f")

        ax.barh(range(len(names_shap)), values_shap, color=colors_shap, height=0.62, zorder=2)
        ax.set_yticks(range(len(names_shap)))
        ax.set_yticklabels(names_shap, fontsize=9.5, color="#ccccde", fontfamily="monospace")
        ax.axvline(0, color="#2a2b38", linewidth=1, linestyle="--", zorder=1)
        ax.set_xlabel("SHAP value  (log-odds)   +→ FRAUD   −→ LEGIT",
                      fontsize=9.5, color="#555568", labelpad=8)
        ax.set_title(
            f"Top 15 Feature Contributions — Exact SHAP (LR)   |   "
            f"P(fraud) = {fraud_prob:.1%}   |   log-odds = {shap_log_odds:+.3f}",
            fontweight="bold", color="#f0ede6", fontsize=11, pad=14
        )
        ax.tick_params(axis='both', colors="#555568", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1f28")
        ax.grid(axis='x', color="#1a1b22", linewidth=0.5, zorder=0)

        fraud_p = mpatches.Patch(color="#e8e5de", label="→ FRAUD")
        legit_p = mpatches.Patch(color="#3a3b4a", label="→ LEGIT")
        ax.legend(handles=[fraud_p, legit_p], fontsize=9.5,
                  facecolor="#0e0f12", edgecolor="#1e1f28", labelcolor="#ccccde",
                  loc="lower right")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Inline SHAP force bars ────────────────────────────────────────────────
    # FIX: guard against max_abs = 0 when all SHAP values are zero
    max_abs  = max((abs(v) for _, v in shap_top), default=0.0)
    max_abs  = max_abs if max_abs > 0 else 1.0   # ← ZeroDivisionError fix

    bars_html = ""
    for name, val in shap_top:
        pct      = min(abs(val) / max_abs * 46, 46)
        color    = "#e8e5de" if val > 0 else "#3a3b4a"
        val_str  = f"{val:+.4f}"
        val_col  = "#e8e5de" if val > 0 else "#888898"
        if val >= 0:
            bar_style = f"left:50%;width:{pct}%;background:{color};"
        else:
            bar_style = f"right:50%;width:{pct}%;background:{color};"

        bars_html += f"""
<div class="shap-row">
  <div class="shap-name" title="{name}">{name}</div>
  <div class="shap-track">
    <div class="shap-mid"></div>
    <div class="shap-bar" style="{bar_style}height:12px;top:4px;border-radius:2px;"></div>
  </div>
  <div class="shap-num" style="color:{val_col};">{val_str}</div>
</div>"""

    if not bars_html:
        bars_html = '<div style="color:#555568;font-size:0.85rem;padding:1rem 0;">No active SHAP features — all input fields were empty.</div>'

    st.markdown(f"""
<div style="background:#0e0f12;border:1px solid #1e1f28;border-radius:14px;
            padding:1.2rem 1.4rem;margin-top:1rem;">
  <div style="font-size:0.65rem;font-weight:800;letter-spacing:0.15em;
              text-transform:uppercase;color:#555568;margin-bottom:1rem;">
    Force Plot — φᵢ per feature (log-odds)
  </div>
  <div style="font-size:0.72rem;color:#444456;margin-bottom:0.8rem;
              display:flex;gap:20px;align-items:center;">
    <span>
      <span style="display:inline-block;width:12px;height:12px;
                   background:#e8e5de;border-radius:2px;vertical-align:middle;
                   margin-right:5px;"></span>Pushes → FRAUD
    </span>
    <span>
      <span style="display:inline-block;width:12px;height:12px;
                   background:#3a3b4a;border-radius:2px;vertical-align:middle;
                   margin-right:5px;"></span>Pushes → LEGIT
    </span>
    <span style="margin-left:auto;font-family:monospace;">
      intercept {shap_intercept:+.3f} + Σφᵢ {float(shap_vals.sum()):+.3f}
      = <b style="color:#ddddee;">{shap_log_odds:+.3f}</b> log-odds
    </span>
  </div>
  {bars_html}
</div>
""", unsafe_allow_html=True)

    # ── Waterfall chart ───────────────────────────────────────────────────────
    if MPL_AVAILABLE and shap_top:
        wf_pairs  = [(n, v) for n, v in shap_top if abs(v) >= 0.005][:10]
        wf_steps  = [("Intercept", shap_intercept)]
        for n, v in wf_pairs:
            wf_steps.append((n[:20], v))
        wf_steps.append(("Final", None))

        running  = shap_intercept
        cum_vals = [running]
        for _, v in wf_pairs:
            running += v
            cum_vals.append(running)
        cum_vals.append(shap_log_odds)

        fig2, ax2 = plt.subplots(figsize=(11, 3.8))
        fig2.patch.set_facecolor("#0e0f12")
        ax2.set_facecolor("#0b0c0f")

        all_y = cum_vals + [0]
        ymin  = min(all_y) - 0.4
        ymax  = max(all_y) + 0.4

        n_steps = len(wf_steps)
        bar_w   = 0.55

        for i, (label, delta) in enumerate(wf_steps):
            if label == "Final":
                bot = min(shap_log_odds, 0)
                top = max(shap_log_odds, 0)
                col = "#e8e5de" if shap_log_odds > 0 else "#3a3b4a"
            elif i == 0:
                bot = min(shap_intercept, 0)
                top = max(shap_intercept, 0)
                col = "#555568"
            else:
                prev = cum_vals[i - 1]
                cur  = cum_vals[i]
                bot  = min(prev, cur)
                top  = max(prev, cur)
                col  = "#e8e5de" if delta > 0 else "#3a3b4a"

            ax2.bar(i, top - bot, bottom=bot, color=col, width=bar_w, zorder=2)

            if delta is not None and abs(delta) >= 0.005:
                ypos = top + 0.06 if delta >= 0 else bot - 0.12
                ax2.text(i, ypos, f"{delta:+.3f}",
                         ha='center', va='bottom', fontsize=7.5,
                         color="#ccccde", fontfamily="monospace")

        ax2.axhline(0, color="#2a2b38", linewidth=0.8, linestyle="--", zorder=1)
        ax2.set_xticks(range(n_steps))
        xlabels = [s[0][:16] for s in wf_steps]
        ax2.set_xticklabels(xlabels, rotation=30, ha='right',
                            fontsize=8, color="#888898", fontfamily="monospace")
        ax2.set_ylabel("log-odds", fontsize=8.5, color="#555568")
        ax2.set_title("SHAP Waterfall — cumulative log-odds buildup",
                      fontsize=10, color="#ccccde", pad=10)
        ax2.tick_params(axis='both', colors="#555568")
        ax2.set_ylim(ymin, ymax)
        ax2.grid(axis='y', color="#1a1b22", linewidth=0.5, zorder=0)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#1e1f28")

        plt.tight_layout(pad=1.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()


    # ═══════════════════════════════════════════════════════════════════════════
    # ④ RISK CONTEXT INTELLIGENCE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="slabel">④ Risk Context Intelligence</div>', unsafe_allow_html=True)

    if fd["free_email"] and fd["urgency"] > 1:
        ctx = ("<b>Pattern match:</b> Free email + urgency language detected. "
               "Matches known mass scam campaign fingerprint in EMSCAD training data.")
    elif sus_sal:
        ctx = ("<b>Salary anomaly:</b> Vague or unrealistically high salary — "
               "classic bait-and-switch. Detected by <code>suspicious_salary()</code> from eda.py.")
    elif len(scam_matches) >= 3:
        ctx = (f"<b>High scam phrase density:</b> {len(scam_matches)} known scam phrases detected. "
               f"Lexical fingerprint matches fraudulent postings. Phrases: {', '.join(scam_matches[:5])}.")
    elif fd["urgency"] > 0:
        ctx = ("<b>Pressure tactic:</b> Urgency language reduces applicant due-diligence "
               "— a documented manipulation technique in the EMSCAD dataset.")
    elif fraud_prob >= FRAUD_THRESHOLD:
        ctx = (f"<b>ML model flagged:</b> P(fraud) = {fraud_prob:.1%} ≥ threshold {FRAUD_THRESHOLD}. "
               f"TF-IDF patterns in the 5000-dim text space match training fraud examples.")
    else:
        ctx = ("<b>No dominant pattern:</b> Behavioral features are clean. "
               "ML probability is the primary signal. Verify company independently.")

    st.markdown(f'<div class="insight">🧠 {ctx}</div>', unsafe_allow_html=True)


    # ═══════════════════════════════════════════════════════════════════════════
    # EXPANDERS
    # ═══════════════════════════════════════════════════════════════════════════
    with st.expander("🔍  Full Rule-Based Explainability Report"):
        any_flag   = False
        flags_html = ""
        if fd["urgency"] > 0:
            hit        = [w for w in URGENCY_WORDS if w in job_description.lower()]
            flags_html += (f'<div class="flag-row"><div class="flag-icon">▪</div>'
                           f'<div>Urgency keywords: <span style="color:#ffffff;">'
                           f'{fd["urgency"]} hit(s)</span> — {", ".join(hit)}</div></div>')
            any_flag = True
        if fd["free_email"]:
            hit_d      = [d for d in FREE_DOMAINS if d in company_profile.lower()]
            flags_html += (f'<div class="flag-row"><div class="flag-icon">▪</div>'
                           f'<div>Free email domain: <span style="color:#ffffff;">'
                           f'{"".join(hit_d)}</span></div></div>')
            any_flag = True
        if fd["salary_missing"]:
            flags_html += ('<div class="flag-row"><div class="flag-icon">◦</div>'
                           '<div>Salary not provided — adds 15 pts via §10 formula</div></div>')
            any_flag = True
        if scam_matches:
            flags_html += (f'<div class="flag-row"><div class="flag-icon">◦</div>'
                           f'<div>Scam phrases matched: {", ".join(scam_matches[:6])}</div></div>')
            any_flag = True
        if caps > 0.15:
            flags_html += (f'<div class="flag-row"><div class="flag-icon">◦</div>'
                           f'<div>High caps ratio: {caps:.0%}</div></div>')
            any_flag = True
        if sus_sal:
            flags_html += (f'<div class="flag-row"><div class="flag-icon">◦</div>'
                           f'<div>Suspicious salary: {salary_range}</div></div>')
            any_flag = True
        if not any_flag:
            flags_html = '<div style="color:#aaaabc;font-size:0.9rem;">✓ No rule-based flags raised</div>'

        st.markdown(f'<div class="trace-box">{flags_html}</div>', unsafe_allow_html=True)
        dc_cls  = "val" if model_decision == "FRAUD" else "ok"
        lvl_cls = "val" if lvl == "HIGH" else ("warn" if lvl == "MEDIUM" else "ok")
        st.markdown(f"""
<div class="trace-box" style="margin-top:0.9rem;">
<span class="key">raw_probability     </span><span class="val">{fraud_prob:.4f}</span>
<span class="key">decision_threshold  </span><span class="val">{FRAUD_THRESHOLD}</span>
<span class="key">model_verdict       </span><span class="{dc_cls}">{model_decision}</span>
<span class="key">shap_intercept      </span><span class="val">{shap_intercept:+.4f}</span>
<span class="key">shap_sum            </span><span class="val">{float(shap_vals.sum()):+.4f}</span>
<span class="key">log_odds            </span><span class="val">{shap_log_odds:+.4f}</span>
<span class="key">composite_score     </span><span class="{lvl_cls}">{risk_score} / 100</span>
<span class="key">risk_level          </span><span class="{lvl_cls}">{lvl}</span>
<span class="key">top_driver          </span><span class="val">{driver}</span>
</div>""", unsafe_allow_html=True)

    with st.expander("📊  Feature Importance (coef_ based · expainabiity_and_insights.py)"):
        importance_pairs = get_feature_importance(fraud_model, feature_names)
        top15 = importance_pairs[:15]
        if PD_AVAILABLE:
            df_imp = pd.DataFrame(top15, columns=["Feature", "Coefficient"])
            df_imp["Direction"] = df_imp["Coefficient"].apply(
                lambda x: "→ FRAUD" if x > 0 else "→ LEGIT"
            )
            st.dataframe(df_imp, use_container_width=True, height=420)
        else:
            for feat, coef in top15:
                st.text(f"{feat:<40} {coef:+.4f}")

    with st.expander("🌐  FastAPI Backend — /predict endpoint"):
        st.markdown("""
<div class="trace-box">
<span class="key">endpoint   </span><span class="val">POST /predict</span>
<span class="key">backend    </span><span class="val">api.py  (FastAPI + uvicorn)</span>
<span class="key">run        </span><span class="val">uvicorn api:app --reload</span>
<span class="key">docs       </span><span class="val">http://localhost:8000/docs</span>
</div>
""", unsafe_allow_html=True)
        st.code('''{
  "title":       "Data Entry Executive",
  "description": "Work from home, earn 50k/month, no experience needed...",
  "company":     "contact: recruiter@gmail.com",
  "salary":      "50000"
}''', language="json")
        st.markdown(
            '<div class="insight" style="margin-top:0.7rem;">'
            '<b>Response includes:</b> fraud_probability, risk_score, risk_level, '
            'verdict, advice, top 10 SHAP features, behavioral signals. '
            'Full schema at <code>/docs</code>.</div>',
            unsafe_allow_html=True
        )

    with st.expander("✅  Defensive Checklist"):
        items = [
            ("Verify company on MCA21 / LinkedIn",       "Official registration eliminates ghost companies"),
            ("Confirm official domain email",             "Legit companies don't use Gmail/Yahoo for hiring"),
            ("Cross-check salary on Glassdoor / Naukri", "Compare claimed CTC with market rates"),
            ("Never pay upfront fees",                    "No legitimate employer charges registration fees"),
            ("Don't share Aadhaar / PAN early",           "Only after receiving a formal offer letter"),
            ("Google: [company] + scam / fraud",          "Many scam companies have reported complaints"),
            ("Verify recruiter on LinkedIn",              "Check profile age, connections, endorsements"),
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


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
  SCAMGUARD-AI &nbsp;·&nbsp; Logistic Regression + TF-IDF &nbsp;·&nbsp;
  4 Models Benchmarked &nbsp;·&nbsp;
  Decision threshold: <b>{FRAUD_THRESHOLD}</b> &nbsp;·&nbsp;
  5-Fold CV AUC: <b>0.96 ± 0.01</b> &nbsp;·&nbsp;
  Trained on EMSCAD dataset &nbsp;·&nbsp;
  Exact SHAP via <b>utils.compute_shap_values()</b> &nbsp;·&nbsp;
  REST API: <b>api.py</b> &nbsp;·&nbsp;
  Not a substitute for manual verification
</div>
""", unsafe_allow_html=True)
