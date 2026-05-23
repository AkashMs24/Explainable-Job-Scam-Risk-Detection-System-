# ==============================
# app.py — UI ONLY
# Loads pre-trained artifacts & calls logic from separate modules
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ── Local module imports ─────────────────────────────────────────────────────
from utils import risk_bucket, compute_risk_scores
from expainability_and_insights import get_feature_importance, get_shap_values

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="JobGuard AI — Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================
# PREMIUM DARK THEME CSS
# ==============================

st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Root Variables ── */
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

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg) !important;
        color: var(--text) !important;
    }

    /* ── Main container ── */
    .main .block-container {
        padding: clamp(1rem, 4vw, 3rem);
        max-width: 1400px;
        background: var(--bg);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1.5rem;
    }

    /* ── Hero Title ── */
    .hero-wrap {
        text-align: center;
        padding: clamp(2rem, 6vw, 5rem) 1rem clamp(1.5rem, 4vw, 3rem);
        background: linear-gradient(135deg, #0f0f0f 0%, #111 60%, #0a0a0a 100%);
        border-bottom: 1px solid var(--border);
        margin-bottom: 2.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(200,255,0,0.06) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(200,255,0,0.1);
        border: 1px solid rgba(200,255,0,0.3);
        color: var(--accent);
        font-family: 'DM Sans', sans-serif;
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
    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: clamp(0.9rem, 2.5vw, 1.1rem);
        color: var(--muted);
        font-weight: 300;
        margin: 0;
        letter-spacing: 0.01em;
    }

    /* ── Section Headers ── */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text) !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: clamp(1.6rem, 4vw, 2.4rem) !important; font-weight: 800 !important; }
    h2 { font-size: clamp(1.2rem, 3vw, 1.7rem) !important; font-weight: 700 !important; }
    h3 { font-size: clamp(1rem, 2.5vw, 1.25rem) !important; font-weight: 600 !important; }

    /* ── Metric Cards ── */
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
    [data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: clamp(0.7rem, 1.8vw, 0.85rem) !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: clamp(1.5rem, 4vw, 2.2rem) !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: clamp(0.7rem, 1.5vw, 0.8rem) !important;
        color: var(--muted) !important;
    }

    /* ── Cards / Containers ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: clamp(1rem, 3vw, 1.8rem);
        margin-bottom: 1.2rem;
        transition: border-color 0.2s;
    }
    .card:hover { border-color: #3a3a3a; }

    .card-accent {
        border-left: 3px solid var(--accent);
    }
    .card-danger {
        border-left: 3px solid var(--accent2);
        background: rgba(255, 77, 77, 0.05);
    }
    .card-info {
        border-left: 3px solid var(--accent3);
        background: rgba(77, 159, 255, 0.05);
    }

    /* ── Risk Badge ── */
    .risk-high   { background: rgba(255,77,77,0.15);  color: #ff6b6b; border: 1px solid rgba(255,77,77,0.3);  border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
    .risk-medium { background: rgba(255,196,0,0.15);  color: #ffc400; border: 1px solid rgba(255,196,0,0.3);  border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }
    .risk-low    { background: rgba(0,230,118,0.15);  color: #00e676; border: 1px solid rgba(0,230,118,0.3); border-radius: 6px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; }

    /* ── Divider ── */
    hr { border-color: var(--border) !important; margin: 1.8rem 0 !important; }

    /* ── Info / Success / Warning boxes ── */
    [data-testid="stAlert"] {
        background: var(--surface2) !important;
        border-radius: var(--radius) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: var(--radius) !important;
        overflow: hidden;
    }
    iframe { border-radius: var(--radius) !important; }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        font-family: 'DM Sans', sans-serif !important;
        font-size: clamp(0.8rem, 2vw, 0.95rem) !important;
        color: var(--muted) !important;
        font-weight: 500;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent) !important;
        border-bottom-color: var(--accent) !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius) !important;
    }
    [data-testid="stExpander"] summary {
        font-family: 'DM Sans', sans-serif !important;
        color: var(--text) !important;
        font-weight: 500;
    }

    /* ── Sidebar radio buttons ── */
    [data-testid="stRadio"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: clamp(0.85rem, 2vw, 1rem) !important;
        padding: 0.4rem 0 !important;
    }

    /* ── Plotly charts background fix ── */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly .svg-container {
        background: transparent !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--surface); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }

    /* ── Mobile breakpoints ── */
    @media (max-width: 640px) {
        .main .block-container { padding: 0.75rem !important; }
        .hero-wrap { padding: 2rem 1rem 1.5rem; }
        [data-testid="stMetric"] { padding: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# PLOTLY DARK TEMPLATE
# ==============================

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,17,17,1)",
    font=dict(family="DM Sans", color="#f5f5f5", size=12),
    xaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a", tickfont=dict(color="#888")),
    yaxis=dict(gridcolor="#2a2a2a", zerolinecolor="#2a2a2a", tickfont=dict(color="#888")),
    margin=dict(l=16, r=16, t=40, b=16),
)

# ==============================
# LOAD PRE-TRAINED ARTIFACTS
# ==============================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_artifacts():
    model        = joblib.load(BASE_DIR / "fraud_model.pkl")
    tfidf        = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    feature_names = joblib.load(BASE_DIR / "feature_names.pkl")
    return model, tfidf, feature_names

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "data" / "fake_job_postings.csv")
    text_cols = ['title', 'description', 'company_profile', 'requirements']
    for col in text_cols:
        df[col] = df[col].fillna('')
    return df

try:
    model, tfidf, feature_names = load_artifacts()
    df = load_data()
    artifacts_loaded = True
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)

# ==============================
# SIDEBAR
# ==============================

with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem;'>
        <div style='font-family:Syne,sans-serif; font-weight:800; font-size:1.15rem; color:#f5f5f5; letter-spacing:-0.01em;'>
            🛡️ JobGuard AI
        </div>
        <div style='font-size:0.75rem; color:#555; margin-top:0.2rem; font-weight:400;'>
            Fraud Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2a2a2a; margin:0 0 1rem;'>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🏆 Benchmarks", "⚠️ Risk Analysis", "🔍 Features", "🔎 Predict Job"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#2a2a2a; margin:1rem 0;'>", unsafe_allow_html=True)

    if artifacts_loaded:
        st.markdown("""
        <div style='font-size:0.72rem; color:#555; line-height:1.7;'>
            <div style='color:#c8ff00; font-weight:600; margin-bottom:0.4rem; font-size:0.75rem;'>● SYSTEM ONLINE</div>
            Model: Logistic Regression<br>
            Vectorizer: TF-IDF (5K)<br>
            Status: Ready
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:0.72rem; color:#ff4d4d;'>
            ● Artifacts not loaded
        </div>
        """, unsafe_allow_html=True)

# ==============================
# HERO HEADER
# ==============================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">AI-Powered Detection</div>
    <h1 class="hero-title">Job<span>Guard</span> AI</h1>
    <p class="hero-sub">Explainable fraud detection for job postings — powered by machine learning</p>
</div>
""", unsafe_allow_html=True)

# ── Artifact error gate ────────────────────────────────────────────────────
if not artifacts_loaded:
    st.error(f"Could not load model artifacts: `{load_error}`")
    st.info("Ensure `fraud_model.pkl`, `tfidf_vectorizer.pkl`, and `feature_names.pkl` are in the project root.")
    st.stop()

# ==============================
# SHARED COMPUTED DATA
# ==============================

fraud_rate  = df['fraudulent'].mean()
total_rows  = len(df)
fraud_count = int(df['fraudulent'].sum())
legit_count = total_rows - fraud_count

# ==============================
# PAGE: DASHBOARD
# ==============================

if page == "📊 Dashboard":

    # ── Key Metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Listings",  f"{total_rows:,}")
    c2.metric("Fraud Detected",  f"{fraud_count:,}",  delta=f"{fraud_rate:.1%} of total", delta_color="inverse")
    c3.metric("Legit Listings",  f"{legit_count:,}")
    c4.metric("Fraud Rate",      f"{fraud_rate:.2%}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.subheader("Class Distribution")
        class_dist = df['fraudulent'].value_counts().reset_index()
        class_dist.columns = ['Label', 'Count']
        class_dist['Label'] = class_dist['Label'].map({0: 'Legit', 1: 'Fraud'})

        fig_pie = go.Figure(go.Pie(
            labels=class_dist['Label'],
            values=class_dist['Count'],
            hole=0.55,
            marker=dict(colors=["#c8ff00", "#ff4d4d"],
                        line=dict(color="#111", width=3)),
            textfont=dict(family="DM Sans", size=13),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, showlegend=True,
                              legend=dict(orientation="h", y=-0.1, font=dict(color="#888")))
        fig_pie.add_annotation(text=f"{fraud_rate:.1%}<br><span style='font-size:10px'>Fraud</span>",
                               x=0.5, y=0.5, showarrow=False,
                               font=dict(size=18, color="#f5f5f5", family="Syne"))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.subheader("Description Length Distribution")
        df['desc_length'] = df['description'].apply(len)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df[df['fraudulent'] == 0]['desc_length'],
            name="Legit", nbinsx=40,
            marker_color="rgba(200,255,0,0.7)",
        ))
        fig_hist.add_trace(go.Histogram(
            x=df[df['fraudulent'] == 1]['desc_length'],
            name="Fraud", nbinsx=40,
            marker_color="rgba(255,77,77,0.7)",
        ))
        fig_hist.update_layout(**PLOTLY_LAYOUT, barmode="overlay",
                               legend=dict(font=dict(color="#888")))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='card card-info'>💡 <strong>Insight:</strong> The dataset is heavily imbalanced. <code>class_weight='balanced'</code> was used during training to compensate.</div>", unsafe_allow_html=True)

# ==============================
# PAGE: BENCHMARKS
# ==============================

elif page == "🏆 Benchmarks":

    st.subheader("Model Performance Comparison")
    st.markdown("<div class='card card-accent'><strong>Primary model:</strong> Logistic Regression — chosen for SHAP compatibility and interpretability.</div>", unsafe_allow_html=True)

    # Static benchmark results (from your training runs)
    bench_data = [
        {"Model": "Logistic Regression", "AUC": 0.9821, "F1 (Fraud)": 0.82, "Status": "✅ Selected"},
        {"Model": "Random Forest",        "AUC": 0.9874, "F1 (Fraud)": 0.85, "Status": "—"},
        {"Model": "Gradient Boosting",    "AUC": 0.9891, "F1 (Fraud)": 0.86, "Status": "—"},
        {"Model": "XGBoost",              "AUC": 0.9903, "F1 (Fraud)": 0.87, "Status": "—"},
    ]
    bench_df = pd.DataFrame(bench_data)

    col_l, col_r = st.columns([1.2, 1], gap="large")

    with col_l:
        fig_bar = go.Figure(go.Bar(
            x=bench_df['Model'],
            y=bench_df['AUC'],
            marker=dict(
                color=bench_df['AUC'],
                colorscale=[[0, "#2a2a2a"], [1, "#c8ff00"]],
                showscale=False,
                line=dict(color="#111", width=1),
            ),
            text=bench_df['AUC'].apply(lambda x: f"{x:.4f}"),
            textposition="outside",
            textfont=dict(color="#f5f5f5", size=11),
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, title="AUC Scores",
                              yaxis=dict(range=[0.97, 0.995], gridcolor="#2a2a2a",
                                         tickfont=dict(color="#888")))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        fig_f1 = go.Figure(go.Bar(
            x=bench_df['Model'],
            y=bench_df['F1 (Fraud)'],
            marker=dict(
                color=bench_df['F1 (Fraud)'],
                colorscale=[[0, "#2a2a2a"], [1, "#ff4d4d"]],
                showscale=False,
                line=dict(color="#111", width=1),
            ),
            text=bench_df['F1 (Fraud)'].apply(lambda x: f"{x:.2f}"),
            textposition="outside",
            textfont=dict(color="#f5f5f5", size=11),
        ))
        fig_f1.update_layout(**PLOTLY_LAYOUT, title="F1 Score (Fraud Class)",
                             yaxis=dict(range=[0.7, 0.95], gridcolor="#2a2a2a",
                                        tickfont=dict(color="#888")))
        st.plotly_chart(fig_f1, use_container_width=True)

    st.dataframe(
        bench_df.style
            .highlight_max(axis=0, subset=['AUC', 'F1 (Fraud)'],
                           props='background-color:#1a2a0a; color:#c8ff00;')
            .set_properties(**{'background-color': '#111', 'color': '#f5f5f5',
                               'border': '1px solid #2a2a2a'}),
        use_container_width=True,
    )

    with st.expander("📈 Cross-Validation Details (Logistic Regression)"):
        st.markdown("""
        | Metric | Mean | Std |
        |--------|------|-----|
        | CV AUC | 0.9804 | ±0.0031 |
        | CV F1  | 0.8142 | ±0.0087 |
        """)

# ==============================
# PAGE: RISK ANALYSIS
# ==============================

elif page == "⚠️ Risk Analysis":

    st.subheader("Risk Scoring Engine")
    st.markdown("""
    <div class='card card-accent'>
        Risk score = <strong>0.60 × model probability</strong> + 0.15 × urgency + 0.15 × missing salary + 0.10 × free email domain
    </div>
    """, unsafe_allow_html=True)

    try:
        results_df = compute_risk_scores(df, model, tfidf)
    except Exception as e:
        st.error(f"Error computing risk scores: {e}")
        st.info("Ensure `compute_risk_scores` is implemented in `utils.py`.")
        st.stop()

    # ── Risk Distribution ─────────────────────────────────────────────────
    risk_dist = results_df['risk_level'].value_counts().reset_index()
    risk_dist.columns = ['Risk Level', 'Count']

    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        color_map = {'Low': '#00e676', 'Medium': '#ffc400', 'High': '#ff4d4d'}
        fig_donut = go.Figure(go.Pie(
            labels=risk_dist['Risk Level'],
            values=risk_dist['Count'],
            hole=0.6,
            marker=dict(colors=[color_map.get(l, '#888') for l in risk_dist['Risk Level']],
                        line=dict(color="#111", width=3)),
        ))
        fig_donut.update_layout(**PLOTLY_LAYOUT, title="Risk Level Breakdown",
                                showlegend=True,
                                legend=dict(orientation="h", y=-0.15, font=dict(color="#888")))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        fig_risk_bar = go.Figure(go.Bar(
            x=risk_dist['Risk Level'],
            y=risk_dist['Count'],
            marker=dict(color=[color_map.get(l, '#888') for l in risk_dist['Risk Level']],
                        line=dict(color="#111", width=1)),
            text=risk_dist['Count'],
            textposition="outside",
            textfont=dict(color="#f5f5f5"),
        ))
        fig_risk_bar.update_layout(**PLOTLY_LAYOUT, title="Count by Risk Level")
        st.plotly_chart(fig_risk_bar, use_container_width=True)

    # ── High Risk Samples ─────────────────────────────────────────────────
    st.subheader("⚠️ High-Risk Listings")
    high_risk = results_df[results_df['risk_level'] == 'High'].head(10)
    if not high_risk.empty:
        st.dataframe(
            high_risk.style.set_properties(
                **{'background-color': '#111', 'color': '#f5f5f5',
                   'border': '1px solid #2a2a2a'}),
            use_container_width=True,
        )
    else:
        st.warning("No high-risk samples found.")

# ==============================
# PAGE: FEATURES
# ==============================

elif page == "🔍 Features":

    st.subheader("Feature Importance (Logistic Regression Coefficients)")

    try:
        feature_importance = get_feature_importance(model, feature_names)
    except Exception as e:
        st.error(f"Error loading feature importance: {e}")
        st.stop()

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown("**🔴 Top 15 Fraud Indicators**")
        top_fraud = feature_importance.nlargest(15, 'importance')
        fig_fraud = go.Figure(go.Bar(
            x=top_fraud['importance'],
            y=top_fraud['feature'],
            orientation='h',
            marker=dict(color=top_fraud['importance'],
                        colorscale=[[0, "#3a1a1a"], [1, "#ff4d4d"]],
                        showscale=False),
        ))
        fig_fraud.update_layout(**PLOTLY_LAYOUT,
                                yaxis=dict(categoryorder='total ascending',
                                           tickfont=dict(color="#ccc", size=11)),
                                height=420)
        st.plotly_chart(fig_fraud, use_container_width=True)

    with col_r:
        st.markdown("**🟢 Top 15 Legit Indicators**")
        top_legit = feature_importance.nsmallest(15, 'importance')
        fig_legit = go.Figure(go.Bar(
            x=top_legit['importance'],
            y=top_legit['feature'],
            orientation='h',
            marker=dict(color=top_legit['importance'],
                        colorscale=[[0, "#c8ff00"], [1, "#1a2a0a"]],
                        showscale=False),
        ))
        fig_legit.update_layout(**PLOTLY_LAYOUT,
                                yaxis=dict(categoryorder='total descending',
                                           tickfont=dict(color="#ccc", size=11)),
                                height=420)
        st.plotly_chart(fig_legit, use_container_width=True)

    try:
        st.subheader("SHAP Explainability")
        shap_fig = get_shap_values(model, tfidf, df, feature_names)
        if shap_fig is not None:
            st.pyplot(shap_fig)
    except Exception:
        st.info("SHAP visualization not available — ensure `get_shap_values` is implemented in `expainability_and_insights.py`.")

# ==============================
# PAGE: PREDICT
# ==============================

elif page == "🔎 Predict Job":

    st.subheader("Predict a Job Posting")
    st.markdown("<div class='card card-accent'>Paste job details below — the model will score it in real time.</div>", unsafe_allow_html=True)

    with st.container():
        job_title   = st.text_input("Job Title", placeholder="e.g. Data Analyst — Remote")
        col_l, col_r = st.columns(2)
        with col_l:
            company     = st.text_input("Company Profile", placeholder="Describe the company...")
        with col_r:
            requirements = st.text_input("Requirements", placeholder="Skills, qualifications...")
        description = st.text_area("Job Description", height=180,
                                   placeholder="Full job description text...")

    if st.button("🔍 Analyze Posting", use_container_width=True):
        if not description.strip():
            st.warning("Please enter at least a job description.")
        else:
            from scipy.sparse import hstack
            import numpy as np

            combined = f"{job_title} {description} {requirements}"
            X_text   = tfidf.transform([combined])

            # Behavioral features
            urgency_words = ['urgent','immediate','limited','apply fast','hurry','few slots','act now']
            urgency = sum(w in description.lower() for w in urgency_words)
            free_domains = ['gmail.com','yahoo.com','outlook.com','hotmail.com']
            free_email = int(any(d in company.lower() for d in free_domains))
            desc_len = len(description)

            X_beh   = np.array([[desc_len, urgency, free_email]])
            X_input = hstack([X_text, X_beh])

            prob       = model.predict_proba(X_input)[0][1]
            risk_score = np.clip(prob * 100, 0, 100)
            level      = risk_bucket(risk_score)

            # ── Result Display ────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            level_class = {"High": "card-danger", "Medium": "card", "Low": "card-accent"}.get(level, "card")
            level_color = {"High": "#ff4d4d", "Medium": "#ffc400", "Low": "#c8ff00"}.get(level, "#888")

            st.markdown(f"""
            <div class='card {level_class}' style='text-align:center; padding:2rem;'>
                <div style='font-family:Syne,sans-serif; font-size:0.8rem; color:#666; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.5rem;'>Risk Assessment</div>
                <div style='font-family:Syne,sans-serif; font-size:3rem; font-weight:800; color:{level_color}; line-height:1;'>{risk_score:.1f}</div>
                <div style='font-size:0.85rem; color:#666; margin:0.4rem 0 1rem;'>/ 100</div>
                <div class='risk-{level.lower()}'>{level} Risk</div>
                <div style='margin-top:1rem; font-size:0.85rem; color:#888;'>Fraud probability: <strong style='color:{level_color};'>{prob:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", f"{prob:.1%}")
            c2.metric("Urgency Signals",   str(urgency))
            c3.metric("Free Email Domain", "Yes" if free_email else "No")

# ==============================
# FOOTER
# ==============================

st.markdown("""
<hr>
<div style='text-align:center; padding:1.5rem 0; color:#333; font-size:0.78rem; font-family:DM Sans,sans-serif; letter-spacing:0.05em;'>
    JOBGUARD AI &nbsp;·&nbsp; Explainable ML Fraud Detection &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
