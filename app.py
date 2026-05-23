# ==============================
# 1. Imports & Paths
# ==============================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy.sparse import hstack
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# ==============================
# STREAMLIT CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🛡️",
    layout="wide",  # Uses full screen width
    initial_sidebar_state="expanded"
)

# Custom CSS for minor mobile tweaks (font sizes, spacing)
st.markdown("""
<style>
    /* Make main header smaller on mobile */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.4rem !important; }
    h3 { font-size: 1.1rem !important; }
    
    /* Ensure metrics are readable */
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; }
    
    /* Add padding to containers for better touch targets */
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 2. Load Data & Cache
# ==============================

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"
SRC_DIR   = BASE_DIR / "src"
SRC_DIR.mkdir(exist_ok=True)

@st.cache_data
def load_and_process_data():
    """
    Loads data and performs all feature engineering.
    Cached to prevent re-running on every interaction.
    """
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()

    df = pd.read_csv(DATA_PATH)

    text_cols = ['title', 'description', 'company_profile', 'requirements']
    for col in text_cols:
        df[col] = df[col].fillna('')

    # ── Class distribution ───────────────
    fraud_rate = df['fraudulent'].mean()
    total_rows = len(df)

    # ==============================
    # 3. Behavioral Feature Engineering
    # ==============================

    df['desc_length'] = df['description'].apply(len)

    urgency_words = [
        'urgent', 'immediate', 'limited',
        'apply fast', 'hurry', 'few slots', 'act now'
    ]

    def urgency_score(text):
        text = text.lower()
        return sum(word in text for word in urgency_words)

    df['urgency_score'] = df['description'].apply(urgency_score)

    free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']

    def free_email_flag(text):
        text = text.lower()
        return int(any(domain in text for domain in free_domains))

    df['free_email'] = df['company_profile'].apply(free_email_flag)

    # ==============================
    # 4. Text Feature Engineering
    # ==============================

    df['combined_text'] = (
        df['title'] + ' ' +
        df['description'] + ' ' +
        df['requirements']
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english'
    )

    X_text = tfidf.fit_transform(df['combined_text'])

    # ==============================
    # 5. Combine Features
    # ==============================

    behavior_features = ['desc_length', 'urgency_score', 'free_email']
    X_behavior        = df[behavior_features].values

    X_final = hstack([X_text, X_behavior])
    y       = df['fraudulent']

    return df, X_final, y, tfidf, behavior_features, fraud_rate, total_rows

# Load Data
df, X_final, y, tfidf, behavior_features, fraud_rate, total_rows = load_and_process_data()

# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X_final,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# 7. Model Benchmarking & Training
# ==============================

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    """
    Trains models. Cached to prevent re-training on refresh.
    """
    benchmark_models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            C=1.0,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        neg_count = int((y_train == 0).sum())
        pos_count = int((y_train == 1).sum())
        benchmark_models["XGBoost"] = XGBClassifier(
            scale_pos_weight=neg_count / max(pos_count, 1),
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1,
        )

    benchmark_results = {}

    for name, mdl in benchmark_models.items():
        mdl.fit(X_train, y_train)
        y_pred_b  = mdl.predict(X_test)
        y_proba_b = mdl.predict_proba(X_test)[:, 1]
        auc       = roc_auc_score(y_test, y_proba_b)
        report    = classification_report(y_test, y_pred_b, output_dict=True)
        f1_fraud  = report['1']['f1-score']

        benchmark_results[name] = {
            "model":     mdl,
            "auc":       auc,
            "f1_fraud":  f1_fraud,
            "report":    classification_report(y_test, y_pred_b),
        }

    return benchmark_results

benchmark_results = train_models(X_train, X_test, y_train, y_test)

# Select LR as primary model
selected_model_name = "Logistic Regression"
model = benchmark_results[selected_model_name]["model"]
final_auc = benchmark_results[selected_model_name]["auc"]

# ==============================
# 8. Cross-Validation
# ==============================

@st.cache_data
def run_cross_validation(X_final, y):
    skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cv  = LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)

    cv_auc = cross_val_score(lr_cv, X_final, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_f1  = cross_val_score(lr_cv, X_final, y, cv=skf, scoring='f1', n_jobs=-1)
    
    return cv_auc, cv_f1

cv_auc, cv_f1 = run_cross_validation(X_final, y)

# ==============================
# 9. Risk Scoring Engine
# ==============================

y_proba  = model.predict_proba(X_test)[:, 1]
test_idx = y_test.index

urgency_norm = (
    df.loc[test_idx, 'urgency_score'] /
    max(df['urgency_score'].max(), 1)
).fillna(0)

if 'salary_range' in df.columns:
    salary_risk = df.loc[test_idx, 'salary_range'].isnull().astype(int)
else:
    salary_risk = pd.Series(np.zeros(len(test_idx)), index=test_idx)

email_risk = df.loc[test_idx, 'free_email']

risk_score = (
    0.60 * y_proba +
    0.15 * urgency_norm.values +
    0.15 * salary_risk.values +
    0.10 * email_risk.values
)

risk_score = np.clip(risk_score * 100, 0, 100)

def risk_bucket(score):
    if score < 30: return "Low"
    elif score < 60: return "Medium"
    else: return "High"

risk_level = [risk_bucket(score) for score in risk_score]

results_df = pd.DataFrame({
    "fraud_probability": y_proba,
    "risk_score":        risk_score,
    "risk_level":        risk_level,
    "actual_label":      y_test.values,
})

# ==============================
# 10. Feature Importance
# ==============================

tfidf_features = list(tfidf.get_feature_names_out())
feature_names  = tfidf_features + behavior_features
coefficients = model.coef_[0]

feature_importance = pd.DataFrame({
    "feature":    feature_names,
    "importance": coefficients,
}).sort_values(by="importance", ascending=False)

# ==========================================================
# STREAMLIT UI (Mobile Responsive Layout)
# ==========================================================

# Header
st.title("🛡️ Fake Job Posting Detector")
st.markdown("Mobile-Responsive Dashboard")

# Sidebar Navigation
st.sidebar.header("Menu")
page = st.sidebar.radio("Select View", ["Dashboard", "Benchmarks", "Risk Analysis", "Features"])

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.header("Overview")
    
    # Responsive Metrics Grid
    # On mobile, these 4 columns will automatically stack vertically
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Jobs", f"{total_rows:,}")
    col2.metric("Fraud Rate", f"{fraud_rate:.2%}")
    col3.metric("Test AUC", f"{final_auc:.4f}")
    col4.metric("CV AUC", f"{cv_auc.mean():.4f}")

    st.divider()

    # Chart: Class Distribution
    st.subheader("Class Distribution")
    class_dist = df['fraudulent'].value_counts().reset_index()
    class_dist.columns = ['Label', 'Count']
    class_dist['Label'] = class_dist['Label'].map({0: 'Legit', 1: 'Fraud'})
    
    fig_pie = px.pie(class_dist, values='Count', names='Label', color_discrete_sequence=['#2ecc71', '#e74c3c'])
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
    # use_container_width=True makes it fit mobile screen perfectly
    st.plotly_chart(fig_pie, use_container_width=True)

    st.info("The dataset is imbalanced. We used `class_weight='balanced'` to handle this.")

# --- PAGE 2: BENCHMARKS ---
elif page == "Benchmarks":
    st.header("Model Benchmarks")
    
    # Create DataFrame for benchmarks
    bench_data = []
    for name, res in benchmark_results.items():
        bench_data.append({
            "Model": name,
            "AUC": res['auc'],
            "F1 (Fraud)": res['f1_fraud']
        })
    bench_df = pd.DataFrame(bench_data)
    
    # Chart: AUC Comparison
    fig_bar = px.bar(bench_df, x='Model', y='AUC', color='AUC', color_continuous_scale='Viridis')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Table: Detailed Metrics
    st.dataframe(bench_df.style.highlight_max(axis=0, subset=['AUC', 'F1 (Fraud)']), use_container_width=True)
    
    st.success("**Selected:** Logistic Regression (Exact SHAP support)")

    # Cross Validation Details in Expander (saves space on mobile)
    with st.expander("View Cross-Validation Details"):
        st.write(f"**CV AUC:** {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        st.write(f"**CV F1:** {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

# --- PAGE 3: RISK ANALYSIS ---
elif page == "Risk Analysis":
    st.header("Risk Scoring Engine")
    
    # Risk Distribution Chart
    risk_dist = results_df['risk_level'].value_counts().reset_index()
    risk_dist.columns = ['Risk Level', 'Count']
    
    fig_risk = px.bar(risk_dist, x='Risk Level', y='Count', color='Risk Level',
                      color_discrete_map={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'})
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Sample High-Risk Predictions
    st.subheader("High-Risk Samples")
    high_risk = results_df[results_df['risk_level'] == 'High']
    if not high_risk.empty:
        # use_container_width=True ensures table scrolls horizontally on mobile
        st.dataframe(high_risk.head(10), use_container_width=True)
    else:
        st.warning("No high-risk samples in current test batch.")

# --- PAGE 4: FEATURES ---
elif page == "Features":
    st.header("Feature Importance")
    
    # Top Fraud Features
    st.subheader("Top 15 Fraud Indicators")
    top_fraud = feature_importance.head(15)
    fig_fraud = px.bar(top_fraud, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Reds')
    fig_fraud.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_fraud, use_container_width=True)
    
    # Top Legit Features
    st.subheader("Top 15 Legit Indicators")
    top_legit = feature_importance.tail(15)
    fig_legit = px.bar(top_legit, x='importance', y='feature', orientation='h', color='importance', color_continuous_scale='Greens')
    fig_legit.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_legit, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Deployed with Streamlit | Mobile Optimized")
