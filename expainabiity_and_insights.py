# ======================================
# expainabiity_and_insights.py
# ======================================
# Run this standalone to:
#   1. Print top-15 feature importances  (coef_ based)
#   2. Compute + print exact SHAP values for a sample posting
#   3. Plot a SHAP bar chart (matplotlib)
#
# SHAP NOTE:
#   For Logistic Regression, SHAP values are mathematically exact:
#       φᵢ = coef[i] × feature_value[i]   (log-odds space)
#       Σ φᵢ + intercept = log-odds = logit(P(fraud))
#   This requires NO shap library — it is computed in utils.py via
#   compute_shap_values(). This is the function app.py calls for ③ SHAP.
#
# WHERE TO UPDATE SHAP IN app.py:
#   Search for the comment  # ── SECTION 3: SHAP  in app.py.
#   The call chain is:
#       X_final, fd  = build_feature_vector(...)   ← already called in §ANALYSIS
#       shap_vals, intercept, log_odds = compute_shap_values(fraud_model, X_final, feature_names)
#       top_pairs = top_shap_features(shap_vals, feature_names, n=15)
#   Then render top_pairs as a bar chart. app.py already does this — no
#   external shap package is required.
# ======================================

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# ── utils.py provides all shared logic ────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    build_feature_vector,
    compute_shap_values,
    top_shap_features,
    get_feature_importance,
    URGENCY_WORDS,
    FREE_DOMAINS,
    FRAUD_THRESHOLD,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR  = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Load artifacts ─────────────────────────────────────────────────────────────
model        = joblib.load(os.path.join(SRC_DIR, "fraud_model.pkl"))
tfidf        = joblib.load(os.path.join(SRC_DIR, "tfidf_vectorizer.pkl"))
feature_names = joblib.load(os.path.join(SRC_DIR, "feature_names.pkl"))

# ── 1. Feature importance (coef_ based — same as training script §11) ─────────
print("=" * 60)
print("FEATURE IMPORTANCE  (coef_ · mirrors training §11)")
print("=" * 60)
importance_pairs = get_feature_importance(model, feature_names)
df_imp = pd.DataFrame(importance_pairs[:15], columns=["Feature", "Coefficient"])
df_imp["Direction"] = df_imp["Coefficient"].apply(lambda x: "→ FRAUD" if x > 0 else "→ LEGIT")
print(df_imp.to_string(index=False))
print()

# ── 2. Exact SHAP on a sample posting ─────────────────────────────────────────
SAMPLE_TITLE   = "Work From Home Data Entry Executive"
SAMPLE_DESC    = ("No experience required. Registration fee of ₹500. "
                  "Weekly payout. Unlimited earning potential. APPLY NOW URGENT.")
SAMPLE_PROFILE = "Contact: recruiter@gmail.com"
SAMPLE_SALARY  = ""

print("=" * 60)
print("EXACT SHAP ANALYSIS  (φᵢ = coef[i] × feature_value[i])")
print("Sample posting:", SAMPLE_TITLE)
print("=" * 60)

X_final, fd = build_feature_vector(tfidf, SAMPLE_TITLE, SAMPLE_DESC,
                                    SAMPLE_PROFILE, SAMPLE_SALARY)

fraud_prob   = model.predict_proba(X_final)[0][1]
shap_vals, intercept, log_odds = compute_shap_values(model, X_final, feature_names)
top_pairs    = top_shap_features(shap_vals, feature_names, n=15)

print(f"\nP(fraud)   = {fraud_prob:.4f}")
print(f"Intercept  = {intercept:.4f}")
print(f"SHAP sum   = {shap_vals.sum():.4f}")
print(f"Log-odds   = {log_odds:.4f}  (intercept + SHAP sum — must equal logit(P))")
print(f"Logit check= {np.log(fraud_prob / (1 - fraud_prob)):.4f}")
print()
print(f"{'Feature':<45} {'SHAP φᵢ':>10}  {'Direction'}")
print("-" * 65)
for name, val in top_pairs:
    direction = "→ FRAUD" if val > 0 else "→ LEGIT"
    print(f"{name:<45} {val:>+10.4f}  {direction}")

# ── 3. SHAP bar chart ──────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    names  = [p[0] for p in top_pairs][::-1]
    values = [p[1] for p in top_pairs][::-1]
    colors = ["#ffffff" if v > 0 else "#555566" for v in values]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0b0c0f")
    ax.set_facecolor("#0f1014")

    bars = ax.barh(range(len(names)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9, color="#c0bdb0")
    ax.axvline(0, color="#333344", linewidth=1, linestyle="--")
    ax.set_xlabel("SHAP value  (log-odds space)  +→FRAUD  −→LEGIT",
                  fontsize=9, color="#888898")
    ax.set_title(f"Exact SHAP — LR  |  P(fraud)={fraud_prob:.1%}  |  log-odds={log_odds:.3f}",
                 fontsize=11, color="#e8e5de", fontweight="bold", pad=14)
    ax.tick_params(colors="#555566")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1c1e24")

    fraud_patch = mpatches.Patch(color="#ffffff", label="Pushes → FRAUD")
    legit_patch = mpatches.Patch(color="#555566", label="Pushes → LEGIT")
    ax.legend(handles=[fraud_patch, legit_patch], fontsize=9,
              facecolor="#0f1014", edgecolor="#1c1e24", labelcolor="#c0bdb0")

    plt.tight_layout()
    plt.savefig("shap_plot.png", dpi=150, bbox_inches="tight")
    print("\nSHAP chart saved → shap_plot.png")
    plt.show()

except ImportError:
    print("\nmatplotlib not installed — skipping chart. Run: pip install matplotlib")
