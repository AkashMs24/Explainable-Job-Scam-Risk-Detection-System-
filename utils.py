# ==============================
# utils.py
# Single source of truth for:
#   - Feature engineering  (mirrors 02_feature_engineering_and_mo...py)
#   - Risk scoring engine  (mirrors §10 of training script)
#   - Explainability       (mirrors expainabiity_and_insights.py)
#   - EDA behavioral sigs  (mirrors eda.py)
#   - SHAP computation     (exact LR formula: φᵢ = coef[i] × feature_value[i])
# Imported by app.py at runtime.
# ==============================

import numpy as np
from scipy.sparse import hstack, issparse

# ── CONSTANTS (from training script & eda.py) ─────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediate", "limited",
    "apply fast", "hurry", "few slots", "act now"
]

FREE_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com"
]

BEHAVIOR_FEATURES = ["desc_length", "urgency_score", "free_email"]

FRAUD_THRESHOLD = 0.35

SCAM_PHRASES = [
    "no experience required", "work from home", "earn up to",
    "processing fee", "registration fee", "unlimited earnings",
    "weekly payout", "data entry", "typing work", "copy paste",
]


# ── BEHAVIORAL FEATURE FUNCTIONS (from 02_feature_engineering_and_mo...py) ────

def urgency_score(text: str) -> int:
    """Count urgency words — exact replica of training script §3."""
    text = str(text).lower()
    return sum(word in text for word in URGENCY_WORDS)


def free_email_flag(text: str) -> int:
    """Return 1 if a free email domain appears — exact replica of training §3."""
    text = str(text).lower()
    return int(any(domain in text for domain in FREE_DOMAINS))


def desc_length(text: str) -> int:
    """Character length of description — exact replica of training §3."""
    return len(str(text))


# ── FEATURE VECTOR BUILDER (mirrors training pipeline exactly) ────────────────

def build_feature_vector(tfidf_vectorizer, job_title: str, job_description: str,
                         company_profile: str, salary_range: str):
    """
    Replicates §4–§5 of training script exactly:
        combined_text = title + description + requirements(≈company_profile)
        X_text        = tfidf.transform(combined_text)           → 5000 dims
        X_behavior    = [desc_length, urgency_score, free_email] →    3 dims
        X_final       = hstack([X_text, X_behavior])             → 5003 dims

    Returns (X_final [sparse, 1×5003], feature_dict).
    """
    combined_text = f"{job_title} {job_description} {company_profile}"
    X_text     = tfidf_vectorizer.transform([combined_text])

    d_len      = desc_length(job_description)
    urg        = urgency_score(job_description)
    free_em    = free_email_flag(company_profile)

    X_behavior = np.array([[d_len, urg, free_em]])
    X_final    = hstack([X_text, X_behavior])

    fd = {
        "desc_length":    d_len,
        "urgency":        urg,
        "free_email":     free_em,
        "salary_missing": int(not str(salary_range).strip()),
    }
    return X_final, fd


# ── RISK SCORING ENGINE (mirrors training script §10 exactly) ─────────────────

def compute_risk_score(fraud_prob: float, fd: dict) -> float:
    """
    Composite score replicating §10:
        risk_score = 0.60 × adj_prob
                   + 0.15 × urgency_norm
                   + 0.15 × salary_missing
                   + 0.10 × free_email
    Clipped to [0, 100].
    The adj_prob remapping keeps the 0–50 range for sub-threshold
    and 50–100 range for super-threshold predictions.
    """
    if fraud_prob >= FRAUD_THRESHOLD:
        adj = 0.5 + (fraud_prob - FRAUD_THRESHOLD) / (1 - FRAUD_THRESHOLD) * 0.5
    else:
        adj = fraud_prob / FRAUD_THRESHOLD * 0.5
    adj = min(adj, 1.0)

    urg_norm = min(fd["urgency"] / max(len(URGENCY_WORDS), 1), 1.0)

    score = (
        0.60 * adj
        + 0.15 * urg_norm
        + 0.15 * fd["salary_missing"]
        + 0.10 * fd["free_email"]
    ) * 100

    # Behavioral floor: if 2+ behavioral signals fire AND model says FRAUD, lift to ≥62
    beh = (0.15 * urg_norm + 0.15 * fd["salary_missing"] + 0.10 * fd["free_email"]) * 100
    if beh >= 20 and fraud_prob >= FRAUD_THRESHOLD:
        score = max(score, 62.0)

    return round(float(np.clip(score, 0, 100)), 2)


def get_risk_level(score: float):
    """Return (level, pill_class, card_class, accent_color, advice)."""
    if score < 30:
        return ("LOW",    "pill-green", "good",   "#aaaaaa",
                "Job appears relatively safe. Verify company details independently.")
    elif score < 60:
        return ("MEDIUM", "pill-amber", "warn",   "#dddddd",
                "Proceed with caution. Do not share documents or pay any fee.")
    else:
        return ("HIGH",   "pill-red",   "danger", "#ffffff",
                "High scam risk. Do NOT apply or share personal information.")


# ── MODEL CONFIDENCE (from expainabiity_and_insights.py) ─────────────────────

def model_confidence(prob: float):
    """Return (label, pill_class) based on distance from threshold."""
    dist = abs(prob - FRAUD_THRESHOLD)
    if dist >= 0.25:   return "High",               "pill-green"
    elif dist >= 0.10: return "Moderate",            "pill-amber"
    return               "Low (borderline)",         "pill-red"


# ── FEATURE IMPORTANCE (from expainabiity_and_insights.py) ───────────────────

def get_feature_importance(model, feature_names):
    """
    Direct coef_-based importance — mirrors expainabiity_and_insights.py.
    Returns sorted list of (feature_name, coefficient), highest abs first.
    """
    coefficients = model.coef_[0]
    n = min(len(coefficients), len(feature_names))
    paired = sorted(
        zip(feature_names[:n], coefficients[:n]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return paired


# ── SHAP FOR LOGISTIC REGRESSION (exact, no shap library needed) ──────────────

def compute_shap_values(model, X_final, feature_names):
    """
    Exact SHAP for Logistic Regression (linear model):
        φᵢ = coef[i] × feature_value[i]   (log-odds space)
        Σ φᵢ + intercept = log-odds = logit(P(fraud))

    This is mathematically exact for linear models — not an approximation.
    No shap library required.

    Returns:
        shap_vals   : np.array shape (n_features,)
        intercept   : float
        log_odds    : float  (intercept + Σ φᵢ)
        top_n_idx   : indices of top-N features by |φᵢ|
    """
    coef = model.coef_[0]
    n = min(len(coef), len(feature_names))
    coef = coef[:n]

    # Extract feature values from sparse or dense matrix
    if issparse(X_final):
        x_vec = np.array(X_final.todense()).flatten()[:n]
    else:
        x_vec = np.array(X_final).flatten()[:n]

    shap_vals = coef * x_vec                      # φᵢ = coef[i] × x[i]
    intercept = float(model.intercept_[0])
    log_odds  = intercept + float(shap_vals.sum())

    return shap_vals, intercept, log_odds


def top_shap_features(shap_vals, feature_names, n=15):
    """
    Return top-n features sorted by |SHAP value|.
    Returns list of (name, shap_value) tuples.
    """
    n_feats = min(len(shap_vals), len(feature_names))
    idx_sorted = np.argsort(np.abs(shap_vals[:n_feats]))[::-1][:n]
    return [(feature_names[i], float(shap_vals[i])) for i in idx_sorted]


# ── ADDITIONAL BEHAVIORAL SIGNALS (from eda.py) ───────────────────────────────

def caps_ratio(text: str) -> float:
    """Proportion of ALL-CAPS words — from eda.py exploration."""
    words = str(text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


def suspicious_salary(salary_text: str) -> int:
    """Flag vague or unrealistically high salary — mirrors eda.py salary analysis."""
    import re
    if not str(salary_text).strip():
        return 0
    sl = str(salary_text).lower()
    if any(w in sl for w in ["unlimited", "upto", "up to", "per day"]):
        return 1
    nums = re.findall(r'\d[\d,]*', salary_text)
    if nums:
        try:
            if max(int(n.replace(",", "")) for n in nums) > 500000:
                return 1
        except Exception:
            pass
    return 0


def top_driver(adj_score: float, fd: dict):
    """
    Identify the single largest contributor to the composite risk score.
    Returns (top_driver_name, contributions_dict).
    adj_score is already in 0–100 pts (the 0.60 × adj × 100 component).
    """
    urg_norm = min(fd["urgency"] / max(len(URGENCY_WORDS), 1), 1.0)
    contribs = {
        "ML model fraud probability": round(adj_score * 0.60, 2),
        "Urgency language":           round(urg_norm * 0.15 * 100, 2),
        "Missing salary info":        round(fd["salary_missing"] * 0.15 * 100, 2),
        "Free/unverified email":      round(fd["free_email"] * 0.10 * 100, 2),
    }
    active = {k: v for k, v in contribs.items() if v > 0}
    top    = max(active, key=active.get) if active else "No dominant driver"
    return top, contribs


def matched_scam_phrases(job_title: str, job_description: str) -> list:
    """Return list of SCAM_PHRASES found in title + description."""
    combined = (job_description + " " + job_title).lower()
    return [p for p in SCAM_PHRASES if p in combined]
