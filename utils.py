# ==============================
# utils.py
# Shared logic extracted from:
#   - 02_feature_engineering_and_mo...py
#   - expainabiity_and_insights.py
#   - eda.py
# Imported by app.py at runtime.
# ==============================

import numpy as np
from scipy.sparse import hstack

# ── CONSTANTS (from 02_feature_engineering_and_mo...py & eda.py) ─────────────
URGENCY_WORDS = [
    "urgent", "immediate", "limited",
    "apply fast", "hurry", "few slots", "act now"
]

FREE_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com"
]

BEHAVIOR_FEATURES = ["desc_length", "urgency_score", "free_email"]

FRAUD_THRESHOLD = 0.35


# ── BEHAVIORAL FEATURE FUNCTIONS (from 02_feature_engineering_and_mo...py) ───

def urgency_score(text: str) -> int:
    """Count how many urgency words appear in text."""
    text = str(text).lower()
    return sum(word in text for word in URGENCY_WORDS)


def free_email_flag(text: str) -> int:
    """Return 1 if a free email domain is found in text."""
    text = str(text).lower()
    return int(any(domain in text for domain in FREE_DOMAINS))


def desc_length(text: str) -> int:
    """Return character length of description."""
    return len(str(text))


# ── FEATURE VECTOR BUILDER (from 02_feature_engineering_and_mo...py) ─────────

def build_feature_vector(tfidf_vectorizer, job_title, job_description,
                         company_profile, salary_range):
    """
    Replicates training pipeline exactly:
      combined_text = title + description + requirements(proxied by company_profile)
      X = hstack([X_text, X_behavior])
    Returns (X_final, feature_dict).
    """
    combined_text = f"{job_title} {job_description} {company_profile}"
    X_text     = tfidf_vectorizer.transform([combined_text])
    d_len      = desc_length(job_description)
    urg        = urgency_score(job_description)
    free_email = free_email_flag(company_profile)
    X_behavior = np.array([[d_len, urg, free_email]])
    X_final    = hstack([X_text, X_behavior])
    fd = {
        "desc_length":    d_len,
        "urgency":        urg,
        "free_email":     free_email,
        "salary_missing": int(not str(salary_range).strip()),
    }
    return X_final, fd


# ── RISK SCORING ENGINE (from 02_feature_engineering_and_mo...py §10) ────────

def compute_risk_score(fraud_prob: float, fd: dict) -> float:
    """
    Composite risk score mirroring training script §10:
      risk_score = 0.60 * adj_prob
                 + 0.15 * urgency_norm
                 + 0.15 * salary_missing
                 + 0.10 * free_email
    Clipped to [0, 100].
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

    beh = (0.15 * urg_norm + 0.15 * fd["salary_missing"] + 0.10 * fd["free_email"]) * 100
    if beh >= 20 and fraud_prob >= FRAUD_THRESHOLD:
        score = max(score, 62.0)

    return round(float(np.clip(score, 0, 100)), 2)


def get_risk_level(score: float):
    """Return (level, pill_class, card_class, accent_color, advice)."""
    if score < 30:
        return "LOW",    "pill-green", "good",    "#9090a8", \
               "Job appears relatively safe. Verify company details independently."
    elif score < 60:
        return "MEDIUM", "pill-amber", "warn",    "#888898", \
               "Proceed with caution. Do not share documents or pay any fee."
    else:
        return "HIGH",   "pill-red",   "danger",  "#ffffff", \
               "High scam risk. Do NOT apply or share personal information."


# ── MODEL CONFIDENCE (from expainabiity_and_insights.py) ─────────────────────

def model_confidence(prob: float):
    """Return (label, pill_class) based on distance from threshold."""
    dist = abs(prob - FRAUD_THRESHOLD)
    if dist >= 0.25:   return "High",           "pill-green"
    elif dist >= 0.10: return "Moderate",        "pill-amber"
    return               "Low (borderline)",     "pill-red"


# ── FEATURE IMPORTANCE (from expainabiity_and_insights.py) ───────────────────

def get_feature_importance(model, feature_names):
    """
    Direct coef_-based importance — mirrors expainabiity_and_insights.py §Feature importance.
    Returns sorted list of (feature_name, coefficient).
    """
    coefficients = model.coef_[0]
    paired = sorted(
        zip(feature_names, coefficients),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return paired


# ── ADDITIONAL BEHAVIORAL SIGNALS (from eda.py) ───────────────────────────────

def caps_ratio(text: str) -> float:
    """Proportion of ALL-CAPS words — from eda.py exploration."""
    words = str(text).split()
    if not words:
        return 0.0
    return sum(1 for w in words if w.isupper() and len(w) > 1) / len(words)


def suspicious_salary(salary_text: str) -> int:
    """
    Flag vague or unrealistically high salary — mirrors eda.py salary analysis.
    """
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
    """Return (top_driver_name, contributions_dict)."""
    urg_norm = min(fd["urgency"] / max(len(URGENCY_WORDS), 1), 1.0)
    contribs = {
        "ML model fraud probability": adj_score * 0.60,
        "Urgency language":           urg_norm * 0.15 * 100,
        "Missing salary info":        fd["salary_missing"] * 0.15 * 100,
        "Free/unverified email":      fd["free_email"] * 0.10 * 100,
    }
    active = {k: v for k, v in contribs.items() if v > 0}
    return (max(active, key=active.get) if active else "No dominant driver"), contribs


# ── SCAM PHRASE LIST (from eda.py exploration) ────────────────────────────────

SCAM_PHRASES = [
    "no experience required", "work from home", "earn up to",
    "processing fee", "registration fee", "unlimited earnings",
    "weekly payout", "data entry", "typing work", "copy paste",
]


def matched_scam_phrases(job_title: str, job_description: str) -> list:
    """Return list of scam phrases found in title + description."""
    combined = (job_description + " " + job_title).lower()
    return [p for p in SCAM_PHRASES if p in combined]
