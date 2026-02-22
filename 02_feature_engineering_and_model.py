# ==============================
# 1. Imports & Paths
# ==============================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from scipy.sparse import hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"
SRC_DIR = BASE_DIR / "src"

SRC_DIR.mkdir(exist_ok=True)


# ==============================
# 2. Load Data
# ==============================

df = pd.read_csv(DATA_PATH)

text_cols = ['title', 'description', 'company_profile', 'requirements']
for col in text_cols:
    df[col] = df[col].fillna('')


# ==============================
# 3. Behavioral Feature Engineering
# ==============================

# Description length
df['desc_length'] = df['description'].apply(len)

# Urgency score
urgency_words = [
    'urgent', 'immediate', 'limited',
    'apply fast', 'hurry', 'few slots', 'act now'
]

def urgency_score(text):
    text = text.lower()
    return sum(word in text for word in urgency_words)

df['urgency_score'] = df['description'].apply(urgency_score)

# Free email flag
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
X_behavior = df[behavior_features].values

X = hstack([X_text, X_behavior])
y = df['fraudulent']


# ==============================
# 6. Train-Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ==============================
# 7. Model Training
# ==============================

model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)

model.fit(X_train, y_train)


# ==============================
# 8. Evaluation
# ==============================

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================
# 9. SAVE MODEL ARTIFACTS (CRITICAL)
# ==============================

tfidf_features = list(tfidf.get_feature_names_out())
feature_names = tfidf_features + behavior_features

joblib.dump(model, SRC_DIR / "fraud_model.pkl")
joblib.dump(tfidf, SRC_DIR / "tfidf_vectorizer.pkl")
joblib.dump(feature_names, SRC_DIR / "feature_names.pkl")


# ==============================
# 10. Risk Scoring Engine (Post-Model)
# ==============================

# Fraud probability
y_proba = model.predict_proba(X_test)[:, 1]
test_idx = y_test.index

# Normalize urgency
urgency_norm = (
    df.loc[test_idx, 'urgency_score'] /
    df['urgency_score'].max()
).fillna(0)

# Salary risk (USED ONLY FOR SCORING, NOT TRAINING)
salary_risk = (
    df.loc[test_idx, 'salary_range']
      .isnull()
      .astype(int)
)

email_risk = df.loc[test_idx, 'free_email']

risk_score = (
    0.60 * y_proba +
    0.15 * urgency_norm +
    0.15 * salary_risk +
    0.10 * email_risk
)

risk_score = np.clip(risk_score * 100, 0, 100)


def risk_bucket(score):
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    else:
        return "High"


risk_level = [risk_bucket(score) for score in risk_score]


results = pd.DataFrame({
    "fraud_probability": y_proba,
    "risk_score": risk_score,
    "risk_level": risk_level,
    "actual_label": y_test.values
})

print("\nSample High-Risk Prediction:")
print(results[results['risk_level'] == 'High'].head(1))


# ==============================
# 11. Feature Importance (Explainability)
# ==============================

coefficients = model.coef_[0]

print("\nFeature alignment check:")
print(len(feature_names), len(coefficients))  # MUST MATCH

feature_importance = pd.DataFrame({
    "feature": feature_names,
    "importance": coefficients
}).sort_values(by="importance", ascending=False)

print("\nTop Important Features:")
print(feature_importance.head(15))