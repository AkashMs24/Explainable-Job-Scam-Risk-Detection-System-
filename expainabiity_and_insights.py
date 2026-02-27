# ======================================
# Explainability & Insights Script
# ======================================

import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack

# ------------------
# Paths
# ------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ------------------
# Load model artifacts
# ------------------
model = joblib.load(os.path.join(SRC_DIR, "fraud_model.pkl"))
tfidf = joblib.load(os.path.join(SRC_DIR, "tfidf_vectorizer.pkl"))
feature_names = joblib.load(os.path.join(SRC_DIR, "feature_names.pkl"))

# ------------------
# Load data
# ------------------
df = pd.read_csv(os.path.join(DATA_DIR, "fake_job_postings.csv"))

text_cols = ['title', 'description', 'company_profile', 'requirements']
for col in text_cols:
    df[col] = df[col].fillna('')

# ------------------
# Behavioral features (EXACT SAME AS TRAINING)
# ------------------

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

behavior_features = ['desc_length', 'urgency_score', 'free_email']

# ------------------
# Text features
# ------------------
df['combined_text'] = (
    df['title'] + ' ' +
    df['description'] + ' ' +
    df['requirements']
)

X_text = tfidf.transform(df['combined_text'])
X_behavior = df[behavior_features].values

X = hstack([X_text, X_behavior])

# ------------------
# Feature importance
# ------------------
coefficients = model.coef_[0]

print("Feature alignment check:")
print(len(feature_names), len(coefficients))  # MUST MATCH

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': coefficients
}).sort_values(by='importance', ascending=False)

print("\nTop Important Features:")
print(feature_importance.head(15))
