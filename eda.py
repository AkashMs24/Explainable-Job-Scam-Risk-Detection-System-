import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)
print(df.head())

df.shape
df['fraudulent'].value_counts()

pd.crosstab(df['has_company_logo'], df['fraudulent'], normalize='index')

df['salary_range'].isnull().groupby(df['fraudulent']).mean()

df['desc_length'] = df['description'].astype(str).apply(len)

df.groupby('fraudulent')['desc_length'].mean()

urgency_words = [
    'urgent', 'immediate', 'limited', 'apply fast',
    'hurry', 'few slots', 'act now'
]

def urgency_score(text):
    text = str(text).lower()
    return sum(word in text for word in urgency_words)

df['urgency_score'] = df['description'].apply(urgency_score)

df.groupby('fraudulent')['urgency_score'].mean()

free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']

def free_email_flag(text):
    text = str(text).lower()
    return any(domain in text for domain in free_domains)

df['free_email'] = df['company_profile'].apply(free_email_flag)
df.groupby('fraudulent')['free_email'].mean()

free_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']

def free_email_flag(text):
    text = str(text).lower()
    return any(domain in text for domain in free_domains)

df['free_email'] = df['company_profile'].apply(free_email_flag)
df.groupby('fraudulent')['free_email'].mean()