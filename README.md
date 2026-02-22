🛡️ SCAMGUARD-AI
Explainable Job Scam Risk Detection System using NLP & Machine Learning

SCAMGUARD-AI is an explainable, decision-support system that detects fraudulent job and internship postings.
Instead of only classifying jobs as fake or real, the system assigns a scam risk score (0–100) and explains why a job is considered risky.

This project is designed with real-world deployment and interpretability in mind, making it suitable for freshers, recruiters, and placement platforms.

🚀 Key Highlights

✅ NLP-based analysis of job descriptions

✅ Behavioral scam indicators (urgency language, free email domains, missing salary)

✅ Risk scoring (0–100) instead of binary classification

✅ Explainable predictions for transparency

✅ Streamlit web application for real-time usage

✅ Deployment-ready ML pipeline

🧠 Why This Project Is Unique

Most student projects stop at fake vs real classification.

SCAMGUARD-AI goes further by:

Combining textual NLP features + behavioral fraud patterns

Prioritizing recall (missing a scam is more dangerous than flagging a real job)

Producing actionable risk scores, not just labels

Providing human-readable explanations

Being built as a decision-support system, not a black-box model

This mirrors how real fraud detection systems are designed in industry.

🏗️ Project Structure
Explainable-Job-Scam-Risk-Detection-System/
│
├── app.py                         # Streamlit web application
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── .gitignore                     # Git hygiene
│
├── fraud_model.pkl                # Trained ML model
├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
├── feature_names.pkl              # Feature names (for explainability)
│
├── 02_feature_engineering_and_model.py
├── eda.py
├── explainability_and_insights.py
│
└── data/
    └── fake_job_postings.csv
📊 Dataset

Source: Kaggle – Real or Fake Job Posting Prediction

Records: ~18,000 job postings

Target variable: fraudulent

0 → Real job

1 → Fake job

The dataset contains job titles, descriptions, company profiles, requirements, and metadata.

⚙️ Methodology

Exploratory Data Analysis (EDA)

Studied patterns in fake job postings

Identified behavioral scam signals

Feature Engineering

TF-IDF text vectorization

Urgency language detection

Free email domain detection

Description length analysis

Model Training

Logistic Regression with class-weight balancing

Focus on recall instead of accuracy

Risk Scoring Engine

Combines ML probability with rule-based indicators

Produces a 0–100 scam risk score

Explainability

Feature importance analysis

Human-readable explanations in the UI

🖥️ How to Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/Explainable-Job-Scam-Risk-Detection-System-using-NLP-and-ML.git
cd Explainable-Job-Scam-Risk-Detection-System-using-NLP-and-ML
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Streamlit App
streamlit run app.py

The app will open in your browser at:

http://localhost:8501
🌐 Deployment

The application is designed to be deployed on Streamlit Community Cloud.

Steps:

Push the project to GitHub

Go to https://share.streamlit.io

Select the repository

Set app.py as the main file

Deploy 🚀

🧪 Example Use Case

Input:

Job title: Data Entry Intern

Description: Urgent hiring! Work from home. Limited slots. Apply immediately.

Company profile: Gmail contact

Salary: Not mentioned

Output:

Scam Risk Score: High (e.g., 80+/100)

Explanation:

Urgency-driven language detected

Salary information missing

Free email domain used

⚠️ Disclaimer

SCAMGUARD-AI is a decision-support system.
Predictions should always be combined with manual verification and human judgment.

📌 Skills Demonstrated

Data Science & Machine Learning

Natural Language Processing (NLP)

Feature Engineering

Model Interpretability

Streamlit Deployment

Real-world problem solving

👤 Author

Akash M S
B.Tech (Data Science)
GitHub: https://github.com/AkashMs24

⭐ Final Note

This project is built with placements and real-world relevance in mind.
It demonstrates not just model building, but thinking like a data scientist in production.
