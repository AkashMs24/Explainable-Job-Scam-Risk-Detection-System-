# ==============================
# setup.py — Package Setup
# ==============================

from setuptools import setup, find_packages

setup(
    name="jobguard-ai",
    version="1.2.0",
    description="Explainable Job Fraud Detection System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Akash MS",
    author_email="akash@example.com",
    url="https://github.com/AkashMs24/explainable-job-scam-risk-detection-system",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "streamlit>=1.35.0",
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "pdfplumber>=0.10.0",
        "pytesseract>=0.3.10",
        "plotly>=5.17.0",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
        "ml": ["xgboost>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="fraud detection, explainable ai, shap, streamlit, fastapi",
)
