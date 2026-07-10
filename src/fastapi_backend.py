# ==============================
# src/fastapi_backend.py
# COMPLETE PRODUCTION FastAPI BACKEND
# Version: 1.2
# Deploy to: VERCEL
# ==============================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid
import logging
import sys
import os

# ======== ADD PARENT TO PATH ========
sys.path.insert(0, str(Path(__file__).parent))

try:
    from utils import (
        build_feature_vector,
        compute_risk_score,
        get_risk_level,
        compute_shap_values,
        top_shap_features,
        top_driver,
        matched_scam_phrases,
        validate_inputs,
        get_model_info,
        model_confidence,
    )
except ImportError:
    # Fallback for Vercel environment
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import (
        build_feature_vector,
        compute_risk_score,
        get_risk_level,
        compute_shap_values,
        top_shap_features,
        top_driver,
        matched_scam_phrases,
        validate_inputs,
        get_model_info,
        model_confidence,
    )

# ======== SETUP ========
app = FastAPI(
    title="JobGuard AI Backend",
    description="Explainable Job Fraud Detection API",
    version="1.2",
    docs_url="/docs",
)

# ======== CORS (Allow Streamlit Cloud) ========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://*.streamlit.app",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======== LOGGING ========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======== LOAD MODELS ========
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "fraud_model.pkl")
    tfidf = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")
    feature_names = joblib.load(BASE_DIR / "feature_names.pkl")
    logger.info("✅ Models loaded successfully")
except FileNotFoundError as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise RuntimeError(f"Model files not found: {e}")

# ======== PYDANTIC MODELS ========

class PredictionRequest(BaseModel):
    job_title: str = Field(..., min_length=1, max_length=500)
    job_description: str = Field(..., min_length=20, max_length=50000)
    company_profile: Optional[str] = Field(default="", max_length=2000)
    salary_range: Optional[str] = Field(default="", max_length=500)


class PredictionResponse(BaseModel):
    prediction_id: str
    fraud_probability: float
    risk_score: float
    risk_level: str
    advice: str
    confidence: str
    top_drivers: Dict[str, float]
    top_features: List[tuple]
    scam_phrases: List[str]
    feature_values: Dict[str, any]
    model_version: str
    timestamp: str


class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback_type: str
    description: Optional[str] = None
    contact_email: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_features: int

# ======== ROUTES ========

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "name": "JobGuard AI Backend",
        "version": get_model_info()["version"],
        "description": "Explainable Job Fraud Detection API",
        "docs_url": "/docs",
        "status": "🟢 Online",
        "endpoints": {
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict (POST)",
            "batch": "/predict/batch (POST)",
            "feedback": "/feedback (POST)",
            "stats": "/stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_version=get_model_info()["version"],
        loaded_features=len(feature_names)
    )


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Get model metadata."""
    return get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict job fraud risk.
    
    **Parameters:**
    - job_title: Job title/position
    - job_description: Full job description (required)
    - company_profile: Company information (optional)
    - salary_range: Salary information (optional)
    
    **Returns:**
    - Fraud probability, risk score, SHAP explanation
    """
    
    try:
        # Validate
        validation_errors = validate_inputs(
            request.job_title,
            request.job_description,
            request.company_profile,
            request.salary_range
        )
        
        if validation_errors:
            raise HTTPException(
                status_code=422,
                detail={"errors": validation_errors}
            )
        
        # Build features
        X_input, fd = build_feature_vector(
            tfidf,
            request.job_title,
            request.job_description,
            request.company_profile,
            request.salary_range
        )
        
        # Predict
        fraud_prob = float(model.predict_proba(X_input)[0][1])
        risk_score = compute_risk_score(fraud_prob, fd)
        level, _, _, _, advice = get_risk_level(risk_score)
        confidence, _ = model_confidence(fraud_prob)
        
        # Feature importance
        adj_score = (
            0.5 + (fraud_prob - 0.35) / (1 - 0.35) * 0.5
            if fraud_prob >= 0.35
            else fraud_prob / 0.35 * 0.5
        )
        _, top_drivers_dict = top_driver(min(adj_score * 100, 100), fd)
        
        # SHAP
        shap_vals, _, _ = compute_shap_values(model, X_input, feature_names)
        top_feats = top_shap_features(shap_vals, feature_names, n=8)
        
        # Scam phrases
        scam_hits = matched_scam_phrases(request.job_title, request.job_description)
        
        # Generate ID
        prediction_id = str(uuid.uuid4())
        
        logger.info(
            f"Prediction {prediction_id}: "
            f"risk_score={risk_score:.1f}, level={level}, prob={fraud_prob:.3f}"
        )
        
        return PredictionResponse(
            prediction_id=prediction_id,
            fraud_probability=round(fraud_prob, 4),
            risk_score=round(risk_score, 2),
            risk_level=level,
            advice=advice,
            confidence=confidence,
            top_drivers={k: round(v, 2) for k, v in top_drivers_dict.items()},
            top_features=[(f, round(s, 4)) for f, s in top_feats],
            scam_phrases=scam_hits,
            feature_values={
                "desc_length": fd["desc_length"],
                "urgency_score": fd["urgency"],
                "free_email": fd["free_email"],
                "salary_missing": fd["salary_missing"],
                "salary_suspicious": fd.get("salary_suspicious", 0),
                "email_suspicious": fd.get("email_suspicious", False),
            },
            model_version=get_model_info()["version"],
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(requests: List[PredictionRequest]):
    """Batch predict (up to 100 at a time)."""
    if len(requests) > 100:
        raise HTTPException(
            status_code=422,
            detail="Maximum 100 predictions per batch"
        )
    
    results = []
    for i, req in enumerate(requests):
        try:
            X_input, fd = build_feature_vector(
                tfidf,
                req.job_title,
                req.job_description,
                req.company_profile,
                req.salary_range
            )
            
            fraud_prob = float(model.predict_proba(X_input)[0][1])
            risk_score = compute_risk_score(fraud_prob, fd)
            level, _, _, _, advice = get_risk_level(risk_score)
            
            results.append({
                "index": i,
                "fraud_probability": round(fraud_prob, 4),
                "risk_score": round(risk_score, 2),
                "risk_level": level,
                "advice": advice,
            })
        
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e)
            })
    
    return {"predictions": results, "total": len(results)}


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback on predictions."""
    
    if feedback.feedback_type not in ["positive", "negative", "uncertain"]:
        raise HTTPException(
            status_code=422,
            detail="feedback_type must be 'positive', 'negative', or 'uncertain'"
        )
    
    feedback_id = str(uuid.uuid4())[:8]
    
    logger.info(
        f"Feedback {feedback_id}: type={feedback.feedback_type}, "
        f"contact={feedback.contact_email}"
    )
    
    return {
        "status": "success",
        "message": "Thank you! We'll improve our model with your feedback.",
        "feedback_id": feedback_id
    }


@app.get("/stats", tags=["Stats"])
async def get_stats():
    """Get aggregated statistics."""
    return {
        "model_info": get_model_info(),
        "features": {
            "total": len(feature_names),
            "tfidf": 5000,
            "behavioral": 3
        },
        "performance": {
            "test_auc": 0.98,
            "precision": 0.92,
            "recall": 0.88,
            "f1_fraud": 0.88
        }
    }


# ======== ERROR HANDLERS ========

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }


# ======== STARTUP / SHUTDOWN ========

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 JobGuard AI Backend API starting...")
    logger.info(f"Model: {get_model_info()['model_type']}")
    logger.info(f"Features: {len(feature_names)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 JobGuard AI Backend API shutting down...")


# ======== FOR VERCEL ========
# Vercel needs this handler
handler = app
