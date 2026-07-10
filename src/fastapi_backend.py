# ==============================
# src/fastapi_backend.py
# COMPLETE PRODUCTION FastAPI BACKEND
# Version: 1.3
# Deploy to: VERCEL
#
# v1.3 wires in the four new src/ modules:
#   - company_verification.py  -> POST /verify/company
#   - email_reputation.py      -> POST /verify/email
#   - uncertainty.py           -> optional block on POST /predict
#   - ab_testing.py            -> POST /predict (auto-assigned arm) +
#                                  GET/POST /experiments/*
#
# IMPORTANT — Vercel/serverless note:
# ab_testing.py keeps its results in an in-memory Python object. That is
# fine on the Docker/Streamlit-Cloud style deployment where one process
# stays warm, but on Vercel each request can hit a fresh, isolated
# serverless instance, so the in-memory experiment log will NOT reliably
# accumulate across requests there. The endpoints are still wired up
# (so nothing breaks / 404s) but if you need real cross-request A/B
# stats on Vercel, back the Experiment with an external store (e.g. a
# tiny KV/Redis/Postgres) instead of the in-memory list.
# ==============================

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

# ======== NEW FEATURE MODULES ========
from company_verification import verify_company_identifier
from email_reputation import check_email_reputation
from uncertainty import estimate_uncertainty
from ab_testing import Experiment

# ======== SETUP ========
app = FastAPI(
    title="JobGuard AI Backend",
    description="Explainable Job Fraud Detection API",
    version="1.3",
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
# No pickle/joblib, no scikit-learn at runtime. Model weights live in the
# plain-text model_weights.json (git-safe), loaded via lite_model.py.
from lite_model import load_lite_model

BASE_DIR = Path(__file__).resolve().parent

try:
    model, tfidf, feature_names = load_lite_model(BASE_DIR / "model_weights.json")
    logger.info("✅ Models loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    raise RuntimeError(f"Model weights not found: {e}")

# Precompute plain-python coef/intercept once at startup (cheap — reused by
# every /predict call that opts into uncertainty, instead of re-converting
# the numpy arrays on every request).
_MODEL_COEF: List[float] = model.coef_[0].tolist()
_MODEL_INTERCEPT: float = float(model.intercept_[0])
_N_VOCAB: int = len(tfidf.vocabulary_)

# ======== A/B EXPERIMENT REGISTRY ========
# In-memory registry of named experiments. See the Vercel note at the top
# of this file about persistence across serverless invocations.
_EXPERIMENTS: Dict[str, Experiment] = {
    "default": Experiment(name="default"),
}
# Risk-score cutoff used to decide "flagged_as_fraud" for A/B bucketing.
_AB_FLAG_THRESHOLD = 50.0


def _get_experiment(name: str) -> Experiment:
    if name not in _EXPERIMENTS:
        _EXPERIMENTS[name] = Experiment(name=name)
    return _EXPERIMENTS[name]


# ======== PYDANTIC MODELS ========

class PredictionRequest(BaseModel):
    job_title: str = Field(..., min_length=1, max_length=500)
    job_description: str = Field(..., min_length=20, max_length=50000)
    company_profile: Optional[str] = Field(default="", max_length=2000)
    salary_range: Optional[str] = Field(default="", max_length=500)
    # --- new, all optional & default-off so /predict stays fast by default ---
    include_uncertainty: bool = Field(
        default=False,
        description="If true, runs a Monte Carlo feature-dropout uncertainty "
                     "pass (~200 forward passes, pure python). Adds latency, "
                     "opt-in only.",
    )
    uncertainty_samples: int = Field(
        default=200, ge=20, le=500,
        description="Number of MC dropout samples if include_uncertainty=true.",
    )
    experiment_name: Optional[str] = Field(
        default=None,
        description="If set, this request is bucketed into an A/B experiment "
                     "arm (created on first use) and logged for later "
                     "/experiments/{name}/summary and .../significance calls.",
    )


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
    feature_values: Dict[str, Any]
    model_version: str
    timestamp: str
    uncertainty: Optional[Dict[str, Any]] = None
    experiment: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    prediction_id: str
    feedback_type: str
    description: Optional[str] = None
    contact_email: Optional[str] = None
    # --- new ---
    experiment_name: Optional[str] = Field(
        default=None,
        description="If the original prediction was logged under an "
                     "experiment, pass the same name here to also record "
                     "ground_truth against that experiment's results.",
    )
    ground_truth_is_fraud: Optional[bool] = Field(
        default=None,
        description="True/False label for the posting, used to update A/B "
                     "significance stats when experiment_name is provided.",
    )


class HealthResponse(BaseModel):
    status: str
    model_version: str
    loaded_features: int


class CompanyVerificationRequest(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=64,
                             description="GSTIN (15 chars) or CIN (21 chars)")


class EmailReputationRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class SignificanceRequest(BaseModel):
    arm_a: str = "control"
    arm_b: str = "treatment"
    metric: str = Field(default="flagged_rate",
                         description="'flagged_rate' or 'accuracy_on_labeled'")

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
            "stats": "/stats",
            "verify_company": "/verify/company (POST)",
            "verify_email": "/verify/email (POST)",
            "experiment_summary": "/experiments/{name}/summary (GET)",
            "experiment_significance": "/experiments/{name}/significance (POST)",
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


def _sparse_to_uncertainty_inputs(X_input):
    """
    Splits the hstacked (tfidf | behavioral) sparse row from
    utils.build_feature_vector into the shapes uncertainty.estimate_uncertainty
    expects: a {feature_index: value} dict for the TF-IDF block, and a
    (desc_length, urgency_score, free_email_flag) tuple for the 3 behavioral
    columns.
    """
    X_coo = X_input.tocoo()
    tfidf_vector: Dict[int, float] = {}
    behavior_values = [0.0, 0.0, 0.0]
    for col, val in zip(X_coo.col, X_coo.data):
        col = int(col)
        if col < _N_VOCAB:
            tfidf_vector[col] = float(val)
        else:
            behavior_idx = col - _N_VOCAB
            if 0 <= behavior_idx < 3:
                behavior_values[behavior_idx] = float(val)
    return tfidf_vector, tuple(behavior_values)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Predict job fraud risk.

    **Parameters:**
    - job_title: Job title/position
    - job_description: Full job description (required)
    - company_profile: Company information (optional)
    - salary_range: Salary information (optional)
    - include_uncertainty: opt-in Monte Carlo confidence-interval pass
    - experiment_name: opt-in A/B bucketing + logging

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

        # --- optional: uncertainty quantification (opt-in, off by default) ---
        uncertainty_block = None
        if request.include_uncertainty:
            tfidf_vector, behavior_values = _sparse_to_uncertainty_inputs(X_input)
            uncertainty_block = estimate_uncertainty(
                tfidf_vector=tfidf_vector,
                behavior_values=behavior_values,
                coef=_MODEL_COEF,
                intercept=_MODEL_INTERCEPT,
                n_samples=request.uncertainty_samples,
            )

        # --- optional: A/B experiment bucketing + logging (opt-in) ---
        experiment_block = None
        if request.experiment_name:
            experiment = _get_experiment(request.experiment_name)
            arm = experiment.assign(prediction_id)
            flagged = risk_score >= _AB_FLAG_THRESHOLD
            experiment.log_result(
                request_id=prediction_id,
                arm=arm,
                fraud_probability=fraud_prob,
                flagged_as_fraud=flagged,
            )
            experiment_block = {
                "experiment_name": request.experiment_name,
                "arm": arm,
                "flagged_as_fraud": flagged,
            }

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
            timestamp=datetime.utcnow().isoformat(),
            uncertainty=uncertainty_block,
            experiment=experiment_block,
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

    ab_updated = False
    if feedback.experiment_name and feedback.ground_truth_is_fraud is not None:
        experiment = _get_experiment(feedback.experiment_name)
        ab_updated = experiment.record_feedback(
            request_id=feedback.prediction_id,
            ground_truth=feedback.ground_truth_is_fraud,
        )

    return {
        "status": "success",
        "message": "Thank you! We'll improve our model with your feedback.",
        "feedback_id": feedback_id,
        "experiment_ground_truth_recorded": ab_updated,
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


# ======== NEW: COMPANY / EMAIL VERIFICATION ========

@app.post("/verify/company", tags=["Verification"])
async def verify_company(request: CompanyVerificationRequest):
    """
    Structural + checksum validation of an Indian GSTIN (15 chars) or
    CIN (21 chars). NOT a live government-database lookup — see
    company_verification.py for the scope note. Cheap/CPU-only, safe to
    call on every request if you want it inline with /predict.
    """
    try:
        return verify_company_identifier(request.identifier)
    except Exception as e:
        logger.error(f"Company verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@app.post("/verify/email", tags=["Verification"])
async def verify_email(request: EmailReputationRequest):
    """
    Email reputation check: free/disposable provider flags (instant) plus
    SPF/DMARC DNS lookups (network I/O, ~up to a few seconds worst case on
    a cold/unresponsive domain). Kept as its OWN endpoint rather than being
    folded into /predict so a slow or non-responsive DNS lookup can never
    add latency to the core fraud-scoring path.
    """
    try:
        return check_email_reputation(request.email)
    except Exception as e:
        logger.error(f"Email reputation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# ======== NEW: A/B EXPERIMENT ENDPOINTS ========

@app.get("/experiments/{name}/summary", tags=["Experiments"])
async def experiment_summary(name: str):
    """Per-arm counts / flagged-rate / avg fraud probability / accuracy so far."""
    experiment = _get_experiment(name)
    return {"experiment_name": name, "summary": experiment.summary()}


@app.post("/experiments/{name}/significance", tags=["Experiments"])
async def experiment_significance(name: str, request: SignificanceRequest):
    """Two-proportion z-test between two arms of a named experiment."""
    experiment = _get_experiment(name)
    return experiment.significance_test(
        arm_a=request.arm_a, arm_b=request.arm_b, metric=request.metric
    )


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
