# ==============================
# src/uncertainty.py
# Approximate Bayesian uncertainty quantification for the fraud probability.
#
# HONEST SCOPE NOTE (say this exact thing in interviews, don't oversell it):
# This is NOT exact Bayesian inference over model parameters (that would
# require a Bayesian logistic regression trained with a prior + MCMC/VI,
# which the existing model wasn't). This is a practical approximation
# technique — a Monte Carlo "feature-dropout" ensemble, in the same family
# as MC Dropout (Gal & Ghahramani, 2016) — that estimates *epistemic*
# uncertainty (how much the prediction depends on any single evidence
# fragment) by repeatedly perturbing the active feature set and observing
# how much the output probability moves. Wide variance = the verdict hinges
# on a few specific words/features (fragile). Narrow variance = many
# independent signals agree (robust).
#
# It gives a genuine, useful signal for this application without requiring
# a full model retrain — which is the honest reason to use it, and a good
# thing to say plainly if asked "is this real Bayesian ML?".
# ==============================

import random
import math
from typing import Dict, List, Tuple


def _predict_from_sparse(coef: List[float], intercept: float, active: Dict[int, float]) -> float:
    z = intercept + sum(coef[idx] * val for idx, val in active.items())
    return 1.0 / (1.0 + math.exp(-z))


def estimate_uncertainty(
    tfidf_vector: Dict[int, float],
    behavior_values: Tuple[float, float, float],
    coef: List[float],
    intercept: float,
    n_samples: int = 200,
    dropout_rate: float = 0.15,
    seed: int = 42,
) -> Dict:
    """
    tfidf_vector: {feature_index: tfidf_value} for the 5000 text features (index 0-4999)
    behavior_values: (desc_length, urgency_score, free_email_flag) -> indices 5000,5001,5002
    coef, intercept: the fitted logistic regression weights (from lite_model / model_weights.json)
    """
    rng = random.Random(seed)

    full_active = dict(tfidf_vector)
    full_active[5000] = behavior_values[0]
    full_active[5001] = behavior_values[1]
    full_active[5002] = behavior_values[2]

    point_estimate = _predict_from_sparse(coef, intercept, full_active)

    active_keys = list(full_active.keys())
    samples = []
    for _ in range(n_samples):
        if not active_keys:
            samples.append(point_estimate)
            continue
        keep_prob = 1.0 - dropout_rate
        perturbed = {
            k: v for k, v in full_active.items()
            if rng.random() < keep_prob
        }
        samples.append(_predict_from_sparse(coef, intercept, perturbed))

    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    variance = sum((s - mean) ** 2 for s in samples) / n
    std = math.sqrt(variance)

    lower_idx = max(0, int(0.025 * n))
    upper_idx = min(n - 1, int(0.975 * n))
    ci_low = samples[lower_idx]
    ci_high = samples[upper_idx]

    ci_width = ci_high - ci_low
    if ci_width < 0.10:
        confidence_label = "High confidence — verdict is stable across feature perturbations"
    elif ci_width < 0.25:
        confidence_label = "Moderate confidence — some sensitivity to specific features"
    else:
        confidence_label = "Low confidence — verdict hinges heavily on a few specific features; recommend manual review"

    return {
        "point_estimate": round(point_estimate, 4),
        "mean_estimate": round(mean, 4),
        "std_dev": round(std, 4),
        "credible_interval_95": [round(ci_low, 4), round(ci_high, 4)],
        "credible_interval_width": round(ci_width, 4),
        "confidence_label": confidence_label,
        "method": "monte_carlo_feature_dropout",
        "n_samples": n_samples,
    }
