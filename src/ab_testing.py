# ==============================
# src/ab_testing.py
# A/B testing framework for comparing fraud-detection configurations
# (e.g. two risk-score thresholds, or "with uncertainty gating" vs "without").
#
# Deterministic hash-based bucketing: the same request_id always lands in the
# same arm, so results are reproducible and don't require storing per-user
# state. Includes a two-proportion z-test so you can report whether an
# observed difference between arms is statistically significant, not just
# "arm B looked a bit better."
# ==============================

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


def assign_arm(request_id: str, arm_names: List[str] = ("control", "treatment")) -> str:
    """Deterministically assigns a request_id to an experiment arm via hashing."""
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % len(arm_names)
    return arm_names[bucket]


@dataclass
class ExperimentResult:
    arm: str
    request_id: str
    fraud_probability: float
    flagged_as_fraud: bool
    ground_truth: Optional[bool] = None  # filled in later via /feedback, if available


@dataclass
class Experiment:
    name: str
    arm_names: List[str] = field(default_factory=lambda: ["control", "treatment"])
    results: List[ExperimentResult] = field(default_factory=list)

    def assign(self, request_id: str) -> str:
        return assign_arm(request_id, self.arm_names)

    def log_result(self, request_id: str, arm: str, fraud_probability: float,
                    flagged_as_fraud: bool, ground_truth: Optional[bool] = None):
        self.results.append(ExperimentResult(
            arm=arm, request_id=request_id, fraud_probability=fraud_probability,
            flagged_as_fraud=flagged_as_fraud, ground_truth=ground_truth,
        ))

    def record_feedback(self, request_id: str, ground_truth: bool) -> bool:
        for r in self.results:
            if r.request_id == request_id:
                r.ground_truth = ground_truth
                return True
        return False

    def summary(self) -> Dict:
        by_arm: Dict[str, List[ExperimentResult]] = {a: [] for a in self.arm_names}
        for r in self.results:
            by_arm.setdefault(r.arm, []).append(r)

        out = {}
        for arm, rows in by_arm.items():
            n = len(rows)
            flagged = sum(1 for r in rows if r.flagged_as_fraud)
            labeled = [r for r in rows if r.ground_truth is not None]
            correct = sum(1 for r in labeled if r.flagged_as_fraud == r.ground_truth)
            out[arm] = {
                "n": n,
                "flagged_rate": round(flagged / n, 4) if n else None,
                "avg_fraud_probability": round(sum(r.fraud_probability for r in rows) / n, 4) if n else None,
                "labeled_count": len(labeled),
                "accuracy_on_labeled": round(correct / len(labeled), 4) if labeled else None,
            }
        return out

    def significance_test(self, arm_a: str, arm_b: str, metric: str = "flagged_rate") -> Dict:
        """
        Two-proportion z-test comparing flagged_rate (or accuracy_on_labeled)
        between two arms. Returns z-statistic and a two-tailed p-value.
        """
        summary = self.summary()
        if arm_a not in summary or arm_b not in summary:
            return {"error": f"Unknown arm(s). Available: {list(summary.keys())}"}

        rows_a = [r for r in self.results if r.arm == arm_a]
        rows_b = [r for r in self.results if r.arm == arm_b]

        if metric == "flagged_rate":
            x_a = sum(1 for r in rows_a if r.flagged_as_fraud)
            x_b = sum(1 for r in rows_b if r.flagged_as_fraud)
            n_a, n_b = len(rows_a), len(rows_b)
        elif metric == "accuracy_on_labeled":
            rows_a = [r for r in rows_a if r.ground_truth is not None]
            rows_b = [r for r in rows_b if r.ground_truth is not None]
            x_a = sum(1 for r in rows_a if r.flagged_as_fraud == r.ground_truth)
            x_b = sum(1 for r in rows_b if r.flagged_as_fraud == r.ground_truth)
            n_a, n_b = len(rows_a), len(rows_b)
        else:
            return {"error": f"Unsupported metric: {metric}"}

        if n_a == 0 or n_b == 0:
            return {"error": "Not enough samples in one or both arms yet."}

        p_a, p_b = x_a / n_a, x_b / n_b
        p_pool = (x_a + x_b) / (n_a + n_b)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b)) if 0 < p_pool < 1 else 0

        if se == 0:
            return {
                "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
                "p_a": round(p_a, 4), "p_b": round(p_b, 4),
                "z_stat": None, "p_value": None,
                "significant_at_0.05": False,
                "note": "No variance in pooled proportion — cannot compute z-test yet.",
            }

        z = (p_a - p_b) / se
        p_value = 2 * (1 - _norm_cdf(abs(z)))

        return {
            "arm_a": arm_a, "arm_b": arm_b, "metric": metric,
            "n_a": n_a, "n_b": n_b,
            "p_a": round(p_a, 4), "p_b": round(p_b, 4),
            "z_stat": round(z, 4),
            "p_value": round(p_value, 4),
            "significant_at_0.05": p_value < 0.05,
        }


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function (no scipy dependency needed)."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
