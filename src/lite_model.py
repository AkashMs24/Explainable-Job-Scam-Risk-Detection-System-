# ==============================
# src/lite_model.py
# Pure-Python replacement for the pickled sklearn model + TfidfVectorizer.
# Loads plain-text model_weights.json (git-safe, no binary corruption risk)
# and exposes the same .transform() / .predict_proba() / .coef_ / .intercept_
# interface that utils.py already expects, so nothing else in the project
# needs to change.
# ==============================

import json
import re
import math
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix


class LiteTfidfVectorizer:
    """Mimics sklearn's TfidfVectorizer.transform() for a fixed, pre-fit vocabulary."""

    def __init__(self, vocabulary: dict, idf: list, stop_words: list):
        self.vocabulary_ = vocabulary
        self.idf_ = np.array(idf, dtype=np.float64)
        self._stop_words = set(stop_words)
        self._token_re = re.compile(r"(?u)\b\w\w+\b")

    def transform(self, texts):
        rows, cols, data = [], [], []
        for row_idx, text in enumerate(texts):
            text = (text or "").lower()
            tokens = [t for t in self._token_re.findall(text) if t not in self._stop_words]

            counts = {}
            for t in tokens:
                idx = self.vocabulary_.get(t)
                if idx is not None:
                    counts[idx] = counts.get(idx, 0) + 1

            # tf-idf (smooth_idf, no sublinear_tf) then L2 normalize
            vec = {idx: c * self.idf_[idx] for idx, c in counts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0

            for idx, v in vec.items():
                rows.append(row_idx)
                cols.append(idx)
                data.append(v / norm)

        return csr_matrix((data, (rows, cols)), shape=(len(texts), len(self.vocabulary_)))


class LiteLogisticRegression:
    """Mimics sklearn's LogisticRegression.predict_proba() / .coef_ / .intercept_."""

    def __init__(self, coef: list, intercept: float):
        self.coef_ = np.array([coef], dtype=np.float64)
        self.intercept_ = np.array([intercept], dtype=np.float64)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        x = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        z = x @ self.coef_[0] + self.intercept_[0]
        p1 = 1.0 / (1.0 + np.exp(-z))
        p0 = 1.0 - p1
        return np.vstack([p0, p1]).T


def load_lite_model(weights_path: Path):
    """Load model_weights.json and return (model, tfidf, feature_names)."""
    with open(weights_path, "r", encoding="utf-8") as f:
        W = json.load(f)

    tfidf = LiteTfidfVectorizer(W["vocabulary"], W["idf"], W["stop_words"])
    model = LiteLogisticRegression(W["coef"], W["intercept"])

    vocab_sorted = sorted(W["vocabulary"], key=lambda w: W["vocabulary"][w])
    feature_names = vocab_sorted + W["behavior_feature_order"]

    return model, tfidf, feature_names
