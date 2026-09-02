"""Combines the tabular fraud score with semantic features from the served
model into one fused score.

Phase 1 skeleton: the semantic side does not exist until Phase 2-3 (the
served model and retrieval are not built yet), so semantic_features is
an empty vector here and the fusion model degenerates to a monotonic
transform of the tabular score alone. The fusion model itself, a small
logistic regression over [tabular_score, *semantic_features], is real
and stays the same shape once semantic features are wired in; only the
input width changes.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

# Bump this when semantic features are wired in (Phase 2-3).
SEMANTIC_FEATURE_DIM = 0


def build_fusion_inputs(
    tabular_scores: np.ndarray, semantic_features: np.ndarray | None = None
) -> np.ndarray:
    """Stacks tabular score and semantic feature vectors into one matrix."""
    tabular_scores = np.asarray(tabular_scores).reshape(-1, 1)
    if semantic_features is None or SEMANTIC_FEATURE_DIM == 0:
        return tabular_scores
    semantic_features = np.asarray(semantic_features).reshape(len(tabular_scores), -1)
    return np.hstack([tabular_scores, semantic_features])


def train_fusion_model(
    tabular_scores: np.ndarray,
    labels: np.ndarray,
    semantic_features: np.ndarray | None = None,
) -> LogisticRegression:
    X = build_fusion_inputs(tabular_scores, semantic_features)
    model = LogisticRegression()
    model.fit(X, labels)
    return model


def fuse_scores(
    model: LogisticRegression,
    tabular_scores: np.ndarray,
    semantic_features: np.ndarray | None = None,
) -> np.ndarray:
    X = build_fusion_inputs(tabular_scores, semantic_features)
    return model.predict_proba(X)[:, 1]
