"""Calibration (isotonic or Platt) of the fused score, plus reliability
diagram generation.

Calibration is a systems property, measurable and real even though the
underlying labels are synthetic: it answers "when this system says 70%,
is it right about 70% of the time on its own validation data," not
"is 70% the true real-world fraud rate."
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_isotonic(scores: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(scores, labels)
    return calibrator


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression()
    calibrator.fit(scores.reshape(-1, 1), labels)
    return calibrator


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    if isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(scores)
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def plot_reliability_diagram(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
) -> Path:
    """Saves a reliability diagram comparing raw vs calibrated scores
    against the diagonal (perfect calibration)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_frac_pos, raw_mean_pred = calibration_curve(labels, raw_scores, n_bins=n_bins, strategy="quantile")
    cal_frac_pos, cal_mean_pred = calibration_curve(
        labels, calibrated_scores, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(raw_mean_pred, raw_frac_pos, marker="o", label="raw tabular score")
    ax.plot(cal_mean_pred, cal_frac_pos, marker="s", label="calibrated score")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title("Reliability diagram (synthetic data, directional)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
