"""End-to-end Phase 1 smoke test: generate synthetic data, train the
tabular model, fuse, calibrate, and emit a reliability diagram.

Run manually:

    python tests/test_phase1_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from data.generate import artifacts_dir, generate_synthetic_transactions
from fusion.calibrate import apply_calibrator, fit_isotonic, plot_reliability_diagram
from fusion.fuse import fuse_scores, train_fusion_model
from tabular.train import train_xgboost


def main() -> None:
    df = generate_synthetic_transactions(n=2000, seed=7)
    assert len(df) == 2000
    assert df["label"].nunique() == 2, "synthetic labels collapsed to one class"

    model, X_val, y_val, tabular_scores, auc = train_xgboost(df, seed=7)
    print(f"[Phase 1 smoke] tabular AUC (directional, synthetic): {auc:.3f}")
    assert 0.5 < auc <= 1.0, f"AUC {auc:.3f} is not better than chance, generator or model is broken"

    fusion_model = train_fusion_model(tabular_scores, y_val.values)
    fused_scores = fuse_scores(fusion_model, tabular_scores)
    assert fused_scores.shape == tabular_scores.shape
    assert np.all((fused_scores >= 0) & (fused_scores <= 1))

    calibrator = fit_isotonic(fused_scores, y_val.values)
    calibrated_scores = apply_calibrator(calibrator, fused_scores)
    assert np.all((calibrated_scores >= 0) & (calibrated_scores <= 1))
    print(f"[Phase 1 smoke] calibrated score range: [{calibrated_scores.min():.3f}, {calibrated_scores.max():.3f}]")

    out_path = artifacts_dir() / "reliability_diagram.png"
    saved_path = plot_reliability_diagram(y_val.values, fused_scores, calibrated_scores, out_path)
    assert saved_path.exists(), "reliability diagram was not written"
    print(f"[Phase 1 smoke] reliability diagram saved to {saved_path}")

    print("[Phase 1 smoke] PASSED")


if __name__ == "__main__":
    main()
