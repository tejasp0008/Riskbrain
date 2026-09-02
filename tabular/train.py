"""Trains the XGBoost tabular fraud scorer on structured transaction features.

Owns the numeric fraud score end to end: structured features in,
calibrated-ready probability out. The language model never touches this
path (see CLAUDE.md). Trained here on synthetic data, so the AUC printed
below is a directional sanity check, not a claim about real-world fraud
detection performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from data.generate import FEATURE_COLUMNS, artifacts_dir, generate_synthetic_transactions


def load_or_generate(csv_path: Path | None, n: int, seed: int) -> pd.DataFrame:
    if csv_path is not None and csv_path.exists():
        return pd.read_csv(csv_path)
    return generate_synthetic_transactions(n=n, seed=seed)


def train_xgboost(df: pd.DataFrame, seed: int = 42):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=seed,
    )
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_scores)

    return model, X_val, y_val, val_scores, auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    df = load_or_generate(csv_path, args.n, args.seed)

    model, X_val, y_val, val_scores, auc = train_xgboost(df, seed=args.seed)
    print(f"[directional, synthetic data] validation AUC: {auc:.3f}")

    out_dir = artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "tabular_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}")

    val_out = X_val.copy()
    val_out["label"] = y_val.values
    val_out["tabular_score"] = val_scores
    val_path = out_dir / "tabular_val_scores.csv"
    val_out.to_csv(val_path, index=False)
    print(f"Saved validation scores to {val_path}")


if __name__ == "__main__":
    main()
