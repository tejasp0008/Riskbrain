"""Synthetic transaction and dispute data generator, grounded in real
Visa/Mastercard reason codes and per-code evidence rules (see schema.py).

This is architectural scaffolding, not a fraud model. The generator
encodes a hand-picked, directionally-plausible risk function (amount,
velocity, device novelty, geo mismatch, off-hours, card-not-present) to
produce labeled data so the tabular pipeline (Phase 1) and later the RAG
corpus (Phase 2) have something real to train and query against. Any
accuracy numbers computed on this data are directional only and prove
the system runs end to end, not that it detects real fraud.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "amount",
    "velocity_1h",
    "velocity_24h",
    "device_is_new",
    "geo_mismatch",
    "is_card_present",
    "hour_of_day",
    "mcc_risk_bucket",
]


def artifacts_dir() -> Path:
    """Resolves the artifacts directory, overridable via env var so the
    same code writes to a local folder or a mounted Drive path in Colab."""
    return Path(os.environ.get("RISKBRAIN_ARTIFACTS_DIR", "./artifacts"))


def generate_synthetic_transactions(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generates n labeled synthetic transactions with structured features.

    The fraud probability is a logistic function of a hand-picked risk
    score built from the features below, then labels are sampled from
    that probability. This creates a learnable but noisy signal, which
    is what a real tabular fraud model deals with.
    """
    rng = np.random.default_rng(seed)

    amount = rng.lognormal(mean=4.0, sigma=1.2, size=n)  # skewed, mostly small
    velocity_1h = rng.poisson(lam=1.2, size=n)
    velocity_24h = velocity_1h + rng.poisson(lam=3.0, size=n)
    device_is_new = rng.binomial(1, 0.15, size=n)
    geo_mismatch = rng.binomial(1, 0.08, size=n)
    is_card_present = rng.binomial(1, 0.55, size=n)
    hour_of_day = rng.integers(0, 24, size=n)
    # 0 = low-risk MCC (groceries, utilities), 1 = high-risk MCC (electronics, gift cards, crypto)
    mcc_risk_bucket = rng.binomial(1, 0.25, size=n)

    off_hours = ((hour_of_day >= 1) & (hour_of_day <= 5)).astype(int)
    log_amount = np.log1p(amount)

    risk_score = (
        0.55 * (log_amount - log_amount.mean()) / log_amount.std()
        + 1.8 * device_is_new
        + 2.6 * geo_mismatch
        + 0.9 * (1 - is_card_present)
        + 1.1 * off_hours
        + 1.3 * mcc_risk_bucket
        + 0.5 * (velocity_1h > 3).astype(int)
        + 0.7 * (velocity_24h > 8).astype(int)
        - 4.9  # base rate anchor, keeps fraud prevalence in a realistic single-digit range
    )
    fraud_prob = 1 / (1 + np.exp(-risk_score))
    label = rng.binomial(1, fraud_prob)

    df = pd.DataFrame(
        {
            "amount": amount,
            "velocity_1h": velocity_1h,
            "velocity_24h": velocity_24h,
            "device_is_new": device_is_new,
            "geo_mismatch": geo_mismatch,
            "is_card_present": is_card_present,
            "hour_of_day": hour_of_day,
            "mcc_risk_bucket": mcc_risk_bucket,
            "label": label,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    df = generate_synthetic_transactions(n=args.n, seed=args.seed)
    out_dir = artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "synthetic_transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows ({df['label'].mean():.1%} positive) to {out_path}")


if __name__ == "__main__":
    main()
