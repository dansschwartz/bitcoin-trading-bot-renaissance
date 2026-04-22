"""
Train a LightGBM model for directional prediction on historical OHLCV data.

Based on production evidence that gradient boosting outperforms deep learning
on crypto price prediction with limited data (Tier 1 in ML taxonomy).

Key differences from deep learning models:
  - Features are FLATTENED (not sequential) — uses most recent bar's features
    plus multi-timeframe momentum (1, 3, 6, 12, 24, 72 bar returns)
  - Binary classification: direction over next 6 bars (30 minutes)
  - Conservative anti-overfit settings (num_leaves=31, max_depth=6, min_data=100)
  - Training takes MINUTES, not hours — no GPU needed
  - Outputs feature importance (top 20) and confidence-filtered accuracy

Usage:
    python -m scripts.training.train_lightgbm
    python -m scripts.training.train_lightgbm --epochs 2000 --lr 0.03
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import build_full_feature_matrix, INPUT_DIM
from scripts.training.training_utils import (
    load_derivatives_csvs,
    load_training_csvs,
    _align_derivatives,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL_HORIZON = 6       # Predict 30-min forward return (6 × 5-min bars)
WARMUP = 80             # Rows needed for indicator warmup + momentum lookback
MOMENTUM_BARS = [1, 3, 6, 12, 24, 72]  # Multi-timeframe momentum horizons

DEFAULTS = {
    "num_boost_round": 2000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "early_stopping_rounds": 50,
    "train_frac": 0.70,
    "val_frac": 0.15,
    "test_frac": 0.15,
}

SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "trained", "lightgbm_model.txt")


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_lightgbm_features(
    pair_dfs: Dict[str, pd.DataFrame],
    derivatives_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    fear_greed_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build flattened feature matrix + binary labels for LightGBM.

    For each bar, takes:
      1. The most recent bar's full feature vector (INPUT_DIM features)
      2. Multi-timeframe momentum returns (1, 3, 6, 12, 24, 72 bar lookbacks)

    Returns:
        X: (N, n_features) float32 array
        y: (N,) int32 array with values {0, 1} (0=down, 1=up)
        feature_names: List of feature names for interpretability
    """
    all_X: List[np.ndarray] = []
    all_y: List[int] = []
    feature_names = None

    for pair, df in pair_dfs.items():
        if len(df) < WARMUP + LABEL_HORIZON + max(MOMENTUM_BARS):
            logger.info(f"  Skipping {pair}: only {len(df)} bars")
            continue

        # Build cross_data for cross-asset features
        cross_data = {p: odf for p, odf in pair_dfs.items() if p != pair}

        # Align derivatives
        deriv_data = _align_derivatives(df, pair, derivatives_dfs, fear_greed_df)

        # Build full feature matrix (N, INPUT_DIM) — no per-window standardization
        feat_matrix = build_full_feature_matrix(
            df, cross_data=cross_data, pair_name=pair,
            derivatives_data=deriv_data,
        )
        if feat_matrix is None:
            logger.info(f"  Skipping {pair}: feature computation failed")
            continue

        close_vals = df["close"].values.astype(float)

        # Build feature names on first pair (from actual feature computation)
        if feature_names is None:
            try:
                from ml_model_loader import (
                    _compute_single_pair_features,
                    _build_cross_features,
                    _build_derivatives_features,
                )
                dummy_feats = _compute_single_pair_features(df.head(60))
                base_names = list(dummy_feats.keys())
            except Exception:
                base_names = []
            # Pad to INPUT_DIM with generic names for cross/derivatives/padding
            while len(base_names) < INPUT_DIM:
                base_names.append(f"feat_{len(base_names)}")
            base_names = base_names[:INPUT_DIM]
            momentum_names = [f"momentum_{h}bar" for h in MOMENTUM_BARS]
            feature_names = base_names + momentum_names

        n_samples = 0
        max_lookback = max(MOMENTUM_BARS)

        for idx in range(WARMUP + max_lookback, len(df) - LABEL_HORIZON):
            current_close = close_vals[idx]
            if current_close <= 0:
                continue

            # 1. Latest bar's full feature vector
            bar_features = feat_matrix[idx]  # (INPUT_DIM,)

            # 2. Multi-timeframe momentum returns
            momentum_feats = []
            for h in MOMENTUM_BARS:
                past_close = close_vals[idx - h]
                if past_close > 0:
                    momentum_feats.append(current_close / past_close - 1.0)
                else:
                    momentum_feats.append(0.0)

            # Combine into flat feature vector
            row = np.concatenate([bar_features, np.array(momentum_feats, dtype=np.float32)])

            # Binary label: 1 if price goes up over next LABEL_HORIZON bars
            future_close = close_vals[idx + LABEL_HORIZON]
            label = 1 if future_close > current_close else 0

            all_X.append(row)
            all_y.append(label)
            n_samples += 1

        logger.info(f"  {pair}: {n_samples:,} samples")

    if not all_X:
        return np.array([]), np.array([]), []

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    up_pct = y.mean() * 100
    logger.info(f"Total: {len(X):,} samples, {X.shape[1]} features, "
                f"label balance: {up_pct:.1f}% up / {100-up_pct:.1f}% down")
    return X, y, feature_names


# ── Training ──────────────────────────────────────────────────────────────────

def train_lightgbm(
    num_boost_round: int = DEFAULTS["num_boost_round"],
    learning_rate: float = DEFAULTS["learning_rate"],
    data_dir: str = "data/training",
    save_path: str = SAVE_PATH,
) -> dict:
    """Train LightGBM directional model.

    Returns:
        Dict with training results
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError(
            "LightGBM not installed. Run: pip install lightgbm"
        )

    logger.info("Loading training data...")
    pair_dfs = load_training_csvs(data_dir=data_dir)
    if not pair_dfs:
        raise RuntimeError(f"No training data found in {data_dir}")

    # Load derivatives (optional)
    derivatives_dfs, fear_greed_df = load_derivatives_csvs(
        data_dir=os.path.join(data_dir, "derivatives")
    )

    logger.info("Building flattened features...")
    X, y, feature_names = build_lightgbm_features(
        pair_dfs,
        derivatives_dfs=derivatives_dfs,
        fear_greed_df=fear_greed_df,
    )
    if len(X) == 0:
        raise RuntimeError("No samples generated")

    # Walk-forward time split: 70/15/15
    n = len(X)
    train_end = int(n * DEFAULTS["train_frac"])
    val_end = int(n * (DEFAULTS["train_frac"] + DEFAULTS["val_frac"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"Split: train={len(X_train):,}, val={len(X_val):,}, test={len(X_test):,}")

    # LightGBM parameters (conservative anti-overfit)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": DEFAULTS["num_leaves"],
        "learning_rate": learning_rate,
        "feature_fraction": DEFAULTS["feature_fraction"],
        "bagging_fraction": DEFAULTS["bagging_fraction"],
        "bagging_freq": DEFAULTS["bagging_freq"],
        "max_depth": DEFAULTS["max_depth"],
        "min_data_in_leaf": DEFAULTS["min_data_in_leaf"],
        "verbose": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

    logger.info(f"Training LightGBM: {num_boost_round} max rounds, "
                f"lr={learning_rate}, leaves={DEFAULTS['num_leaves']}, "
                f"depth={DEFAULTS['max_depth']}")

    callbacks = [
        lgb.early_stopping(stopping_rounds=DEFAULTS["early_stopping_rounds"]),
        lgb.log_evaluation(period=100),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # ── Evaluation ────────────────────────────────────────────────────────────

    results = {}

    for split_name, X_s, y_s in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        probs = model.predict(X_s)
        preds = (probs > 0.5).astype(int)
        acc = (preds == y_s).mean()

        # Confidence-filtered accuracy (only predictions where prob > 0.6 or < 0.4)
        confident_mask = (probs > 0.6) | (probs < 0.4)
        n_confident = confident_mask.sum()
        if n_confident > 0:
            confident_preds = (probs[confident_mask] > 0.5).astype(int)
            confident_acc = (confident_preds == y_s[confident_mask]).mean()
            confident_pct = n_confident / len(y_s) * 100
        else:
            confident_acc = 0.0
            confident_pct = 0.0

        results[f"{split_name}_acc"] = float(acc)
        results[f"{split_name}_confident_acc"] = float(confident_acc)
        results[f"{split_name}_confident_pct"] = float(confident_pct)

        logger.info(
            f"  {split_name:5s}: acc={acc:.4f}, "
            f"confident_acc={confident_acc:.4f} "
            f"({n_confident:,}/{len(y_s):,} = {confident_pct:.1f}% of predictions)"
        )

    # ── Feature Importance ────────────────────────────────────────────────────

    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(
        zip(feature_names, importance),
        key=lambda x: -x[1],
    )

    print(f"\n{'='*60}")
    print(f"{'TOP 20 FEATURE IMPORTANCE (gain)':^60}")
    print(f"{'='*60}")
    for i, (fname, imp) in enumerate(feat_imp[:20]):
        bar = "#" * min(int(imp / max(importance) * 40), 40)
        print(f"  {i+1:2d}. {fname:<25s} {imp:>10.1f}  {bar}")

    # ── Save Model ────────────────────────────────────────────────────────────

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save_model(save_path)
    size_kb = os.path.getsize(save_path) / 1024
    logger.info(f"Saved model to {save_path} ({size_kb:.0f} KB)")

    # ── Summary ───────────────────────────────────────────────────────────────

    metadata = {
        "model": "lightgbm",
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "best_iteration": model.best_iteration,
        "num_features": len(feature_names),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_names": feature_names,
        "momentum_bars": MOMENTUM_BARS,
        **results,
    }

    print(f"\n{'='*60}")
    print(f"{'LIGHTGBM TRAINING RESULTS':^60}")
    print(f"{'='*60}")
    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Model size: {size_kb:.0f} KB")
    print()
    print(f"  {'Split':<8} {'Accuracy':>10} {'Conf>60% Acc':>14} {'% Confident':>13}")
    print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*13}")
    for split_name in ["train", "val", "test"]:
        acc = results[f"{split_name}_acc"]
        cacc = results[f"{split_name}_confident_acc"]
        cpct = results[f"{split_name}_confident_pct"]
        print(f"  {split_name:<8} {acc:>10.4f} {cacc:>14.4f} {cpct:>12.1f}%")

    print(f"\n  Key insight: confident_acc on TEST is the real metric.")
    print(f"  Only trade when model confidence > 60%.")

    return metadata


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train LightGBM directional model")
    parser.add_argument("--rounds", type=int, default=DEFAULTS["num_boost_round"],
                        help="Max boosting rounds")
    parser.add_argument("--lr", type=float, default=DEFAULTS["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--data-dir", default="data/training",
                        help="Directory with training CSVs")
    args = parser.parse_args()

    results = train_lightgbm(
        num_boost_round=args.rounds,
        learning_rate=args.lr,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
