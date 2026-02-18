"""
Shared training utilities for ML model training scripts.

Provides dataset generation, walk-forward splitting, loss functions,
and training/validation loops used by all three model training scripts.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path so we can import ml_model_loader
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import build_feature_sequence, build_full_feature_matrix, INPUT_DIM

logger = logging.getLogger(__name__)

SEQ_LEN = 30  # Must match build_feature_sequence default
# INPUT_DIM is imported from ml_model_loader (98)
LABEL_HORIZON = 6   # Predict 30-min forward return (6 × 5-min bars)
LABEL_SCALE = 100   # Scaling: 0.5% return → label 0.5, clipped to [-1, 1]


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_sequences(
    pair_dfs: Dict[str, pd.DataFrame],
    seq_len: int = SEQ_LEN,
    stride: int = 1,
    cross_asset: bool = True,
    derivatives_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    fear_greed_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate feature sequences and labels from OHLCV DataFrames.

    Uses vectorized feature computation: builds the full feature matrix once
    per pair, then slices windows — ~200x faster than per-window computation.

    Args:
        pair_dfs: Dict of pair → DataFrame with columns [timestamp, open, high, low, close, volume]
        seq_len: Sequence length for features
        stride: Step size between windows (1 = every bar)
        cross_asset: If True, pass cross-pair data for 15 cross-asset features.
                     If False, cross-asset feature slots are zero-padded (backward compat).
        derivatives_dfs: Optional dict of pair → DataFrame with derivatives columns
            (funding_rate, open_interest, long_short_ratio, taker_buy_vol, taker_sell_vol).
            Timestamps must overlap with pair_dfs. Merged by nearest timestamp.
        fear_greed_df: Optional DataFrame with columns [timestamp, fear_greed].
            Daily data forward-filled to 5-min bars.

    Returns:
        X: (N, seq_len, INPUT_DIM) feature array
        y: (N,) label array where y ∈ {-1, +1}
    """
    all_X: List[np.ndarray] = []
    all_y: List[float] = []
    warmup = 50  # rows needed for indicator warmup

    for pair, df in pair_dfs.items():
        min_rows = seq_len + warmup + 1
        if len(df) < min_rows:
            logger.info(f"  Skipping {pair}: only {len(df)} bars (need {min_rows})")
            continue

        # Build cross_data for this pair (full split data from other pairs)
        cross_data = None
        if cross_asset and len(pair_dfs) > 1:
            cross_data = {p: odf for p, odf in pair_dfs.items() if p != pair}

        # Build derivatives_data for this pair (aligned to price timestamps)
        derivatives_data = _align_derivatives(df, pair, derivatives_dfs, fear_greed_df)

        # Compute ALL features for the ENTIRE pair at once (vectorized)
        feat_matrix = build_full_feature_matrix(
            df, cross_data=cross_data, pair_name=pair,
            derivatives_data=derivatives_data,
        )
        if feat_matrix is None:
            logger.info(f"  Skipping {pair}: feature computation failed")
            continue

        close_vals = df["close"].values.astype(float)
        n_samples = 0

        # Slide windows through the pre-computed feature matrix
        for end_idx in range(warmup + seq_len, len(df) - LABEL_HORIZON + 1, stride):
            start_idx = end_idx - seq_len

            # Future price: LABEL_HORIZON bars after the window ends
            future_idx = end_idx + LABEL_HORIZON - 1
            if future_idx >= len(df):
                break

            # Slice window from pre-computed features
            window = feat_matrix[start_idx:end_idx]  # (seq_len, INPUT_DIM)

            # Per-window standardization (same as build_feature_sequence)
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-8
            window = (window - mean) / std

            # Soft label: 6-bar forward return, scaled and clipped to [-1, 1]
            current_close = close_vals[end_idx - 1]
            future_close = close_vals[future_idx]

            if current_close <= 0:
                continue

            ret = (future_close / current_close) - 1.0
            label = float(np.clip(ret * LABEL_SCALE, -1.0, 1.0))

            all_X.append(window)
            all_y.append(label)
            n_samples += 1

        logger.info(f"  {pair}: generated {n_samples} sequences")

    if not all_X:
        return np.array([]), np.array([])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    logger.info(f"Total: {len(X)} sequences (dim={X.shape[-1]}), label balance: "
                f"{(y > 0).sum()} up / {(y < 0).sum()} down "
                f"({(y > 0).mean()*100:.1f}% / {(y < 0).mean()*100:.1f}%)")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_split(
    pair_dfs: Dict[str, pd.DataFrame],
    train_frac: float = 0.7,
    val_frac: float = 0.13,
    test_frac: float = 0.17,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Split each pair's data into train/val/test by time (walk-forward).

    Args:
        pair_dfs: Dict of pair → DataFrame sorted by timestamp
        train_frac: Fraction of data for training (default 0.7 ≈ 21/30 days)
        val_frac: Fraction for validation (default 0.13 ≈ 4/30 days)
        test_frac: Fraction for test (default 0.17 ≈ 5/30 days)

    Returns:
        (train_dfs, val_dfs, test_dfs) — each is a dict of pair → DataFrame
    """
    train_dfs = {}
    val_dfs = {}
    test_dfs = {}

    for pair, df in pair_dfs.items():
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_dfs[pair] = df.iloc[:train_end].copy().reset_index(drop=True)
        val_dfs[pair] = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_dfs[pair] = df.iloc[val_end:].copy().reset_index(drop=True)

        logger.info(f"  {pair}: train={len(train_dfs[pair])}, "
                     f"val={len(val_dfs[pair])}, test={len(test_dfs[pair])}")

    return train_dfs, val_dfs, test_dfs


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class DirectionalLoss(nn.Module):
    """v6: BCE + separation margin + magnitude floor (strengthened for full-data).

    Three components prevent the "predict zero" collapse:

    1. BCE on sign with 20x logit scaling: converts small regression outputs
       into meaningful logits. A prediction of 0.05 → logit 1.0 → 73% prob.

    2. Separation margin: measures mean(pred|target>0) - mean(pred|target<0).
       If separation < margin, adds strong loss. At pred=0 for all samples,
       separation=0 → loss += sep_weight * margin per batch.

    3. Magnitude floor: pushes |pred| above 0.01 minimum. Directly opposes
       weight decay shrinking all outputs toward zero.

    v6 vs v5: Doubled margin (0.05→0.10), doubled sep_weight (5→10),
    increased mag_weight (2→5). At collapse, v6 penalty = 1.05 vs v5's 0.27.
    Combined with per-model weight_decay (1e-5 for QT), prevents transformer
    attention collapse on full 680K+ sample datasets.
    """

    def __init__(self, logit_scale: float = 20.0, margin: float = 0.10):
        super().__init__()
        self.logit_scale = logit_scale
        self.margin = margin

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target

        # 1. Direction: BCE on sign with strong logit scaling
        target_is_positive = (target > 0).float()
        bce = F.binary_cross_entropy_with_logits(
            pred * self.logit_scale, target_is_positive)

        # 2. Separation margin: force pred|up > pred|down by at least margin
        pos_mask = target > 0
        neg_mask = target <= 0
        if pos_mask.any() and neg_mask.any():
            separation = pred[pos_mask].mean() - pred[neg_mask].mean()
            sep_loss = F.relu(self.margin - separation)
        else:
            sep_loss = torch.tensor(0.0, device=pred.device)

        # 3. Magnitude floor: push |pred| above minimum threshold
        mag_loss = F.relu(0.01 - pred.abs()).mean()

        return bce + 10.0 * sep_loss + 5.0 * mag_loss


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def directional_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute directional accuracy: fraction where sign(pred) == sign(target).

    Args:
        predictions: Model predictions array
        targets: True labels array (values in {-1, +1})

    Returns:
        Accuracy as float in [0, 1]
    """
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
    # Treat zero predictions as wrong
    return float(np.mean(pred_sign == target_sign))


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & VALIDATION LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> float:
    """Run one training epoch.

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)

        # Handle models that return (prediction, auxiliary) tuples
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output

        pred = pred.squeeze(-1)
        loss = criterion(pred, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one validation epoch.

    Returns:
        (val_loss, directional_accuracy)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            output = model(X_batch)
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output

            pred = pred.squeeze(-1)
            loss = criterion(pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    if not all_preds:
        return avg_loss, 0.5  # No data — return chance-level accuracy

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    dir_acc = directional_accuracy(preds, targets)

    return avg_loss, dir_acc


def evaluate_on_dataset(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[float, float]:
    """Evaluate a model on a dataset, returning loss and directional accuracy.

    Args:
        model: Trained model
        X: Feature array (N, seq_len, 83)
        y: Label array (N,)
        device: torch device
        batch_size: Batch size for evaluation

    Returns:
        (loss, directional_accuracy)
    """
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)
    return validate_epoch(model, loader, criterion, device)


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_model_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """Save model state dict with optional metadata.

    Saves in the same format as the existing trained models (just state_dict),
    so load_trained_models() can load them directly.

    Args:
        model: PyTorch model
        path: File path to save to
        metadata: Optional metadata (saved alongside but state_dict is primary)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save just the state_dict for compatibility with load_trained_models()
    torch.save(model.state_dict(), path)

    # Save metadata separately if provided
    if metadata:
        meta_path = path.replace(".pth", "_meta.json")
        import json
        # Convert non-serializable types
        serializable = {}
        for k, v in metadata.items():
            if isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            elif isinstance(v, datetime):
                serializable[k] = v.isoformat()
            else:
                serializable[k] = v
        with open(meta_path, "w") as f:
            json.dump(serializable, f, indent=2)

    logger.info(f"Saved model to {path} ({os.path.getsize(path) / 1024:.1f} KB)")


def load_model_checkpoint(path: str) -> Tuple[dict, Optional[Dict]]:
    """Load a model checkpoint and optional metadata.

    Args:
        path: Path to .pth file

    Returns:
        (state_dict, metadata_dict_or_None)
    """
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    metadata = None
    meta_path = path.replace(".pth", "_meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path) as f:
            metadata = json.load(f)

    return state_dict, metadata


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_training_csvs(
    pairs: List[str] = None,
    data_dir: str = "data/training",
) -> Dict[str, pd.DataFrame]:
    """Load training CSVs from data/training/ directory.

    Args:
        pairs: List of pairs to load (default: all available CSVs)
        data_dir: Directory containing CSV files

    Returns:
        Dict of pair → DataFrame
    """
    if not os.path.exists(data_dir):
        logger.error(f"Training data directory not found: {data_dir}")
        return {}

    pair_dfs = {}
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    for csv_file in sorted(csv_files):
        pair = csv_file.replace(".csv", "")
        if pairs and pair not in pairs:
            continue

        df = pd.read_csv(os.path.join(data_dir, csv_file))
        if len(df) > 0:
            pair_dfs[pair] = df
            logger.info(f"  Loaded {pair}: {len(df)} bars")

    return pair_dfs


def load_derivatives_csvs(
    pairs: List[str] = None,
    data_dir: str = "data/training/derivatives",
) -> Tuple[Dict[str, pd.DataFrame], Optional[pd.DataFrame]]:
    """Load derivatives CSVs and Fear & Greed history for training.

    Args:
        pairs: List of pairs to load (default: all available)
        data_dir: Directory containing derivatives CSVs

    Returns:
        (derivatives_dfs, fear_greed_df) where:
        - derivatives_dfs: Dict of pair → DataFrame with derivatives columns
        - fear_greed_df: DataFrame with [timestamp, fear_greed] or None
    """
    derivatives_dfs: Dict[str, pd.DataFrame] = {}
    fear_greed_df = None

    if not os.path.exists(data_dir):
        logger.info(f"No derivatives data directory: {data_dir}")
        return derivatives_dfs, fear_greed_df

    # Load per-pair derivatives
    for csv_file in sorted(os.listdir(data_dir)):
        if csv_file.endswith("_derivatives.csv"):
            pair = csv_file.replace("_derivatives.csv", "")
            if pairs and pair not in pairs:
                continue
            df = pd.read_csv(os.path.join(data_dir, csv_file))
            if len(df) > 0:
                derivatives_dfs[pair] = df
                logger.info(f"  Derivatives {pair}: {len(df)} rows")

    # Load Fear & Greed history
    fng_path = os.path.join(data_dir, "fear_greed_history.csv")
    if os.path.exists(fng_path):
        fear_greed_df = pd.read_csv(fng_path)
        if len(fear_greed_df) > 0:
            logger.info(f"  Fear & Greed: {len(fear_greed_df)} daily values")
        else:
            fear_greed_df = None

    return derivatives_dfs, fear_greed_df


def _align_derivatives(
    price_df: pd.DataFrame,
    pair: str,
    derivatives_dfs: Optional[Dict[str, pd.DataFrame]],
    fear_greed_df: Optional[pd.DataFrame],
) -> Optional[Dict[str, 'pd.Series']]:
    """Align derivatives data to price DataFrame timestamps.

    Returns dict suitable for _build_derivatives_features(), or None if no data.
    """
    if derivatives_dfs is None and fear_greed_df is None:
        return None

    n_rows = len(price_df)
    result: Dict[str, pd.Series] = {}

    # Align per-pair derivatives (funding_rate, OI, LS, taker volumes)
    if derivatives_dfs and pair in derivatives_dfs:
        deriv_df = derivatives_dfs[pair].copy()

        if 'timestamp' in deriv_df.columns and 'timestamp' in price_df.columns:
            # Merge_asof: for each price bar, find the nearest prior derivatives row
            price_ts = price_df[['timestamp']].copy()
            price_ts['timestamp'] = price_ts['timestamp'].astype(int)
            deriv_df['timestamp'] = deriv_df['timestamp'].astype(int)

            # Ensure sorted
            price_ts = price_ts.sort_values('timestamp')
            deriv_df = deriv_df.sort_values('timestamp')

            merged = pd.merge_asof(
                price_ts, deriv_df,
                on='timestamp', direction='backward',
            )

            for col in ['funding_rate', 'open_interest', 'long_short_ratio',
                        'taker_buy_vol', 'taker_sell_vol']:
                if col in merged.columns:
                    result[col] = merged[col].reset_index(drop=True)

    # Align Fear & Greed (daily → forward-fill to 5-min bars)
    if fear_greed_df is not None and 'timestamp' in price_df.columns:
        fng = fear_greed_df.copy()
        fng['timestamp'] = fng['timestamp'].astype(int)
        fng = fng.sort_values('timestamp')

        price_ts = price_df[['timestamp']].copy()
        price_ts['timestamp'] = price_ts['timestamp'].astype(int)
        price_ts = price_ts.sort_values('timestamp')

        merged_fng = pd.merge_asof(
            price_ts, fng,
            on='timestamp', direction='backward',
        )
        if 'fear_greed' in merged_fng.columns:
            result['fear_greed'] = merged_fng['fear_greed'].reset_index(drop=True)

    return result if result else None


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader]:
    """Create PyTorch DataLoaders from numpy arrays.

    Args:
        X_train: Training features (N, seq_len, 83)
        y_train: Training labels (N,)
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size

    Returns:
        (train_loader, val_loader)
    """
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
