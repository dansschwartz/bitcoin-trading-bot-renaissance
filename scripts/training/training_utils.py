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

from ml_model_loader import build_feature_sequence

logger = logging.getLogger(__name__)

SEQ_LEN = 30  # Must match build_feature_sequence default
INPUT_DIM = 83  # Feature dimension (46 real + padding to 83)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_sequences(
    pair_dfs: Dict[str, pd.DataFrame],
    seq_len: int = SEQ_LEN,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate feature sequences and labels from OHLCV DataFrames.

    Slides a window through each pair's data, calling build_feature_sequence()
    for each window. Label is the sign of the next-bar return.

    Args:
        pair_dfs: Dict of pair → DataFrame with columns [timestamp, open, high, low, close, volume]
        seq_len: Sequence length for features
        stride: Step size between windows (1 = every bar)

    Returns:
        X: (N, seq_len, 83) feature array
        y: (N,) label array where y ∈ {-1, +1}
    """
    all_X = []
    all_y = []

    for pair, df in pair_dfs.items():
        # Need seq_len + 50 (for indicator warmup in build_feature_sequence) + 1 (for label)
        min_rows = seq_len + 51
        if len(df) < min_rows:
            logger.info(f"  Skipping {pair}: only {len(df)} bars (need {min_rows})")
            continue

        n_samples = 0
        close_vals = df["close"].values.astype(float)

        # Slide window through data
        # build_feature_sequence takes a df and uses tail(seq_len + 50)
        # We need one bar after the window for the label
        for start_idx in range(0, len(df) - seq_len - 50, stride):
            end_idx = start_idx + seq_len + 50
            if end_idx >= len(df):
                break

            window_df = df.iloc[start_idx:end_idx].copy()
            features = build_feature_sequence(window_df, seq_len=seq_len)

            if features is None or features.shape != (seq_len, INPUT_DIM):
                continue

            # Label: direction of next bar's return
            # The last bar in the feature window corresponds to df.iloc[end_idx - 1]
            # The "next bar" is df.iloc[end_idx]
            if end_idx >= len(df):
                break

            current_close = close_vals[end_idx - 1]
            next_close = close_vals[end_idx]

            if current_close <= 0:
                continue

            ret = (next_close / current_close) - 1.0
            label = 1.0 if ret > 0 else -1.0

            all_X.append(features)
            all_y.append(label)
            n_samples += 1

        logger.info(f"  {pair}: generated {n_samples} sequences")

    if not all_X:
        return np.array([]), np.array([])

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.float32)
    logger.info(f"Total: {len(X)} sequences, label balance: "
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
    """MSE loss with directional accuracy penalty.

    loss = mse_loss + directional_weight * directional_penalty
    directional_penalty = mean(relu(-pred * target))

    The penalty term activates when prediction and target have opposite signs,
    encouraging the model to at least get the direction right.
    """

    def __init__(self, directional_weight: float = 0.3):
        super().__init__()
        self.directional_weight = directional_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target

        mse = F.mse_loss(pred, target)
        # Penalize wrong direction: when pred*target < 0, relu(-pred*target) > 0
        directional_penalty = torch.mean(F.relu(-pred * target))
        return mse + self.directional_weight * directional_penalty


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
    criterion = DirectionalLoss(directional_weight=0.3)
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
