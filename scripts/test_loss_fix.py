#!/usr/bin/env python3
"""
Quick local test: compare old DirectionalLoss vs new sign-aware loss.

Trains quantum_transformer on 10K BTC samples for 10 epochs.
Goal: confirm accuracy rises above 50.5% with new loss before Colab run.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import (
    INPUT_DIM, TrainedQuantumTransformer, build_full_feature_matrix,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LEN = 30
LABEL_HORIZON = 6
LABEL_SCALE = 100
WARMUP = 50
N_SAMPLES = 10_000
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-3


# ── Loss functions ───────────────────────────────────────────────────────────

class OldDirectionalLoss(nn.Module):
    """The loss that fails: Huber + directional penalty (vanishes at pred≈0)."""
    def __init__(self, dw=0.3):
        super().__init__()
        self.dw = dw

    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target
        huber = F.smooth_l1_loss(pred, target)
        dir_penalty = torch.mean(F.relu(-pred * target))
        return huber + self.dw * dir_penalty


class NewSignAwareLoss(nn.Module):
    """Fixed loss: Huber + BCE on sign prediction.

    The BCE term provides strong log-scale gradients for direction,
    even when predictions are near zero. The pred * 10 scaling turns
    a tiny prediction (0.05) into a meaningful logit (0.5 → 62% prob).
    """
    def __init__(self, sign_weight=0.5):
        super().__init__()
        self.sign_weight = sign_weight

    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target

        # Regression component: Huber
        huber = F.smooth_l1_loss(pred, target)

        # Direction component: BCE on sign
        # pred * 10 → amplifies small preds into meaningful logits
        target_is_positive = (target > 0).float()
        sign_loss = F.binary_cross_entropy_with_logits(pred * 10.0, target_is_positive)

        return (1 - self.sign_weight) * huber + self.sign_weight * sign_loss


# ── Data generation ──────────────────────────────────────────────────────────

def generate_small_dataset(n_samples=N_SAMPLES, stride=3):
    """Generate a small dataset from BTC data for local testing."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD_5m_historical.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} bars, close range: ${df['close'].min():,.0f} → ${df['close'].max():,.0f}")

    # Use only last portion if we want fewer samples
    feat_matrix = build_full_feature_matrix(df)
    if feat_matrix is None:
        raise RuntimeError("Feature computation failed")
    print(f"  Feature matrix: {feat_matrix.shape}")

    close_vals = df['close'].values.astype(float)
    all_X, all_y = [], []

    for end_idx in range(WARMUP + SEQ_LEN, len(df) - LABEL_HORIZON + 1, stride):
        if len(all_X) >= n_samples:
            break

        start_idx = end_idx - SEQ_LEN
        future_idx = end_idx + LABEL_HORIZON - 1
        if future_idx >= len(df):
            break

        window = feat_matrix[start_idx:end_idx]
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-8
        window = (window - mean) / std

        current_close = close_vals[end_idx - 1]
        future_close = close_vals[future_idx]
        if current_close <= 0:
            continue

        ret = future_close / current_close - 1.0
        label = float(np.clip(ret * LABEL_SCALE, -1.0, 1.0))

        all_X.append(window)
        all_y.append(label)

    X = np.array(all_X[:n_samples], dtype=np.float32)
    y = np.array(all_y[:n_samples], dtype=np.float32)
    print(f"  Generated {len(X):,} samples")
    print(f"  Label stats: mean={y.mean():.4f}, std={y.std():.4f}, "
          f"up={100*(y>0).mean():.1f}%, down={100*(y<0).mean():.1f}%")
    return X, y


# ── Training loop ────────────────────────────────────────────────────────────

def train_and_eval(loss_name, criterion, X_train, y_train, X_val, y_val):
    """Train model for EPOCHS and return final accuracy."""
    model = TrainedQuantumTransformer(input_dim=INPUT_DIM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n{'='*60}")
    print(f"Training with {loss_name}")
    print(f"{'='*60}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss, n_b = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            output = model(X_b)
            pred = output[0] if isinstance(output, tuple) else output
            pred = pred.squeeze(-1)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1
        train_loss = total_loss / max(n_b, 1)
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(DEVICE)
                output = model(X_b)
                pred = output[0] if isinstance(output, tuple) else output
                all_preds.append(pred.squeeze(-1).cpu().numpy())
                all_targets.append(y_b.numpy())

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        dir_acc = float(np.mean(np.sign(preds) == np.sign(targets)))

        # Prediction stats
        pred_std = float(np.std(preds))
        pred_mean = float(np.mean(preds))

        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={train_loss:.4f}, "
              f"acc={dir_acc:.3f}, "
              f"pred_mean={pred_mean:.4f}, pred_std={pred_std:.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    return dir_acc, pred_std


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"INPUT_DIM: {INPUT_DIM}")

    # Generate data
    X, y = generate_small_dataset(n_samples=N_SAMPLES, stride=5)

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"\nTrain: {len(X_train):,}, Val: {len(X_val):,}")

    # Test 1: Old loss (should stay at ~49.7%)
    old_acc, old_std = train_and_eval(
        "OLD: Huber + relu(-pred*target)",
        OldDirectionalLoss(dw=0.3),
        X_train, y_train, X_val, y_val,
    )

    # Test 2: New loss (should climb above 50.5%)
    new_acc, new_std = train_and_eval(
        "NEW: Huber + BCE(sign)",
        NewSignAwareLoss(sign_weight=0.5),
        X_train, y_train, X_val, y_val,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Old loss: acc={old_acc:.3f}, pred_std={old_std:.4f}")
    print(f"New loss: acc={new_acc:.3f}, pred_std={new_std:.4f}")
    print(f"Improvement: {(new_acc - old_acc)*100:+.1f} percentage points")

    if new_acc > 0.505:
        print("\n✓ New loss passed threshold (>50.5%). Safe to run on Colab.")
    else:
        print("\n✗ New loss did NOT pass threshold. Need further investigation.")


if __name__ == "__main__":
    main()
