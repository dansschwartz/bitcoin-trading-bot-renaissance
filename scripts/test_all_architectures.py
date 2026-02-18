#!/usr/bin/env python3
"""
Comprehensive architecture test: train all 6 models locally on 10K samples.

Tests each architecture with two loss approaches:
  1. Classification: pure BCE on direction (sigmoid output)
  2. Hybrid: BCE direction + Huber magnitude

Goal: find which architectures can learn >50.5% directional accuracy
and which ones collapse to constant predictions.

Key insight: Large overparameterized models collapse with regression loss
because predicting 0 (the label mean) is a trivial local minimum.
Classification (BCE) loss provides strong gradients at the decision
boundary, preventing this collapse.
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
    INPUT_DIM,
    TrainedQuantumTransformer,
    TrainedBidirectionalLSTM,
    TrainedDilatedCNN,
    TrainedCNN,
    TrainedGRU,
    build_full_feature_matrix,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LEN = 30
LABEL_HORIZON = 6
LABEL_SCALE = 100
WARMUP = 50
N_SAMPLES = 10_000
EPOCHS = 15
BATCH_SIZE = 64
LR = 3e-4  # Lower LR for large models (was 1e-3)


# ── Loss functions ───────────────────────────────────────────────────────────

class ClassificationLoss(nn.Module):
    """Pure BCE on sign direction — forces model to predict up/down."""
    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target
        target_is_positive = (target > 0).float()
        # pred is raw logit; scale by 5 to make small preds meaningful
        return F.binary_cross_entropy_with_logits(pred * 5.0, target_is_positive)


class HybridLoss(nn.Module):
    """BCE sign + Huber magnitude — direction matters most, magnitude is secondary."""
    def __init__(self, sign_weight=0.7):
        super().__init__()
        self.sign_weight = sign_weight

    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target
        target_is_positive = (target > 0).float()
        bce = F.binary_cross_entropy_with_logits(pred * 5.0, target_is_positive)
        huber = F.smooth_l1_loss(pred, target)
        return self.sign_weight * bce + (1 - self.sign_weight) * huber


# ── Data generation ──────────────────────────────────────────────────────────

def generate_small_dataset(n_samples=N_SAMPLES, stride=3):
    """Generate a small dataset from BTC data for local testing."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD_5m_historical.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} bars, close range: "
          f"${df['close'].min():,.0f} → ${df['close'].max():,.0f}")

    feat_matrix = build_full_feature_matrix(df)
    if feat_matrix is None:
        raise RuntimeError("Feature computation failed")
    print(f"  Feature matrix: {feat_matrix.shape}")

    # Global normalization stats (compute on full dataset, then apply per window)
    global_mean = feat_matrix.mean(axis=0, keepdims=True)
    global_std = feat_matrix.std(axis=0, keepdims=True) + 1e-8

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

        # Per-window standardization (same as Colab)
        w_mean = window.mean(axis=0, keepdims=True)
        w_std = window.std(axis=0, keepdims=True) + 1e-8
        window = (window - w_mean) / w_std

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

def train_and_eval(model_name, model, criterion, X_train, y_train, X_val, y_val,
                   lr=LR, epochs=EPOCHS, warmup_epochs=3):
    """Train model and return final accuracy + prediction std."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Linear warmup + cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'='*70}")

    best_acc = 0.0
    best_std = 0.0

    for epoch in range(epochs):
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

        # For directional accuracy: positive pred = up, negative pred = down
        dir_acc = float(np.mean(np.sign(preds) == np.sign(targets)))
        pred_std = float(np.std(preds))
        pred_mean = float(np.mean(preds))

        if dir_acc > best_acc:
            best_acc = dir_acc
            best_std = pred_std

        # Print every epoch for first 3, then every 3rd epoch, plus last
        if epoch < 3 or (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            collapsed = " *** COLLAPSED ***" if pred_std < 0.001 else ""
            print(f"  Epoch {epoch+1:2d}/{epochs}: loss={train_loss:.4f}, "
                  f"acc={dir_acc:.3f}, "
                  f"pred_mean={pred_mean:+.4f}, pred_std={pred_std:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{collapsed}")

    return best_acc, best_std


# ── Model factory ────────────────────────────────────────────────────────────

def get_models():
    """Return list of (name, model) tuples to test."""
    return [
        ("QuantumTransformer (4.6M)", TrainedQuantumTransformer(input_dim=INPUT_DIM)),
        ("BiLSTM (2.5M)", TrainedBidirectionalLSTM(input_dim=INPUT_DIM)),
        ("DilatedCNN", TrainedDilatedCNN(input_dim=INPUT_DIM)),
        ("SimpleCNN", TrainedCNN(input_dim=INPUT_DIM)),
        ("BiGRU", TrainedGRU(input_dim=INPUT_DIM)),
    ]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"INPUT_DIM: {INPUT_DIM}")
    print(f"LR: {LR}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Warmup: 3 epochs, then cosine decay")

    # Generate data
    X, y = generate_small_dataset(n_samples=N_SAMPLES, stride=5)

    # 80/20 split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"\nTrain: {len(X_train):,}, Val: {len(X_val):,}")

    results = {}

    # Test each architecture with both loss types
    for loss_name, criterion in [
        ("BCE-classification", ClassificationLoss()),
        ("Hybrid(0.7*BCE+0.3*Huber)", HybridLoss(sign_weight=0.7)),
    ]:
        print(f"\n\n{'#'*70}")
        print(f"# LOSS: {loss_name}")
        print(f"{'#'*70}")

        for model_name, model in get_models():
            full_name = f"{model_name} + {loss_name}"
            t0 = time.time()
            acc, pred_std = train_and_eval(
                full_name, model, criterion,
                X_train, y_train, X_val, y_val,
            )
            elapsed = time.time() - t0
            results[full_name] = {
                'acc': acc,
                'pred_std': pred_std,
                'time': elapsed,
                'collapsed': pred_std < 0.001,
            }

    # Summary
    print(f"\n\n{'='*70}")
    print(f"{'RESULTS SUMMARY':^70}")
    print(f"{'='*70}")
    print(f"{'Model + Loss':<50} {'Acc':>6} {'PredStd':>8} {'Status':>12} {'Time':>6}")
    print(f"{'-'*50} {'-'*6} {'-'*8} {'-'*12} {'-'*6}")

    passing = []
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        status = "COLLAPSED" if r['collapsed'] else ("PASS" if r['acc'] > 0.505 else "LOW")
        marker = ">>>" if r['acc'] > 0.505 and not r['collapsed'] else "   "
        print(f"{marker} {name:<47} {r['acc']:.3f} {r['pred_std']:.4f} {status:>12} {r['time']:5.0f}s")
        if r['acc'] > 0.505 and not r['collapsed']:
            passing.append(name)

    print(f"\n{'='*70}")
    if passing:
        print(f"PASSING (>50.5% accuracy, non-collapsed): {len(passing)}")
        for name in passing:
            print(f"  + {name}")
        print(f"\nSafe to proceed with Colab retraining using these architectures.")
    else:
        print("NO ARCHITECTURES PASSED. Need further investigation.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
