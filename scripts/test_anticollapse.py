#!/usr/bin/env python3
"""
Test the anti-collapse loss (v6) + optimizer (v7) on a larger local dataset.

v6 loss: BCE(pred*20) + 10.0*separation_margin(0.10) + 5.0*magnitude_floor

v7 optimizer (QT only):
  - weight_decay=0 (wd=1e-5 per step × 10K+ batches/epoch = 10% shrinkage/epoch
    on full data, which degenerates attention → softmax uniform → pred=0)
  - Differential LR: attention params at 0.1x base LR
  - Other models: weight_decay=1e-4 (unchanged)

Previous failures on full Colab data (680K samples):
  v3-v5: various loss fixes, all failed because root cause was optimizer
  v6: stronger loss penalty (1.05 vs 0.27), but still collapsed at epoch 5
      because wd=1e-5 compounds to 10% shrinkage/epoch on 10K+ batches
"""

import os
import sys
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ml_model_loader import (
    INPUT_DIM, TrainedQuantumTransformer, TrainedBidirectionalLSTM,
    TrainedDilatedCNN, TrainedCNN, TrainedGRU, build_full_feature_matrix,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEQ_LEN = 30
LABEL_HORIZON = 6
LABEL_SCALE = 100
WARMUP = 50
N_SAMPLES = 50_000
EPOCHS = 20
BATCH_SIZE = 64
LR = 3e-4
WARMUP_EPOCHS = 3


class DirectionalLossV6(nn.Module):
    """v6: BCE + separation margin + magnitude floor (strengthened)."""
    def __init__(self, logit_scale=20.0, margin=0.10):
        super().__init__()
        self.logit_scale = logit_scale
        self.margin = margin

    def forward(self, pred, target):
        pred = pred.squeeze(-1) if pred.dim() > 1 else pred
        target = target.squeeze(-1) if target.dim() > 1 else target

        # 1. BCE on sign with strong logit scaling
        target_pos = (target > 0).float()
        bce = F.binary_cross_entropy_with_logits(
            pred * self.logit_scale, target_pos)

        # 2. Separation margin (doubled: margin=0.10, weight=10.0)
        pos_mask = target > 0
        neg_mask = target <= 0
        if pos_mask.any() and neg_mask.any():
            separation = pred[pos_mask].mean() - pred[neg_mask].mean()
            sep_loss = F.relu(self.margin - separation)
        else:
            sep_loss = torch.tensor(0.0, device=pred.device)

        # 3. Magnitude floor (weight=5.0, up from 2.0)
        mag_loss = F.relu(0.01 - pred.abs()).mean()

        return bce + 10.0 * sep_loss + 5.0 * mag_loss


def generate_dataset(n_samples=N_SAMPLES, stride=3):
    """Generate dataset from BTC data."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD_5m_historical.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(PROJECT_ROOT, "data", "training", "BTC-USD.csv")
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} bars")

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


def train_and_eval(model_name, model, criterion, X_train, y_train, X_val, y_val):
    """Train model and track collapse."""
    model = model.to(DEVICE)

    # v7: QT gets wd=0 + differential LR (attention at 0.1x)
    if "Quantum" in model_name:
        attn_params = []
        other_params = []
        for pname, p in model.named_parameters():
            if any(k in pname for k in ['attention', 'pos_encoding', 'skip_enhancement']):
                attn_params.append(p)
            else:
                other_params.append(p)
        optimizer = torch.optim.AdamW([
            {'params': attn_params, 'lr': LR * 0.1, 'weight_decay': 0},
            {'params': other_params, 'lr': LR, 'weight_decay': 0},
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(EPOCHS - WARMUP_EPOCHS, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  {model_name} ({n_params:,} params)")
    print(f"{'='*70}")

    best_acc = 0.0
    collapsed_at = None

    for epoch in range(EPOCHS):
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
        pred_std = float(np.std(preds))
        pred_mean = float(np.mean(preds))

        # Also track separation
        pos_preds = preds[targets > 0]
        neg_preds = preds[targets <= 0]
        sep = pos_preds.mean() - neg_preds.mean() if len(pos_preds) > 0 and len(neg_preds) > 0 else 0.0

        if dir_acc > best_acc:
            best_acc = dir_acc

        if pred_std < 0.001 and collapsed_at is None:
            collapsed_at = epoch + 1

        collapsed = " *** COLLAPSED ***" if pred_std < 0.001 else ""
        print(f"  Epoch {epoch+1:2d}/{EPOCHS}: loss={train_loss:.4f}, "
              f"acc={dir_acc:.3f}, "
              f"pred_mean={pred_mean:+.4f}, pred_std={pred_std:.4f}, "
              f"sep={sep:+.4f}, "
              f"lr={optimizer.param_groups[0]['lr']:.2e}{collapsed}")

    return best_acc, pred_std, collapsed_at


def main():
    print(f"Device: {DEVICE}")
    print(f"N_SAMPLES: {N_SAMPLES:,}")
    print(f"LR: {LR}, Warmup: {WARMUP_EPOCHS} epochs")
    print(f"Loss: v6 (BCE*20 + 10.0*sep_margin(0.10) + 5.0*mag_floor)")

    X, y = generate_dataset(n_samples=N_SAMPLES, stride=2)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"\nTrain: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"Train batches: {len(X_train) // BATCH_SIZE}")

    criterion = DirectionalLossV6(logit_scale=20.0, margin=0.10)

    results = {}
    for name, model in [
        ("QuantumTransformer", TrainedQuantumTransformer(input_dim=INPUT_DIM)),
        ("BiLSTM", TrainedBidirectionalLSTM(input_dim=INPUT_DIM)),
        ("DilatedCNN", TrainedDilatedCNN(input_dim=INPUT_DIM)),
        ("SimpleCNN", TrainedCNN(input_dim=INPUT_DIM)),
        ("BiGRU", TrainedGRU(input_dim=INPUT_DIM)),
    ]:
        t0 = time.time()
        acc, pred_std, collapsed_at = train_and_eval(
            name, model, criterion, X_train, y_train, X_val, y_val)
        elapsed = time.time() - t0
        results[name] = {
            'acc': acc, 'pred_std': pred_std,
            'collapsed_at': collapsed_at, 'time': elapsed,
        }

    print(f"\n\n{'='*70}")
    print(f"{'RESULTS (v6: BCE*20 + 10*sep(0.10) + 5*mag)':^70}")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'Best Acc':>8} {'PredStd':>8} {'Collapsed':>12} {'Time':>6}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*12} {'-'*6}")

    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        col = f"epoch {r['collapsed_at']}" if r['collapsed_at'] else "NO"
        status = "PASS" if r['acc'] > 0.505 and not r['collapsed_at'] else "FAIL"
        print(f"  {name:<23} {r['acc']:.3f}    {r['pred_std']:.4f}    {col:<10}   {r['time']:5.0f}s  [{status}]")


if __name__ == "__main__":
    main()
