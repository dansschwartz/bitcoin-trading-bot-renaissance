#!/usr/bin/env python3
"""
Train the VAE Anomaly Detector on historical market data.

Uses the same 83-feature pipeline as the ML models (build_feature_sequence).
Trains on OHLCV data from:
  1. Local five_minute_bars table (primary)
  2. Coinbase REST API candles (supplement if needed)

Usage:
    python train_vae.py                  # Train on local DB data
    python train_vae.py --fetch 7        # Fetch 7 days from Coinbase first
    python train_vae.py --epochs 200     # Custom epoch count
"""

import argparse
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from vae_anomaly_detector import VariationalAutoEncoder
from ml_model_loader import build_feature_sequence, build_full_feature_matrix, INPUT_DIM as ML_INPUT_DIM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("train_vae")

DB_PATH = "data/renaissance_bot.db"
OUTPUT_PATH = "models/trained/vae_anomaly_detector.pth"
INPUT_DIM = ML_INPUT_DIM  # 98 (from ml_model_loader)
LATENT_DIM = 32
SEQ_LEN = 30  # Must match build_feature_sequence default


def load_bars_from_db(pairs: list = None, min_bars: int = 60) -> dict:
    """Load OHLCV bars from five_minute_bars table, grouped by pair."""
    if not os.path.exists(DB_PATH):
        logger.warning(f"Database not found: {DB_PATH}")
        return {}

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT pair, COUNT(*) as cnt
        FROM five_minute_bars
        GROUP BY pair
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_bars,))
    available = {r['pair']: r['cnt'] for r in c.fetchall()}
    logger.info(f"Available pairs in DB: {available}")

    if pairs:
        available = {k: v for k, v in available.items() if k in pairs}

    pair_dfs = {}
    for pair, cnt in available.items():
        c.execute("""
            SELECT bar_start, open, high, low, close, volume
            FROM five_minute_bars
            WHERE pair = ?
            ORDER BY bar_start ASC
        """, (pair,))
        rows = c.fetchall()
        df = pd.DataFrame([dict(r) for r in rows])
        df = df.rename(columns={'bar_start': 'timestamp'})
        pair_dfs[pair] = df
        logger.info(f"  {pair}: {len(df)} bars")

    conn.close()
    return pair_dfs


def fetch_coinbase_candles(pair: str = "BTC-USD", days: int = 7) -> pd.DataFrame:
    """Fetch historical candles from Coinbase REST API."""
    try:
        import requests
    except ImportError:
        logger.error("requests library required for Coinbase fetch")
        return pd.DataFrame()

    granularity = 300  # 5 min
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    all_candles = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(hours=5), end)  # Max 300 candles per request
        url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
        params = {
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
            "granularity": granularity,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                candles = resp.json()
                all_candles.extend(candles)
            else:
                logger.warning(f"Coinbase API {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"Coinbase fetch error: {e}")

        current_start = current_end
        time.sleep(0.3)  # Rate limit

    if not all_candles:
        return pd.DataFrame()

    # Coinbase format: [timestamp, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    logger.info(f"Fetched {len(df)} candles for {pair} from Coinbase")
    return df


def generate_training_samples(pair_dfs: dict) -> np.ndarray:
    """Generate feature samples from OHLCV data using vectorized feature builder.

    Computes features once per pair (vectorized), then samples every 5th row.
    Much faster than per-window computation for large datasets.
    """
    all_samples = []

    for pair, df in pair_dfs.items():
        if len(df) < SEQ_LEN + 20:
            logger.info(f"  Skipping {pair}: only {len(df)} bars (need {SEQ_LEN + 20})")
            continue

        # Build cross_data (full data from other pairs)
        cross_data = None
        if len(pair_dfs) > 1:
            cross_data = {p: odf for p, odf in pair_dfs.items() if p != pair}

        # Compute ALL features for entire pair at once
        feat_matrix = build_full_feature_matrix(df, cross_data=cross_data, pair_name=pair)
        if feat_matrix is None:
            logger.info(f"  Skipping {pair}: feature computation failed")
            continue

        # Sample every 5th row after warmup period
        stride = 5
        warmup = 50
        n_samples = 0
        for idx in range(warmup, len(feat_matrix), stride):
            sample = feat_matrix[idx, :]  # (INPUT_DIM,)
            all_samples.append(sample)
            n_samples += 1

        logger.info(f"  {pair}: generated {n_samples} training samples")

    if not all_samples:
        return np.array([])

    samples = np.array(all_samples, dtype=np.float32)
    logger.info(f"Total training samples: {samples.shape[0]} x {samples.shape[1]}")
    return samples


def train_vae(samples: np.ndarray, epochs: int = 100, batch_size: int = 64,
              lr: float = 1e-3) -> VariationalAutoEncoder:
    """Train VAE on feature samples."""
    n = samples.shape[0]
    if n < 50:
        raise ValueError(f"Only {n} samples — need at least 50 for training")

    # Train/val split (80/20)
    split = int(0.8 * n)
    indices = np.random.permutation(n)
    train_data = torch.FloatTensor(samples[indices[:split]])
    val_data = torch.FloatTensor(samples[indices[split:]])

    train_loader = DataLoader(TensorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data), batch_size=batch_size)

    model = VariationalAutoEncoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience_limit = 20

    logger.info(f"Training VAE: {n} samples ({split} train, {n - split} val), "
                f"{epochs} epochs, batch_size={batch_size}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = VariationalAutoEncoder.loss_function(recon, batch, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= split

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                recon, mu, logvar = model(batch)
                loss = VariationalAutoEncoder.loss_function(recon, batch, mu, logvar)
                val_loss += loss.item()
        val_loss /= (n - split)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch + 1:3d}/{epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"best={best_val_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= patience_limit:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # Compute reconstruction error statistics on validation set
    recon_errors = []
    with torch.no_grad():
        for (batch,) in val_loader:
            recon, mu, logvar = model(batch)
            per_sample_error = torch.mean((batch - recon) ** 2, dim=1)
            recon_errors.extend(per_sample_error.numpy().tolist())

    recon_errors = np.array(recon_errors)
    logger.info(f"Validation reconstruction errors:")
    logger.info(f"  mean={recon_errors.mean():.6f}, std={recon_errors.std():.6f}")
    logger.info(f"  p50={np.percentile(recon_errors, 50):.6f}, "
                f"p95={np.percentile(recon_errors, 95):.6f}, "
                f"p99={np.percentile(recon_errors, 99):.6f}")
    logger.info(f"  Suggested anomaly_threshold (p99 * 3): {np.percentile(recon_errors, 99) * 3:.6f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train VAE Anomaly Detector")
    parser.add_argument("--fetch", type=int, default=0,
                        help="Fetch N days of candles from Coinbase before training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Specific pairs to train on (e.g., BTC-USD ETH-USD)")
    args = parser.parse_args()

    # Collect data
    pair_dfs = load_bars_from_db(pairs=args.pairs)

    # Optionally supplement with Coinbase candles
    if args.fetch > 0:
        fetch_pairs = args.pairs or ["BTC-USD", "ETH-USD", "SOL-USD"]
        for pair in fetch_pairs:
            if '-' not in pair:
                continue
            df = fetch_coinbase_candles(pair, days=args.fetch)
            if not df.empty:
                if pair in pair_dfs:
                    # Merge, dedup by timestamp
                    combined = pd.concat([pair_dfs[pair], df]).drop_duplicates('timestamp')
                    pair_dfs[pair] = combined.sort_values('timestamp').reset_index(drop=True)
                    logger.info(f"  Merged {pair}: {len(pair_dfs[pair])} total bars")
                else:
                    pair_dfs[pair] = df

    if not pair_dfs:
        logger.error("No data available for training. Run the bot first or use --fetch.")
        return

    # Generate features
    samples = generate_training_samples(pair_dfs)
    if samples.size == 0:
        logger.error("No training samples generated. Need more data.")
        return

    # Train
    model = train_vae(samples, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    logger.info(f"Saved VAE weights to {OUTPUT_PATH}")
    logger.info(f"Model size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
