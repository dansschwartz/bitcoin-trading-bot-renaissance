# ML Model Retraining Guide

**Date:** 2026-04-23
**Context:** All 7 ML models need retraining after DirectionalLoss hyperparameter fix
**Full analysis:** See `docs/ML_ACCURACY_INVESTIGATION.md`

---

## Why Retraining Is Needed

The ML accuracy investigation (2026-04-22) found **6 root causes** for why all 7 models
achieve ~47-49% directional accuracy (worse than random). Five of the six were fixed at
the inference layer. **Finding 5 — DirectionalLoss hyperparameter mismatch — requires
model retraining to take effect.**

### What Was Wrong (Finding 5)

All production models were trained with v6 loss parameters:
```python
criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)  # v6 — WRONG for tanh outputs
```

These parameters were designed for **unbounded model outputs**, but all models use
`tanh` activation (bounded to `[-1, 1]`). With `logit_scale=20.0`:
- Any prediction >0.05 is treated as 99.99% confident
- Gradient signal is destroyed — the model can't learn nuanced directional predictions
- Models collapse to near-zero predictions to minimize loss

### What Was Fixed

All training scripts now use v7 loss parameters:
```python
criterion = DirectionalLoss(logit_scale=3.0, margin=0.25)  # v7 — calibrated for tanh
```

With `logit_scale=3.0`:
- `pred=0.5` → logit=1.5 → 82% probability (reasonable)
- `pred=1.0` → logit=3.0 → 95% probability (strong conviction)
- Gradients flow naturally through the full prediction range

### Other Inference Fixes Applied (don't require retraining)

| Finding | Fix | Commit |
|---------|-----|--------|
| 1. LightGBM feature mismatch | Raw feature cache + momentum at inference | `602fc81` |
| 2. Debiaser inverts predictions | Debiaser/normalizer disabled | `5740ae7` |
| 3. Distribution shift | Std floor (1e-4) added | `0d72efe` |
| 4. Features zero-padded | Feature zero-rate logging | `f884d77` |
| 5. DirectionalLoss params | Training scripts updated to v7 | `4ad29f1` |
| 6. Feature staleness | 15-minute staleness cutoff | `0a86ebc` |

---

## Expected Improvements After Retraining

| Metric | Before (v6 loss) | Expected (v7 loss) |
|--------|-------------------|--------------------|
| Directional accuracy | 45-49% | >52-55% |
| Prediction magnitude | Near-zero (collapsed) | Meaningful directional signals |
| Collapse recoveries per training | 3-5 | 0-1 |
| Gradient flow | Destroyed by 20x scaling | Natural through tanh range |

---

## Step-by-Step: Retraining in Google Colab

### Option A: Using the Colab Notebook (GPU — recommended)

1. **Sync code to Google Drive:**
   ```bash
   ./retrain.sh sync
   ```

2. **Open Colab:**
   ```bash
   ./retrain.sh train
   ```
   This opens the notebook in your browser.

3. **In Colab:**
   - Runtime → Change runtime type → **T4 GPU**
   - Runtime → Run all (Ctrl+F9)
   - Wait ~30-60 minutes for training to complete

4. **Deploy trained models:**
   ```bash
   ./retrain.sh deploy
   ```
   This pulls models from Google Drive and optionally restarts the bot.

**Or do it all in one command:**
```bash
./retrain.sh full
```

### Option B: Using the Standalone Notebook

1. Upload `notebooks/retrain_98dim_colab.ipynb` to Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Click "Run All"
4. Download the `trained_models_98dim.zip` when done
5. Extract to `models/trained/`

### Option C: Local Training (CPU/MPS — slower)

```bash
# From project root:
python scripts/training/train_all.py
```

This trains all 7 models in the correct order:
- Phase 1: 5 base models (quantum_transformer, bidirectional_lstm, dilated_cnn, cnn, gru)
- Phase 2: Meta-ensemble (stacking layer, depends on all 5 base models)
- Phase 3: VAE anomaly detector

Estimated time on M-series Mac (MPS): ~2-3 hours
Estimated time on CPU: ~4-6 hours

### Option D: Weekly Retraining Script (with safety gates)

```bash
# Interactive (prompts before deploying)
python -m scripts.training.retrain_weekly

# Auto-deploy if accuracy improves
python -m scripts.training.retrain_weekly --auto

# Use full historical data (first-time training)
python -m scripts.training.retrain_weekly --full-history

# Custom rolling window
python -m scripts.training.retrain_weekly --rolling-days 180
```

The weekly script adds safety gates: new models only deploy if
`new_accuracy >= old_accuracy - 1%`.

---

## Validating the Retrained Models

After training completes, verify:

### 1. Check training metadata
```bash
python3 -c "
import json
with open('models/trained/training_metadata.json') as f:
    meta = json.load(f)
for name, info in meta.items():
    acc = info.get('directional_accuracy', 0)
    acc_str = f'{acc:.1%}' if isinstance(acc, float) else 'N/A'
    print(f'{name:25s} | accuracy: {acc_str}')
"
```

**Expected:** All models >52% directional accuracy on test set.

### 2. Check model files exist
```bash
ls -la models/trained/*.pth models/trained/*.pkl
```

**Expected files:**
- `best_quantum_transformer_model.pth` (~18 MB)
- `best_bidirectional_lstm_model.pth` (~16 MB)
- `best_dilated_cnn_model.pth` (~2.4 MB)
- `best_cnn_model.pth` (~1.8 MB)
- `best_gru_model.pth` (~2.1 MB)
- `best_meta_ensemble_model.pth` (~117 KB)
- `vae_anomaly_detector.pth` (~575 KB)
- `best_lightgbm_model.pkl` (~1 MB)
- `lightgbm_model.txt` (text representation)

### 3. Check prediction quality (after bot restart)
```bash
# Wait 5+ minutes for predictions to accumulate, then:
.venv/bin/python3 -c "
import sqlite3
conn = sqlite3.connect('file:data/trading.db?mode=ro', uri=True)
rows = conn.execute('''
    SELECT model_name, COUNT(*) as n,
           AVG(CASE WHEN correct THEN 1.0 ELSE 0.0 END) as acc
    FROM prediction_history
    WHERE timestamp > datetime('now', '-1 hour')
    GROUP BY model_name
''').fetchall()
for r in rows:
    print(f'{r[0]:25s} | {r[1]:5d} predictions | {r[2]:.1%} accuracy')
"
```

### 4. Check predictions are non-zero
```bash
.venv/bin/python3 -c "
import sqlite3, numpy as np
conn = sqlite3.connect('file:data/trading.db?mode=ro', uri=True)
rows = conn.execute('''
    SELECT prediction FROM prediction_history
    WHERE timestamp > datetime('now', '-1 hour')
    LIMIT 100
''').fetchall()
preds = [r[0] for r in rows]
print(f'Predictions: {len(preds)}')
print(f'Mean abs: {np.mean(np.abs(preds)):.4f}')
print(f'Std: {np.std(preds):.4f}')
print(f'Near-zero (<0.01): {sum(1 for p in preds if abs(p) < 0.01)}/{len(preds)}')
"
```

**Expected:** Mean abs prediction >0.05, not all near-zero.

---

## Deploying Retrained Models

### From Google Drive (after Colab training)
```bash
./retrain.sh deploy
```

### Manual deployment
```bash
# 1. Back up existing models
cp -r models/trained models/trained_backup_$(date +%Y%m%d)

# 2. Copy new model files to models/trained/
cp /path/to/new/models/*.pth models/trained/
cp /path/to/new/models/*.pkl models/trained/
cp /path/to/new/models/*.json models/trained/

# 3. Restart the bot
pkill -f "python.*renaissance_trading_bot" || true
sleep 2
.venv/bin/python3 renaissance_trading_bot.py > logs/bot_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Bot restarted (PID: $!)"
```

---

## Training Pipeline Architecture

```
data/training/*.csv (30+ days of 5-min OHLCV per pair)
        │
        ▼
┌──────────────────────────┐
│  training_utils.py       │
│  - load_training_csvs()  │
│  - generate_sequences()  │  → (N, 30, 98) sequences + labels
│  - walk_forward_split()  │  → 70% train / 13% val / 17% test
│  - DirectionalLoss v7    │  → logit_scale=3.0, margin=0.25
└──────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Phase 1: 5 Base Models (independent, parallel)  │
│  - train_quantum_transformer.py → .pth           │
│  - train_bidirectional_lstm.py  → .pth           │
│  - train_dilated_cnn.py        → .pth           │
│  - train_cnn.py                → .pth           │
│  - train_gru.py                → .pth           │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Phase 2: Meta-Ensemble (depends on Phase 1)     │
│  - train_meta_ensemble.py → .pth                │
│  - Stacking layer: [83-dim features | 5 preds]  │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Phase 3: VAE Anomaly Detector                   │
│  - train_vae.py → vae_anomaly_detector.pth      │
└──────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────┐
│  Separate: LightGBM (gradient boosting)          │
│  - train_lightgbm.py → .pkl + .txt              │
│  - 104 features (98 raw + 6 momentum)            │
│  - Binary classification, no GPU needed          │
└──────────────────────────────────────────────────┘
```

---

## Files Modified in This Fix

| File | Change |
|------|--------|
| `scripts/training/training_utils.py:208` | DirectionalLoss defaults now `logit_scale=3.0, margin=0.25` |
| `scripts/training/train_quantum_transformer.py:149` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/train_bidirectional_lstm.py:126` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/train_dilated_cnn.py:126` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/train_cnn.py:122` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/train_gru.py:122` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/train_meta_ensemble.py:270` | Uses `DirectionalLoss()` (v7 defaults) |
| `scripts/training/retrain_weekly.py:190` | Fixed `DirectionalLoss(0.3)` → `DirectionalLoss()` |
| `notebooks/retrain_98dim_colab.ipynb` | Meta-ensemble cell: `logit_scale=20.0` → `3.0` |
| `retrain.sh` | Auto-detect BOT_DIR, copy `.pkl` files in deploy |

---

## Troubleshooting

### "No training data found"
```bash
# Download fresh data:
python -m scripts.training.fetch_training_data --days 30
```

### "CUDA out of memory" in Colab
- Reduce batch size: edit `BATCH_SIZE = 64` → `32` in the notebook
- Or use a smaller GPU runtime (T4 has 16GB, usually sufficient)

### Model accuracy below 50% after retraining
- Check data quality: `wc -l data/training/*.csv`
- Ensure at least 30 days of data per pair
- Try `--full-history` for more training data
- Check for NaN/inf in features: the training scripts log this

### Safety gate rejects new model
The weekly retraining script keeps the old model if the new one is >1% worse.
This is intentional. Check the training logs for why accuracy dropped.
