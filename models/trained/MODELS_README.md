# Trained Models Registry

## Model Inventory

### Deep Learning Models (DirectionalLoss)

These 6 models use PyTorch with a custom `DirectionalLoss` function that penalizes
wrong-direction predictions more heavily than magnitude errors.

| Model | File | Architecture | Retrained |
|-------|------|-------------|-----------|
| Quantum Transformer | `best_quantum_transformer_model.pth` | Transformer with quantum-inspired attention | 2026-04-24 (v7) |
| CNN | `best_cnn_model.pth` | 1D convolutional network | 2026-04-24 (v7) |
| Bidirectional LSTM | `best_bidirectional_lstm_model.pth` | Bi-LSTM with attention | 2026-04-24 (v7) |
| Dilated CNN | `best_dilated_cnn_model.pth` | Dilated causal convolutions | 2026-04-24 (v7) |
| GRU | `best_gru_model.pth` | Gated Recurrent Unit | 2026-04-24 (v7) |
| Meta Ensemble | `best_meta_ensemble_model.pth` | Stacking ensemble of above 5 models | 2026-04-24 (v7) |

### LightGBM Models (Binary Objective)

These models use standard LightGBM binary classification and were **not** affected
by the DirectionalLoss v6/v7 change.

| Model | File | Purpose | Trained |
|-------|------|---------|---------|
| LightGBM (directional) | `lightgbm_model.txt` | Direction prediction | 2026-04-24 (Colab) |
| Crash LightGBM (BTC) | `crash_lightgbm_model.txt` | Crash probability | 2026-03-01 |
| Crash BTC 2-bar | `../crash/btc_2bar_model.txt` | BTC crash, 10min horizon | 2026-03-01 |
| Crash ETH 2-bar | `../crash/eth_2bar_model.txt` | ETH crash, 10min horizon | 2026-03-01 |
| Crash ETH 1-bar | `../crash/eth_1bar_model.txt` | ETH crash, 5min horizon | 2026-03-01 |
| Crash SOL 2-bar | `../crash/sol_2bar_model.txt` | SOL crash, 10min horizon | 2026-03-01 |
| Crash XRP 2-bar | `../crash/xrp_2bar_model.txt` | XRP crash, 10min horizon | 2026-03-01 |

## v7 Retraining (2026-04-24)

### What Changed

The `DirectionalLoss` function was updated from v6 to v7:

- **v6 (old):** `logit_scale=1.0, margin=0.0` — effectively standard MSE loss with
  no directional penalty. Models converged to ~50% directional accuracy (no better
  than random).
- **v7 (new):** `logit_scale=3.0, margin=0.25` — applies a sigmoid-scaled penalty
  that amplifies wrong-direction errors by 3x and requires predictions to exceed a
  0.25 margin before being counted as directionally correct.

### Why

The ML accuracy audit (see `docs/ML_ACCURACY_INVESTIGATION.md`) found all 6 DL models
at 47-49% directional accuracy — worse than a coin flip. Root cause: the v6
DirectionalLoss parameters made the directional component contribute essentially zero
gradient, so models optimized only for magnitude (MSE), ignoring direction entirely.

### Training Details

- **Platform:** Google Colab (T4 GPU)
- **Epochs:** 100 (up from 5 in v6 training)
- **Dataset:** Same 35,775 train / 6,257 val / 8,326 test samples
- **Device:** CUDA (v6 was trained on CPU)
- **Accuracy:** Not captured per-model from Colab output; meta files show `null`
  for accuracy fields pending live evaluation

### Models NOT Retrained

The LightGBM models (both directional and crash) use standard LightGBM binary/cross-entropy
objectives, not DirectionalLoss. They were unaffected by the v6 bug and were not retrained.

- `lightgbm_model.txt` — Was retrained separately on 2026-04-24 via Colab with
  updated features (98 input dims, 294 features after last/mean/std prep)
- `crash_lightgbm_model.txt` and `models/crash/*` — Trained 2026-03-01, unchanged

## How to Retrain

See `scripts/training/RETRAINING_GUIDE.md` for full instructions.

Quick summary:
1. Upload training notebook to Google Colab
2. Ensure `DirectionalLoss` uses v7 params: `logit_scale=3.0, margin=0.25`
3. Train for 100 epochs minimum
4. Download `.pth` files and replace in `models/trained/`
5. Update the corresponding `_meta.json` files with new training date and metrics
