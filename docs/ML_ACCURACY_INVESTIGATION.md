# ML Model Accuracy Investigation

**Date:** 2026-04-22
**Status:** All 7 ML models below 50% directional accuracy on 170K+ predictions
**Verdict:** Multiple compounding issues — no single "smoking gun"

---

## Executive Summary

All 7 ML models (quantum_transformer, bidirectional_lstm, dilated_cnn, cnn, gru,
lightgbm, meta_ensemble) achieve ~47-49% directional accuracy on live predictions,
which is **worse than random**. After thorough code review of both the training
pipeline and inference pipeline, I identified **6 root causes** ranging from
critical feature mismatches to subtle distribution shifts.

The core problem: **models are trained on historical data with rich cross-asset
and derivatives features, but at inference they often receive degraded/missing
versions of those same features, causing out-of-distribution inputs that the
models respond to with near-zero predictions.**

---

## Finding 1: LightGBM Training/Inference Feature Mismatch (CRITICAL)

**Severity:** P0 — This alone could explain LightGBM's sub-50% accuracy

### Training features (train_lightgbm.py:69-175)
```python
# Training uses: INPUT_DIM features (98) + 6 momentum returns = 104 features
bar_features = feat_matrix[idx]  # (98,) — single bar, RAW (no standardization)
momentum_feats = [current_close / past_close - 1.0 for h in MOMENTUM_BARS]
row = np.concatenate([bar_features, np.array(momentum_feats)])  # (104,)
```

### Inference features (ml_model_loader.py:1752-1765)
```python
# Inference uses: [last, mean, std] = INPUT_DIM * 3 = 294 features
def _prepare_lgb_features(features):
    last = features[-1, :]           # (98,) — last bar
    mean = features.mean(axis=0)     # (98,) — window average
    std = features.std(axis=0)       # (98,) — window volatility
    return np.concatenate([last, mean, std]).reshape(1, -1)  # (1, 294)
```

### The Mismatch

| Aspect | Training | Inference |
|--------|----------|-----------|
| Feature count | 104 (98 + 6 momentum) | 294 (98 × 3) |
| Feature composition | Raw features + momentum returns | Last/mean/std of standardized window |
| Standardization | None (raw `build_full_feature_matrix`) | Per-window z-score applied first |
| Momentum features | Explicit 1,3,6,12,24,72-bar returns | Not present |

**Impact:** LightGBM at inference receives a completely different feature
structure than what it was trained on. Feature indices 98-103 contain
momentum returns during training but mean-of-feature-0 through mean-of-feature-5
at inference. The model's learned decision boundaries are meaningless.

**File references:**
- `scripts/training/train_lightgbm.py:69-175` (training features)
- `ml_model_loader.py:1752-1765` (`_prepare_lgb_features` — inference)
- `ml_model_loader.py:1788-1807` (`predict_lightgbm` — auto-pads/truncates)

---

## Finding 2: Debiaser and Normalizer Corrupt Predictions During Warmup (HIGH)

**Severity:** P1 — Actively inverts correct predictions for first ~200 samples

### The Code (ml_model_loader.py:154-251)

```python
# PredictionDebiaser (line 170-186)
def debias(self, model_name, raw_prediction):
    self._ema[model_name] = alpha * raw_prediction + (1-alpha) * self._ema[model_name]
    self._count[model_name] += 1
    if self._count[model_name] < self._warmup:  # warmup = 200
        return raw_prediction  # RAW during warmup
    return raw_prediction - self._ema[model_name]  # DEBIASED after warmup

# ModelPredictionNormalizer (line 220-234)
def normalize(self, model_name, prediction):
    s['mean'] += alpha * (prediction - s['mean'])
    s['var'] += alpha * ((prediction - s['mean'])**2 - s['var'])
    if s['count'] < self.min_samples:  # min_samples = 200
        return prediction  # PASSTHROUGH during warmup
    std = max(s['var'] ** 0.5, 1e-8)
    return (prediction - s['mean']) / std  # Z-SCORE after warmup
```

### The Problem

After warmup (200+ predictions per model), predictions are z-score normalized.
This transforms the model's output from `[-1, 1]` (tanh bounded) to standard
normal distribution, potentially **inverting the sign** of predictions.

Example: If a model consistently predicts +0.02 (slight bullish), after warmup:
- mean ≈ 0.02, std ≈ 0.01
- Prediction of +0.015 → z-score = (0.015 - 0.02) / 0.01 = **-0.5** (inverted!)

The debiaser subtracts the EMA, and the normalizer divides by running std.
Together they can turn a mildly bullish prediction into a bearish one.

**File references:**
- `ml_model_loader.py:154-201` (PredictionDebiaser)
- `ml_model_loader.py:206-246` (ModelPredictionNormalizer)
- `ml_model_loader.py:1567-1568` (applied in predict_with_models)

---

## Finding 3: Per-Window Standardization Creates Distribution Shift (HIGH)

**Severity:** P1 — Models see different feature distributions in quiet vs volatile markets

### The Standardization (ml_model_loader.py:1261-1264)
```python
# build_feature_sequence — applied at inference
feat_arr = feat_df.tail(seq_len).values  # (30, 98)
mean = feat_arr.mean(axis=0, keepdims=True)
std = feat_arr.std(axis=0, keepdims=True) + 1e-8
feat_arr = (feat_arr - mean) / std
```

### Training does the same (training_utils.py:111-114)
```python
window = feat_matrix[start_idx:end_idx]  # (30, 98)
mean = window.mean(axis=0, keepdims=True)
std = window.std(axis=0, keepdims=True) + 1e-8
window = (window - mean) / std
```

### Why This Still Causes Problems

The standardization is **consistent** between training and inference, but it
creates a **regime-dependent distribution shift**:

1. **Training data** spans months with diverse volatility regimes. The model
   learns from windows where std ranges from 0.0001 to 0.05.

2. **Inference** during quiet markets: std ≈ 0.00001 for return features.
   Division by near-zero amplifies noise by 1000x, creating features
   the model never saw during training.

3. **Inference** during crashes: std ≈ 0.05 for return features. Features
   are compressed, losing the signal the model learned from.

The model was effectively trained on a **mixture of scaling regimes** but
at inference always sees exactly one regime's scaling. This creates
systematic prediction errors that cancel out to ~50% accuracy.

---

## Finding 4: Cross-Asset and Derivatives Features Often Zero at Inference (HIGH)

**Severity:** P1 — Up to 26 of 98 features are zero-padded at inference

### Cross-Asset Features (15 features)

At training (`training_utils.py:80-83`):
```python
# All pairs available from CSV data
cross_data = {p: odf for p, odf in pair_dfs.items() if p != pair}
```

At inference (`renaissance_trading_bot.py:4377-4387`):
```python
# Only pairs that have accumulated 30+ bars in memory
cross_data = {}
for _pid in cycle_pairs:
    _cdf = _tech._to_dataframe()
    if _cdf is None or len(_cdf) < 30:
        _cdf = self._load_price_df_from_db(_pid, limit=300)
    if _cdf is not None and len(_cdf) > 0:
        cross_data[_pid] = _cdf
```

**Problem:** During early bot operation or after restarts, many pairs may not
have 30+ bars of data. Cross-asset features default to zeros. The model was
trained with rich cross-asset data but at inference gets zeros for these 15
features.

### Derivatives Features (7 + 4 = 11 features)

At training: Historical derivatives CSVs with funding rates, OI, etc.
At inference: Real-time Binance derivatives API — may be unavailable,
returning zeros for all 11 features.

### Feature Audit Evidence

The code at `ml_model_loader.py:1226-1237` logs feature health:
```python
active_cols = [c for c in feat_df.columns if feat_df[c].abs().sum() > 0]
zero_cols = [c for c in feat_df.columns if c not in active_cols]
logger.info(f"[FEATURE AUDIT] {_pair_key}: {len(active_cols)}/{n_total} active, ...")
```

When 26 of 72 real features are zero (plus 26 padding zeros = 52/98 zeros),
the model is effectively operating on **half its feature space**.

---

## Finding 5: DirectionalLoss Hyperparameter Mismatch Between Training Scripts (MEDIUM)

**Severity:** P2 — Training may use suboptimal loss configuration

### In train_quantum_transformer.py:149
```python
criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)
```

### In DirectionalLoss class definition (training_utils.py:207)
```python
class DirectionalLoss(nn.Module):
    def __init__(self, logit_scale: float = 3.0, margin: float = 0.25):
```

### In evaluate_on_dataset (training_utils.py:373)
```python
criterion = DirectionalLoss(logit_scale=20.0, margin=0.10)
```

The default DirectionalLoss uses `logit_scale=3.0, margin=0.25` (v7 parameters
designed for tanh-bounded outputs), but the training script overrides to
`logit_scale=20.0, margin=0.10` (v6 parameters designed for unbounded outputs).

The docstring explicitly warns:
> "v6 used 20x, which made all predictions 99.99% confident and destroyed
>  gradients for tanh-bounded outputs."

Yet the training script still uses `logit_scale=20.0`. This means during training,
the BCE component treats any prediction >0.05 as 99.99% confident of "up",
destroying gradient signal for the model to learn nuanced directional predictions.

**File references:**
- `scripts/training/train_quantum_transformer.py:149`
- `scripts/training/training_utils.py:207` (default = 3.0)
- `scripts/training/training_utils.py:373` (evaluate uses 20.0)

---

## Finding 6: Feature Staleness from Cached/In-Memory Data (MEDIUM)

**Severity:** P2 — Predictions may lag reality by 5-30 minutes

### The Data Path

1. Binance spot API fetches OHLCV data every cycle (~10 seconds)
2. `TradingTechnical` maintains a rolling buffer of ~300 bars
3. `_to_dataframe()` converts to DataFrame for ML features
4. `build_feature_sequence()` takes the last 300 bars, computes features
   from the last 30 bars

### The Staleness Problem

- 5-minute bars mean each bar represents 5 minutes of data
- The most recent bar may be **up to 5 minutes old** (it's the current
  incomplete bar or the last closed bar)
- Cross-asset data from other pairs may be even more stale if their
  fetch cycle hasn't run yet
- After bot restarts, `_to_dataframe()` may return data loaded from the
  database, which could be hours old

The forensic audit found predictions up to 126 minutes old. Even with
the 15-minute staleness fix, the **features themselves** can still be
built from stale underlying data.

---

## Recommendations

### Immediate (Fix Without Retraining)

1. **Fix LightGBM feature mismatch** — Either change `_prepare_lgb_features()`
   to match training format (98 raw + 6 momentum), or retrain LightGBM with
   [last, mean, std] format. The current mismatch is a clear bug.

2. **Disable debiaser/normalizer or extend warmup** — The z-score normalization
   after 200 samples actively corrupts predictions. Either disable it, or
   increase warmup to the full session lifetime.

3. **Log feature zero-rate** — Add persistent logging of how many features
   are zero at each prediction. This will quantify Finding 4.

### Short-Term (Retrain Required)

4. **Fix DirectionalLoss parameters** — Use `logit_scale=3.0, margin=0.25`
   (the v7 defaults) in all training scripts. The current 20.0 destroys
   gradients for tanh-bounded outputs.

5. **Train with realistic feature dropout** — Randomly zero out cross-asset
   and derivatives features during training (20-50% dropout rate) so models
   learn to handle missing features gracefully.

6. **Add feature distribution monitoring** — Track running statistics of
   each feature's mean/std at inference and compare to training-time
   distributions. Alert when drift exceeds 2σ.

### Long-Term (Architecture Changes)

7. **Replace per-window standardization** — Use a global running z-score
   (EMA of mean/std) instead of per-window standardization. This eliminates
   the regime-dependent scaling problem.

8. **Separate cross-asset model** — Cross-asset features are unreliable
   at inference. Consider a model variant trained without them.

---

## Appendix: Model Accuracy by Model (from Forensic Audit)

| Model | Evaluated | Correct | Accuracy |
|-------|-----------|---------|----------|
| lightgbm | 27,966 | 13,724 | 49.1% |
| meta_ensemble | 27,966 | 13,655 | 48.8% |
| dilated_cnn | 19,885 | 9,642 | 48.5% |
| cnn | 27,966 | 13,354 | 47.8% |
| bidirectional_lstm | 19,885 | 9,500 | 47.8% |
| quantum_transformer | 27,966 | 13,093 | 46.8% |

All models are within 2.3% of 50% (random), consistent with out-of-distribution
inference producing near-zero predictions that are scored as coin flips.
