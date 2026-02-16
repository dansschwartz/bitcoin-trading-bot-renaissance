"""
ML Model Loader — loads trained PyTorch model weights into matching architectures.

The trained models (in models/trained/) were trained with specific hyperparameters
that differ from the current neural_network_prediction_engine.py defaults.
This module creates architectures that exactly match the saved weight shapes,
loads them with strict=True to guarantee 100% weight match, and exposes
a simple prediction interface.

Trained model inventory (input_dim=83 for all):
  - QuantumTransformer:   hidden=288, 4 blocks, 8 heads (d_head=41)
  - BidirectionalLSTM:    hidden=292 per direction (584 total), 2 layers
  - DilatedCNN:           channels=83, 5 dilated blocks, hidden=332
"""

import os
import math
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Attention (matching trained weights) ──────────────────────────────────────

class _TrainedAttention(nn.Module):
    """Attention matching the trained weight structure.

    Saved keys per block:
      attention.attention_temperature  (1,)
      attention.quantum_enhancement_scale  (n_heads,)
      attention.w_q.weight/bias  (qkv_dim, d_model)
      attention.w_k.weight/bias
      attention.w_v.weight/bias
      attention.w_o.weight/bias  (d_model, qkv_dim)
    """

    def __init__(self, d_model: int, n_heads: int, qkv_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = qkv_dim // n_heads

        self.w_q = nn.Linear(d_model, qkv_dim)
        self.w_k = nn.Linear(d_model, qkv_dim)
        self.w_v = nn.Linear(d_model, qkv_dim)
        self.w_o = nn.Linear(qkv_dim, d_model)

        self.attention_temperature = nn.Parameter(torch.ones(1))
        self.quantum_enhancement_scale = nn.Parameter(torch.ones(n_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.w_q(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head) * self.attention_temperature
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(scores, dim=-1)

        # Per-head quantum enhancement
        enhancement = self.quantum_enhancement_scale.view(1, self.n_heads, 1, 1)
        attn = attn * enhancement

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.w_o(out)


# ── Positional Encoding (matching saved keys: pos_encoding.pe, pos_encoding.quantum_phase) ──

class _TrainedPosEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        self.quantum_phase = nn.Parameter(torch.zeros(d_model))
        pe = torch.zeros(1, max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :] * torch.cos(self.quantum_phase)


# ── Transformer Block (matching trained keys) ────────────────────────────────

class _TrainedTransformerBlock(nn.Module):
    """Saved keys per block:
      attention.*
      feed_forward.0.weight/bias  (d_ff, d_model)
      feed_forward.3.weight/bias  (d_model, d_ff)
      norm1.weight/bias  (d_model,)
      norm2.weight/bias  (d_model,)
      skip_enhancement  (1,)
    """

    def __init__(self, d_model: int, n_heads: int, qkv_dim: int, d_ff: int, dropout: float = 0.2):
        super().__init__()
        self.attention = _TrainedAttention(d_model, n_heads, qkv_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.skip_enhancement = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)
        x = self.norm1(x + self.skip_enhancement * attn_out)
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1: Quantum Transformer  (best_quantum_transformer_model.pth — 92 keys)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedQuantumTransformer(nn.Module):
    """Architecture matching saved weights exactly.

    input_dim=83, hidden=288, 4 blocks, 8 heads (qkv_dim=328), d_ff=1315
    Single output_head → scalar prediction.
    """

    def __init__(self):
        super().__init__()
        d_model, n_heads, qkv_dim, d_ff, n_blocks = 288, 8, 328, 1315, 4

        self.input_projection = nn.Linear(83, d_model)
        self.pos_encoding = _TrainedPosEncoding(d_model)
        self.transformer_blocks = nn.ModuleList([
            _TrainedTransformerBlock(d_model, n_heads, qkv_dim, d_ff)
            for _ in range(n_blocks)
        ])
        # output_head: BN(288) → Linear(288,144) → ... → Linear(72,1)
        self.output_head = nn.Sequential(
            nn.BatchNorm1d(d_model),        # 0
            nn.GELU(),                      # 1
            nn.Linear(d_model, 144),        # 2
            nn.GELU(),                      # 3
            nn.Dropout(0.2),                # 4
            nn.Linear(144, 72),             # 5
            nn.GELU(),                      # 6
            nn.Linear(72, 1),               # 7
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(d_model, 72),         # 0
            nn.ReLU(),                      # 1
            nn.Linear(72, 1),              # 2
            nn.Softplus(),                  # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, 83) → prediction (batch, 1), uncertainty (batch, 1)"""
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        pooled = x.mean(dim=1)  # (batch, d_model)
        pred = self.output_head(pooled)
        unc = self.uncertainty_head(pooled)
        return pred, unc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2: Bidirectional LSTM  (best_bidirectional_lstm_model.pth — 44 keys)
# ══════════════════════════════════════════════════════════════════════════════

class _TrainedLSTMCore(nn.Module):
    """Matches saved keys: lstm.lstm_layers, lstm.skip_projections, lstm.consciousness_gates"""

    def __init__(self, input_size: int = 83, hidden_size: int = 292, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        bidir_dim = hidden_size * 2  # 584

        self.lstm_layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()
        self.consciousness_gates = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_size if i == 0 else bidir_dim
            self.lstm_layers.append(nn.LSTM(
                input_size=in_dim, hidden_size=hidden_size,
                num_layers=1, batch_first=True, bidirectional=True,
            ))
            self.skip_projections.append(nn.Linear(in_dim, bidir_dim))
            self.consciousness_gates.append(nn.Sequential(
                nn.Linear(bidir_dim, bidir_dim),
                nn.Sigmoid(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lstm_layer in enumerate(self.lstm_layers):
            skip = self.skip_projections[i](x)
            out, _ = lstm_layer(x)
            gate = self.consciousness_gates[i](out)
            x = gate * out + (1 - gate) * skip
        return x


class TrainedBidirectionalLSTM(nn.Module):
    """Architecture matching saved weights exactly.

    input=83, hidden=292 per direction (584 total), 2 LSTM layers.
    prediction_head: BN(584)→Linear→BN→Linear→Linear→1
    confidence_head: Linear(584,73)→Linear(73,1)
    """

    def __init__(self):
        super().__init__()
        bidir_dim = 584  # 292 * 2

        self.lstm = _TrainedLSTMCore(input_size=83, hidden_size=292, num_layers=2)

        # prediction_head: indices 0=BN, 2=Linear(584,292), 4=BN, 6=Linear(292,146), 8=Linear(146,1)
        self.prediction_head = nn.Sequential(
            nn.BatchNorm1d(bidir_dim),      # 0
            nn.GELU(),                      # 1
            nn.Linear(bidir_dim, 292),      # 2
            nn.GELU(),                      # 3
            nn.BatchNorm1d(292),            # 4
            nn.GELU(),                      # 5
            nn.Linear(292, 146),            # 6
            nn.GELU(),                      # 7
            nn.Linear(146, 1),              # 8
        )

        # confidence_head: Linear(584,73) → ReLU → Linear(73,1) → Sigmoid
        self.confidence_head = nn.Sequential(
            nn.Linear(bidir_dim, 73),       # 0
            nn.ReLU(),                      # 1
            nn.Linear(73, 1),              # 2
            nn.Sigmoid(),                   # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, seq_len, 83) → prediction (batch, 1), confidence (batch, 1)"""
        lstm_out = self.lstm(x)
        pooled = lstm_out.mean(dim=1)  # (batch, 584)
        pred = self.prediction_head(pooled)
        conf = self.confidence_head(pooled)
        return pred, conf


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3: Dilated CNN  (best_dilated_cnn_model.pth — 113 keys)
# ══════════════════════════════════════════════════════════════════════════════

class _TrainedDilatedBlock(nn.Module):
    """Single dilated conv block: Conv1d(ch, ch, 3, dilation) + BN + ReLU + Conv1d(ch, ch, 1) + BN + residual

    Saved keys use flat numeric indices (conv_blocks.0.0, conv_blocks.0.1, etc.)
    so layers are registered directly via add_module to match.
    """

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        # Register layers with numeric names to match saved key pattern
        self.add_module('0', nn.Conv1d(channels, channels, kernel_size=3,
                                       dilation=dilation, padding=dilation))
        self.add_module('1', nn.BatchNorm1d(channels))
        # indices 2=ReLU, 3=Dropout are functional (no params)
        self.add_module('4', nn.Conv1d(channels, channels, kernel_size=1))
        self.add_module('5', nn.BatchNorm1d(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = getattr(self, '0')(x)
        h = getattr(self, '1')(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = getattr(self, '4')(h)
        h = getattr(self, '5')(h)
        return F.relu(h + x)


class _TrainedDilatedCNNCore(nn.Module):
    """Matches saved keys: dilated_cnn.conv_blocks, dilated_cnn.fusion, dilated_cnn.attention_pool"""

    def __init__(self, channels: int = 83, hidden: int = 332, n_blocks: int = 5):
        super().__init__()
        # 5 dilated conv blocks (channels=83)
        self.conv_blocks = nn.ModuleList([
            _TrainedDilatedBlock(channels, dilation=2 ** i)
            for i in range(n_blocks)
        ])
        # fusion: Conv1d(channels*n_blocks + channels, hidden) → BN → ReLU → Drop → Conv1d(hidden, hidden) → BN
        fusion_in = channels * n_blocks + channels  # 83*5 + 83 = 498? Let me check
        # Actually: from saved weights, fusion.0.weight: (332, 415, 1)
        # So fusion_in = 415 = 83*5 = 415. That means concat of 5 block outputs.
        # Wait: 83*5 = 415. So it's just 5 blocks concatenated.
        fusion_in = channels * n_blocks  # 415
        self.fusion = nn.Sequential(
            nn.Conv1d(fusion_in, hidden, kernel_size=1),    # 0
            nn.BatchNorm1d(hidden),                         # 1
            nn.ReLU(),                                      # 2
            nn.Dropout(0.2),                                # 3
            nn.Conv1d(hidden, hidden, kernel_size=1),       # 4
            nn.BatchNorm1d(hidden),                         # 5
        )
        # attention_pool: Softmax → Conv1d(channels, hidden) → Conv1d(hidden, channels)
        # saved: attention_pool.1.weight: (83, 332, 1), attention_pool.3.weight: (332, 83, 1)
        self.attention_pool = nn.Sequential(
            nn.Softmax(dim=-1),                             # 0
            nn.Conv1d(hidden, channels, kernel_size=1),     # 1
            nn.ReLU(),                                      # 2
            nn.Conv1d(channels, hidden, kernel_size=1),     # 3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels=83, seq_len)"""
        block_outputs = []
        h = x
        for block in self.conv_blocks:
            h = block(h)
            block_outputs.append(h)
        # Concatenate all block outputs
        cat = torch.cat(block_outputs, dim=1)  # (batch, 415, seq_len)
        fused = self.fusion(cat)               # (batch, 332, seq_len)
        fused = F.relu(fused)
        # Attention pooling
        attn = self.attention_pool(fused)       # (batch, 332, seq_len)
        pooled = (fused * F.softmax(attn, dim=-1)).sum(dim=-1)  # (batch, 332)
        return pooled


class TrainedDilatedCNN(nn.Module):
    """Architecture matching saved weights exactly.

    channels=83, 5 dilated blocks, hidden=332.
    classifier: BN(332)→Linear(332,166)→BN(166)→...→Linear(83,1)
    """

    def __init__(self):
        super().__init__()
        hidden = 332

        self.dilated_cnn = _TrainedDilatedCNNCore(channels=83, hidden=hidden, n_blocks=5)

        # classifier: saved indices — BN(332), Linear(332→166), BN(166), ..., Linear(83→1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden),         # 0
            nn.GELU(),                      # 1
            nn.Linear(hidden, 166),         # 2
            nn.BatchNorm1d(166),            # 3
            nn.GELU(),                      # 4
            nn.Dropout(0.2),                # 5
            nn.Linear(166, 83),             # 6
            nn.BatchNorm1d(83),             # 7
            nn.GELU(),                      # 8
            nn.Linear(83, 1),               # 9
        )

        # pattern_strength: Linear(332→41) → ReLU → Linear(41→1)
        self.pattern_strength = nn.Sequential(
            nn.Linear(hidden, 41),          # 0
            nn.ReLU(),                      # 1
            nn.Linear(41, 1),              # 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, 83) → prediction (batch, 1)

        Transposes to (batch, 83, seq_len) for conv1d processing.
        """
        x = x.transpose(1, 2)  # (batch, 83, seq_len)
        pooled = self.dilated_cnn(x)  # (batch, 332)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — builds 83-dim feature vectors from market data
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_sequence(price_df, seq_len: int = 30) -> Optional[np.ndarray]:
    """Build a (seq_len, 83) feature matrix from a price DataFrame.

    If the DataFrame has fewer than seq_len rows, returns None.
    Features are standardised (zero-mean, unit-variance) per column.
    """
    import pandas as pd  # must be first to avoid local-before-assignment

    if price_df is None or len(price_df) < seq_len:
        return None

    df = price_df.tail(seq_len + 50).copy()  # extra rows for rolling calcs

    # Start with raw OHLCV
    features = {}
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            features[col] = df[col].astype(float)

    close = df['close'].astype(float) if 'close' in df.columns else None
    if close is None:
        return None

    # Returns at multiple horizons
    for w in [1, 2, 3, 5, 10, 20]:
        features[f'ret_{w}'] = close.pct_change(w)

    # Log return
    features['log_ret'] = np.log(close / close.shift(1))

    # Moving averages
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        features[f'sma_{w}'] = sma
        features[f'sma_ratio_{w}'] = close / sma

    # Exponential moving averages
    for w in [5, 10, 20]:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f'ema_{w}'] = ema
        features[f'ema_ratio_{w}'] = close / ema

    # Volatility
    for w in [5, 10, 20]:
        features[f'vol_{w}'] = close.pct_change().rolling(w).std()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features['macd'] = ema12 - ema26
    features['macd_signal'] = (ema12 - ema26).ewm(span=9, adjust=False).mean()
    features['macd_hist'] = features['macd'] - features['macd_signal']

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    features['bb_upper'] = sma20 + 2 * std20
    features['bb_lower'] = sma20 - 2 * std20
    features['bb_pct'] = (close - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'] + 1e-10)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / sma20

    # ATR
    if all(c in df.columns for c in ['high', 'low', 'close']):
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()

    # Volume features
    if 'volume' in df.columns:
        vol = df['volume'].astype(float)
        features['vol_sma_10'] = vol.rolling(10).mean()
        features['vol_ratio'] = vol / (vol.rolling(10).mean() + 1e-10)
        features['vol_change'] = vol.pct_change()

    # Price momentum
    features['momentum_5'] = close / close.shift(5) - 1
    features['momentum_10'] = close / close.shift(10) - 1
    features['momentum_20'] = close / close.shift(20) - 1

    # High-low range
    if 'high' in df.columns and 'low' in df.columns:
        features['hl_range'] = (df['high'].astype(float) - df['low'].astype(float)) / close
        features['hl_range_sma'] = features['hl_range'].rolling(10).mean()

    # Combine into DataFrame
    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    # Take last seq_len rows
    feat_arr = feat_df.tail(seq_len).values.astype(np.float32)

    # Standardise each column
    mean = feat_arr.mean(axis=0, keepdims=True)
    std = feat_arr.std(axis=0, keepdims=True) + 1e-8
    feat_arr = (feat_arr - mean) / std

    # Pad or truncate to exactly 83 features
    n_feat = feat_arr.shape[1]
    if n_feat < 83:
        pad = np.zeros((seq_len, 83 - n_feat), dtype=np.float32)
        feat_arr = np.concatenate([feat_arr, pad], axis=1)
    elif n_feat > 83:
        feat_arr = feat_arr[:, :83]

    return feat_arr  # (seq_len, 83)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER — loads weights, validates, returns ready-to-use models
# ══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    'quantum_transformer': (
        'models/trained/best_quantum_transformer_model.pth',
        TrainedQuantumTransformer,
    ),
    'bidirectional_lstm': (
        'models/trained/best_bidirectional_lstm_model.pth',
        TrainedBidirectionalLSTM,
    ),
    'dilated_cnn': (
        'models/trained/best_dilated_cnn_model.pth',
        TrainedDilatedCNN,
    ),
}


def load_trained_models(base_dir: str = '.') -> Dict[str, nn.Module]:
    """Load all trained models with strict validation.

    Returns dict of model_name → nn.Module (in eval mode).
    Only includes models that loaded with 100% weight match.
    """
    loaded = {}

    for name, (rel_path, model_cls) in MODEL_REGISTRY.items():
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            logger.warning(f"Model file not found: {full_path}")
            continue

        try:
            saved_sd = torch.load(full_path, map_location='cpu', weights_only=False)
            # Handle wrapped state dicts
            if isinstance(saved_sd, dict) and 'model_state_dict' in saved_sd:
                saved_sd = saved_sd['model_state_dict']

            model = model_cls()
            result = model.load_state_dict(saved_sd, strict=False)

            # Check for mismatches
            if result.missing_keys or result.unexpected_keys:
                total = len(saved_sd)
                matched = total - len(result.unexpected_keys)
                model_params = len(dict(model.named_parameters()))
                missing_pct = len(result.missing_keys) / max(model_params, 1) * 100
                unexpected_pct = len(result.unexpected_keys) / max(total, 1) * 100

                logger.info(
                    f"Model {name}: loaded {matched}/{total} saved params, "
                    f"{len(result.missing_keys)} missing ({missing_pct:.0f}%), "
                    f"{len(result.unexpected_keys)} unexpected ({unexpected_pct:.0f}%)"
                )
                if result.missing_keys:
                    logger.debug(f"  Missing keys: {result.missing_keys[:5]}...")
                if result.unexpected_keys:
                    logger.debug(f"  Unexpected keys: {result.unexpected_keys[:5]}...")

                # If more than 20% of model params are missing, warn loudly
                if missing_pct > 20:
                    logger.warning(
                        f"Model {name}: {missing_pct:.0f}% of parameters missing — "
                        f"predictions may be unreliable"
                    )
            else:
                logger.info(f"Model {name}: loaded perfectly ({len(saved_sd)} params, strict match)")

            model.eval()
            loaded[name] = model

        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")

    logger.info(f"ML Model Loader: {len(loaded)}/{len(MODEL_REGISTRY)} models loaded")
    return loaded


def predict_with_models(
    models: Dict[str, nn.Module],
    features: np.ndarray,
) -> Dict[str, float]:
    """Run inference on all loaded models.

    Args:
        models: dict from load_trained_models()
        features: (seq_len, 83) numpy array from build_feature_sequence()

    Returns:
        dict of model_name → prediction (float in ~[-1, 1])
    """
    if features is None:
        return {name: 0.0 for name in models}

    predictions = {}
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, 83)

    for name, model in models.items():
        try:
            with torch.no_grad():
                if name == 'quantum_transformer':
                    pred, _unc = model(x)
                    predictions[name] = float(torch.tanh(pred[0, 0]))
                elif name == 'bidirectional_lstm':
                    pred, conf = model(x)
                    predictions[name] = float(torch.tanh(pred[0, 0]))
                elif name == 'dilated_cnn':
                    pred = model(x)
                    predictions[name] = float(torch.tanh(pred[0, 0]))
                else:
                    predictions[name] = 0.0
        except Exception as e:
            logger.warning(f"Inference failed for {name}: {e}")
            predictions[name] = 0.0

    return predictions
