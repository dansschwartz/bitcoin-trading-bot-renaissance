"""
ML Model Loader — loads trained PyTorch model weights into matching architectures.

The trained models (in models/trained/) were trained with specific hyperparameters
that differ from the current neural_network_prediction_engine.py defaults.
This module creates architectures that exactly match the saved weight shapes,
loads them with strict=True to guarantee 100% weight match, and exposes
a simple prediction interface.

Feature dimension: INPUT_DIM=98 (49 single-pair + 15 cross-asset, padded)
  - 49 single-pair features: OHLCV, returns, MAs, RSI, MACD, BB, ATR, volume
  - 15 cross-asset features: lead signals, correlations, spreads, market-wide
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

# ── Feature dimension constants ───────────────────────────────────────────────

INPUT_DIM = 98          # Current feature dimension (padded to 98)
INPUT_DIM_LEGACY = 83   # Previous feature dimension (for reference / weight loading)
N_CROSS_FEATURES = 15   # Number of cross-asset features
N_DERIVATIVES_FEATURES = 7  # funding_rate_z, oi_change_pct, long_short_ratio,
                            # taker_buy_sell_ratio, has_derivatives_data,
                            # fear_greed_norm, fear_greed_roc
# Total real features: 46 single-pair + 15 cross-asset + 7 derivatives = 68, padded to 98

# ── Cross-asset lead signal configuration ─────────────────────────────────────

LEAD_SIGNALS = {
    'BTC-USD':  {'primary': 'ETH-USD',  'secondary': 'SOL-USD'},
    'ETH-USD':  {'primary': 'BTC-USD',  'secondary': 'LINK-USD'},
    'SOL-USD':  {'primary': 'BTC-USD',  'secondary': 'ETH-USD'},
    'LINK-USD': {'primary': 'ETH-USD',  'secondary': 'BTC-USD'},
    'AVAX-USD': {'primary': 'ETH-USD',  'secondary': 'BTC-USD'},
    'DOGE-USD': {'primary': 'BTC-USD',  'secondary': 'ETH-USD'},
}

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

    input_dim=98, hidden=288, 4 blocks, 8 heads (qkv_dim=328), d_ff=1315
    Single output_head → scalar prediction.
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        d_model, n_heads, qkv_dim, d_ff, n_blocks = 288, 8, 328, 1315, 4

        self.input_projection = nn.Linear(input_dim, d_model)
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
        """x: (batch, seq_len, input_dim) → prediction (batch, 1), uncertainty (batch, 1)"""
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

    def __init__(self, input_size: int = INPUT_DIM, hidden_size: int = 292, num_layers: int = 2):
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

    input=98, hidden=292 per direction (584 total), 2 LSTM layers.
    prediction_head: BN(584)→Linear→BN→Linear→Linear→1
    confidence_head: Linear(584,73)→Linear(73,1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        bidir_dim = 584  # 292 * 2

        self.lstm = _TrainedLSTMCore(input_size=input_dim, hidden_size=292, num_layers=2)

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
        """x: (batch, seq_len, input_dim) → prediction (batch, 1), confidence (batch, 1)"""
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

    def __init__(self, channels: int = INPUT_DIM, hidden: int = 332, n_blocks: int = 5):
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

    channels=input_dim, 5 dilated blocks, hidden=332.
    classifier: BN(332)→Linear(332,166)→BN(166)→...→Linear(input_dim,1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        hidden = 332

        self.dilated_cnn = _TrainedDilatedCNNCore(channels=input_dim, hidden=hidden, n_blocks=5)

        # classifier: BN(332)→Linear→BN→...→Linear(input_dim, 1)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden),         # 0
            nn.GELU(),                      # 1
            nn.Linear(hidden, 166),         # 2
            nn.BatchNorm1d(166),            # 3
            nn.GELU(),                      # 4
            nn.Dropout(0.2),                # 5
            nn.Linear(166, input_dim),      # 6
            nn.BatchNorm1d(input_dim),      # 7
            nn.GELU(),                      # 8
            nn.Linear(input_dim, 1),        # 9
        )

        # pattern_strength: Linear(332→41) → ReLU → Linear(41→1)
        self.pattern_strength = nn.Sequential(
            nn.Linear(hidden, 41),          # 0
            nn.ReLU(),                      # 1
            nn.Linear(41, 1),              # 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)

        Transposes to (batch, input_dim, seq_len) for conv1d processing.
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        pooled = self.dilated_cnn(x)  # (batch, 332)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4: Simple CNN  (best_cnn_model.pth — fresh architecture for 83 features)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedCNN(nn.Module):
    """Simple Conv1d model for input_dim-feature input.

    4 conv layers with increasing then decreasing channels,
    global average pooling, and a small classifier head.
    Input: (batch, seq_len, input_dim)  Output: (batch, 1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)"""
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.conv_layers(x)  # (batch, 64, seq_len)
        x = x.mean(dim=-1)  # global avg pool → (batch, 64)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 5: Bidirectional GRU  (best_gru_model.pth — fresh architecture)
# ══════════════════════════════════════════════════════════════════════════════

class TrainedGRU(nn.Module):
    """Bidirectional GRU for input_dim-feature input.

    2-layer BiGRU with hidden_size=134 (268 total), plus a classifier head.
    Input: (batch, seq_len, input_dim)  Output: (batch, 1)
    """

    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=134,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        bidir_dim = 268  # 134 * 2
        self.classifier = nn.Sequential(
            nn.Linear(bidir_dim, 134),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(134, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) → prediction (batch, 1)"""
        out, _ = self.gru(x)  # (batch, seq_len, 268)
        pooled = out.mean(dim=1)  # (batch, 268)
        return self.classifier(pooled)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 6: Meta-Ensemble  (stacking layer over 5 base models)
# ══════════════════════════════════════════════════════════════════════════════

N_BASE_MODELS = 6  # QT, BiLSTM, DilatedCNN, CNN, GRU, LightGBM
BASE_MODEL_NAMES = [
    'quantum_transformer', 'bidirectional_lstm', 'dilated_cnn', 'cnn', 'gru', 'lightgbm',
]


class TrainedMetaEnsemble(nn.Module):
    """Meta-learning stacking layer that learns which base models to trust.

    Takes input_dim-dim market features + N base model predictions as input.
    Learns context-dependent model weighting: in trending markets, trust
    momentum models more; in choppy markets, trust mean-reversion models.

    Input: (batch, input_dim + n_models)
    Output: (prediction, confidence) tuple

    The n_models parameter auto-detects from saved weights when loading,
    supporting both 5-model (legacy) and 6-model (with LightGBM) ensembles.
    """

    def __init__(self, input_dim: int = INPUT_DIM, n_models: int = N_BASE_MODELS):
        super().__init__()
        self._input_dim = input_dim
        self._n_models = n_models

        # Feature extractor — understands market context
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),  # 0
            nn.BatchNorm1d(128),        # 1
            nn.GELU(),                  # 2
            nn.Dropout(0.2),            # 3
            nn.Linear(128, 64),         # 4
            nn.BatchNorm1d(64),         # 5
            nn.GELU(),                  # 6
        )

        # Weight generator — context-dependent per-model attention weights
        self.weight_generator = nn.Sequential(
            nn.Linear(64, 32),          # 0
            nn.GELU(),                  # 1
            nn.Linear(32, n_models),    # 2  → n_models weights
        )

        # Final predictor — combines context + base predictions
        self.final_predictor = nn.Sequential(
            nn.Linear(64 + n_models, 32),  # 0
            nn.GELU(),                     # 1
            nn.Dropout(0.1),               # 2
            nn.Linear(32, 1),              # 3
        )

        # Confidence estimator — how sure is the ensemble
        self.confidence_estimator = nn.Sequential(
            nn.Linear(64 + n_models, 16),  # 0
            nn.ReLU(),                     # 1
            nn.Linear(16, 1),              # 2
            nn.Sigmoid(),                  # 3
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (batch, input_dim + n_models) → prediction (batch, 1), confidence (batch, 1)"""
        features = x[:, :self._input_dim]
        model_preds = x[:, self._input_dim:]  # (batch, n_models)

        # Extract market context
        ctx = self.feature_extractor(features)  # (batch, 64)

        # Generate per-model attention weights
        weights = F.softmax(self.weight_generator(ctx), dim=-1)  # (batch, n_models)

        # Combine context + raw model predictions
        combined = torch.cat([ctx, model_preds], dim=-1)  # (batch, 64 + n_models)

        # Final prediction (learns both from weighted combo and direct features)
        pred = self.final_predictor(combined)  # (batch, 1)
        conf = self.confidence_estimator(combined)  # (batch, 1)

        return pred, conf


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — 46 scale-invariant single-pair + 15 cross-asset features
# No raw prices or raw volumes — all returns, ratios, z-scores, bounded indicators.
# Ensures identical feature distributions whether BTC is $3K or $130K.
# ══════════════════════════════════════════════════════════════════════════════


def _compute_single_pair_features(
    df: 'pd.DataFrame',
) -> Dict[str, 'pd.Series']:
    """Compute 46 scale-invariant single-pair features from OHLCV data.

    CRITICAL: No feature depends on absolute price level or raw volume.
    Every feature is a return, ratio, z-score, or bounded indicator.
    This ensures the model sees identical feature distributions whether
    BTC is $3K or $130K.

    Returns:
        Dict of feature_name → pd.Series (46 features)
    """
    close = df['close'].astype(float)
    _open = df['open'].astype(float) if 'open' in df.columns else close
    high = df['high'].astype(float) if 'high' in df.columns else close
    low = df['low'].astype(float) if 'low' in df.columns else close
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    features: Dict[str, 'pd.Series'] = {}

    # ── Group 1: Candle shape (5 features) — replaces raw OHLCV ──────────
    features['open_gap'] = np.log(_open / (close.shift(1) + 1e-10))
    features['upper_wick'] = (high - np.maximum(_open, close)) / (close + 1e-10)
    features['lower_wick'] = (np.minimum(_open, close) - low) / (close + 1e-10)
    features['body'] = (close - _open) / (close + 1e-10)
    # Volume: rolling z-score (100-bar window)
    if vol is not None:
        vol_mean = vol.rolling(100, min_periods=10).mean()
        vol_std = vol.rolling(100, min_periods=10).std()
        features['volume_z'] = (vol - vol_mean) / (vol_std + 1e-10)
    else:
        features['volume_z'] = close * 0.0

    # ── Group 2: Returns at multiple horizons (7 features) ────────────────
    for w in [1, 2, 3, 5, 10, 20]:
        features[f'ret_{w}'] = close.pct_change(w)
    features['log_ret'] = np.log(close / close.shift(1))

    # ── Group 3: SMA distance + slope (8 features) — replaces raw SMA ────
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        features[f'sma_dist_{w}'] = (close - sma) / (sma + 1e-10)
        features[f'sma_slope_{w}'] = sma.pct_change(3)

    # ── Group 4: EMA distance + slope (6 features) — replaces raw EMA ────
    for w in [5, 10, 20]:
        ema = close.ewm(span=w, adjust=False).mean()
        features[f'ema_dist_{w}'] = (close - ema) / (ema + 1e-10)
        features[f'ema_slope_{w}'] = ema.pct_change(3)

    # ── Group 5: Realized volatility (3 features) ────────────────────────
    pct_ret = close.pct_change()
    for w in [5, 10, 20]:
        features[f'vol_{w}'] = pct_ret.rolling(w).std()

    # ── Group 6: RSI, rescaled to [-1, 1] (1 feature) ────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss_s + 1e-10)
    features['rsi_norm'] = (100 - (100 / (1 + rs)) - 50) / 50

    # ── Group 7: MACD normalized by price (3 features) ────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    features['macd_pct'] = macd / (close + 1e-10)
    features['macd_signal_pct'] = macd_signal / (close + 1e-10)
    features['macd_hist_pct'] = (macd - macd_signal) / (close + 1e-10)

    # ── Group 8: Bollinger Bands — z-score + width (4 features) ──────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower + 1e-10
    features['bb_pct'] = (close - bb_lower) / bb_range
    features['bb_width'] = bb_range / (sma20 + 1e-10)
    features['bb_upper_dist'] = (bb_upper - close) / (close + 1e-10)
    features['bb_lower_dist'] = (close - bb_lower) / (close + 1e-10)

    # ── Group 9: ATR as % of price (1 feature) ───────────────────────────
    import pandas as pd
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    features['atr_pct'] = tr.rolling(14).mean() / (close + 1e-10)

    # ── Group 10: Volume ratios (3 features) ─────────────────────────────
    if vol is not None:
        features['vol_ratio'] = vol / (vol.rolling(10, min_periods=1).mean() + 1e-10)
        features['vol_change'] = vol.pct_change()
        features['vol_trend'] = (vol.rolling(5, min_periods=1).mean()
                                 / (vol.rolling(20, min_periods=1).mean() + 1e-10))
    else:
        features['vol_ratio'] = close * 0.0
        features['vol_change'] = close * 0.0
        features['vol_trend'] = close * 0.0 + 1.0

    # ── Group 11: Price momentum (3 features) ────────────────────────────
    features['momentum_5'] = close / close.shift(5) - 1
    features['momentum_10'] = close / close.shift(10) - 1
    features['momentum_20'] = close / close.shift(20) - 1

    # ── Group 12: Range features (2 features) ────────────────────────────
    features['hl_range'] = (high - low) / (close + 1e-10)
    hl_range = features['hl_range']
    features['hl_range_norm'] = hl_range / (hl_range.rolling(10, min_periods=1).mean() + 1e-10)

    # Total: 5 + 7 + 8 + 6 + 3 + 1 + 3 + 4 + 1 + 3 + 3 + 2 = 46
    return features


def _build_cross_features(
    close: 'pd.Series',
    volume: 'pd.Series',
    cross_data: Dict[str, 'pd.DataFrame'],
    pair_name: str,
) -> Dict[str, 'pd.Series']:
    """Compute 15 cross-asset features from other pairs' data.

    Features are computed using only past data (no future leakage).

    Args:
        close: This pair's close price series (aligned index with cross_data)
        volume: This pair's volume series
        cross_data: Dict of pair_name → DataFrame with at least [close, volume] columns
        pair_name: This pair's name (e.g. 'BTC-USD')

    Returns:
        Dict of feature_name → pd.Series (15 features)
    """
    import pandas as pd

    feats: Dict[str, pd.Series] = {}
    log_ret = np.log(close / close.shift(1))

    # ── Lead signals (5 features) ─────────────────────────────────────────
    lead_cfg = LEAD_SIGNALS.get(pair_name, {})
    primary_pair = lead_cfg.get('primary')
    secondary_pair = lead_cfg.get('secondary')

    for role, leader_pair, horizons in [
        ('primary', primary_pair, [1, 3, 6]),
        ('secondary', secondary_pair, [1, 3]),
    ]:
        if leader_pair and leader_pair in cross_data:
            leader_close = cross_data[leader_pair]['close'].astype(float)
            leader_log_ret = np.log(leader_close / leader_close.shift(1))
            for h in horizons:
                # h-bar log return of the leader
                feat_name = f'lead_{role}_ret_{h}'
                feats[feat_name] = leader_log_ret.rolling(h).sum()
        else:
            # Fill zeros for missing leader
            for h in ([1, 3, 6] if role == 'primary' else [1, 3]):
                feats[f'lead_{role}_ret_{h}'] = pd.Series(0.0, index=close.index)

    # ── Cross-pair correlations (4 features) ──────────────────────────────
    for ref_name, ref_label in [('BTC-USD', 'btc'), ('ETH-USD', 'eth')]:
        if ref_name in cross_data and ref_name != pair_name:
            ref_close = cross_data[ref_name]['close'].astype(float)
            ref_log_ret = np.log(ref_close / ref_close.shift(1))
            corr_50 = log_ret.rolling(50).corr(ref_log_ret)
            feats[f'corr_{ref_label}_50'] = corr_50
            # Z-score of correlation over 200-bar window
            corr_mean = corr_50.rolling(200).mean()
            corr_std = corr_50.rolling(200).std()
            feats[f'corr_z_{ref_label}'] = (corr_50 - corr_mean) / (corr_std + 1e-10)
        else:
            feats[f'corr_{ref_label}_50'] = pd.Series(0.0, index=close.index)
            feats[f'corr_z_{ref_label}'] = pd.Series(0.0, index=close.index)

    # ── Spread features (2 features) ──────────────────────────────────────
    for ref_name, ref_label in [('BTC-USD', 'btc'), ('ETH-USD', 'eth')]:
        if ref_name in cross_data and ref_name != pair_name:
            ref_close = cross_data[ref_name]['close'].astype(float)
            log_spread = np.log(close / (ref_close + 1e-10))
            spread_mean = log_spread.rolling(100).mean()
            spread_std = log_spread.rolling(100).std()
            feats[f'spread_{ref_label}_z'] = (log_spread - spread_mean) / (spread_std + 1e-10)
        else:
            feats[f'spread_{ref_label}_z'] = pd.Series(0.0, index=close.index)

    # ── Market-wide features (4 features, same for all pairs) ─────────────
    all_rets = []
    all_vol_zs = []
    for p, cdf in cross_data.items():
        c = cdf['close'].astype(float)
        r = np.log(c / c.shift(1))
        all_rets.append(r)
        if 'volume' in cdf.columns:
            v = cdf['volume'].astype(float)
            vol_ma = v.rolling(20).mean()
            all_vol_zs.append(v / (vol_ma + 1e-10) - 1.0)

    # Also include self
    all_rets.append(log_ret)
    if volume is not None:
        vol_ma_self = volume.rolling(20).mean()
        all_vol_zs.append(volume / (vol_ma_self + 1e-10) - 1.0)

    if all_rets:
        ret_df = pd.concat(all_rets, axis=1)
        feats['mkt_avg_ret'] = ret_df.mean(axis=1)
        feats['mkt_dispersion'] = ret_df.std(axis=1)
        feats['mkt_breadth'] = (ret_df > 0).mean(axis=1)
    else:
        feats['mkt_avg_ret'] = pd.Series(0.0, index=close.index)
        feats['mkt_dispersion'] = pd.Series(0.0, index=close.index)
        feats['mkt_breadth'] = pd.Series(0.5, index=close.index)

    if all_vol_zs:
        vol_df = pd.concat(all_vol_zs, axis=1)
        feats['mkt_avg_vol_z'] = vol_df.mean(axis=1)
    else:
        feats['mkt_avg_vol_z'] = pd.Series(0.0, index=close.index)

    return feats


def _build_derivatives_features(
    n_rows: int,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
) -> Dict[str, 'pd.Series']:
    """Compute 7 derivatives + sentiment features.

    Args:
        n_rows: Number of rows in the price DataFrame (for index alignment)
        derivatives_data: Dict with optional keys:
            - 'funding_rate': pd.Series of raw funding rates
            - 'open_interest': pd.Series of open interest values
            - 'long_short_ratio': pd.Series of long/short account ratios
            - 'taker_buy_vol': pd.Series of taker buy volume
            - 'taker_sell_vol': pd.Series of taker sell volume
            - 'fear_greed': pd.Series of Fear & Greed index (0-100)
            All series must be aligned to the same index as price_df.
            Missing keys → zeros with has_derivatives_data=0.

    Returns:
        Dict of 7 feature_name → pd.Series
    """
    import pandas as pd

    idx = pd.RangeIndex(n_rows)
    feats: Dict[str, pd.Series] = {}

    has_deriv = False

    if derivatives_data is not None:
        # ── Funding rate z-score (50-bar window) ────────────────────────
        fr = derivatives_data.get('funding_rate')
        if fr is not None and len(fr) > 0:
            fr = fr.astype(float)
            fr_mean = fr.rolling(50, min_periods=5).mean()
            fr_std = fr.rolling(50, min_periods=5).std()
            feats['funding_rate_z'] = (fr - fr_mean) / (fr_std + 1e-10)
            has_deriv = True
        else:
            feats['funding_rate_z'] = pd.Series(0.0, index=idx)

        # ── Open interest 5-bar % change ────────────────────────────────
        oi = derivatives_data.get('open_interest')
        if oi is not None and len(oi) > 0:
            oi = oi.astype(float)
            feats['oi_change_pct'] = oi.pct_change(5)
            has_deriv = True
        else:
            feats['oi_change_pct'] = pd.Series(0.0, index=idx)

        # ── Long/short ratio (raw, already scale-invariant) ─────────────
        ls = derivatives_data.get('long_short_ratio')
        if ls is not None and len(ls) > 0:
            feats['long_short_ratio'] = ls.astype(float)
            has_deriv = True
        else:
            feats['long_short_ratio'] = pd.Series(0.0, index=idx)

        # ── Taker buy/sell ratio ────────────────────────────────────────
        buy_vol = derivatives_data.get('taker_buy_vol')
        sell_vol = derivatives_data.get('taker_sell_vol')
        if buy_vol is not None and sell_vol is not None and len(buy_vol) > 0:
            bv = buy_vol.astype(float)
            sv = sell_vol.astype(float)
            feats['taker_buy_sell_ratio'] = bv / (sv + 1e-10)
            has_deriv = True
        else:
            feats['taker_buy_sell_ratio'] = pd.Series(0.0, index=idx)

        # ── Fear & Greed (normalized + 3-day ROC) ──────────────────────
        fg = derivatives_data.get('fear_greed')
        if fg is not None and len(fg) > 0:
            fg = fg.astype(float)
            feats['fear_greed_norm'] = fg / 100.0
            # 3-day ROC: for 5-min bars, 3 days = 3 * 288 = 864 bars
            # But FnG is daily (forward-filled), so diff(864) gives ~3-day change
            feats['fear_greed_roc'] = fg.diff(864) / 100.0
        else:
            feats['fear_greed_norm'] = pd.Series(0.0, index=idx)
            feats['fear_greed_roc'] = pd.Series(0.0, index=idx)

    else:
        # No derivatives data at all — fill zeros
        feats['funding_rate_z'] = pd.Series(0.0, index=idx)
        feats['oi_change_pct'] = pd.Series(0.0, index=idx)
        feats['long_short_ratio'] = pd.Series(0.0, index=idx)
        feats['taker_buy_sell_ratio'] = pd.Series(0.0, index=idx)
        feats['fear_greed_norm'] = pd.Series(0.0, index=idx)
        feats['fear_greed_roc'] = pd.Series(0.0, index=idx)

    # ── Binary flag: model knows when derivatives data is present ────
    feats['has_derivatives_data'] = pd.Series(
        1.0 if has_deriv else 0.0, index=idx
    )

    return feats


def build_feature_sequence(
    price_df,
    seq_len: int = 30,
    cross_data: Optional[Dict[str, 'pd.DataFrame']] = None,
    pair_name: Optional[str] = None,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
) -> Optional[np.ndarray]:
    """Build a (seq_len, INPUT_DIM) feature matrix from a price DataFrame.

    All features are scale-invariant: returns, ratios, z-scores, bounded
    indicators.  No raw prices or raw volumes.  Per-window standardization
    is applied on top so every sample has zero mean / unit variance.

    Args:
        price_df: DataFrame with OHLCV columns
        seq_len: Number of time steps in the output sequence
        cross_data: Optional dict of pair_name → DataFrame with [close, volume] columns.
        pair_name: This pair's identifier (e.g. 'BTC-USD').
        derivatives_data: Optional dict of feature_name → pd.Series for derivatives
            features (funding_rate, open_interest, long_short_ratio, taker_buy_vol,
            taker_sell_vol, fear_greed). Series must be aligned to price_df index.

    Returns:
        (seq_len, INPUT_DIM) float32 array, or None if insufficient data
    """
    import pandas as pd

    if price_df is None or len(price_df) < seq_len:
        return None

    df = price_df.tail(seq_len + 50).copy()

    # Trim derivatives_data to match
    if derivatives_data is not None:
        n_tail = len(df)
        derivatives_data = {
            k: v.tail(n_tail).reset_index(drop=True) if hasattr(v, 'tail') else v
            for k, v in derivatives_data.items()
        }

    if cross_data is not None:
        _cross = {}
        for p, cdf in cross_data.items():
            if p == pair_name:
                continue
            _cross[p] = cdf.tail(seq_len + 50).copy().reset_index(drop=True)
        cross_data = _cross if _cross else None

    df = df.reset_index(drop=True)
    close = df['close'].astype(float) if 'close' in df.columns else None
    if close is None:
        return None
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    # 46 scale-invariant single-pair features
    features = _compute_single_pair_features(df)

    # 15 cross-asset features (already scale-invariant)
    if cross_data is not None and pair_name is not None:
        cross_feats = _build_cross_features(close, vol, cross_data, pair_name)
        features.update(cross_feats)

    # 7 derivatives + sentiment features
    deriv_feats = _build_derivatives_features(len(df), derivatives_data)
    features.update(deriv_feats)

    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

    # Take last seq_len rows
    feat_arr = feat_df.tail(seq_len).values.astype(np.float32)

    # Per-window standardization (zero-mean, unit-variance per column)
    mean = feat_arr.mean(axis=0, keepdims=True)
    std = feat_arr.std(axis=0, keepdims=True) + 1e-8
    feat_arr = (feat_arr - mean) / std

    # Pad or truncate to exactly INPUT_DIM
    n_feat = feat_arr.shape[1]
    if n_feat < INPUT_DIM:
        pad = np.zeros((seq_len, INPUT_DIM - n_feat), dtype=np.float32)
        feat_arr = np.concatenate([feat_arr, pad], axis=1)
    elif n_feat > INPUT_DIM:
        feat_arr = feat_arr[:, :INPUT_DIM]

    return feat_arr  # (seq_len, INPUT_DIM)


def build_full_feature_matrix(
    price_df,
    cross_data: Optional[Dict[str, 'pd.DataFrame']] = None,
    pair_name: Optional[str] = None,
    derivatives_data: Optional[Dict[str, 'pd.Series']] = None,
) -> Optional[np.ndarray]:
    """Build a (N, INPUT_DIM) feature matrix for the ENTIRE DataFrame at once.

    Uses the same scale-invariant features as build_feature_sequence().
    Does NOT apply per-window standardization — that is done when slicing
    windows in generate_sequences().

    Args:
        price_df: DataFrame with OHLCV columns (entire history for one pair)
        cross_data: Optional dict of pair_name → DataFrame for cross-asset features
        pair_name: This pair's identifier
        derivatives_data: Optional dict of feature_name → pd.Series for derivatives
            features. Series must be aligned to price_df index.

    Returns:
        (N, INPUT_DIM) float32 array where N = len(price_df), or None
    """
    import pandas as pd

    if price_df is None or len(price_df) < 30:
        return None

    df = price_df.copy().reset_index(drop=True)

    if cross_data is not None:
        _cross = {p: cdf.copy().reset_index(drop=True)
                  for p, cdf in cross_data.items() if p != pair_name}
        cross_data = _cross if _cross else None

    # Reset derivatives_data index to match df
    if derivatives_data is not None:
        derivatives_data = {
            k: v.reset_index(drop=True) if hasattr(v, 'reset_index') else v
            for k, v in derivatives_data.items()
        }

    close = df['close'].astype(float) if 'close' in df.columns else None
    if close is None:
        return None
    vol = df['volume'].astype(float) if 'volume' in df.columns else None

    # 46 scale-invariant single-pair features
    features = _compute_single_pair_features(df)

    # 15 cross-asset features
    if cross_data is not None and pair_name is not None:
        cross_feats = _build_cross_features(close, vol, cross_data, pair_name)
        features.update(cross_feats)

    # 7 derivatives + sentiment features
    deriv_feats = _build_derivatives_features(len(df), derivatives_data)
    features.update(deriv_feats)

    feat_df = pd.DataFrame(features, index=df.index)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
    feat_arr = feat_df.values.astype(np.float32)

    # Pad or truncate to INPUT_DIM
    n_feat = feat_arr.shape[1]
    if n_feat < INPUT_DIM:
        pad = np.zeros((len(feat_arr), INPUT_DIM - n_feat), dtype=np.float32)
        feat_arr = np.concatenate([feat_arr, pad], axis=1)
    elif n_feat > INPUT_DIM:
        feat_arr = feat_arr[:, :INPUT_DIM]

    return feat_arr  # (N, INPUT_DIM)


# ══════════════════════════════════════════════════════════════════════════════
# LOADER — loads weights, validates, returns ready-to-use models
# ══════════════════════════════════════════════════════════════════════════════


def _detect_input_dim(model_name: str, state_dict: dict) -> Optional[int]:
    """Auto-detect the input dimension a saved model was trained with.

    Inspects the first layer's weight shape to infer the original input_dim.
    Returns None if detection fails (caller should use default).
    """
    try:
        # Each model type has a different "first layer" key
        key_map = {
            'quantum_transformer': 'input_projection.weight',
            'bidirectional_lstm': 'lstm.lstm_layers.0.weight_ih_l0',
            'dilated_cnn': 'dilated_cnn.conv_blocks.0.0.weight',
            'cnn': 'conv_layers.0.weight',
            'gru': 'gru.weight_ih_l0',
            'meta_ensemble': 'feature_extractor.0.weight',
        }
        key = key_map.get(model_name)
        if key and key in state_dict:
            w = state_dict[key]
            if model_name == 'quantum_transformer':
                # input_projection: Linear(input_dim, d_model) → weight shape (d_model, input_dim)
                return w.shape[1]
            elif model_name == 'bidirectional_lstm':
                # LSTM weight_ih: shape (4*hidden, input_size)
                return w.shape[1]
            elif model_name == 'dilated_cnn':
                # Conv1d(channels, channels, 3) → weight shape (channels, channels, 3)
                return w.shape[0]
            elif model_name == 'cnn':
                # Conv1d(input_dim, 128, 3) → weight shape (128, input_dim, 3)
                return w.shape[1]
            elif model_name == 'gru':
                # GRU weight_ih: shape (3*hidden, input_size)
                return w.shape[1]
            elif model_name == 'meta_ensemble':
                # Linear(input_dim, 128) → weight shape (128, input_dim)
                return w.shape[1]
    except Exception:
        pass
    return None


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
    'cnn': (
        'models/trained/best_cnn_model.pth',
        TrainedCNN,
    ),
    'gru': (
        'models/trained/best_gru_model.pth',
        TrainedGRU,
    ),
    'meta_ensemble': (
        'models/trained/best_meta_ensemble_model.pth',
        TrainedMetaEnsemble,
    ),
}


def load_trained_models(base_dir: str = '.', input_dim: int = INPUT_DIM) -> Dict[str, nn.Module]:
    """Load all trained models with strict validation.

    Args:
        base_dir: Base directory for model weight files
        input_dim: Feature dimension for model constructors (default INPUT_DIM=98).
                   Use INPUT_DIM_LEGACY=83 to load old pre-trained weights exactly.

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

            # Auto-detect input_dim from saved weights to handle legacy models
            detected_dim = _detect_input_dim(name, saved_sd)
            use_dim = detected_dim if detected_dim is not None else input_dim

            # For meta_ensemble, detect n_models from saved weight_generator shape
            if name == 'meta_ensemble':
                wg_key = 'weight_generator.2.weight'
                detected_n_models = saved_sd[wg_key].shape[0] if wg_key in saved_sd else N_BASE_MODELS
                model = model_cls(input_dim=use_dim, n_models=detected_n_models)
                if detected_n_models != N_BASE_MODELS:
                    logger.info(f"Meta-ensemble: detected {detected_n_models} base models from saved weights "
                                f"(current N_BASE_MODELS={N_BASE_MODELS})")
            else:
                model = model_cls(input_dim=use_dim)
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

    # Also load LightGBM if available (non-PyTorch, separate path)
    lgbm = load_lightgbm_model(base_dir)
    if lgbm is not None:
        loaded['lightgbm'] = lgbm

    return loaded


def predict_with_models(
    models: Dict[str, nn.Module],
    features: np.ndarray,
    price_series: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Run inference on all loaded models (5 DL + LightGBM + meta-ensemble).

    Args:
        models: dict from load_trained_models()
        features: (seq_len, INPUT_DIM) numpy array from build_feature_sequence()
        price_series: Optional close prices (unused, kept for API compat)

    Returns:
        (predictions, confidences) where:
          predictions: dict of model_name → prediction (float in ~[-1, 1])
          confidences: dict of model_name → confidence (float in [0, 1])
            For deep learning models, confidence = abs(prediction) (higher signal = more confident).
            For LightGBM, confidence = distance from 0.5 probability (calibrated).
            For meta_ensemble, confidence comes from the confidence head.
    """
    if features is None:
        empty = {name: 0.0 for name in models}
        return empty, {name: 0.5 for name in models}

    predictions = {}
    confidences = {}
    x = torch.FloatTensor(features).unsqueeze(0)  # (1, seq_len, INPUT_DIM)

    # Run base models first (all take sequence input)
    for name, model in models.items():
        if name in ('meta_ensemble', 'lightgbm'):
            continue  # Handle separately
        try:
            with torch.no_grad():
                # Match feature dim to model's expected input
                model_x = _match_feature_dim(x, model, name)
                output = model(model_x)
                if isinstance(output, tuple):
                    pred = output[0]  # (prediction, uncertainty/confidence)
                else:
                    pred = output
                pred_val = float(torch.tanh(pred[0, 0]))
                predictions[name] = pred_val
                confidences[name] = min(abs(pred_val) + 0.5, 0.95)
        except Exception as e:
            logger.warning(f"Inference failed for {name}: {e}")
            predictions[name] = 0.0
            confidences[name] = 0.0

    # Run LightGBM if loaded (non-PyTorch, uses flattened features + momentum)
    if 'lightgbm' in models:
        lgbm_pred, lgbm_conf = predict_lightgbm(
            models['lightgbm'], features, price_series
        )
        predictions['lightgbm'] = lgbm_pred
        confidences['lightgbm'] = lgbm_conf

    # Run meta-ensemble if loaded (uses base model predictions + features)
    if 'meta_ensemble' in models:
        try:
            with torch.no_grad():
                meta_model = models['meta_ensemble']
                meta_dim = meta_model._input_dim
                n_models = meta_model._n_models

                # Build base model predictions list matching what the meta-ensemble expects
                # If meta-ensemble was trained with 5 models, only pass the 5 DL predictions
                # If trained with 6, include LightGBM
                if n_models >= N_BASE_MODELS:
                    # New 6-model meta-ensemble: pass all including lightgbm
                    base_pred_list = [predictions.get(name, 0.0) for name in BASE_MODEL_NAMES]
                else:
                    # Legacy 5-model meta-ensemble: pass only DL predictions
                    base_pred_list = [
                        predictions.get(name, 0.0)
                        for name in BASE_MODEL_NAMES if name != 'lightgbm'
                    ]

                # Ensure we have exactly n_models predictions
                while len(base_pred_list) < n_models:
                    base_pred_list.append(0.0)
                base_pred_list = base_pred_list[:n_models]

                feat_vec = torch.FloatTensor(features[-1, :meta_dim]).unsqueeze(0)  # (1, meta_dim)
                base_preds = torch.FloatTensor([base_pred_list])  # (1, n_models)
                meta_input = torch.cat([feat_vec, base_preds], dim=-1)  # (1, meta_dim + n_models)
                pred, conf = meta_model(meta_input)
                predictions['meta_ensemble'] = float(torch.tanh(pred[0, 0]))
                confidences['meta_ensemble'] = float(conf[0, 0])
        except Exception as e:
            logger.warning(f"Inference failed for meta_ensemble: {e}")
            predictions['meta_ensemble'] = 0.0
            confidences['meta_ensemble'] = 0.0

    return predictions, confidences


def _match_feature_dim(
    x: torch.Tensor,
    model: nn.Module,
    model_name: str,
) -> torch.Tensor:
    """Pad or truncate feature tensor to match model's expected input dimension.

    This handles the case where a model was trained with INPUT_DIM_LEGACY=83
    but features are now INPUT_DIM=98, or vice versa.
    """
    feat_dim = x.shape[-1]

    # Detect model's expected input dim from its first layer
    expected_dim = None
    try:
        if model_name == 'quantum_transformer':
            expected_dim = model.input_projection.in_features
        elif model_name == 'bidirectional_lstm':
            expected_dim = model.lstm.lstm_layers[0].input_size
        elif model_name == 'dilated_cnn':
            expected_dim = model.dilated_cnn.conv_blocks[0]._modules['0'].in_channels
        elif model_name == 'cnn':
            expected_dim = model.conv_layers[0].in_channels
        elif model_name == 'gru':
            expected_dim = model.gru.input_size
    except Exception:
        pass

    if expected_dim is None or expected_dim == feat_dim:
        return x

    if feat_dim > expected_dim:
        # Truncate extra features
        return x[:, :, :expected_dim]
    else:
        # Pad with zeros
        pad = torch.zeros(x.shape[0], x.shape[1], expected_dim - feat_dim)
        return torch.cat([x, pad], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# LIGHTGBM — gradient boosting model (non-PyTorch)
# Trained in Colab with [last, mean, std] feature preparation → 294 features.
# Trees find interaction effects and threshold rules that neural nets miss,
# structurally diversifying the ensemble.
# ══════════════════════════════════════════════════════════════════════════════

LIGHTGBM_PKL_PATH = os.path.join('models', 'trained', 'best_lightgbm_model.pkl')
LIGHTGBM_TXT_PATH = os.path.join('models', 'trained', 'lightgbm_model.txt')
LIGHTGBM_META_PATH = os.path.join('models', 'trained', 'lightgbm_meta.json')


def load_lightgbm_model(base_dir: str = '.') -> Optional[object]:
    """Load a trained LightGBM model (pickle or Booster .txt format).

    Tries pickle format first (from Colab training), falls back to .txt Booster.
    Returns the model object, or None if not found/not installed.
    """
    import json as _json
    import pickle as _pickle

    # Try pickle format first (Colab-trained model)
    pkl_path = os.path.join(base_dir, LIGHTGBM_PKL_PATH)
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                model = _pickle.load(f)
            # Load metadata if available
            meta_path = os.path.join(base_dir, LIGHTGBM_META_PATH)
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = _json.load(f)
            logger.info(f"LightGBM model loaded (pkl): {pkl_path} "
                        f"(best_iter={meta.get('best_iteration', '?')}, "
                        f"test_acc={meta.get('test_accuracy', '?')})")
            return model
        except Exception as e:
            logger.error(f"Failed to load LightGBM from pkl: {e}")

    # Fallback: .txt Booster format
    txt_path = os.path.join(base_dir, LIGHTGBM_TXT_PATH)
    if os.path.exists(txt_path):
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=txt_path)
            logger.info(f"LightGBM model loaded (txt): {txt_path} "
                        f"({model.num_trees()} trees, {model.num_feature()} features)")
            return model
        except ImportError:
            logger.debug("LightGBM not installed — skipping lightgbm model")
            return None
        except Exception as e:
            logger.error(f"Failed to load LightGBM from txt: {e}")
            return None

    logger.debug(f"LightGBM model not found at {pkl_path} or {txt_path}")
    return None


def _prepare_lgb_features(features: np.ndarray) -> np.ndarray:
    """Convert (seq_len, INPUT_DIM) sequence to (1, INPUT_DIM*3) for LightGBM.

    Concatenates:
    - Last timestep features (most recent market state)
    - Mean across window (trend context)
    - Std across window (volatility context)

    This matches the training pipeline's prepare_lgb_features().
    """
    last = features[-1, :]           # (INPUT_DIM,) — most recent bar
    mean = features.mean(axis=0)     # (INPUT_DIM,) — window average
    std = features.std(axis=0)       # (INPUT_DIM,) — window volatility
    return np.concatenate([last, mean, std]).reshape(1, -1)  # (1, INPUT_DIM*3)


def predict_lightgbm(
    model: object,
    features: Optional[np.ndarray],
    price_series: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Run inference on LightGBM model.

    Args:
        model: LightGBM model from load_lightgbm_model()
        features: (seq_len, INPUT_DIM) array from build_feature_sequence()
        price_series: Unused (kept for API compatibility)

    Returns:
        (prediction, confidence) where:
          prediction: float in [-1, 1] (mapped from probability)
          confidence: float in [0, 1] (distance from 0.5, doubled)
    """
    if features is None or model is None:
        return 0.0, 0.0

    try:
        # Flatten sequence to LightGBM format: [last, mean, std]
        lgb_input = _prepare_lgb_features(features)  # (1, INPUT_DIM*3)
        lgb_input = np.nan_to_num(lgb_input, nan=0.0, posinf=0.0, neginf=0.0)

        # Auto-detect input format: if model expects more features than we have,
        # pad with zeros; if fewer, truncate (handles legacy vs new format)
        expected_features = None
        if hasattr(model, 'num_feature'):
            expected_features = model.num_feature()
        elif hasattr(model, 'n_features_'):
            expected_features = model.n_features_

        if expected_features is not None:
            n_have = lgb_input.shape[1]
            if n_have < expected_features:
                pad = np.zeros((1, expected_features - n_have), dtype=np.float32)
                lgb_input = np.concatenate([lgb_input, pad], axis=1)
            elif n_have > expected_features:
                lgb_input = lgb_input[:, :expected_features]

        # LightGBM predict returns probability of class 1 (up)
        prob = float(model.predict(lgb_input)[0])

        # Map probability [0, 1] → prediction [-1, 1]
        prediction = float(np.clip((prob - 0.5) * 2.0, -1.0, 1.0))

        # Confidence: how far from uncertain (0.5)
        confidence = abs(prob - 0.5) * 2.0  # 0.5→0, 0.7→0.4, 1.0→1.0

        return prediction, confidence

    except Exception as e:
        logger.warning(f"LightGBM inference failed: {e}")
        return 0.0, 0.0
