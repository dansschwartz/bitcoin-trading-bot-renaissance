"""
Shared constants and configuration — SINGLE SOURCE OF TRUTH.
Every researcher imports from here. Never hardcode system-specific values.
"""
import os
from pathlib import Path
from typing import Dict, List

# ── SYSTEM TOPOLOGY ──
PROJECT_ROOT = Path(os.environ.get(
    "RENAISSANCE_ROOT",
    os.path.expanduser("~/Downloads/bitcoin-trading-bot-renaissance")
))
DB_PATH = PROJECT_ROOT / "data" / "trading.db"
TRAINING_DIR = PROJECT_ROOT / "data" / "training"
MODEL_DIR = PROJECT_ROOT / "models" / "trained"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "research_snapshots" / "latest"

# ── TRADING UNIVERSE ──
PAIRS: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD", "AVAX-USD", "LINK-USD"]

PAIR_TO_BINANCE: Dict[str, str] = {
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
    "DOGE-USD": "DOGEUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT",
}

LEAD_LAG: Dict[str, Dict[str, str]] = {
    "BTC-USD":  {"primary": "ETH-USD",  "secondary": "SOL-USD"},
    "ETH-USD":  {"primary": "BTC-USD",  "secondary": "LINK-USD"},
    "SOL-USD":  {"primary": "BTC-USD",  "secondary": "ETH-USD"},
    "LINK-USD": {"primary": "ETH-USD",  "secondary": "BTC-USD"},
    "AVAX-USD": {"primary": "ETH-USD",  "secondary": "BTC-USD"},
    "DOGE-USD": {"primary": "BTC-USD",  "secondary": "ETH-USD"},
}

# ── TIME CONSTANTS ──
BAR_MINUTES = 5
BARS_PER_HOUR = 12
BARS_PER_DAY = 288
BARS_PER_WEEK = 2016
LOOKAHEAD_BARS = 6
FUNDING_PERIOD_HOURS = 8
FUNDING_BARS = 96

# ── FEATURE PIPELINE ──
FEATURE_DIM = 98
SEQUENCE_LENGTH = 60
FEATURE_GROUPS = {
    "candle_shape": (0, 5), "returns": (5, 12), "sma": (12, 20),
    "ema": (20, 26), "technical": (26, 31), "volatility": (31, 35),
    "volume": (35, 40), "momentum": (40, 43), "microstructure": (43, 46),
    "cross_asset": (46, 61), "derivatives": (61, 68), "padding": (68, 98),
}

# ── MODEL REGISTRY ──
MODELS = {
    "quantum_transformer": {"params": 4_600_000, "type": "attention", "file": "quantum_transformer.pth"},
    "bilstm":              {"params":   800_000, "type": "rnn",       "file": "bilstm.pth"},
    "dilated_cnn":         {"params":   500_000, "type": "cnn",       "file": "dilated_cnn.pth"},
    "cnn":                 {"params":   300_000, "type": "cnn",       "file": "cnn.pth"},
    "gru":                 {"params":   600_000, "type": "rnn",       "file": "gru.pth"},
    "meta_ensemble":       {"params":   100_000, "type": "stacker",   "file": "meta_ensemble.pth"},
    "vae":                 {"params":   400_000, "type": "vae",       "file": "vae.pth"},
    "lightgbm":            {"params":     "N/A", "type": "gbm",       "file": "best_lightgbm_model.pkl"},
}

# ── SAFETY LIMITS (IMMUTABLE) ──
ABSOLUTE_MAX_POSITION_USD = 10_000
ABSOLUTE_MAX_DRAWDOWN_PCT = 0.10
ABSOLUTE_MAX_LEVERAGE = 5.0
ABSOLUTE_MAX_DAILY_TRADES = 200
MIN_CIRCUIT_BREAKER_DRAWDOWN = 0.03

# ── FEES ──
FEES = {
    "mexc":    {"maker_bps": 0.0, "taker_bps": 1.0},
    "binance": {"maker_bps": 2.0, "taker_bps": 4.0},
}

# ── TARGETS ──
TARGET_WIN_RATE = 0.53
TARGET_SHARPE = 0.5
TARGET_MAX_DRAWDOWN = 0.05
MIN_TRADES_FOR_SIGNIFICANCE = 500
MEDALLION_WIN_RATE = 0.5075
MEDALLION_DEVIL_RATIO = 0.50

# ── DB HELPERS ──
def get_db_connection(readonly: bool = True):
    import sqlite3
    # Try multiple DB names
    for name in ["trading.db", "renaissance_bot.db"]:
        path = PROJECT_ROOT / "data" / name
        if path.exists():
            uri = f"file:{path}?mode=ro" if readonly else str(path)
            conn = sqlite3.connect(uri, uri=readonly, timeout=10.0)
            conn.row_factory = sqlite3.Row
            return conn
    raise FileNotFoundError(f"No database found in {PROJECT_ROOT / 'data'}")

def get_snapshot_db():
    import sqlite3
    snap_db = SNAPSHOT_DIR / "trading_snapshot.db"
    if not snap_db.exists():
        raise FileNotFoundError(f"No snapshot at {snap_db}")
    conn = sqlite3.connect(f"file:{snap_db}?mode=ro", uri=True, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn
