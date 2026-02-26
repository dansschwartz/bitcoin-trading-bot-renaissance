"""Standard data loading for all researchers. Pre-configured for our data format."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from knowledge._base import PAIRS, TRAINING_DIR, SNAPSHOT_DIR, get_db_connection, get_snapshot_db

def load_pair_csv(pair: str, source: str = "training", nrows: Optional[int] = None) -> pd.DataFrame:
    """Load 5-min OHLCV for a pair. Returns DataFrame with DatetimeIndex."""
    base = TRAINING_DIR if source == "training" else SNAPSHOT_DIR / "training_data"
    for pattern in [f"{pair}_5m.csv", f"{pair.replace('-','')}_5m.csv", f"{pair}.csv"]:
        path = base / pattern
        if path.exists():
            break
    else:
        available = [f.name for f in base.iterdir() if f.suffix == '.csv'] if base.exists() else []
        raise FileNotFoundError(f"No CSV for {pair} in {base}. Available: {available}")

    df = pd.read_csv(path, nrows=nrows)
    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ("timestamp", "time", "date", "datetime"): col_map[col] = "timestamp"
        elif lower in ("open", "o"): col_map[col] = "open"
        elif lower in ("high", "h"): col_map[col] = "high"
        elif lower in ("low", "l"): col_map[col] = "low"
        elif lower in ("close", "c", "price"): col_map[col] = "close"
        elif lower in ("volume", "vol", "v"): col_map[col] = "volume"
    df = df.rename(columns=col_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
    return df

def load_all_pairs(source: str = "training", nrows: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """Load all 6 pairs -> {"BTC-USD": df, ...}"""
    result = {}
    for pair in PAIRS:
        try:
            result[pair] = load_pair_csv(pair, source=source, nrows=nrows)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
    return result

def get_returns(df: pd.DataFrame, method: str = "log", periods: int = 1) -> pd.Series:
    """Compute returns from OHLCV DataFrame."""
    close = df["close"] if "close" in df.columns else df
    if method == "log":
        return np.log(close / close.shift(periods))
    return close.pct_change(periods)

def get_aligned_returns(pairs: Optional[List[str]] = None, source: str = "training",
                        nrows: Optional[int] = None) -> pd.DataFrame:
    """Returns for multiple pairs, aligned by timestamp. Columns = pair names."""
    pairs = pairs or PAIRS
    data = load_all_pairs(source=source, nrows=nrows)
    returns = {pair: get_returns(df) for pair, df in data.items() if pair in pairs}
    return pd.DataFrame(returns).dropna()
