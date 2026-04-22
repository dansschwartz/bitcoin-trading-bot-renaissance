"""
4-Hour Neural Network Oracle
=============================
Runs every 4 hours aligned to candle boundaries (00/04/08/12/16/20 UTC).
Fetches latest 4H candle from Binance, computes 36 features, runs ensemble
of 6 MLP models, outputs Buy/Hold/Sell signal to the database.

Any strategy can read the signal via get_latest_signal(asset).
"""

import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pickle
import requests

logger = logging.getLogger('oracle')

# ── Configuration ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCALER_PATH = str(PROJECT_ROOT / 'models' / 'oracle' / 'oracle_scaler.pkl')
MODEL_DIR = str(PROJECT_ROOT / 'models' / 'oracle')
DEFAULT_DB_PATH = str(PROJECT_ROOT / 'data' / 'renaissance_bot.db')

MODELS = [
    {'file': 'model_final_2_1.h5', 'bw': 2, 'fw': 1, 'weight': 0.10},
    {'file': 'model_final_2_2.h5', 'bw': 2, 'fw': 2, 'weight': 0.20},
    {'file': 'model_final_3_2.h5', 'bw': 3, 'fw': 2, 'weight': 0.15},
    {'file': 'model_final_4_2.h5', 'bw': 4, 'fw': 2, 'weight': 0.25},
    {'file': 'model_final_5_1.h5', 'bw': 5, 'fw': 1, 'weight': 0.10},
    {'file': 'model_final_5_2.h5', 'bw': 5, 'fw': 2, 'weight': 0.20},
]

ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

SIGNAL_MAP = {-1: 'BUY', 0: 'HOLD', 1: 'SELL'}

FEATURE_COLS = [
    'Z_score', 'RSI', 'boll', 'ULTOSC', 'pct_change', 'zsVol',
    'PR_MA_Ratio_short', 'MA_Ratio_short', 'MA_Ratio', 'PR_MA_Ratio',
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3WHITESOLDIERS',
    'CDLABANDONEDBABY', 'CDLBELTHOLD', 'CDLCOUNTERATTACK',
    'CDLDARKCLOUDCOVER', 'CDLDRAGONFLYDOJI', 'CDLENGULFING',
    'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGRAVESTONEDOJI',
    'CDLHANGINGMAN', 'CDLHARAMICROSS', 'CDLINVERTEDHAMMER',
    'CDLMARUBOZU', 'CDLMORNINGDOJISTAR', 'CDLMORNINGSTAR',
    'CDLPIERCING', 'CDLRISEFALL3METHODS', 'CDLSHOOTINGSTAR',
    'CDLSPINNINGTOP', 'CDLUPSIDEGAP2CROWS',
    'DayOfWeek', 'Month', 'Hourly',
]

MIN_CANDLES_NEEDED = 120


class OracleService:
    """4-hour neural network oracle — shared signal service.

    Models are loaded lazily during prediction and unloaded after
    to save memory on constrained VPS (2GB RAM).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()
        self._load_scaler()
        # Count available models but DON'T load them yet (saves ~500MB RAM)
        self.model_count = sum(
            1 for m in MODELS
            if os.path.exists(os.path.join(MODEL_DIR, m['file']))
        )
        if self.model_count == 0:
            raise RuntimeError("No oracle model files found!")
        logger.info(f"Oracle: {self.model_count} model files available (lazy-load)")

    def _init_db(self) -> None:
        """Create oracle tables."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oracle_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                asset TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL,
                model_votes TEXT,
                candle_close REAL,
                candle_time TEXT,
                features_json TEXT,
                UNIQUE(asset, candle_time)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oracle_asset_time
            ON oracle_signals(asset, timestamp DESC)
        """)
        conn.commit()
        conn.close()
        logger.info("Oracle DB tables initialized")

    def _load_scaler(self) -> None:
        """Load the fitted StandardScaler."""
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Oracle scaler not found: {SCALER_PATH}")
        with open(SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        logger.info(f"Oracle scaler loaded: {len(self.scaler.mean_)} features")

    def _load_models_for_prediction(self) -> List[Dict[str, Any]]:
        """Load all pretrained Keras models (called only during prediction)."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import keras

        loaded: List[Dict[str, Any]] = []
        for m in MODELS:
            path = os.path.join(MODEL_DIR, m['file'])
            if os.path.exists(path):
                model = keras.models.load_model(path)
                loaded.append({
                    'model': model,
                    'weight': m['weight'],
                    'name': m['file'],
                    'bw': m['bw'],
                    'fw': m['fw'],
                })
            else:
                logger.warning(f"Oracle model not found: {path}")

        if not loaded:
            raise RuntimeError("No oracle models loaded!")
        logger.info(f"Oracle: {len(loaded)} models loaded for prediction")
        return loaded

    @staticmethod
    def _unload_models() -> None:
        """Free Keras/TF memory after prediction."""
        try:
            import keras
            keras.backend.clear_session()
        except Exception as e:
            logger.warning(f"Failed: import keras: {e}")
        import gc
        gc.collect()

    # ── Data fetching ─────────────────────────────────────────────────────────

    def _fetch_recent_candles(self, symbol: str,
                              count: int = MIN_CANDLES_NEEDED
                              ) -> Optional[pd.DataFrame]:
        """Fetch recent 4H candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol, 'interval': '4h', 'limit': count}
        try:
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Binance API error: {resp.status_code}")
                return None

            data = resp.json()
            df = pd.DataFrame(data).iloc[:, :6]
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
            df["Date"] = pd.to_datetime(df["Date"], unit="ms")
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col])
            df['Asset_name'] = symbol
            return df
        except Exception as e:
            logger.error(f"Failed to fetch candles for {symbol}: {e}")
            return None

    # ── Feature computation ───────────────────────────────────────────────────

    def _compute_features(self, df: pd.DataFrame):
        """Compute all 36 features using the paper's exact pipeline."""
        from oracle.technical_analysis_lib import TecnicalAnalysis

        data = df.copy()
        data = TecnicalAnalysis.compute_oscillators(data)
        data = TecnicalAnalysis.find_patterns(data)
        data = TecnicalAnalysis.add_timely_data(data)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)

        if len(data) == 0:
            return None

        latest = data.iloc[[-1]]
        features = latest[FEATURE_COLS].values
        features_scaled = self.scaler.transform(features)
        return features_scaled, latest

    # ── Prediction ────────────────────────────────────────────────────────────

    def _predict_ensemble(self, features_scaled: np.ndarray,
                          models: List[Dict[str, Any]]):
        """
        Run all models and produce weighted ensemble prediction.
        Each model outputs class probabilities for 3 classes.
        Class mapping: 0→BUY(-1), 1→HOLD(0), 2→SELL(1)

        Uses weighted probability averaging (not hard argmax) so that
        moderate BUY/SELL probabilities across models can accumulate
        into actionable signals instead of always losing to HOLD.
        """
        # Weighted average of raw probabilities across all models
        avg_probs = np.zeros(3)  # [BUY, HOLD, SELL]
        total_weight = 0.0
        individual_votes = []

        for m in models:
            raw_pred = m['model'].predict(features_scaled, verbose=0)
            probs = raw_pred[0]  # shape (3,)
            avg_probs += probs * m['weight']
            total_weight += m['weight']

            # Individual model vote (for logging)
            pred_class = int(np.argmax(probs)) - 1
            signal = SIGNAL_MAP[pred_class]
            individual_votes.append({
                'model': m['name'],
                'signal': signal,
                'weight': m['weight'],
                'probs': [round(float(p), 4) for p in probs],
            })

        if total_weight > 0:
            avg_probs /= total_weight

        # Pick signal from averaged probabilities
        # avg_probs[0]=BUY, avg_probs[1]=HOLD, avg_probs[2]=SELL
        ensemble_class = int(np.argmax(avg_probs)) - 1
        winner = SIGNAL_MAP[ensemble_class]
        confidence = float(avg_probs[np.argmax(avg_probs)])

        logger.info(
            f"Oracle ensemble probs: BUY={avg_probs[0]:.3f} "
            f"HOLD={avg_probs[1]:.3f} SELL={avg_probs[2]:.3f} "
            f"-> {winner} ({confidence:.1%})"
        )
        return winner, confidence, individual_votes

    # ── Main prediction ──────────────────────────────────────────────────────

    def predict_now(self) -> Dict[str, Dict[str, Any]]:
        """Run prediction for all assets right now.

        Loads models into memory, runs predictions, then unloads
        to keep memory usage low between 4-hour prediction cycles.
        """
        results = {}

        # Fetch data + compute features first (before loading heavy models)
        asset_features = {}
        for symbol in ASSETS:
            logger.info(f"Oracle fetching data for {symbol}...")
            df = self._fetch_recent_candles(symbol)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for {symbol}")
                continue
            result = self._compute_features(df)
            if result is None:
                logger.warning(f"Feature computation failed for {symbol}")
                continue
            asset_features[symbol] = result

        if not asset_features:
            logger.warning("No assets had sufficient data for prediction")
            return results

        # Load models, predict, then immediately unload
        try:
            models = self._load_models_for_prediction()
        except Exception as e:
            logger.error(f"Failed to load oracle models: {e}")
            return results

        conn = sqlite3.connect(self.db_path, timeout=30)
        try:
            for symbol, (features_scaled, latest_row) in asset_features.items():
                signal, confidence, votes = self._predict_ensemble(
                    features_scaled, models)
                candle_close = float(latest_row['Close'].iloc[0])
                candle_time = str(latest_row['Date'].iloc[0])

                try:
                    conn.execute("""
                        INSERT INTO oracle_signals
                        (timestamp, asset, signal, confidence, model_votes,
                         candle_close, candle_time, features_json)
                        VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(asset, candle_time) DO UPDATE SET
                            signal = excluded.signal,
                            confidence = excluded.confidence,
                            model_votes = excluded.model_votes,
                            timestamp = excluded.timestamp
                    """, (
                        symbol, signal, confidence,
                        json.dumps(votes),
                        candle_close, candle_time,
                        json.dumps(dict(zip(FEATURE_COLS,
                                            features_scaled[0].tolist()))),
                    ))
                    conn.commit()
                except Exception as e:
                    logger.error(f"DB write failed: {e}")

                results[symbol] = {
                    'signal': signal,
                    'confidence': confidence,
                    'votes': votes,
                    'candle_close': candle_close,
                    'candle_time': candle_time,
                }

                logger.info(
                    f"ORACLE[{symbol}] -> {signal} "
                    f"(confidence={confidence:.0%}) "
                    f"close=${candle_close:,.2f} "
                    f"candle={candle_time}"
                )
        finally:
            conn.close()
            # Free Keras/TF memory after prediction
            self._unload_models()
            logger.info("Oracle models unloaded to free memory")

        return results

    # ── Public API ────────────────────────────────────────────────────────────

    def get_latest_signal(self, asset: str) -> Dict[str, Any]:
        """
        Get the most recent oracle signal for an asset.
        Called by straddle engine, arb module, polymarket, etc.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        row = conn.execute("""
            SELECT signal, confidence, candle_close, timestamp
            FROM oracle_signals
            WHERE asset = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (asset,)).fetchone()
        conn.close()

        if not row:
            return {'signal': 'HOLD', 'confidence': 0.0,
                    'candle_close': 0, 'age_minutes': 999}

        try:
            signal_time = datetime.fromisoformat(row[3])
            if signal_time.tzinfo is None:
                signal_time = signal_time.replace(tzinfo=timezone.utc)
            age_minutes = (datetime.now(timezone.utc) - signal_time
                          ).total_seconds() / 60
        except Exception:
            age_minutes = 999

        return {
            'signal': row[0],
            'confidence': row[1],
            'candle_close': row[2],
            'age_minutes': round(age_minutes, 1),
        }

    def get_signal_history(self, asset: str, hours: int = 48
                          ) -> List[Dict[str, Any]]:
        """Get recent signal history for dashboard."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        rows = conn.execute("""
            SELECT timestamp, signal, confidence, candle_close, model_votes
            FROM oracle_signals
            WHERE asset = ? AND timestamp > datetime('now', ? || ' hours')
            ORDER BY timestamp DESC
        """, (asset, f"-{hours}")).fetchall()
        conn.close()

        return [{
            'timestamp': r[0], 'signal': r[1],
            'confidence': r[2], 'candle_close': r[3],
            'model_votes': json.loads(r[4]) if r[4] else [],
        } for r in rows]

    def get_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get latest signal for all assets (for dashboard)."""
        signals = {}
        for asset in ASSETS:
            signals[asset] = {
                'current': self.get_latest_signal(asset),
                'history': self.get_signal_history(asset, hours=48),
            }
        return signals

    # ── Scheduling ────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Run prediction aligned to 4-hour candle boundaries."""
        import asyncio

        while True:
            now = datetime.now(timezone.utc)
            hour = now.hour
            next_boundary_hour = ((hour // 4) + 1) * 4
            if next_boundary_hour >= 24:
                next_boundary = now.replace(
                    hour=0, minute=0, second=30, microsecond=0
                ) + timedelta(days=1)
            else:
                next_boundary = now.replace(
                    hour=next_boundary_hour, minute=0, second=30,
                    microsecond=0
                )

            wait_seconds = (next_boundary - now).total_seconds()
            if wait_seconds > 0:
                logger.info(
                    f"Oracle waiting {wait_seconds/60:.1f}min until next "
                    f"4H boundary ({next_boundary.strftime('%H:%M UTC')})"
                )
                await asyncio.sleep(wait_seconds)

            try:
                results = self.predict_now()
                for asset, result in results.items():
                    logger.info(
                        f"ORACLE SIGNAL: {asset} -> {result['signal']} "
                        f"({result['confidence']:.0%})"
                    )
            except Exception as e:
                logger.error(f"Oracle prediction failed: {e}", exc_info=True)

            await asyncio.sleep(60)
