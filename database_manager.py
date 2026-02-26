# database_manager.py
import sqlite3
import json
import logging
import os
import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MarketData:
    """Market data structure"""
    price: float
    volume: float
    bid: float
    ask: float
    spread: float
    timestamp: datetime
    source: str = "coinbase"
    product_id: str = "BTC-USD"


@dataclass
class SentimentData:
    """Sentiment data structure"""
    overall_sentiment: float
    twitter_sentiment: float
    reddit_sentiment: float
    fear_greed_index: int
    confidence: float
    timestamp: datetime
    sources: Dict[str, Any]


class DatabaseManager:
    def __init__(self, config: Dict):
        self.db_path = config['path']
        self.backup_interval = config.get('backup_interval', 3600)
        self.logger = logging.getLogger(f"{__name__}.DatabaseManager")

    @contextmanager
    def _get_connection(self):
        """Context manager for safe SQLite connections with WAL mode and timeout."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def init_database(self):
        """Initialize database with all required tables"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, price REAL NOT NULL,
                volume REAL NOT NULL, bid REAL NOT NULL, ask REAL NOT NULL,
                spread REAL NOT NULL, timestamp TEXT NOT NULL, source TEXT NOT NULL,
                product_id TEXT DEFAULT 'BTC-USD')''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT, decision_id INTEGER UNIQUE,
                product_id TEXT NOT NULL, t_entry TEXT NOT NULL, entry_price REAL NOT NULL,
                t_exit TEXT NOT NULL, exit_price REAL NOT NULL, horizon_min INTEGER NOT NULL,
                ret_pct REAL NOT NULL, correct INTEGER NOT NULL,
                FOREIGN KEY (decision_id) REFERENCES decisions(id))''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, overall_sentiment REAL NOT NULL,
                twitter_sentiment REAL NOT NULL, reddit_sentiment REAL NOT NULL,
                fear_greed_index INTEGER NOT NULL, confidence REAL NOT NULL,
                timestamp TEXT NOT NULL, sources TEXT NOT NULL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS onchain_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT, active_addresses INTEGER,
                transaction_count INTEGER, hash_rate REAL, network_health REAL,
                timestamp TEXT NOT NULL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, action TEXT NOT NULL, confidence REAL NOT NULL,
                position_size REAL NOT NULL, weighted_signal REAL NOT NULL,
                reasoning TEXT NOT NULL, feature_vector TEXT, vae_loss REAL,
                hmm_regime TEXT)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, side TEXT NOT NULL, size REAL NOT NULL,
                price REAL NOT NULL, status TEXT NOT NULL, algo_used TEXT,
                slippage REAL, execution_time REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS ml_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL, model_name TEXT NOT NULL,
                prediction REAL NOT NULL, confidence REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS open_positions (
                position_id TEXT PRIMARY KEY, product_id TEXT NOT NULL,
                side TEXT NOT NULL, size REAL NOT NULL, entry_price REAL NOT NULL,
                stop_loss_price REAL, take_profit_price REAL,
                opened_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN')''')

            # Add P&L tracking columns (backward-compatible migration)
            for col_def in [
                "close_price REAL",
                "closed_at TEXT",
                "realized_pnl REAL",
                "exit_reason TEXT",
                "hold_duration_seconds REAL",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE open_positions ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Medallion Intelligence: Expanded data capture tables
            cursor.execute('''CREATE TABLE IF NOT EXISTS funding_rate_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                funding_rate REAL NOT NULL, exchange TEXT,
                predicted_rate REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS open_interest_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                open_interest REAL NOT NULL, change_24h_pct REAL)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS liquidation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, symbol TEXT NOT NULL,
                direction TEXT NOT NULL, risk_score REAL,
                funding_rate_percentile REAL, long_short_ratio REAL,
                recommended_action TEXT)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS ghost_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, param_set TEXT NOT NULL,
                product_id TEXT NOT NULL, action TEXT NOT NULL,
                entry_price REAL NOT NULL, exit_price REAL,
                pnl_pct REAL, exit_reason TEXT, cycles_held INTEGER)''')

            cursor.execute('''CREATE TABLE IF NOT EXISTS signal_throttle_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, action TEXT NOT NULL,
                signal_name TEXT NOT NULL, accuracy REAL,
                sample_count INTEGER, product_id TEXT)''')

            # ── Position Snapshots: per-minute P&L tracking for early-exit calibration ──
            cursor.execute('''CREATE TABLE IF NOT EXISTS position_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                snapshot_time TEXT NOT NULL,
                hold_seconds REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl_pct REAL NOT NULL,
                unrealized_pnl_usd REAL,
                side TEXT NOT NULL,
                confidence REAL,
                regime TEXT)''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_pos ON position_snapshots(position_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_time ON position_snapshots(snapshot_time)')

            # ── Decision Audit Log: flat, fully-indexed pipeline trace ──
            cursor.execute('''CREATE TABLE IF NOT EXISTS decision_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                product_id TEXT NOT NULL,
                cycle_number INTEGER,

                -- Stage 1: Market Snapshot
                price REAL, bid REAL, ask REAL, spread_bps REAL,
                volume_24h_usd REAL, order_book_depth_usd REAL,

                -- Stage 2: Raw Signals
                sig_order_flow REAL, sig_order_book REAL, sig_volume REAL,
                sig_macd REAL, sig_rsi REAL, sig_bollinger REAL,
                sig_alternative REAL, sig_volume_profile REAL,
                sig_stat_arb REAL, sig_lead_lag REAL, sig_fractal REAL,
                sig_entropy REAL, sig_ml_ensemble REAL, sig_ml_cnn REAL,
                sig_quantum REAL, sig_correlation_divergence REAL,
                sig_garch_vol REAL, sig_medallion_analog REAL,
                raw_signal_count INTEGER,

                -- Stage 3: Regime
                regime_label TEXT, regime_confidence REAL,
                regime_classifier TEXT, bar_gap_ratio REAL, gap_poisoned INTEGER,

                -- Stage 4: Weights & Fusion
                weighted_signal REAL, signal_contributions TEXT,
                weight_adjustments TEXT,

                -- Stage 5: ML Predictions
                ml_ensemble_score REAL, ml_confidence REAL,
                ml_model_count INTEGER, ml_agreement_pct REAL,
                ml_predictions_json TEXT, ml_scale_factor REAL,

                -- Stage 6: Confluence
                confluence_boost REAL, confluence_active_count INTEGER,

                -- Stage 7: Confidence
                signal_strength REAL, signal_consensus REAL,
                raw_confidence REAL, regime_conf_boost REAL,
                ml_conf_boost REAL, final_confidence REAL,

                -- Stage 8: Risk Gates
                gate_cost_prescreen INTEGER,
                gate_confidence INTEGER,
                gate_regime_filter INTEGER, gate_regime_filter_detail TEXT,
                gate_ml_agreement INTEGER, gate_ml_agreement_detail TEXT,
                gate_anti_churn INTEGER, gate_anti_churn_detail TEXT,
                gate_signal_reversal INTEGER,
                gate_anti_stacking INTEGER,
                gate_risk_regime INTEGER,
                gate_daily_loss INTEGER,
                gate_vae INTEGER, gate_vae_detail TEXT,
                gate_risk_gateway INTEGER,
                gate_health_monitor INTEGER,
                blocked_by TEXT,

                -- Stage 9: Position Sizing
                kelly_fraction REAL, applied_fraction REAL,
                edge REAL, effective_edge REAL, win_probability REAL,
                position_usd REAL, position_units REAL,
                market_impact_bps REAL, capacity_pct REAL,
                sizing_chain_json TEXT,
                buy_threshold REAL, sell_threshold REAL,
                garch_pos_multiplier REAL,

                -- Stage 10: Final Decision
                final_action TEXT, final_position_size REAL,

                -- Stage 11: Execution
                execution_mode TEXT, devil_trade_id TEXT,

                -- Stage 12: Feature Vector
                feature_vector_hash TEXT, feature_top5_json TEXT,

                -- Stage 13: System State
                drawdown_pct REAL, daily_pnl REAL,
                account_balance REAL, open_positions_count INTEGER,
                scan_tier INTEGER,

                -- Outcome (filled asynchronously)
                outcome_1bar REAL, outcome_6bar REAL,
                outcome_12bar REAL, outcome_24bar REAL,
                outcome_evaluated_at TEXT
            )''')

            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_product_ts ON decision_audit_log(product_id, timestamp DESC)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_action ON decision_audit_log(final_action)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_blocked ON decision_audit_log(blocked_by)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_cycle ON decision_audit_log(cycle_number)')

            # Add outcome columns to ml_predictions (backward-compatible migration)
            for col_def in [
                "price_at_prediction REAL",
                "actual_return_1bar REAL",
                "actual_return_6bar REAL",
                "actual_direction INTEGER",
                "is_correct INTEGER",
                "evaluated_at TEXT",
                "price_at_evaluation REAL",
            ]:
                try:
                    cursor.execute(f"ALTER TABLE ml_predictions ADD COLUMN {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            conn.commit()
            self.logger.info("Database initialized successfully with expanded metrics support")

    async def store_market_data(self, data: MarketData):
        """Store market data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = data.timestamp.isoformat() if hasattr(data.timestamp, 'isoformat') else datetime.now(timezone.utc).isoformat()

                cursor.execute('''
                    INSERT INTO market_data (price, volume, bid, ask, spread, timestamp, source, product_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.price,
                    data.volume,
                    data.bid,
                    data.ask,
                    data.spread,
                    ts,
                    data.source,
                    data.product_id
                ))

                conn.commit()
            self.logger.debug("Market data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing market data: {e}")

    async def store_sentiment_data(self, data: SentimentData):
        """Store sentiment data"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                def json_serial(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    raise TypeError(f"Type {type(obj)} not serializable")

                cursor.execute('''
                    INSERT INTO sentiment_data (overall_sentiment, twitter_sentiment, reddit_sentiment,
                                              fear_greed_index, confidence, timestamp, sources)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data.overall_sentiment,
                    data.twitter_sentiment,
                    data.reddit_sentiment,
                    data.fear_greed_index,
                    data.confidence,
                    data.timestamp.isoformat(),
                    json.dumps(data.sources, default=json_serial)
                ))

                conn.commit()
            self.logger.debug("Sentiment data stored successfully")

        except Exception as e:
            self.logger.error(f"Error storing sentiment data: {e}")

    # Tables that may be queried via get_recent_data / cleanup_old_data
    ALLOWED_TABLES = frozenset({
        "market_data", "labels", "sentiment_data", "onchain_data",
        "decisions", "trades", "ml_predictions", "open_positions",
        "daily_candles", "data_refresh_log",
    })

    # Map tables without a 'timestamp' column to their time column
    _TIME_COL = {
        "labels": "t_entry",
        "open_positions": "opened_at",
        "daily_candles": "date",
        "data_refresh_log": "last_refresh",
    }

    async def get_recent_data(self, table: str, hours: int = 24) -> List[Dict]:
        """Get recent data from specified table"""
        if table not in self.ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                time_col = self._TIME_COL.get(table, "timestamp")
                cursor.execute(
                    f"SELECT * FROM {table} "
                    f"WHERE datetime({time_col}) > datetime('now', ? || ' hours') "
                    f"ORDER BY {time_col} DESC",
                    (f"-{int(hours)}",)
                )

                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving data: {e}")
            return []

    async def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                tables = ['market_data', 'sentiment_data', 'onchain_data', 'decisions', 'trades', 'ml_predictions']

                for table in tables:
                    if table not in self.ALLOWED_TABLES:
                        continue
                    time_col = self._TIME_COL.get(table, "timestamp")
                    cursor.execute(
                        f"DELETE FROM {table} "
                        f"WHERE datetime({time_col}) < datetime('now', ? || ' days')",
                        (f"-{int(days)}",)
                    )

                conn.commit()
            self.logger.info(f"Cleaned up data older than {days} days")

        except Exception as e:
            self.logger.error(f"Error cleaning up data: {e}")

    async def store_decision(self, decision_data: Dict[str, Any]):
        """Store trading decision with expanded metrics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Use UTC for persistence consistency
                ts = decision_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                # Handle non-serializable objects in reasoning
                reasoning = decision_data.get('reasoning', {})
                def json_serial(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.bool_, np.integer)):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    raise TypeError(f"Type {type(obj)} not serializable")

                # Extract expanded metrics
                feature_vector = decision_data.get('feature_vector')
                if isinstance(feature_vector, np.ndarray):
                    feature_vector = json.dumps(feature_vector.tolist())
                elif feature_vector is not None:
                    feature_vector = str(feature_vector)

                vae_loss = decision_data.get('vae_loss')
                hmm_regime = decision_data.get('hmm_regime')

                cursor.execute('''
                    INSERT INTO decisions (timestamp, product_id, action, confidence, position_size, weighted_signal, reasoning, feature_vector, vae_loss, hmm_regime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ts,
                    decision_data.get('product_id'),
                    decision_data.get('action'),
                    decision_data.get('confidence'),
                    decision_data.get('position_size'),
                    decision_data.get('weighted_signal'),
                    json.dumps(reasoning, default=json_serial),
                    feature_vector,
                    vae_loss,
                    hmm_regime
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing decision: {e}")

    async def store_trade(self, trade_data: Dict[str, Any]):
        """Store executed trade"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = trade_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                cursor.execute('''
                    INSERT INTO trades (timestamp, product_id, side, size, price, status, algo_used, slippage, execution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    ts,
                    trade_data.get('product_id'),
                    trade_data.get('side'),
                    trade_data.get('size'),
                    trade_data.get('price'),
                    trade_data.get('status'),
                    trade_data.get('algo_used'),
                    trade_data.get('slippage'),
                    trade_data.get('execution_time')
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing trade: {e}")

    async def store_ml_prediction(self, prediction_data: Dict[str, Any]):
        """Store ML model prediction"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                ts = prediction_data.get('timestamp')
                if not ts:
                    ts = datetime.now(timezone.utc).isoformat()
                elif isinstance(ts, datetime):
                    ts = ts.isoformat()

                cursor.execute('''
                    INSERT INTO ml_predictions (timestamp, product_id, model_name, prediction, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    ts,
                    prediction_data.get('product_id'),
                    prediction_data.get('model_name'),
                    prediction_data.get('prediction'),
                    prediction_data.get('confidence'),
                ))

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing ML prediction: {e}")

    # ──────────────────────────────────────────────
    #  Position State Recovery
    # ──────────────────────────────────────────────

    async def save_position(self, position_data: Dict[str, Any]):
        """Upsert an open position for state recovery."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO open_positions
                        (position_id, product_id, side, size, entry_price,
                         stop_loss_price, take_profit_price, opened_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_data['position_id'],
                    position_data['product_id'],
                    position_data['side'],
                    position_data['size'],
                    position_data['entry_price'],
                    position_data.get('stop_loss_price'),
                    position_data.get('take_profit_price'),
                    position_data.get('opened_at', datetime.now(timezone.utc).isoformat()),
                    position_data.get('status', 'OPEN'),
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error saving position: {e}")

    async def close_position_record(
        self,
        position_id: str,
        close_price: float = 0.0,
        realized_pnl: float = 0.0,
        exit_reason: str = "",
    ) -> None:
        """Mark a position as CLOSED and persist exit data (price, P&L, reason, hold time)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Compute hold duration from opened_at
                hold_duration: float = 0.0
                row = cursor.execute(
                    "SELECT opened_at FROM open_positions WHERE position_id = ?",
                    (position_id,),
                ).fetchone()
                if row and row[0]:
                    try:
                        opened = datetime.fromisoformat(row[0])
                        hold_duration = (datetime.now(timezone.utc) - opened.replace(tzinfo=timezone.utc)).total_seconds()
                    except (ValueError, TypeError):
                        pass
                closed_at = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    """UPDATE open_positions
                       SET status = 'CLOSED',
                           close_price = ?,
                           closed_at = ?,
                           realized_pnl = ?,
                           exit_reason = ?,
                           hold_duration_seconds = ?
                       WHERE position_id = ?""",
                    (close_price, closed_at, realized_pnl, exit_reason, hold_duration, position_id),
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error closing position record: {e}")

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Retrieve all OPEN positions for state recovery on restart."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM open_positions WHERE status = 'OPEN'")
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting open positions: {e}")
            return []

    async def get_daily_pnl(self, date_str: str) -> float:
        """Sum realized PnL from today's trades (approximated from trade records)."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COALESCE(SUM(
                        CASE WHEN side = 'SELL' THEN size * price
                             WHEN side = 'BUY'  THEN -size * price
                             ELSE 0 END
                    ), 0.0)
                    FROM trades
                    WHERE date(timestamp) = ?
                ''', (date_str,))
                result = cursor.fetchone()
                return float(result[0]) if result else 0.0
        except Exception as e:
            self.logger.error(f"Error getting daily PnL: {e}")
            return 0.0

    # ──────────────────────────────────────────────
    #  Decision Audit Log
    # ──────────────────────────────────────────────

    # Canonical column order for decision_audit_log INSERT
    _AUDIT_COLUMNS = [
        'timestamp', 'product_id', 'cycle_number',
        # Market snapshot
        'price', 'bid', 'ask', 'spread_bps',
        'volume_24h_usd', 'order_book_depth_usd',
        # Raw signals
        'sig_order_flow', 'sig_order_book', 'sig_volume',
        'sig_macd', 'sig_rsi', 'sig_bollinger',
        'sig_alternative', 'sig_volume_profile',
        'sig_stat_arb', 'sig_lead_lag', 'sig_fractal',
        'sig_entropy', 'sig_ml_ensemble', 'sig_ml_cnn',
        'sig_quantum', 'sig_correlation_divergence',
        'sig_garch_vol', 'sig_medallion_analog',
        'raw_signal_count',
        # Regime
        'regime_label', 'regime_confidence',
        'regime_classifier', 'bar_gap_ratio', 'gap_poisoned',
        # Weights & Fusion
        'weighted_signal', 'signal_contributions', 'weight_adjustments',
        # ML Predictions
        'ml_ensemble_score', 'ml_confidence',
        'ml_model_count', 'ml_agreement_pct',
        'ml_predictions_json', 'ml_scale_factor',
        # Confluence
        'confluence_boost', 'confluence_active_count',
        # Confidence
        'signal_strength', 'signal_consensus',
        'raw_confidence', 'regime_conf_boost',
        'ml_conf_boost', 'final_confidence',
        # Risk Gates
        'gate_cost_prescreen',
        'gate_confidence',
        'gate_regime_filter', 'gate_regime_filter_detail',
        'gate_ml_agreement', 'gate_ml_agreement_detail',
        'gate_anti_churn', 'gate_anti_churn_detail',
        'gate_signal_reversal',
        'gate_anti_stacking',
        'gate_risk_regime',
        'gate_daily_loss',
        'gate_vae', 'gate_vae_detail',
        'gate_risk_gateway',
        'gate_health_monitor',
        'blocked_by',
        # Position Sizing
        'kelly_fraction', 'applied_fraction',
        'edge', 'effective_edge', 'win_probability',
        'position_usd', 'position_units',
        'market_impact_bps', 'capacity_pct',
        'sizing_chain_json',
        'buy_threshold', 'sell_threshold',
        'garch_pos_multiplier',
        # Final Decision
        'final_action', 'final_position_size',
        # Execution
        'execution_mode', 'devil_trade_id',
        # Feature Vector
        'feature_vector_hash', 'feature_top5_json',
        # System State
        'drawdown_pct', 'daily_pnl',
        'account_balance', 'open_positions_count',
        'scan_tier',
    ]

    async def store_audit_log(self, audit_data: Dict[str, Any]) -> None:
        """Insert one row into decision_audit_log from a flat dict."""
        try:
            values = [audit_data.get(col) for col in self._AUDIT_COLUMNS]
            placeholders = ', '.join(['?'] * len(self._AUDIT_COLUMNS))
            col_names = ', '.join(self._AUDIT_COLUMNS)
            with self._get_connection() as conn:
                conn.execute(
                    f"INSERT INTO decision_audit_log ({col_names}) VALUES ({placeholders})",
                    values,
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error storing audit log: {e}")

    async def evaluate_audit_outcomes(self, lookback_minutes: int = 60) -> int:
        """Fill outcome columns for audit rows where enough time has elapsed.

        Compares decision-time price to current price for the product.
        Returns number of rows updated.
        """
        updated = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Find rows that need outcome evaluation (outcome_1bar IS NULL and old enough)
                rows = cursor.execute('''
                    SELECT id, product_id, price, timestamp
                    FROM decision_audit_log
                    WHERE outcome_1bar IS NULL
                      AND price IS NOT NULL
                      AND datetime(timestamp) < datetime('now', ? || ' minutes')
                    ORDER BY id
                    LIMIT 500
                ''', (f"-{lookback_minutes}",)).fetchall()

                if not rows:
                    return 0

                # Get latest prices per product from market_data
                products = list({r[1] for r in rows})
                latest_prices: Dict[str, float] = {}
                for pid in products:
                    px_row = cursor.execute('''
                        SELECT price FROM market_data
                        WHERE product_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (pid,)).fetchone()
                    if px_row:
                        latest_prices[pid] = float(px_row[0])

                now_ts = datetime.now(timezone.utc).isoformat()
                for row_id, pid, entry_price, ts in rows:
                    current_px = latest_prices.get(pid)
                    if not current_px or entry_price <= 0:
                        continue
                    ret = (current_px - entry_price) / entry_price
                    cursor.execute('''
                        UPDATE decision_audit_log
                        SET outcome_1bar = ?, outcome_evaluated_at = ?
                        WHERE id = ?
                    ''', (round(ret, 6), now_ts, row_id))
                    updated += 1

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error evaluating audit outcomes: {e}")
        return updated

    async def evaluate_ml_outcomes(self, lookback_minutes: int = 60) -> int:
        """Fill outcome columns for ml_predictions that lack evaluation.

        Returns number of rows updated.
        """
        updated = 0
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                rows = cursor.execute('''
                    SELECT id, product_id, prediction, price_at_prediction, timestamp
                    FROM ml_predictions
                    WHERE is_correct IS NULL
                      AND price_at_prediction IS NOT NULL
                      AND datetime(timestamp) < datetime('now', ? || ' minutes')
                    ORDER BY id
                    LIMIT 500
                ''', (f"-{lookback_minutes}",)).fetchall()

                if not rows:
                    return 0

                products = list({r[1] for r in rows})
                latest_prices: Dict[str, float] = {}
                for pid in products:
                    px_row = cursor.execute('''
                        SELECT price FROM market_data
                        WHERE product_id = ?
                        ORDER BY timestamp DESC LIMIT 1
                    ''', (pid,)).fetchone()
                    if px_row:
                        latest_prices[pid] = float(px_row[0])

                now_ts = datetime.now(timezone.utc).isoformat()
                for row_id, pid, prediction, entry_px, ts in rows:
                    current_px = latest_prices.get(pid)
                    if not current_px or not entry_px or entry_px <= 0:
                        continue
                    actual_return = (current_px - entry_px) / entry_px
                    actual_dir = 1 if actual_return > 0.001 else (-1 if actual_return < -0.001 else 0)
                    pred_dir = 1 if prediction > 0 else (-1 if prediction < 0 else 0)
                    is_correct = 1 if (pred_dir != 0 and pred_dir == actual_dir) else 0
                    cursor.execute('''
                        UPDATE ml_predictions
                        SET actual_return_1bar = ?, actual_direction = ?,
                            is_correct = ?, evaluated_at = ?,
                            price_at_evaluation = ?
                        WHERE id = ?
                    ''', (round(actual_return, 6), actual_dir, is_correct, now_ts, current_px, row_id))
                    updated += 1

                conn.commit()
        except Exception as e:
            self.logger.error(f"Error evaluating ML outcomes: {e}")
        return updated