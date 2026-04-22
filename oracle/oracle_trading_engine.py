"""
Oracle Trading Engine
======================
Exact replication of the paper's calc_cum_ret_s1 strategy.

LONG only. Enter on BUY, exit on SELL or 10% stop loss.
One position per pair. Compound returns. 0.1% fees.

Uses oracle_service.py for signals. Does NOT run its own models.

Reference: Parente, Rizzuti, Trerotola (2023) — "A profitable
  trading algorithm for cryptocurrencies using a Neural Network model"
"""

import asyncio
import logging
import sqlite3
import requests
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger('oracle_trader')


# ══════════════════════════════════════════════════════
# CONFIGURATION — MATCH THE PAPER EXACTLY
# ══════════════════════════════════════════════════════

COMMISSION_FEE = 0.001          # 0.1% per trade
STOP_LOSS = 0.10                # 10% (paper's best from Table 4)

DEFAULT_PAIRS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT',
    'LINKUSDT', 'ADAUSDT', 'XRPUSDT', 'DOTUSDT', 'MATICUSDT',
]

DEFAULT_WALLET_SIZE = 5000.0    # $5,000 per pair
MAX_DAILY_LOSS_PCT = 0.15       # Halt pair if down 15% in a day

# Hardcoded safety — not configurable
ABSOLUTE_MAX_WALLET_SIZE = 10000
ABSOLUTE_MAX_STOP_LOSS = 0.15
ABSOLUTE_MAX_PAIRS = 20
ABSOLUTE_MAX_TOTAL_CAPITAL = 200000


# ══════════════════════════════════════════════════════
# WALLET (per-pair)
# ══════════════════════════════════════════════════════

@dataclass
class OracleWallet:
    """Independent wallet for one trading pair."""
    pair: str
    initial_capital: float
    capital: float
    position_open: bool = False
    entry_price: float = 0.0
    entry_time: str = ''
    entry_capital: float = 0.0
    stop_loss_price: float = 0.0
    stop_loss_pct: float = STOP_LOSS
    fee: float = COMMISSION_FEE
    daily_loss_usd: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl_usd: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_capital: float = 0.0
    status: str = 'observation'
    last_signal: str = 'HOLD'
    last_signal_time: str = ''
    current_price: float = 0.0


# ══════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════

class OracleTradingEngine:
    """Paper-exact trading engine using oracle signals."""

    def __init__(self, config: dict = None, oracle=None, db_path: str = ''):
        config = config or {}
        self.oracle = oracle
        self.db_path = db_path
        self.wallets: Dict[str, OracleWallet] = {}
        self.fee = config.get('commission_fee', COMMISSION_FEE)
        self.stop_loss = min(
            config.get('stop_loss', STOP_LOSS),
            ABSOLUTE_MAX_STOP_LOSS,
        )

        # Initialize wallets for each pair
        pairs = config.get('pairs', DEFAULT_PAIRS)[:ABSOLUTE_MAX_PAIRS]
        wallet_size = min(
            config.get('wallet_size', DEFAULT_WALLET_SIZE),
            ABSOLUTE_MAX_WALLET_SIZE,
        )
        total_capital = wallet_size * len(pairs)
        if total_capital > ABSOLUTE_MAX_TOTAL_CAPITAL:
            wallet_size = ABSOLUTE_MAX_TOTAL_CAPITAL / len(pairs)

        initial_status = config.get('initial_status', 'observation')

        for pair in pairs:
            self.wallets[pair] = OracleWallet(
                pair=pair,
                initial_capital=wallet_size,
                capital=wallet_size,
                peak_capital=wallet_size,
                stop_loss_pct=self.stop_loss,
                fee=self.fee,
                status=initial_status,
            )
            logger.info(
                f"Oracle wallet: {pair} | ${wallet_size:,.0f} | "
                f"stop={self.stop_loss:.0%} | fee={self.fee:.1%}"
            )

        self._init_db()
        self._load_state_from_db()
        self._running = False
        self._last_daily_reset = datetime.now(timezone.utc).date()

    # ══════════════════════════════════════════════════════
    # DATABASE
    # ══════════════════════════════════════════════════════

    def _get_db(self) -> sqlite3.Connection:
        """Get a DB connection."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        """Create oracle trading tables."""
        if not self.db_path:
            return
        conn = self._get_db()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oracle_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                capital_before REAL,
                capital_after REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                fee_paid REAL,
                signal TEXT,
                signal_confidence REAL,
                exit_reason TEXT,
                entry_time TEXT,
                exit_time TEXT,
                hold_bars INTEGER,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oracle_wallet_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                capital REAL,
                position_open INTEGER,
                entry_price REAL,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl_usd REAL,
                win_rate REAL,
                max_drawdown_pct REAL,
                status TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_oracle_trades_pair
            ON oracle_trades(pair, timestamp DESC)
        """)
        conn.commit()
        conn.close()
        logger.info("Oracle trading DB tables initialized")

    def _load_state_from_db(self) -> None:
        """Restore wallet state from DB on restart."""
        if not self.db_path:
            return
        conn = self._get_db()
        for pair, wallet in self.wallets.items():
            # Get latest snapshot
            row = conn.execute("""
                SELECT capital, position_open, entry_price, total_trades,
                       winning_trades, total_pnl_usd, max_drawdown_pct, status
                FROM oracle_wallet_snapshots
                WHERE pair = ?
                ORDER BY timestamp DESC LIMIT 1
            """, (pair,)).fetchone()
            if row:
                wallet.capital = row[0]
                wallet.position_open = bool(row[1])
                wallet.entry_price = row[2] or 0.0
                wallet.total_trades = row[3] or 0
                wallet.winning_trades = row[4] or 0
                wallet.total_pnl_usd = row[5] or 0.0
                wallet.max_drawdown_pct = row[6] or 0.0
                wallet.status = row[7] or 'observation'
                wallet.peak_capital = max(wallet.capital, wallet.initial_capital)
                if wallet.position_open and wallet.entry_price > 0:
                    wallet.stop_loss_price = wallet.entry_price * (1 - wallet.stop_loss_pct)
                    # Find entry_time from last OPEN trade
                    entry_row = conn.execute("""
                        SELECT entry_time FROM oracle_trades
                        WHERE pair = ? AND action = 'OPEN'
                        ORDER BY timestamp DESC LIMIT 1
                    """, (pair,)).fetchone()
                    if entry_row:
                        wallet.entry_time = entry_row[0] or ''
                    wallet.entry_capital = wallet.capital
                logger.info(
                    f"Oracle wallet restored: {pair} | "
                    f"capital=${wallet.capital:,.2f} | "
                    f"trades={wallet.total_trades} | "
                    f"{'OPEN' if wallet.position_open else 'flat'}"
                )
        conn.close()

    # ══════════════════════════════════════════════════════
    # PRICE FETCHING
    # ══════════════════════════════════════════════════════

    def _get_current_price(self, symbol: str) -> Optional[dict]:
        """Get current price data from Binance."""
        try:
            url = "https://api.binance.com/api/v3/ticker/24hr"
            resp = requests.get(url, params={'symbol': symbol}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    'close': float(data['lastPrice']),
                    'low': float(data['lowPrice']),
                    'high': float(data['highPrice']),
                    'bid': float(data['bidPrice']),
                    'ask': float(data['askPrice']),
                }
        except Exception as e:
            logger.debug(f"Price fetch failed for {symbol}: {e}")
        return None

    # ══════════════════════════════════════════════════════
    # CORE LOGIC — MATCHES calc_cum_ret_s1 EXACTLY
    # ══════════════════════════════════════════════════════

    def _process_signal(self, wallet: OracleWallet) -> None:
        """
        Process the latest oracle signal for one pair.

        Paper logic:
          - BUY signal + no position -> enter LONG at close
          - SELL signal + position open -> exit at close
          - Stop loss hit -> exit at stop price
          - HOLD -> do nothing
          - BUY + already in position -> do nothing (no doubling)
        """
        if wallet.status == 'halted':
            return

        # Get latest oracle signal
        signal_data = self.oracle.get_latest_signal(wallet.pair)
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        age_minutes = signal_data.get('age_minutes', 999)

        # Ignore stale signals (older than 4.5 hours)
        if age_minutes > 270:
            return

        wallet.last_signal = signal
        wallet.last_signal_time = datetime.now(timezone.utc).isoformat()

        # Get current price
        price_data = self._get_current_price(wallet.pair)
        if not price_data:
            return

        current_price = price_data['close']
        wallet.current_price = current_price

        # ── STOP LOSS CHECK (always check first, like the paper) ──
        if wallet.position_open:
            pct_change = (current_price - wallet.entry_price) / wallet.entry_price

            if pct_change < -wallet.stop_loss_pct:
                # Close at exactly the stop loss price (paper's logic)
                exit_price = wallet.entry_price * (1 - wallet.stop_loss_pct)
                self._close_position(wallet, exit_price, 'stop_loss',
                                     signal, confidence)
                return

        # ── SIGNAL PROCESSING ──
        if signal == 'BUY':
            if not wallet.position_open:
                self._open_position(wallet, current_price, signal, confidence)
            return

        if signal == 'SELL':
            if wallet.position_open:
                self._close_position(wallet, current_price, 'sell_signal',
                                     signal, confidence)
            return

        # HOLD — do nothing

    # ══════════════════════════════════════════════════════
    # POSITION MANAGEMENT
    # ══════════════════════════════════════════════════════

    def _open_position(self, wallet: OracleWallet, price: float,
                       signal: str, confidence: float) -> None:
        """Enter LONG position. Invest full wallet capital."""
        wallet.position_open = True
        wallet.entry_price = price
        wallet.entry_capital = wallet.capital
        wallet.entry_time = datetime.now(timezone.utc).isoformat()
        wallet.stop_loss_price = price * (1 - wallet.stop_loss_pct)

        mode = "LIVE" if wallet.status == 'live' else "OBS"
        logger.info(
            f"ORACLE[{wallet.pair}] OPEN [{mode}] "
            f"price=${price:,.2f} "
            f"capital=${wallet.capital:,.2f} "
            f"stop=${wallet.stop_loss_price:,.2f} "
            f"(-{wallet.stop_loss_pct:.0%}) "
            f"signal={signal} conf={confidence:.0%}"
        )

        if self.db_path:
            try:
                conn = self._get_db()
                conn.execute("""
                    INSERT INTO oracle_trades
                    (pair, action, price, capital_before, signal,
                     signal_confidence, entry_time, timestamp)
                    VALUES (?, 'OPEN', ?, ?, ?, ?, ?, datetime('now'))
                """, (wallet.pair, price, wallet.capital, signal,
                      confidence, wallet.entry_time))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"DB error on open: {e}")

    def _close_position(self, wallet: OracleWallet, exit_price: float,
                        exit_reason: str, signal: str,
                        confidence: float) -> None:
        """
        Close LONG position. Apply fees. Compound returns.

        Paper's exact P&L formula:
          capital *= 1 + (((exit_price * (1 - fee)) -
                           (entry_price * (1 + fee))) /
                          (entry_price * (1 + fee)))
        """
        entry_price = wallet.entry_price
        fee = wallet.fee

        # Paper's exact compounding formula
        trade_return = (((exit_price * (1 - fee)) -
                         (entry_price * (1 + fee))) /
                        (entry_price * (1 + fee)))

        capital_before = wallet.capital
        wallet.capital *= (1 + trade_return)
        pnl_usd = wallet.capital - capital_before
        pnl_pct = trade_return

        # Fee estimate for logging
        fee_paid = (entry_price * fee + exit_price * fee) * \
                   (capital_before / entry_price)

        # Stats
        wallet.total_trades += 1
        wallet.total_pnl_usd += pnl_usd
        if pnl_usd > 0:
            wallet.winning_trades += 1

        # Peak and drawdown
        if wallet.capital > wallet.peak_capital:
            wallet.peak_capital = wallet.capital
        current_dd = (wallet.peak_capital - wallet.capital) / wallet.peak_capital
        if current_dd > wallet.max_drawdown_pct:
            wallet.max_drawdown_pct = current_dd

        # Daily loss tracking
        if pnl_usd < 0:
            wallet.daily_loss_usd += abs(pnl_usd)
            max_daily = wallet.initial_capital * MAX_DAILY_LOSS_PCT
            if wallet.daily_loss_usd >= max_daily:
                logger.warning(
                    f"ORACLE[{wallet.pair}] HALTED: "
                    f"daily loss ${wallet.daily_loss_usd:.2f} "
                    f">= ${max_daily:.2f}"
                )
                wallet.status = 'halted'

        # Hold time
        hold_bars = 0
        hold_time = '?'
        if wallet.entry_time:
            try:
                entry_dt = datetime.fromisoformat(wallet.entry_time)
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                hold_hours = (datetime.now(timezone.utc) - entry_dt
                              ).total_seconds() / 3600
                hold_bars = int(hold_hours / 4)
                hold_time = f"{hold_hours:.1f}h ({hold_bars} bars)"
            except Exception:
                pass

        # Reset position
        wallet.position_open = False
        wallet.entry_price = 0.0
        wallet.stop_loss_price = 0.0

        win_rate = (wallet.winning_trades / wallet.total_trades * 100
                    if wallet.total_trades > 0 else 0)

        mode = "LIVE" if wallet.status == 'live' else "OBS"
        logger.info(
            f"ORACLE[{wallet.pair}] CLOSE [{mode}] "
            f"reason={exit_reason} "
            f"entry=${entry_price:,.2f} exit=${exit_price:,.2f} "
            f"return={pnl_pct:+.2%} "
            f"P&L=${pnl_usd:+.2f} "
            f"capital=${wallet.capital:,.2f} "
            f"held={hold_time} "
            f"record={wallet.winning_trades}/{wallet.total_trades} "
            f"({win_rate:.0f}%)"
        )

        if self.db_path:
            try:
                conn = self._get_db()
                conn.execute("""
                    INSERT INTO oracle_trades
                    (pair, action, price, capital_before, capital_after,
                     pnl_usd, pnl_pct, fee_paid, signal, signal_confidence,
                     exit_reason, entry_time, exit_time, hold_bars, timestamp)
                    VALUES (?, 'CLOSE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, datetime('now'))
                """, (wallet.pair, exit_price, capital_before,
                      wallet.capital, pnl_usd, pnl_pct, fee_paid,
                      signal, confidence, exit_reason,
                      wallet.entry_time,
                      datetime.now(timezone.utc).isoformat(),
                      hold_bars))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"DB error on close: {e}")

        wallet.entry_time = ''

    # ══════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════

    async def run_forever(self) -> None:
        """Check signals every 5 minutes."""
        self._running = True
        logger.info(
            f"Oracle Trading Engine started: {len(self.wallets)} pairs"
        )

        while self._running:
            try:
                # Daily loss reset at midnight UTC
                today = datetime.now(timezone.utc).date()
                if today != self._last_daily_reset:
                    self._reset_daily_losses()
                    self._last_daily_reset = today

                for wallet in self.wallets.values():
                    self._process_signal(wallet)

                self._save_wallet_snapshots()

            except Exception as e:
                logger.error(f"Oracle trading loop error: {e}",
                             exc_info=True)

            await asyncio.sleep(300)  # 5 minutes

    def _save_wallet_snapshots(self) -> None:
        """Save current wallet state to DB for dashboard."""
        if not self.db_path:
            return
        try:
            conn = self._get_db()
            for pair, w in self.wallets.items():
                wr = (w.winning_trades / w.total_trades * 100
                      if w.total_trades > 0 else 0)
                conn.execute("""
                    INSERT INTO oracle_wallet_snapshots
                    (pair, capital, position_open, entry_price,
                     total_trades, winning_trades, total_pnl_usd,
                     win_rate, max_drawdown_pct, status, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                """, (pair, w.capital, int(w.position_open),
                      w.entry_price, w.total_trades, w.winning_trades,
                      w.total_pnl_usd, wr, w.max_drawdown_pct,
                      w.status))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Snapshot save error: {e}")

    def _reset_daily_losses(self) -> None:
        """Reset daily loss counters at midnight UTC."""
        for wallet in self.wallets.values():
            wallet.daily_loss_usd = 0.0
            if wallet.status == 'halted':
                wallet.status = 'observation'
                logger.info(f"ORACLE[{wallet.pair}] un-halted for new day")

    # ══════════════════════════════════════════════════════
    # STATUS (for dashboard)
    # ══════════════════════════════════════════════════════

    def get_status(self) -> dict:
        """Full status for dashboard display."""
        wallets = {}
        total_capital = 0.0
        total_pnl = 0.0
        total_initial = 0.0
        positions_open = 0

        for pair, w in self.wallets.items():
            wr = (w.winning_trades / w.total_trades * 100
                  if w.total_trades > 0 else 0)
            unrealized = 0.0
            if w.position_open and w.entry_price > 0 and w.current_price > 0:
                unrealized = round(
                    (w.current_price - w.entry_price) / w.entry_price * 100, 2
                )
            wallets[pair] = {
                'pair': pair,
                'capital': round(w.capital, 2),
                'initial': w.initial_capital,
                'pnl_usd': round(w.total_pnl_usd, 2),
                'pnl_pct': round(
                    (w.capital - w.initial_capital) / w.initial_capital * 100, 2
                ),
                'position_open': w.position_open,
                'entry_price': w.entry_price,
                'stop_loss_price': w.stop_loss_price,
                'current_price': w.current_price,
                'unrealized_pnl': unrealized,
                'total_trades': w.total_trades,
                'winning_trades': w.winning_trades,
                'win_rate': round(wr, 1),
                'max_drawdown_pct': round(w.max_drawdown_pct * 100, 2),
                'status': w.status,
                'last_signal': w.last_signal,
            }
            total_capital += w.capital
            total_initial += w.initial_capital
            total_pnl += w.total_pnl_usd
            if w.position_open:
                positions_open += 1

        return {
            'wallets': wallets,
            'summary': {
                'total_capital': round(total_capital, 2),
                'total_initial': total_initial,
                'total_pnl': round(total_pnl, 2),
                'total_return_pct': round(
                    (total_capital - total_initial) / total_initial * 100, 2
                ) if total_initial > 0 else 0,
                'positions_open': positions_open,
                'total_pairs': len(self.wallets),
                'active_pairs': sum(
                    1 for w in self.wallets.values()
                    if w.status != 'halted'
                ),
            },
        }
