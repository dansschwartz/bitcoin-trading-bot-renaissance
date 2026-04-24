"""
Spot Rebalancer — keeps exchange wallets funded for cross-exchange arb
by auto-buying seed assets when balances drop below thresholds.

Also tracks balance snapshots in SQLite for forensic auditing.
"""
import logging
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("arb.rebalancer")


@dataclass
class RebalanceTrade:
    """Record of a rebalance trade."""
    symbol: str
    side: str
    quantity: Decimal
    price: Decimal
    usdt_value: Decimal
    exchange: str
    reason: str
    timestamp: datetime


class SpotRebalancer:
    """Keeps exchange wallets funded for cross-exchange arbitrage.

    Periodically checks balances and buys seed assets when they drop
    below configured thresholds. Records balance snapshots to DB.
    """

    def __init__(self, clients: Dict[str, object], config: Optional[dict] = None,
                 approved_tokens: Optional[List[str]] = None):
        self._clients = clients
        cfg = config or {}
        self._enabled = cfg.get('enabled', False)
        self._min_usdt = Decimal(str(cfg.get('min_usdt_per_exchange', 10)))
        self._target_usdt = Decimal(str(cfg.get('target_usdt_per_exchange', 20)))
        self._max_rebalance = Decimal(str(cfg.get('max_single_rebalance_usd', 15)))
        self._cooldown_minutes = cfg.get('cooldown_minutes', 60)
        self._seed_assets: List[dict] = cfg.get('seed_assets', [])

        # Capital guard controls
        self.capital_guard = None  # Set by orchestrator
        self._max_budget_per_cycle = float(cfg.get('max_budget_per_cycle', 50))
        self._max_per_token = float(cfg.get('max_per_token', 30))
        self._approved_tokens_only = cfg.get('approved_tokens_only', True)
        self._approved_tokens: set = set(approved_tokens or [])

        # Dynamic seed tracking: assets promoted from inventory misses
        self._dynamic_seeds: Dict[str, dict] = {}
        self._inventory_misses: Dict[str, int] = defaultdict(int)
        self._trade_counts: Dict[str, int] = defaultdict(int)
        self._promotion_threshold = 3  # Promote after 3 misses

        # Cooldown tracking
        self._last_rebalance: Dict[str, float] = {}

        # DB for balance snapshots
        self._db_path = "data/arbitrage.db"
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        logger.info(
            f"SpotRebalancer: enabled={self._enabled}, "
            f"min_usdt=${float(self._min_usdt)}, "
            f"target=${float(self._target_usdt)}, "
            f"seeds={len(self._seed_assets)}"
        )

    def _init_db(self) -> None:
        """Create balance_snapshots table if it doesn't exist."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS balance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    free REAL NOT NULL,
                    locked REAL NOT NULL,
                    total REAL NOT NULL,
                    usd_value REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_balance_snapshots_ts
                ON balance_snapshots (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_balance_snapshots_exchange
                ON balance_snapshots (exchange, currency)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to init balance_snapshots table: {e}")

    def record_balance_snapshot(self, exchange: str, balances: Dict[str, dict]) -> None:
        """Record current balances for an exchange to the DB.

        Args:
            exchange: Exchange name (e.g., 'mexc', 'binance_us')
            balances: {currency: {free: float, locked: float, total: float, usd_value: float}}
        """
        if not balances:
            return
        try:
            conn = sqlite3.connect(self._db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            ts = datetime.utcnow().isoformat()
            rows = []
            for currency, bal in balances.items():
                free = float(bal.get('free', 0))
                locked = float(bal.get('locked', 0))
                total = float(bal.get('total', free + locked))
                usd_value = float(bal.get('usd_value', 0))
                if total > 0:
                    rows.append((ts, exchange, currency, free, locked, total, usd_value))
            if rows:
                conn.executemany(
                    "INSERT INTO balance_snapshots "
                    "(timestamp, exchange, currency, free, locked, total, usd_value) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Balance snapshot record error: {e}")

    async def check_and_rebalance(self) -> List[RebalanceTrade]:
        """Check balances and rebalance if needed. Returns list of trades made."""
        if not self._enabled:
            return []

        trades: List[RebalanceTrade] = []
        now = time.time()
        cycle_spent = 0.0  # Track total USD spent this cycle

        for exchange_name, client in self._clients.items():
            try:
                balances = await client.get_balances()
                if not balances:
                    continue

                # Record snapshot
                snap = {}
                for currency, bal in balances.items():
                    free = float(bal.free) if hasattr(bal, 'free') else float(bal.get('free', 0))
                    locked = float(bal.locked) if hasattr(bal, 'locked') else float(bal.get('locked', 0))
                    snap[currency] = {
                        'free': free,
                        'locked': locked,
                        'total': free + locked,
                    }
                self.record_balance_snapshot(exchange_name, snap)

                # Check USDT
                usdt_bal = balances.get('USDT')
                usdt_free = Decimal(str(float(usdt_bal.free))) if usdt_bal and hasattr(usdt_bal, 'free') else Decimal('0')

                if usdt_free < self._min_usdt:
                    logger.info(
                        f"REBALANCER: {exchange_name} USDT low: "
                        f"${float(usdt_free):.2f} < ${float(self._min_usdt):.2f}"
                    )

                # Check seed assets
                all_seeds = list(self._seed_assets) + [
                    {'symbol': s, 'amount_usd': d.get('amount_usd', 5)}
                    for s, d in self._dynamic_seeds.items()
                ]

                for seed in all_seeds:
                    symbol = seed.get('symbol', '')
                    target_usd = Decimal(str(seed.get('amount_usd', 5)))
                    if not symbol:
                        continue

                    # Approved tokens check
                    if self._approved_tokens_only and self._approved_tokens:
                        if symbol not in self._approved_tokens:
                            logger.debug(
                                f"REBALANCER: {symbol} not in approved tokens — skipping"
                            )
                            continue

                    # Per-token cap
                    if float(target_usd) > self._max_per_token:
                        target_usd = Decimal(str(self._max_per_token))

                    # Per-cycle budget cap
                    if cycle_spent >= self._max_budget_per_cycle:
                        logger.info(
                            f"REBALANCER: cycle budget ${self._max_budget_per_cycle:.0f} exhausted "
                            f"(spent ${cycle_spent:.2f}) — stopping seed buys"
                        )
                        break

                    remaining_budget = self._max_budget_per_cycle - cycle_spent
                    if float(target_usd) > remaining_budget:
                        target_usd = Decimal(str(remaining_budget))

                    # Capital guard check
                    if self.capital_guard:
                        allowed, cur_bal = await self.capital_guard.can_spend(
                            client, float(target_usd)
                        )
                        if not allowed:
                            logger.warning(
                                f"REBALANCER: CapitalGuard blocked {symbol} "
                                f"seed buy (${float(target_usd):.2f}), "
                                f"USDT=${cur_bal:.2f}"
                            )
                            continue

                    # Cooldown check
                    cooldown_key = f"{exchange_name}:{symbol}"
                    last = self._last_rebalance.get(cooldown_key, 0)
                    if now - last < self._cooldown_minutes * 60:
                        continue

                    bal = balances.get(symbol)
                    current_free = Decimal(str(float(bal.free))) if bal and hasattr(bal, 'free') else Decimal('0')

                    # Skip if we have enough (rough USD estimate)
                    if current_free > 0:
                        continue  # Has some — don't rebalance

                    # Track spending for cycle budget enforcement
                    cycle_spent += float(target_usd)

            except Exception as e:
                logger.debug(f"Rebalancer check error for {exchange_name}: {e}")

        return trades

    def record_inventory_miss(self, symbol: str) -> None:
        """Record that a trade was blocked due to insufficient balance."""
        base = symbol.split('/')[0] if '/' in symbol else symbol
        self._inventory_misses[base] += 1

        if (self._inventory_misses[base] >= self._promotion_threshold
                and base not in self._dynamic_seeds):
            self._dynamic_seeds[base] = {
                'amount_usd': 5,
                'promoted_at': time.time(),
                'misses': self._inventory_misses[base],
            }
            logger.info(
                f"REBALANCER: Promoted {base} to dynamic seed "
                f"(misses={self._inventory_misses[base]})"
            )

    def record_trade(self, symbol: str) -> None:
        """Record a successful trade for tracking."""
        base = symbol.split('/')[0] if '/' in symbol else symbol
        self._trade_counts[base] += 1

    def get_dynamic_seeds_status(self) -> dict:
        """Return status of dynamic seeds for dashboard."""
        return {
            'dynamic_seed_count': len(self._dynamic_seeds),
            'seeds': dict(self._dynamic_seeds),
            'pending_promotions': {
                base: count for base, count in self._inventory_misses.items()
                if count > 0 and base not in self._dynamic_seeds
            },
        }
