"""
Arbitrage Risk Engine — non-negotiable limits for arbitrage operations.

Every trade must pass through this gate. Default is DO NOTHING.
If anything is unexpected, reject. Never guess. Never assume.
"""
import logging
from collections import deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional

logger = logging.getLogger("arb.risk")


class ArbitrageRiskEngine:

    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}

        # Hard limits
        self.max_single_arb_usd = Decimal(str(cfg.get('max_single_arb_usd', 500)))
        self.max_total_exposure_usd = Decimal(str(cfg.get('max_total_exposure_usd', 5000)))
        self.max_one_sided_exposure_usd = Decimal(str(cfg.get('max_one_sided_usd', 200)))
        self.max_daily_loss_usd = Decimal(str(cfg.get('max_daily_loss_usd', 100)))
        self.max_funding_position_usd = Decimal(str(cfg.get('max_funding_position_usd', 2000)))
        self.max_trades_per_hour = int(cfg.get('max_trades_per_hour', 100))
        self.max_consecutive_losses = int(cfg.get('max_consecutive_losses', 10))

        # Exchange health
        self.max_api_latency_ms = int(cfg.get('max_api_latency_ms', 500))
        self.max_book_staleness_sec = int(cfg.get('max_book_staleness_sec', 3))
        self.min_exchange_balance_usd = Decimal(str(cfg.get('min_exchange_balance_usd', 100)))

        # State
        self._daily_pnl = Decimal('0')
        self._daily_pnl_reset: datetime = datetime.utcnow()
        self._trade_times: deque = deque(maxlen=200)
        self._consecutive_losses = 0
        self._current_exposure: Dict[str, Decimal] = {}  # exchange -> USD
        self._one_sided_exposure = Decimal('0')
        self._halted = False
        self._halt_reason = ""

        logger.info(
            f"ArbitrageRiskEngine initialized: "
            f"max_single=${float(self.max_single_arb_usd)}, "
            f"max_exposure=${float(self.max_total_exposure_usd)}, "
            f"max_daily_loss=${float(self.max_daily_loss_usd)}"
        )

    def approve_arbitrage(self, signal) -> bool:
        """Gate: approve or reject an arbitrage signal."""
        if self._halted:
            logger.debug(f"Rejected — halted: {self._halt_reason}")
            return False

        # Reset daily PnL at midnight
        self._check_daily_reset()

        # Daily loss limit
        if self._daily_pnl < -self.max_daily_loss_usd:
            self._halt("Daily loss limit exceeded")
            return False

        # Consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._halt(f"Consecutive losses: {self._consecutive_losses}")
            return False

        # Trade rate limit
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        recent_trades = sum(1 for t in self._trade_times if t > hour_ago)
        if recent_trades >= self.max_trades_per_hour:
            logger.debug("Rejected — trade rate limit")
            return False

        # Single trade size
        notional = signal.recommended_quantity * (signal.buy_price + signal.sell_price) / 2
        if notional > self.max_single_arb_usd:
            logger.debug(f"Rejected — trade size ${float(notional):.2f} > limit ${float(self.max_single_arb_usd)}")
            return False

        # Total exposure check
        total_exp = sum(self._current_exposure.values())
        if total_exp + notional > self.max_total_exposure_usd:
            logger.debug("Rejected — total exposure limit")
            return False

        # Order book freshness
        if hasattr(signal, 'confidence') and signal.confidence < Decimal('0.3'):
            logger.debug("Rejected — low confidence signal")
            return False

        self._trade_times.append(now)
        return True

    def approve_funding_arb(self, symbol: str, notional_usd: Decimal) -> bool:
        """Gate for funding rate arbitrage positions."""
        if self._halted:
            return False

        if notional_usd > self.max_funding_position_usd:
            return False

        total_exp = sum(self._current_exposure.values())
        if total_exp + notional_usd > self.max_total_exposure_usd:
            return False

        return True

    def record_trade_result(self, profit_usd: Decimal, one_sided: bool = False):
        """Record the result of an executed trade."""
        self._daily_pnl += profit_usd

        if profit_usd < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if one_sided:
            self._one_sided_exposure += abs(profit_usd)
            if self._one_sided_exposure > self.max_one_sided_exposure_usd:
                self._halt("One-sided exposure limit exceeded")

    def update_exposure(self, exchange: str, amount_usd: Decimal):
        self._current_exposure[exchange] = amount_usd

    def reset_halt(self):
        """Manual reset after investigation."""
        self._halted = False
        self._halt_reason = ""
        self._consecutive_losses = 0
        logger.info("Risk engine halt RESET")

    def _halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        logger.error(f"RISK HALT: {reason}")

    def _check_daily_reset(self):
        now = datetime.utcnow()
        if now.date() > self._daily_pnl_reset.date():
            logger.info(f"Daily PnL reset — yesterday: ${float(self._daily_pnl):.2f}")
            self._daily_pnl = Decimal('0')
            self._daily_pnl_reset = now
            # Auto-reset halt if it was due to daily loss
            if self._halted and "Daily loss" in self._halt_reason:
                self.reset_halt()

    def get_status(self) -> dict:
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "daily_pnl_usd": float(self._daily_pnl),
            "consecutive_losses": self._consecutive_losses,
            "total_exposure_usd": float(sum(self._current_exposure.values())),
            "one_sided_exposure_usd": float(self._one_sided_exposure),
            "trades_last_hour": sum(
                1 for t in self._trade_times
                if t > datetime.utcnow() - timedelta(hours=1)
            ),
        }
