"""
Arbitrage Cost Model — estimates TOTAL cost of executing an arbitrage trade.

Includes: exchange fees, spread cost, slippage, timing cost.
CRITICAL: If we get this wrong, we think we're profiting when we're losing.
The cost model LEARNS from actual execution data over time.
"""
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, List

from ..exchanges.base import SpeedTier

logger = logging.getLogger("arb.costs")


@dataclass
class CostEstimate:
    total_cost_bps: Decimal
    buy_fee_bps: Decimal
    sell_fee_bps: Decimal
    buy_slippage_bps: Decimal
    sell_slippage_bps: Decimal
    timing_cost_bps: Decimal
    taker_maker_cost_bps: Decimal
    taker_taker_cost_bps: Decimal


class ArbitrageCostModel:

    # Fee schedules — MUST be verified monthly
    FEES: Dict[str, Dict[str, Dict[str, Decimal]]] = {
        "mexc": {
            "spot": {
                "maker": Decimal('0.0000'),   # 0.00% — OUR EDGE
                "taker": Decimal('0.0005'),   # 0.05%
            },
            "futures": {
                "maker": Decimal('0.0000'),
                "taker": Decimal('0.0002'),
            },
        },
        "binance": {
            "spot": {
                "maker": Decimal('0.00075'),  # 0.075% with BNB discount
                "taker": Decimal('0.00075'),
            },
            "futures": {
                "maker": Decimal('0.0002'),
                "taker": Decimal('0.0005'),
            },
        },
    }

    def __init__(self):
        self._cost_history: deque = deque(maxlen=10000)
        self._slippage_history: Dict[str, deque] = {}  # per symbol+exchange
        self._estimation_errors: deque = deque(maxlen=1000)

    def estimate_arbitrage_cost(
        self,
        symbol: str,
        buy_exchange: str,
        sell_exchange: str,
        buy_price: Decimal,
        sell_price: Decimal,
        quantity: Optional[Decimal] = None,
        market_type: str = "spot",
        speed_tier: SpeedTier = SpeedTier.MAKER_MAKER,
    ) -> CostEstimate:
        """Estimate total roundtrip cost of an arbitrage trade."""
        buy_fee_maker = self.FEES[buy_exchange][market_type]["maker"]
        sell_fee_maker = self.FEES[sell_exchange][market_type]["maker"]
        buy_fee_taker = self.FEES[buy_exchange][market_type]["taker"]
        sell_fee_taker = self.FEES[sell_exchange][market_type]["taker"]

        buy_slippage = self._estimate_slippage(symbol, buy_exchange, quantity)
        sell_slippage = self._estimate_slippage(symbol, sell_exchange, quantity)

        timing_cost = Decimal('0.5')  # Conservative 0.5 bps

        # Maker-maker (default — our preferred mode)
        total_mm = (
            buy_fee_maker * 10000 + sell_fee_maker * 10000
            + buy_slippage + sell_slippage + timing_cost
        )

        # Taker-maker (when we need speed on buy side)
        total_tm = (
            buy_fee_taker * 10000 + sell_fee_maker * 10000
            + buy_slippage + sell_slippage + timing_cost
        )

        # Taker-taker (fastest, most expensive)
        total_tt = (
            buy_fee_taker * 10000 + sell_fee_taker * 10000
            + buy_slippage + sell_slippage + timing_cost
        )

        # Select total_cost_bps based on requested speed tier
        if speed_tier == SpeedTier.TAKER_TAKER:
            selected_total = total_tt
        elif speed_tier == SpeedTier.TAKER_MAKER:
            selected_total = total_tm
        else:
            selected_total = total_mm

        return CostEstimate(
            total_cost_bps=selected_total,
            buy_fee_bps=buy_fee_maker * 10000,
            sell_fee_bps=sell_fee_maker * 10000,
            buy_slippage_bps=buy_slippage,
            sell_slippage_bps=sell_slippage,
            timing_cost_bps=timing_cost,
            taker_maker_cost_bps=total_tm,
            taker_taker_cost_bps=total_tt,
        )

    def _estimate_slippage(
        self, symbol: str, exchange: str, quantity: Optional[Decimal]
    ) -> Decimal:
        """Estimate slippage. Learns from execution history."""
        key = f"{symbol}_{exchange}"

        # Use learned slippage if we have enough data
        if key in self._slippage_history and len(self._slippage_history[key]) >= 10:
            recent = list(self._slippage_history[key])[-50:]
            avg = sum(recent) / len(recent)
            return avg * Decimal('1.2')  # 20% safety margin

        # Default conservative estimates
        base_slippage = {
            "mexc": Decimal('1.5'),
            "binance": Decimal('0.5'),
        }
        return base_slippage.get(exchange, Decimal('2.0'))

    def update_from_execution(self, trade_result: dict):
        """After every trade, compare estimated vs realized cost. LEARN."""
        estimated = trade_result.get('estimated_cost_bps', Decimal('0'))
        realized = trade_result.get('realized_cost_bps', Decimal('0'))
        error = realized - estimated

        self._cost_history.append({
            "symbol": trade_result.get('symbol'),
            "exchange": trade_result.get('exchange'),
            "quantity": trade_result.get('quantity'),
            "estimated": estimated,
            "realized": realized,
            "error": error,
            "timestamp": datetime.utcnow(),
        })

        # Update slippage model
        actual_slippage = trade_result.get('actual_slippage_bps')
        if actual_slippage is not None:
            key = f"{trade_result['symbol']}_{trade_result['exchange']}"
            if key not in self._slippage_history:
                self._slippage_history[key] = deque(maxlen=200)
            self._slippage_history[key].append(actual_slippage)

        self._estimation_errors.append(error)

        if abs(error) > Decimal('2.0'):
            logger.warning(
                f"Cost estimation error: {trade_result.get('symbol')} "
                f"on {trade_result.get('exchange')} — "
                f"estimated={float(estimated):.2f}bps, realized={float(realized):.2f}bps, "
                f"error={float(error):.2f}bps"
            )

    def get_model_accuracy(self) -> dict:
        """Return stats on cost estimation accuracy."""
        if not self._estimation_errors:
            return {"samples": 0, "mean_error_bps": 0, "max_error_bps": 0}
        errors = [float(e) for e in self._estimation_errors]
        return {
            "samples": len(errors),
            "mean_error_bps": sum(abs(e) for e in errors) / len(errors),
            "max_error_bps": max(abs(e) for e in errors),
            "bias_bps": sum(errors) / len(errors),  # positive = underestimate costs
        }
