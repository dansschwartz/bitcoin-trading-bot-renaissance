"""
Realistic Cross-Exchange Paper Fill — models costs NOT already captured by
the paper fill simulation.

Each exchange client's _paper_fill() already deducts exchange-specific fees
(MEXC 0% maker, Binance 0.075% taker, KuCoin 0.10% taker) from the
OrderResult.fee_amount, and _calculate_actual_profit() subtracts those.

This module adds ONLY the costs that paper fills DON'T model:
1. Amortized rebalancing: ~$0.01/trade (periodic USDT inventory transfers)

NOT a cost per trade (simultaneous execution):
- Withdrawal fee: NO withdrawal happens during a trade
- Transfer time slippage: both legs execute within milliseconds
- Exchange fees: already deducted in paper fill
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

logger = logging.getLogger("arb.realistic_fill")


@dataclass
class RealisticCostBreakdown:
    """Breakdown of realistic costs applied to a cross-exchange paper fill."""
    withdrawal_fee_usd: Decimal     # Amortized rebalancing cost (NOT per-trade withdrawal)
    taker_fee_usd: Decimal          # Binance taker fee on the Binance leg
    adverse_move_usd: Decimal       # Reserved for future use (0 for simultaneous)
    total_realistic_cost_usd: Decimal
    realistic_profit_usd: Decimal   # original profit - total cost
    edge_survived: bool             # True if realistic profit > 0


class RealisticCrossExchangeFill:
    """Applies realistic cost adjustments to cross-exchange paper fills.

    Exchange fees (MEXC 0% maker, Binance 0.075% taker, KuCoin 0.10% taker)
    are already deducted by each client's _paper_fill() and subtracted in
    _calculate_actual_profit(). This module adds ONLY the residual costs
    that paper fills don't capture: amortized inventory rebalancing.
    """

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('realistic_fill', {})

        # Amortized rebalancing cost per trade
        # Periodic USDT rebalancing between exchanges costs ~$10-20/day
        # across hundreds of trades = negligible per trade
        self.rebalance_cost_per_trade = Decimal(str(
            cfg.get('rebalance_cost_per_trade', 0.01)
        ))

        # Stats
        self._total_applied = 0
        self._total_cost_usd = Decimal('0')
        self._edge_survived_count = 0

    def calculate_realistic_costs(
        self,
        symbol: str,
        trade_size_usd: Decimal,
        paper_profit_usd: Decimal,
        buy_exchange: str = 'binance',
        sell_exchange: str = 'mexc',
    ) -> RealisticCostBreakdown:
        """Calculate realistic costs for a simultaneous cross-exchange fill.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            trade_size_usd: Notional value of the trade
            paper_profit_usd: Profit as calculated by the naive paper fill
            buy_exchange: Exchange where buy order was placed
            sell_exchange: Exchange where sell order was placed

        Returns:
            RealisticCostBreakdown with all cost components
        """
        # Exchange fees are already deducted in paper fill — don't double-count.
        # Only add costs that paper fills don't model.
        taker_fee = Decimal('0')  # Already in paper_profit via _paper_fill() fees

        # Amortized rebalancing cost (NOT a withdrawal fee per trade)
        rebalance_cost = self.rebalance_cost_per_trade

        # Total realistic cost (just rebalancing — fees already deducted)
        total_cost = rebalance_cost

        # Adjusted profit
        realistic_profit = paper_profit_usd - total_cost
        edge_survived = realistic_profit > 0

        # Update stats
        self._total_applied += 1
        self._total_cost_usd += total_cost
        if edge_survived:
            self._edge_survived_count += 1

        return RealisticCostBreakdown(
            withdrawal_fee_usd=rebalance_cost,  # DB compat: rebalance cost in this field
            taker_fee_usd=taker_fee,
            adverse_move_usd=Decimal('0'),  # Not applicable for simultaneous execution
            total_realistic_cost_usd=total_cost,
            realistic_profit_usd=realistic_profit,
            edge_survived=edge_survived,
        )

    def get_stats(self) -> dict:
        """Return statistics on realistic cost adjustments."""
        return {
            'total_applied': self._total_applied,
            'total_realistic_cost_usd': float(self._total_cost_usd),
            'avg_cost_per_trade': float(
                self._total_cost_usd / max(1, self._total_applied)
            ),
            'edge_survived_count': self._edge_survived_count,
            'edge_survival_rate': (
                self._edge_survived_count / max(1, self._total_applied)
            ),
            'rebalance_cost_per_trade': float(self.rebalance_cost_per_trade),
        }
