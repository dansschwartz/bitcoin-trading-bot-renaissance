"""
Realistic Cross-Exchange Paper Fill — models the REAL costs for simultaneous
execution (inventory-based arb, NOT transfer-based arb).

The system uses SynchronizedExecutor — both legs fire at the same time.
There is NO token transfer between exchanges during a trade. USDT is
pre-funded on both exchanges.

REAL costs per trade:
1. Taker fee on Binance leg (0.075% with BNB discount)
2. MEXC maker fee: 0% (LIMIT_MAKER)
3. Amortized rebalancing: ~$0.01/trade (periodic inventory transfers)

NOT a cost per trade (simultaneous execution):
- Withdrawal fee: NO withdrawal happens during a trade
- Transfer time slippage: both legs execute within milliseconds
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

    The system uses inventory-based arb (both legs fire simultaneously).
    The only real cost per trade is the Binance taker fee (0.075% with BNB
    or 0.1% without). MEXC charges 0% for LIMIT_MAKER orders.

    Periodic inventory rebalancing costs are amortized as a tiny per-trade cost.
    """

    # Binance fee rates
    BINANCE_TAKER_FEE = Decimal('0.00075')   # 0.075% with BNB discount
    MEXC_MAKER_FEE = Decimal('0')            # 0% maker (LIMIT_MAKER)

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('realistic_fill', {})

        # Amortized rebalancing cost per trade
        # Periodic USDT rebalancing between exchanges costs ~$10-20/day
        # across hundreds of trades = negligible per trade
        self.rebalance_cost_per_trade = Decimal(str(
            cfg.get('rebalance_cost_per_trade', 0.01)
        ))

        # Whether Binance BNB fee discount is active
        self.bnb_discount = cfg.get('bnb_discount', True)
        self.binance_taker_fee = (
            self.BINANCE_TAKER_FEE if self.bnb_discount
            else Decimal('0.001')  # 0.1% without BNB
        )

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
        # Taker fee on Binance leg(s) only
        # MEXC = 0% maker (LIMIT_MAKER), Binance = 0.075% (with BNB)
        taker_fee = Decimal('0')
        if buy_exchange == 'binance':
            taker_fee += trade_size_usd * self.binance_taker_fee
        if sell_exchange == 'binance':
            taker_fee += trade_size_usd * self.binance_taker_fee

        # Amortized rebalancing cost (NOT a withdrawal fee per trade)
        rebalance_cost = self.rebalance_cost_per_trade

        # Total realistic cost
        total_cost = taker_fee + rebalance_cost

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
            'binance_taker_fee_bps': float(self.binance_taker_fee * 10000),
            'rebalance_cost_per_trade': float(self.rebalance_cost_per_trade),
        }
