"""
Realistic Cross-Exchange Paper Fill — models the REAL costs that the naive
paper fill engine ignores, eliminating phantom P&L.

Cross-exchange arb requires inventory on BOTH exchanges. After each trade,
inventory drifts: the buy-side accumulates base tokens, the sell-side
accumulates USDT. Eventually you must rebalance (withdraw + deposit),
which costs real money in:

1. Withdrawal fees (network/exchange)
2. Taker fees (at least one leg needs speed in practice)
3. Adverse price movement (spread decay during execution window)
4. Opportunity cost of locked capital during transfer

This module wraps execution results with realistic cost adjustments.
"""
import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

logger = logging.getLogger("arb.realistic_fill")


@dataclass
class RealisticCostBreakdown:
    """Breakdown of realistic costs applied to a cross-exchange paper fill."""
    withdrawal_fee_usd: Decimal     # Amortized withdrawal fee for rebalancing
    taker_fee_usd: Decimal          # Extra fee for the taker leg
    adverse_move_usd: Decimal       # Spread decay during execution window
    total_realistic_cost_usd: Decimal
    realistic_profit_usd: Decimal   # original profit - total cost
    edge_survived: bool             # True if realistic profit > 0


class RealisticCrossExchangeFill:
    """Applies realistic cost adjustments to cross-exchange paper fills.

    The key insight: cross-exchange arb looks much more profitable on paper
    than in practice because the paper engine assumes:
    - 0% fee on both sides (MEXC maker + Binance maker)
    - Instant simultaneous execution
    - No inventory rebalancing costs
    - No adverse price movement

    In reality:
    - At least one leg is effectively taker (speed matters)
    - Withdrawal fees eat 10-100bps on small trades
    - Spread can decay 1-3bps during execution window
    - You need to rebalance every N trades
    """

    # Withdrawal fees by token (in USD) — sourced from MEXC/Binance fee pages
    # These are the cheapest network for each token
    DEFAULT_WITHDRAWAL_FEES: Dict[str, float] = {
        # Major tokens
        'BTC': 8.00,       # Lightning or on-chain (~0.0001 BTC)
        'ETH': 2.50,       # Arbitrum/Optimism bridge
        'SOL': 0.01,       # Solana native (practically free)
        'BNB': 0.05,       # BSC native
        'XRP': 0.25,       # XRP Ledger
        'DOGE': 2.00,      # DOGE network
        'ADA': 1.00,       # Cardano native
        'AVAX': 0.10,      # Avalanche C-Chain
        'LINK': 0.30,      # Arbitrum
        'DOT': 1.00,       # Polkadot native
        'NEAR': 0.10,      # NEAR native
        'ALGO': 0.10,      # Algorand
        'ATOM': 0.01,      # Cosmos
        'FIL': 0.10,       # Filecoin
        'IMX': 0.50,       # Immutable X
        'ARB': 0.10,       # Arbitrum native
        'OP': 0.10,        # Optimism native
        'APE': 0.50,       # ERC-20
        'SAND': 0.50,      # ERC-20
        'MANA': 0.50,      # ERC-20
        'GALA': 0.50,      # ERC-20
        'ENJ': 0.50,       # ERC-20
        'USDT': 1.00,      # TRC-20 (cheapest stable transfer)
    }
    DEFAULT_FEE_FALLBACK = 1.50     # Conservative default for unlisted tokens

    def __init__(self, config: Optional[dict] = None):
        cfg = (config or {}).get('realistic_fill', {})

        # How many trades before needing to rebalance inventory
        self.trades_per_rebalance = cfg.get('trades_per_rebalance', 15)

        # Adverse price movement model (bps lost during execution window)
        self.adverse_move_bps = Decimal(str(cfg.get('adverse_move_bps', 1.5)))

        # Taker fee penalty — in practice, one leg must cross the spread
        # MEXC taker = 5bps, Binance taker = 7.5bps
        # We model a blended penalty assuming one leg is taker ~50% of the time
        self.taker_penalty_bps = Decimal(str(cfg.get('taker_penalty_bps', 3.0)))

        # Custom withdrawal fees from config (override defaults)
        custom_fees = cfg.get('withdrawal_fees', {})
        self.withdrawal_fees = dict(self.DEFAULT_WITHDRAWAL_FEES)
        self.withdrawal_fees.update(custom_fees)

        # Stats
        self._total_applied = 0
        self._total_cost_usd = Decimal('0')
        self._edge_survived_count = 0

    def calculate_realistic_costs(
        self,
        symbol: str,
        trade_size_usd: Decimal,
        paper_profit_usd: Decimal,
    ) -> RealisticCostBreakdown:
        """Calculate realistic costs for a cross-exchange paper fill.

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            trade_size_usd: Notional value of the trade
            paper_profit_usd: Profit as calculated by the naive paper fill

        Returns:
            RealisticCostBreakdown with all cost components
        """
        base = symbol.split('/')[0] if '/' in symbol else symbol

        # 1. Amortized withdrawal fee
        # After each trade, you need to eventually move tokens back.
        # The withdrawal fee is amortized over trades_per_rebalance trades.
        raw_withdrawal_fee = Decimal(str(
            self.withdrawal_fees.get(base, self.DEFAULT_FEE_FALLBACK)
        ))
        # Also need to rebalance USDT back to the buy exchange
        usdt_withdrawal_fee = Decimal(str(
            self.withdrawal_fees.get('USDT', 1.00)
        ))
        # Total rebalance cost = withdraw base + withdraw USDT
        total_rebalance_fee = raw_withdrawal_fee + usdt_withdrawal_fee
        amortized_withdrawal = total_rebalance_fee / self.trades_per_rebalance

        # 2. Taker fee penalty
        # In practice, cross-exchange arb requires speed. At least one leg
        # must cross the spread (taker). The paper engine assumes both legs
        # are maker, which is unrealistic.
        taker_fee_usd = trade_size_usd * (self.taker_penalty_bps / Decimal('10000'))

        # 3. Adverse price movement
        # During the ~100-500ms execution window, the spread can narrow.
        # This is especially true for pairs discovered by the scanner —
        # other bots are also watching these spreads.
        adverse_move_usd = trade_size_usd * (self.adverse_move_bps / Decimal('10000'))

        # Total realistic cost
        total_cost = amortized_withdrawal + taker_fee_usd + adverse_move_usd

        # Adjusted profit
        realistic_profit = paper_profit_usd - total_cost
        edge_survived = realistic_profit > 0

        # Update stats
        self._total_applied += 1
        self._total_cost_usd += total_cost
        if edge_survived:
            self._edge_survived_count += 1

        return RealisticCostBreakdown(
            withdrawal_fee_usd=amortized_withdrawal,
            taker_fee_usd=taker_fee_usd,
            adverse_move_usd=adverse_move_usd,
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
            'trades_per_rebalance': self.trades_per_rebalance,
            'adverse_move_bps': float(self.adverse_move_bps),
            'taker_penalty_bps': float(self.taker_penalty_bps),
        }
