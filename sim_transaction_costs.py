"""Transaction cost model — "The Devil".

Maker/taker fees, slippage (base + vol + volume), bid-ask half-spread,
funding rates, regime-dependent multipliers.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_config import TradeCost


class SimTransactionCostModel:
    """Realistic per-trade cost calculation.

    Cost components:
    - Maker / taker fee
    - Slippage: base_bps + vol_coeff * volatility * sqrt(participation) + vol_slippage * sqrt(participation)
    - Half bid-ask spread
    - Funding rate (for perpetuals; proportional to holding days)
    - Regime multiplier (normal=1x, volatile=2x, crisis=3x)
    """

    REGIME_MULTIPLIERS = {
        "normal": 1.0,
        "volatile": 2.0,
        "crisis": 3.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or {}
        self.logger = logger or logging.getLogger(__name__)

        self.maker_fee = cfg.get("maker_fee", 0.001)
        self.taker_fee = cfg.get("taker_fee", 0.002)
        self.base_slippage_bps = cfg.get("base_slippage_bps", 5.0)
        self.vol_slippage_coeff = cfg.get("vol_slippage_coeff", 0.1)
        self.volume_slippage_coeff = cfg.get("volume_slippage_coeff", 0.05)
        self.half_spread_bps = cfg.get("half_spread_bps", 3.0)
        self.funding_rate_daily = cfg.get("funding_rate_daily", 0.0001)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_cost(
        self,
        trade_size_usd: float,
        price: float,
        volatility: float = 0.02,
        daily_volume: float = 1e9,
        is_maker: bool = False,
        holding_days: int = 0,
        regime: str = "normal",
    ) -> TradeCost:
        """Compute full trade cost breakdown.

        Args:
            trade_size_usd: Notional size of the trade in USD.
            price: Current asset price.
            volatility: Recent daily return volatility.
            daily_volume: Estimated daily USD volume of the asset.
            is_maker: Whether this is a maker (limit) order.
            holding_days: Days position is held (for funding cost).
            regime: One of 'normal', 'volatile', 'crisis'.
        """
        abs_size = abs(trade_size_usd)

        # --- Fee ---
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee_cost = abs_size * fee_rate

        # --- Slippage (sqrt market impact model) ---
        participation = min(abs_size / max(daily_volume, 1.0), 0.3)
        base_slip = self.base_slippage_bps * 1e-4 * abs_size
        vol_slip = self.vol_slippage_coeff * volatility * np.sqrt(participation) * abs_size
        vol_dep_slip = self.volume_slippage_coeff * np.sqrt(participation) * abs_size
        slippage = base_slip + vol_slip + vol_dep_slip

        # --- Half-spread ---
        half_spread = self.half_spread_bps * 1e-4 * abs_size

        # --- Funding ---
        funding = abs_size * self.funding_rate_daily * max(holding_days, 0)

        # --- Regime multiplier ---
        regime_mult = self.REGIME_MULTIPLIERS.get(regime, 1.0)
        total = (fee_cost + slippage + half_spread + funding) * regime_mult

        return TradeCost(
            maker_fee=fee_cost if is_maker else 0.0,
            taker_fee=fee_cost if not is_maker else 0.0,
            slippage=slippage * regime_mult,
            half_spread=half_spread * regime_mult,
            funding_cost=funding * regime_mult,
            total=total,
            breakdown={
                "fee": fee_cost,
                "base_slippage": base_slip,
                "vol_slippage": vol_slip,
                "volume_slippage": vol_dep_slip,
                "half_spread": half_spread,
                "funding": funding,
                "regime_multiplier": regime_mult,
            },
        )

    def cost_in_bps(self, cost: TradeCost, trade_size_usd: float) -> float:
        """Express total cost in basis points of trade size."""
        if trade_size_usd == 0:
            return 0.0
        return cost.total / abs(trade_size_usd) * 1e4

    def apply_costs_to_returns(
        self,
        returns: np.ndarray,
        trade_mask: np.ndarray,
        prices: np.ndarray,
        volatilities: np.ndarray,
        volumes: np.ndarray,
        trade_size_usd: float = 10_000.0,
        regime: str = "normal",
    ) -> np.ndarray:
        """Subtract transaction costs from returns where *trade_mask* is True.

        Returns a copy of *returns* with costs deducted at trade points.
        """
        adj = returns.copy()
        for i in range(len(returns)):
            if trade_mask[i]:
                cost = self.calculate_cost(
                    trade_size_usd=trade_size_usd,
                    price=prices[i] if i < len(prices) else prices[-1],
                    volatility=volatilities[i] if i < len(volatilities) else 0.02,
                    daily_volume=volumes[i] if i < len(volumes) else 1e9,
                    regime=regime,
                )
                adj[i] -= cost.total / max(trade_size_usd, 1.0)
        return adj
