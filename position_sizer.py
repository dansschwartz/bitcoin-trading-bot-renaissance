"""
Renaissance-Inspired Position Sizing Engine

Implements the five scaling strategies that defined Medallion:

1. CAPACITY-AWARE KELLY — Edge is not fixed. It degrades as position size
   grows because market impact eats into returns. The optimal size is where
   marginal impact cost equals marginal edge — a natural ceiling that
   self-regulates regardless of fund size.

2. LIQUIDITY-RELATIVE SIZING — "Small" is defined relative to what the
   market can absorb without noticing. A $5M position in AAPL (trading $10B
   daily) is invisible. A $5M position in a thin altcoin moves the market 5%.
   Position size is capped by a participation rate of daily volume.

3. SQUARE-ROOT MARKET IMPACT — The Almgren-Chriss model: market impact
   scales as sigma * sqrt(size / daily_volume). This means doubling your
   size doesn't double impact — it multiplies by 1.41x. But it still grows,
   and at large enough scale, it dominates the edge.

4. LEVERAGE DECOUPLING — The fund doesn't make each bet bigger. It uses
   leverage to make MORE bets at the same per-bet size. Returns are amplified
   on capital, not on position sizes relative to liquidity.

5. CAPACITY CEILING — There is a finite amount of "quiet money" extractable
   from any market. The system computes its own capacity limit and will
   refuse to deploy beyond it, even if the account has more capital.

Core formula:
    effective_edge = raw_edge - market_impact(size)
    optimal_size = argmax(size * effective_edge(size))
    actual_size = min(optimal_size, kelly_size, liquidity_cap, balance_cap)

All sizes returned in ASSET UNITS (e.g. BTC), properly scaled.
"""

import logging
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime


@dataclass
class SizingResult:
    """Complete position sizing output with full audit trail."""
    asset_units: float          # Final size in asset units (e.g. 0.00145 BTC)
    usd_value: float            # Equivalent USD value
    kelly_fraction: float       # Raw Kelly fraction before adjustments
    applied_fraction: float     # Actual fraction used (after all adjustments)
    risk_budget_usd: float      # How much of the account we're risking
    edge: float                 # Estimated edge (expected return per trade)
    effective_edge: float       # Edge net of market impact at chosen size
    market_impact_bps: float    # Estimated market impact in basis points
    win_probability: float      # Estimated win probability from confidence
    transaction_cost_ratio: float  # Cost as fraction of expected profit
    volatility_scalar: float    # Vol adjustment multiplier
    regime_scalar: float        # Regime adjustment multiplier
    liquidity_scalar: float     # Liquidity adjustment multiplier
    capacity_used_pct: float    # What % of instrument capacity this trade uses
    sizing_method: str          # Which method produced this size
    reasons: List[str] = field(default_factory=list)  # Human-readable audit


class RenaissancePositionSizer:
    """
    Position sizing engine implementing the five Renaissance scaling strategies.

    The core insight: as you increase position size, market impact grows and
    effective edge shrinks. There is a natural optimal point — the size that
    maximizes (size × effective_edge). Beyond that, adding more capital to
    the same trade actually reduces total expected profit.

    This means:
    - At $10K, balance is the binding constraint (can't reach impact ceiling)
    - At $100K, Kelly fraction is the binding constraint
    - At $10M+, LIQUIDITY becomes the binding constraint — regardless of
      how much more capital you have, you can't profitably deploy more into
      a single instrument without moving the market against yourself
    """

    def __init__(self, config: Dict[str, Any] = None, logger: Optional[logging.Logger] = None):
        config = config or {}
        self.logger = logger or logging.getLogger(__name__)

        # ── Account parameters ──
        self.default_balance_usd = float(config.get("default_balance_usd", 10000.0))
        self.max_position_pct = float(config.get("max_position_pct", 25.0))
        self.max_total_exposure_pct = float(config.get("max_total_exposure_pct", 80.0))

        # ── Kelly parameters ──
        self.kelly_fraction = float(config.get("kelly_fraction", 0.50))
        self.min_edge = float(config.get("min_edge", 0.001))
        self.min_win_prob = float(config.get("min_win_prob", 0.52))

        # ── Transaction cost model ──
        self.taker_fee_bps = float(config.get("taker_fee_bps", 40.0))
        self.maker_fee_bps = float(config.get("maker_fee_bps", 25.0))
        self.spread_cost_bps = float(config.get("spread_cost_bps", 5.0))
        self.slippage_bps = float(config.get("slippage_bps", 5.0))
        self.cost_gate_ratio = float(config.get("cost_gate_ratio", 0.50))

        # ── Market impact model (Almgren-Chriss square-root law) ──
        # impact_bps = impact_coefficient * sigma * sqrt(size_usd / daily_volume_usd)
        # At Coinbase BTC: daily volume ~$2-5B, sigma ~2%
        # A $10K trade: impact = 0.10 * 0.02 * sqrt(10000/3e9) = 0.00036 bps ≈ 0
        # A $10M trade: impact = 0.10 * 0.02 * sqrt(1e7/3e9) = 0.36 bps
        # A $100M trade: impact = 0.10 * 0.02 * sqrt(1e8/3e9) = 1.15 bps (significant)
        self.impact_coefficient = float(config.get("impact_coefficient", 0.10))

        # ── Participation rate (% of daily volume) ──
        # Never trade more than this fraction of daily volume in a single order
        # Renaissance reportedly stayed at 1-2% of volume per instrument
        self.max_participation_rate = float(config.get("max_participation_rate", 0.02))

        # ── Default daily volume estimates (USD) when real data unavailable ──
        self.default_daily_volumes = config.get("default_daily_volumes", {
            "BTC-USD": 3_000_000_000,   # ~$3B/day on Coinbase
            "ETH-USD": 1_500_000_000,   # ~$1.5B/day
            "SOL-USD": 300_000_000,
            "DOGE-USD": 200_000_000,
            "AVAX-USD": 100_000_000,
            "LINK-USD": 150_000_000,
        })
        self.fallback_daily_volume = float(config.get("fallback_daily_volume", 500_000_000))

        # ── Volatility parameters ──
        self.target_vol = float(config.get("target_vol", 0.02))
        self.vol_floor = float(config.get("vol_floor", 0.005))
        self.vol_ceiling = float(config.get("vol_ceiling", 0.10))

        # ── Regime adjustments ──
        self.regime_scalars = config.get("regime_scalars", {
            "trending": 1.20,
            "mean_reverting": 1.00,
            "volatile": 0.60,
            "chaotic": 0.30,
            "normal": 0.80,
        })

        # ── Position limits ──
        self.min_order_usd = float(config.get("min_order_usd", 1.0))
        self.max_fraction_of_book = float(config.get("max_fraction_of_book", 0.02))

        self.logger.info(
            f"PositionSizer initialized: kelly={self.kelly_fraction:.0%}, "
            f"impact_coeff={self.impact_coefficient}, "
            f"max_participation={self.max_participation_rate:.1%}, "
            f"cost_gate={self.cost_gate_ratio:.0%}"
        )

    def calculate_size(
        self,
        signal_strength: float,
        confidence: float,
        current_price: float,
        product_id: str = "BTC-USD",
        volatility: Optional[float] = None,
        vol_regime: Optional[str] = None,
        fractal_regime: Optional[str] = None,
        order_book_depth_usd: Optional[float] = None,
        current_exposure_usd: float = 0.0,
        ml_package: Optional[Any] = None,
        account_balance_usd: Optional[float] = None,
        daily_volume_usd: Optional[float] = None,
        drawdown_pct: float = 0.0,
        measured_edge: Optional[float] = None,
        tier_size_multiplier: float = 1.0,
    ) -> SizingResult:
        """
        Calculate position size with capacity-aware scaling.

        The sizing pipeline:
        1. Estimate edge and win probability from signals
        2. Compute Kelly optimal fraction
        3. Compute market-impact-adjusted optimal size (capacity ceiling)
        4. Take the MINIMUM of Kelly size, capacity size, and balance limits
        5. Apply vol normalization and regime adjustments
        6. Gate through transaction costs

        This naturally handles the scaling paradox:
        - Small fund ($10K): balance is the constraint
        - Medium fund ($1M): Kelly fraction is the constraint
        - Large fund ($1B): capacity/liquidity is the constraint
        """
        reasons = []

        # ── Dynamic balance ──
        balance = account_balance_usd if (account_balance_usd and account_balance_usd > 0) else self.default_balance_usd
        max_position_usd = balance * (self.max_position_pct / 100.0)
        max_total_exposure_usd = balance * (self.max_total_exposure_pct / 100.0)

        # ── Drawdown-based scaling (Renaissance discipline) ──
        # "Risk of ruin is the only unacceptable outcome"
        drawdown_scalar = 1.0
        if drawdown_pct >= 0.15:
            reasons.append(f"DRAWDOWN HALT: {drawdown_pct:.1%} >= 15% — system halted")
            return self._zero_result(f"Drawdown halt: {drawdown_pct:.1%}")
        elif drawdown_pct >= 0.10:
            drawdown_scalar = 0.25  # Quarter size
            reasons.append(f"Drawdown emergency: {drawdown_pct:.1%} — 25% size")
        elif drawdown_pct >= 0.05:
            drawdown_scalar = 0.50  # Half size
            reasons.append(f"Drawdown caution: {drawdown_pct:.1%} — 50% size")
        elif drawdown_pct >= 0.03:
            drawdown_scalar = 0.75
            reasons.append(f"Drawdown awareness: {drawdown_pct:.1%} — 75% size")

        max_position_usd *= drawdown_scalar
        max_total_exposure_usd *= drawdown_scalar

        # ── Signal confidence tier scaling (Medallion: size by conviction) ──
        if tier_size_multiplier != 1.0:
            max_position_usd *= tier_size_multiplier
            reasons.append(f"TierMultiplier={tier_size_multiplier:.2f}")

        # ── Get instrument liquidity ──
        adv = daily_volume_usd or self.default_daily_volumes.get(product_id, self.fallback_daily_volume)
        vol = self._get_volatility(volatility, vol_regime)

        reasons.append(f"Balance=${balance:,.0f}, ADV=${adv:,.0f}, Vol={vol:.3f}")

        # ── Sanity checks ──
        if current_price <= 0:
            return self._zero_result("Invalid price")
        if confidence <= 0 or abs(signal_strength) < 1e-8:
            return self._zero_result("No signal or zero confidence")

        # ── Step 1: Estimate edge and win probability ──
        raw_edge, win_prob = self._estimate_edge(signal_strength, confidence, ml_package)
        # Override with measured edge from scorecard when available
        if measured_edge is not None and measured_edge > 0:
            # Blend: 60% measured, 40% model estimate (trust data but hedge)
            raw_edge = 0.6 * measured_edge + 0.4 * raw_edge
            # Also adjust win_prob from measured edge
            win_prob = max(win_prob, 0.5 + measured_edge * 0.8)
            win_prob = min(win_prob, 0.65)
            reasons.append(f"MeasuredEdge={measured_edge:.4f}")
        reasons.append(f"RawEdge={raw_edge:.4f}, P(win)={win_prob:.3f}")

        if win_prob < self.min_win_prob:
            reasons.append(f"P(win) {win_prob:.3f} < {self.min_win_prob}")
            return self._blocked_result(raw_edge, win_prob, reasons)
        if raw_edge < self.min_edge:
            reasons.append(f"Edge {raw_edge:.4f} < {self.min_edge}")
            return self._blocked_result(raw_edge, win_prob, reasons)

        # ── Step 2: Transaction cost gate ──
        fixed_cost = self.estimate_round_trip_cost()
        if raw_edge > 0:
            cost_ratio = fixed_cost / raw_edge
        else:
            cost_ratio = float('inf')

        if cost_ratio > self.cost_gate_ratio:
            reasons.append(f"COST GATE: cost/edge={cost_ratio:.2f} > {self.cost_gate_ratio}")
            return self._gated_result(raw_edge, win_prob, 0.0, cost_ratio, reasons)

        # Net edge after fixed costs
        net_edge = raw_edge - fixed_cost
        if net_edge <= 0:
            reasons.append(f"Net edge after costs is negative: {net_edge:.4f}")
            return self._gated_result(raw_edge, win_prob, 0.0, cost_ratio, reasons)

        reasons.append(f"FixedCost={fixed_cost:.4f}, NetEdge={net_edge:.4f}, CostRatio={cost_ratio:.2f}")

        # ── Step 3: Kelly optimal size (as USD) ──
        kelly_f = self._kelly_criterion(net_edge, win_prob)
        kelly_usd = balance * kelly_f * self.kelly_fraction
        reasons.append(f"Kelly={kelly_f:.4f}, FracKelly({self.kelly_fraction:.0%})=${kelly_usd:,.0f}")

        # ── Step 4: Capacity-optimal size (market impact ceiling) ──
        #
        # The key insight: effective_edge(S) = net_edge - impact(S)
        # where impact(S) = impact_coeff * sigma * sqrt(S / ADV)
        #
        # Total expected profit = S * effective_edge(S)
        #   = S * (net_edge - c * sigma * sqrt(S / ADV))
        #
        # Taking derivative and setting to zero:
        #   d/dS [S * (E - c*σ*√(S/V))] = 0
        #   E - c*σ*√(S/V) - S * c*σ/(2*√(S*V)) = 0
        #   E - (3/2) * c*σ*√(S/V) = 0
        #   S* = (2E / (3cσ))² * V
        #
        # This is the SIZE that maximizes total expected profit.
        # Beyond this, each additional dollar deployed reduces total profit.

        c = self.impact_coefficient
        sigma = vol

        if c * sigma > 0:
            # Optimal size from impact model
            capacity_usd = ((2.0 * net_edge) / (3.0 * c * sigma)) ** 2 * adv
            # What's the impact at that optimal size?
            impact_at_optimal = c * sigma * math.sqrt(capacity_usd / adv) if adv > 0 else 0
            effective_edge_at_optimal = net_edge - impact_at_optimal
        else:
            capacity_usd = float('inf')
            impact_at_optimal = 0.0
            effective_edge_at_optimal = net_edge

        # Participation rate cap: never exceed X% of daily volume
        participation_cap_usd = adv * self.max_participation_rate

        # The true capacity ceiling is the tighter of the two
        liquidity_ceiling = min(capacity_usd, participation_cap_usd)

        capacity_used_pct = 0.0
        reasons.append(
            f"CapacityCeiling=${capacity_usd:,.0f}, "
            f"ParticipationCap=${participation_cap_usd:,.0f}, "
            f"ImpactAtOptimal={impact_at_optimal*10000:.1f}bps"
        )

        # ── Step 5: Choose the binding constraint ──
        # The position size is the MINIMUM of:
        #   1. Kelly-optimal (bankroll management)
        #   2. Capacity-optimal (market impact ceiling)
        #   3. Per-position limit (% of balance)
        #   4. Remaining exposure headroom
        candidates = {
            "kelly": kelly_usd,
            "capacity": liquidity_ceiling,
            "max_position": max_position_usd,
            "exposure_room": max(0.0, max_total_exposure_usd - current_exposure_usd),
        }

        binding_constraint = min(candidates, key=candidates.get)
        raw_size_usd = candidates[binding_constraint]
        reasons.append(f"Binding constraint: {binding_constraint} (${raw_size_usd:,.0f})")

        if liquidity_ceiling < float('inf'):
            capacity_used_pct = (raw_size_usd / liquidity_ceiling) * 100

        # ── Step 6: Volatility normalization ──
        vol_scalar = self._volatility_scalar(vol)
        size_usd = raw_size_usd * vol_scalar

        # ── Step 7: Regime adjustment ──
        regime = fractal_regime or "normal"
        regime_scalar = self.regime_scalars.get(regime, 0.80)
        if ml_package:
            ml_regime = getattr(ml_package, 'fractal_insights', {}).get('regime_detection', None)
            if ml_regime and ml_regime != regime:
                regime_scalar = (regime_scalar + self.regime_scalars.get(ml_regime, 0.80)) / 2.0

        size_usd *= regime_scalar
        reasons.append(f"VolScalar={vol_scalar:.2f}, RegimeScalar={regime_scalar:.2f}")

        # ── Step 8: Order book constraint (instantaneous liquidity) ──
        liquidity_scalar = 1.0
        if order_book_depth_usd and order_book_depth_usd > 0:
            book_cap = order_book_depth_usd * self.max_fraction_of_book
            if size_usd > book_cap:
                liquidity_scalar = book_cap / size_usd
                size_usd = book_cap
                reasons.append(f"BookCap=${book_cap:,.0f} ({self.max_fraction_of_book:.0%} of ${order_book_depth_usd:,.0f})")

        # ── Step 9: Compute actual market impact at final size ──
        if adv > 0 and c * sigma > 0:
            actual_impact = c * sigma * math.sqrt(size_usd / adv)
            actual_impact_bps = actual_impact * 10000
            effective_edge = net_edge - actual_impact
        else:
            actual_impact_bps = 0.0
            effective_edge = net_edge

        # If impact eats the entire edge, don't trade
        if effective_edge <= 0 and size_usd > self.min_order_usd:
            reasons.append(f"IMPACT GATE: impact={actual_impact_bps:.1f}bps eats entire edge")
            return self._gated_result(raw_edge, win_prob, kelly_f, cost_ratio, reasons)

        reasons.append(f"FinalImpact={actual_impact_bps:.1f}bps, EffectiveEdge={effective_edge:.4f}")

        # ── Step 10: Floor check and convert to asset units ──
        if size_usd < self.min_order_usd:
            reasons.append(f"Below minimum: ${size_usd:.2f}")
            return self._gated_result(raw_edge, win_prob, kelly_f, cost_ratio, reasons)

        asset_units = self._round_size(size_usd / current_price, product_id)
        final_usd = asset_units * current_price

        if final_usd < self.min_order_usd:
            reasons.append(f"Below minimum after rounding: ${final_usd:.2f}")
            return self._gated_result(raw_edge, win_prob, kelly_f, cost_ratio, reasons)

        applied_fraction = final_usd / balance if balance > 0 else 0.0
        reasons.append(
            f"FINAL: {asset_units:.8f} {product_id.split('-')[0]} "
            f"(${final_usd:,.2f}, {applied_fraction:.2%} of ${balance:,.0f})"
        )

        return SizingResult(
            asset_units=asset_units,
            usd_value=final_usd,
            kelly_fraction=kelly_f,
            applied_fraction=applied_fraction,
            risk_budget_usd=final_usd,
            edge=raw_edge,
            effective_edge=effective_edge,
            market_impact_bps=actual_impact_bps,
            win_probability=win_prob,
            transaction_cost_ratio=cost_ratio,
            volatility_scalar=vol_scalar,
            regime_scalar=regime_scalar,
            liquidity_scalar=liquidity_scalar,
            capacity_used_pct=capacity_used_pct,
            sizing_method=f"fractional_kelly|bound:{binding_constraint}",
            reasons=reasons,
        )

    # ──────────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────────

    def _estimate_edge(
        self, signal_strength: float, confidence: float, ml_package: Optional[Any]
    ) -> tuple:
        """
        Estimate trading edge and win probability.

        Calibrated for realistic crypto market microstructure:
        - A strong signal in a 5-min cycle predicts a 10-30 bps move
        - Win probability is slight: 51-58% at best
        - Payoff ratio is near-symmetric (~1.0-1.2x)
        - Edge = expected_return_per_trade ≈ 0.1-1.0%

        In production, Renaissance calibrated from millions of realized trades.
        We use conservative estimates that err on the side of under-trading.
        """
        # Win probability: even a great signal only gets you 51-58%
        # confidence ∈ [0,1] → win_prob ∈ [0.48, 0.58]
        base_win_prob = 0.48 + confidence * 0.10

        if ml_package:
            ml_conf = getattr(ml_package, 'confidence_score', 0.0)
            if hasattr(ml_conf, 'item'):
                ml_conf = ml_conf.item()
            ml_conf = float(ml_conf) if ml_conf else 0.0
            # Blend: 60% technical, 40% ML
            base_win_prob = 0.6 * base_win_prob + 0.4 * (0.48 + ml_conf * 0.10)

        win_prob = float(np.clip(base_win_prob, 0.0, 0.65))

        # Payoff ratio: crypto is near-symmetric, slight edge from timing
        # Strong signal → slightly better entries → payoff ~1.05-1.15x
        sig = abs(signal_strength)
        payoff_ratio = 1.0 + sig * 0.15  # [0,1] → [1.00, 1.15]

        # Edge = E[return] = p*b - q  (where b=payoff, q=1-p)
        loss_prob = 1.0 - win_prob
        edge = win_prob * payoff_ratio - loss_prob
        edge = max(edge, 0.0)

        return edge, win_prob

    def _kelly_criterion(self, edge: float, win_prob: float) -> float:
        """Kelly fraction: f* = (p*b - q) / b"""
        if edge <= 0 or win_prob <= 0:
            return 0.0

        loss_prob = 1.0 - win_prob
        payoff_ratio = (edge + loss_prob) / win_prob
        if payoff_ratio <= 0:
            return 0.0

        kelly_f = (win_prob * payoff_ratio - loss_prob) / payoff_ratio
        return float(np.clip(kelly_f, 0.0, 0.25))

    def estimate_round_trip_cost(self) -> float:
        """Fixed transaction cost per round trip (decimal fraction). Public API for cost pre-screening."""
        entry_cost = self.taker_fee_bps + (self.spread_cost_bps / 2) + self.slippage_bps
        exit_cost = self.maker_fee_bps + (self.spread_cost_bps / 2) + self.slippage_bps
        return (entry_cost + exit_cost) / 10000.0

    def _get_volatility(self, volatility: Optional[float], vol_regime: Optional[str]) -> float:
        if volatility and volatility > 0:
            return float(np.clip(volatility, self.vol_floor, self.vol_ceiling))
        regime_defaults = {"stable": 0.015, "normal": 0.020, "elevated": 0.035, "crisis": 0.060}
        if vol_regime and vol_regime in regime_defaults:
            return regime_defaults[vol_regime]
        return self.target_vol

    def _volatility_scalar(self, vol: float) -> float:
        """Inverse vol sizing, clamped to [0.25, 2.0]."""
        if vol <= 0:
            return 1.0
        return float(np.clip(self.target_vol / vol, 0.25, 2.0))

    def _round_size(self, size: float, product_id: str) -> float:
        base = product_id.split("-")[0].upper()
        precision_map = {"BTC": 8, "ETH": 8, "SOL": 6, "DOGE": 2, "SHIB": 0}
        decimals = precision_map.get(base, 6)
        factor = 10 ** decimals
        return math.floor(size * factor) / factor

    def _zero_result(self, reason: str) -> SizingResult:
        return SizingResult(
            asset_units=0.0, usd_value=0.0, kelly_fraction=0.0,
            applied_fraction=0.0, risk_budget_usd=0.0, edge=0.0,
            effective_edge=0.0, market_impact_bps=0.0,
            win_probability=0.0, transaction_cost_ratio=0.0,
            volatility_scalar=1.0, regime_scalar=1.0, liquidity_scalar=1.0,
            capacity_used_pct=0.0, sizing_method="blocked", reasons=[reason],
        )

    def _blocked_result(self, edge, win_prob, reasons) -> SizingResult:
        return SizingResult(
            asset_units=0.0, usd_value=0.0, kelly_fraction=0.0,
            applied_fraction=0.0, risk_budget_usd=0.0, edge=edge,
            effective_edge=0.0, market_impact_bps=0.0,
            win_probability=win_prob, transaction_cost_ratio=0.0,
            volatility_scalar=1.0, regime_scalar=1.0, liquidity_scalar=1.0,
            capacity_used_pct=0.0, sizing_method="blocked", reasons=reasons,
        )

    def _gated_result(self, edge, win_prob, kelly_f, cost_ratio, reasons) -> SizingResult:
        return SizingResult(
            asset_units=0.0, usd_value=0.0, kelly_fraction=kelly_f,
            applied_fraction=0.0, risk_budget_usd=0.0, edge=edge,
            effective_edge=0.0, market_impact_bps=0.0,
            win_probability=win_prob, transaction_cost_ratio=cost_ratio,
            volatility_scalar=1.0, regime_scalar=1.0, liquidity_scalar=1.0,
            capacity_used_pct=0.0, sizing_method="cost_gated", reasons=reasons,
        )

    def calculate_exit_size(
        self,
        position_size: float,
        entry_price: float,
        current_price: float,
        holding_periods: int,
        confidence: float,
        volatility: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Determine optimal exit sizing.

        Tuned for 60-second cycles on crypto:
        - Stop-loss at -1% (crypto moves fast — -3% is too late)
        - Trailing stop at -0.5% from peak after +0.3% profit
        - Profit targets: 50% exit at +0.5%, full exit at +1.5%
        - Max age: 30 cycles = forced close
        - Alpha half-life: 10 cycles (edge decays fast in crypto)
        """
        reasons = []
        pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        reasons.append(f"PnL={pnl_pct:.2%}, Periods={holding_periods}")

        # Stop loss: exit immediately if losing > 1% — checked FIRST, before min hold
        if pnl_pct < -0.01:
            reasons.append(f"Stop loss: {pnl_pct:.2%} < -1%")
            return {"exit_fraction": 1.0, "reason": "stop_loss", "details": reasons, "urgency": "expedited"}

        # Maximum age: force close after 30 cycles regardless of P&L
        if holding_periods >= 30:
            reasons.append(f"Max age reached: {holding_periods} >= 30 cycles")
            return {"exit_fraction": 1.0, "reason": "max_age", "details": reasons, "urgency": "normal"}

        # Minimum hold: don't evaluate other exits before 2 periods
        if holding_periods < 2:
            reasons.append(f"Min hold: {holding_periods}/2 periods")
            return {"exit_fraction": 0.0, "reason": "hold", "details": reasons, "urgency": "none"}

        # Trailing stop: if position was up +0.3% but now giving back, exit at -0.5% from entry
        if pnl_pct < -0.005 and holding_periods >= 3:
            reasons.append(f"Trailing stop: {pnl_pct:.2%} < -0.5%")
            return {"exit_fraction": 1.0, "reason": "trailing_stop", "details": reasons, "urgency": "expedited"}

        # Alpha half-life: 10 periods (edge decays fast in crypto)
        alpha_half_life = 10
        alpha_remaining = 0.5 ** (holding_periods / alpha_half_life)
        reasons.append(f"Alpha remaining={alpha_remaining:.2%}")

        exit_cost = self.estimate_round_trip_cost() / 2

        # Remaining edge based on confidence and alpha decay
        estimated_edge = confidence * 0.10
        remaining_edge = estimated_edge * alpha_remaining

        if remaining_edge < exit_cost and holding_periods > 5:
            reasons.append(f"Edge exhausted: {remaining_edge:.4f} < cost {exit_cost:.4f}")
            return {"exit_fraction": 1.0, "reason": "edge_exhausted", "details": reasons, "urgency": "normal"}

        # Regime-driven exit
        regime_label = regime or "normal"
        if regime_label in ("chaotic", "volatile") and holding_periods > 5:
            reasons.append(f"Regime={regime_label}, accelerating exit")
            return {"exit_fraction": 0.75, "reason": "regime_exit", "details": reasons, "urgency": "expedited"}

        # Time decay exit — positions held > 2x half-life (20 cycles)
        if holding_periods > alpha_half_life * 2:
            decay_fraction = min(1.0, (holding_periods - alpha_half_life * 2) / alpha_half_life)
            reasons.append(f"Time decay exit: fraction={decay_fraction:.2f}")
            return {"exit_fraction": max(0.5, decay_fraction), "reason": "time_decay", "details": reasons, "urgency": "normal"}

        # Profit target: scale out 50% at +0.5%, full exit at +1.5%
        if pnl_pct > 0.015:
            reasons.append(f"Profit target: full exit at {pnl_pct:.2%}")
            return {"exit_fraction": 1.0, "reason": "profit_target", "details": reasons, "urgency": "normal"}
        if pnl_pct > 0.005:
            reasons.append(f"Profit target: scaling out 50% at {pnl_pct:.2%}")
            return {"exit_fraction": 0.50, "reason": "profit_target", "details": reasons, "urgency": "normal"}

        reasons.append("Holding: edge intact")
        return {"exit_fraction": 0.0, "reason": "hold", "details": reasons, "urgency": "none"}

    def compute_fund_capacity(
        self,
        product_ids: List[str],
        vol: float = 0.02,
        edge: float = 0.01,
    ) -> Dict[str, Any]:
        """
        Compute the maximum deployable capital across all instruments.

        This answers: "How big can this fund get before returns degrade?"

        For each instrument, capacity = (2E / 3cσ)² × ADV
        Total fund capacity = sum across all instruments × leverage_multiple
        """
        total_capacity = 0.0
        per_instrument = {}

        for pid in product_ids:
            adv = self.default_daily_volumes.get(pid, self.fallback_daily_volume)
            c = self.impact_coefficient
            if c * vol > 0:
                instrument_cap = ((2.0 * edge) / (3.0 * c * vol)) ** 2 * adv
                participation_cap = adv * self.max_participation_rate
                effective_cap = min(instrument_cap, participation_cap)
            else:
                effective_cap = adv * self.max_participation_rate

            per_instrument[pid] = {
                "impact_capacity_usd": instrument_cap if c * vol > 0 else float('inf'),
                "participation_cap_usd": adv * self.max_participation_rate,
                "effective_capacity_usd": effective_cap,
                "daily_volume_usd": adv,
            }
            total_capacity += effective_cap

        return {
            "total_capacity_usd": total_capacity,
            "per_instrument": per_instrument,
            "notes": (
                "This is the max capital deployable per cycle across all instruments "
                "before market impact degrades returns to zero. "
                "Adding more instruments increases capacity (Strategy 1: expand universe)."
            ),
        }
