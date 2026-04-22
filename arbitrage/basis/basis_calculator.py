"""
Basis Calculator — pure functions for spot-futures basis analysis.

Basis = (futures_price - spot_price) / spot_price
  Contango: futures > spot (positive basis) — normal in bullish markets
  Backwardation: futures < spot (negative basis) — fear/liquidation cascades

This module contains NO I/O, no DB, no network calls — just math.
"""
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional


@dataclass
class BasisSnapshot:
    """Point-in-time basis measurement for a single symbol."""
    symbol: str
    spot_price: Decimal
    futures_price: Decimal
    basis_abs: Decimal          # futures - spot (signed)
    basis_pct: Decimal          # basis_abs / spot * 100
    basis_bps: Decimal          # basis_abs / spot * 10000
    direction: str              # 'contango' | 'backwardation' | 'flat'
    annualized_basis_pct: Decimal  # Extrapolated to annual rate
    funding_rate: Optional[Decimal] = None  # From MEXC contract API if available
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BasisOpportunity:
    """Evaluated basis trading opportunity."""
    snapshot: BasisSnapshot
    is_profitable: bool
    signal: str                 # 'sell_basis' | 'buy_basis' | 'none'
    edge_bps: Decimal           # basis_bps minus threshold
    estimated_daily_yield_usd: float
    annualized_yield_pct: float
    risk_notes: list = field(default_factory=list)


def calculate_basis(
    symbol: str,
    spot_price: Decimal,
    futures_price: Decimal,
    funding_rate: Optional[Decimal] = None,
    days_to_expiry: int = 90,
) -> BasisSnapshot:
    """Calculate basis metrics from spot and futures prices.

    Args:
        symbol: e.g. "BTC/USDT"
        spot_price: Current spot price
        futures_price: Current futures/perpetual price
        funding_rate: Optional funding rate from the contract API
        days_to_expiry: For annualization (perpetuals use 90 as convention)

    Returns:
        BasisSnapshot with all calculated metrics
    """
    if spot_price <= 0:
        raise ValueError(f"spot_price must be positive, got {spot_price}")
    if futures_price <= 0:
        raise ValueError(f"futures_price must be positive, got {futures_price}")

    basis_abs = futures_price - spot_price
    basis_pct = (basis_abs / spot_price) * Decimal("100")
    basis_bps = (basis_abs / spot_price) * Decimal("10000")

    # Direction
    if basis_bps > Decimal("0.5"):
        direction = "contango"
    elif basis_bps < Decimal("-0.5"):
        direction = "backwardation"
    else:
        direction = "flat"

    # Annualize: (basis_pct / days_to_expiry) * 365
    if days_to_expiry > 0:
        annualized = (basis_pct / Decimal(str(days_to_expiry))) * Decimal("365")
    else:
        annualized = Decimal("0")

    return BasisSnapshot(
        symbol=symbol,
        spot_price=spot_price,
        futures_price=futures_price,
        basis_abs=basis_abs,
        basis_pct=basis_pct,
        basis_bps=basis_bps,
        direction=direction,
        annualized_basis_pct=annualized,
        funding_rate=funding_rate,
    )


def evaluate_opportunity(
    snapshot: BasisSnapshot,
    min_basis_bps: Decimal = Decimal("5"),
    position_size_usd: float = 1000.0,
    estimated_round_trip_cost_bps: Decimal = Decimal("3"),
) -> BasisOpportunity:
    """Evaluate whether a basis snapshot represents a tradeable opportunity.

    Strategy:
      - Contango (futures > spot): sell basis = short futures + long spot
      - Backwardation (futures < spot): buy basis = long futures + short spot

    Args:
        snapshot: Current basis measurement
        min_basis_bps: Minimum basis in bps to consider
        position_size_usd: Notional position size for yield calculation
        estimated_round_trip_cost_bps: Estimated fees for opening + closing

    Returns:
        BasisOpportunity with signal and yield estimates
    """
    abs_basis_bps = abs(snapshot.basis_bps)
    edge_bps = abs_basis_bps - min_basis_bps
    is_profitable = edge_bps > estimated_round_trip_cost_bps

    risk_notes = []

    # Determine signal
    if not is_profitable:
        signal = "none"
    elif snapshot.direction == "contango":
        signal = "sell_basis"  # Short futures, long spot — profit as basis converges
    elif snapshot.direction == "backwardation":
        signal = "buy_basis"   # Long futures, short spot
    else:
        signal = "none"

    # Yield estimates
    daily_yield = float(abs_basis_bps) / 10000.0 * position_size_usd / 90.0
    annual_yield = float(snapshot.annualized_basis_pct)

    # Risk notes
    if abs_basis_bps > Decimal("100"):
        risk_notes.append("extreme_basis_over_100bps")
    if snapshot.direction == "backwardation":
        risk_notes.append("backwardation_higher_risk")
    if snapshot.funding_rate is not None:
        # Funding rate alignment check
        if snapshot.direction == "contango" and snapshot.funding_rate > 0:
            risk_notes.append("funding_aligned_contango")
        elif snapshot.direction == "backwardation" and snapshot.funding_rate < 0:
            risk_notes.append("funding_aligned_backwardation")

    return BasisOpportunity(
        snapshot=snapshot,
        is_profitable=is_profitable,
        signal=signal,
        edge_bps=edge_bps,
        estimated_daily_yield_usd=daily_yield,
        annualized_yield_pct=annual_yield,
        risk_notes=risk_notes,
    )
