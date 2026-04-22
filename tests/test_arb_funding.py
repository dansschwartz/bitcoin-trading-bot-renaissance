"""
Tests for arbitrage/funding/funding_rate_arb.py

Covers:
  - FundingRateArbitrage initialization
  - scan_opportunities: finding differential, annualization, direction
  - Delta-neutral position logic (short high, long low)
  - Position opening with risk approval
  - Position monitoring and auto-close on APR collapse
  - Max positions limit
  - Stats reporting
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from arbitrage.funding.funding_rate_arb import (
    FundingRateArbitrage, FundingOpportunity, FundingPosition, FUTURES_PAIRS,
)
from arbitrage.exchanges.base import FundingRate


def _make_funding_rate(exchange, symbol, rate):
    return FundingRate(
        exchange=exchange, symbol=symbol,
        current_rate=Decimal(rate),
        predicted_rate=Decimal(rate),
        next_funding_time=datetime.utcnow(),
        timestamp=datetime.utcnow(),
    )


class TestFundingOpportunity:
    def test_dataclass_creation(self):
        opp = FundingOpportunity(
            symbol="BTC/USDT",
            mexc_rate=Decimal('0.0003'),
            binance_rate=Decimal('0.0001'),
            differential=Decimal('0.0002'),
            annual_apr=Decimal('21.9'),
            short_exchange="mexc",
            long_exchange="binance",
        )
        assert opp.short_exchange == "mexc"
        assert opp.long_exchange == "binance"


class TestFundingRateArbitrage:
    @pytest.fixture
    def setup_arb(self):
        mexc = AsyncMock()
        binance = AsyncMock()
        risk_engine = MagicMock()
        risk_engine.approve_funding_arb.return_value = True

        arb = FundingRateArbitrage(mexc, binance, risk_engine)
        return arb, mexc, binance, risk_engine

    def test_initialization(self, setup_arb):
        arb, _, _, _ = setup_arb
        assert arb.MIN_DIFFERENTIAL_APR == Decimal('5.0')
        assert arb.CLOSE_DIFFERENTIAL_APR == Decimal('2.0')
        assert arb.MAX_LEVERAGE == Decimal('3.0')
        assert arb._opportunities_found == 0

    @pytest.mark.asyncio
    async def test_scan_finds_opportunity_above_threshold(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        # MEXC rate much higher than Binance => short MEXC, long Binance
        mexc.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("mexc", "BTC/USDT", "0.001")
        )
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.0001")
        )

        opportunities = await arb.scan_opportunities()

        assert len(opportunities) >= 1
        opp = opportunities[0]
        assert opp.short_exchange == "mexc"
        assert opp.long_exchange == "binance"
        assert opp.differential == Decimal('0.0009')
        # APR = 0.0009 * 3 * 365 * 100 = 98.55
        assert opp.annual_apr > arb.MIN_DIFFERENTIAL_APR

    @pytest.mark.asyncio
    async def test_scan_ignores_below_threshold(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        # Very small differential
        mexc.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("mexc", "BTC/USDT", "0.0001")
        )
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.00009")
        )

        opportunities = await arb.scan_opportunities()

        # APR = 0.00001 * 3 * 365 * 100 = 1.095 < 5.0
        btc_opps = [o for o in opportunities if o.symbol == "BTC/USDT"]
        assert len(btc_opps) == 0

    @pytest.mark.asyncio
    async def test_scan_direction_binance_higher(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        # Binance rate higher => short Binance, long MEXC
        mexc.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("mexc", "BTC/USDT", "0.0001")
        )
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.001")
        )

        opportunities = await arb.scan_opportunities()
        btc_opps = [o for o in opportunities if o.symbol == "BTC/USDT"]
        assert len(btc_opps) >= 1
        assert btc_opps[0].short_exchange == "binance"
        assert btc_opps[0].long_exchange == "mexc"

    @pytest.mark.asyncio
    async def test_evaluate_and_open_position(self, setup_arb):
        arb, _, _, risk_engine = setup_arb

        opp = FundingOpportunity(
            symbol="BTC/USDT",
            mexc_rate=Decimal('0.001'),
            binance_rate=Decimal('0.0001'),
            differential=Decimal('0.0009'),
            annual_apr=Decimal('98.55'),
            short_exchange="mexc",
            long_exchange="binance",
        )

        await arb._evaluate_and_open(opp)

        assert "BTC/USDT" in arb._positions
        pos = arb._positions["BTC/USDT"]
        assert pos.short_exchange == "mexc"
        assert pos.long_exchange == "binance"
        assert pos.is_open is True
        assert pos.quantity == arb.MAX_POSITION_USD

    @pytest.mark.asyncio
    async def test_max_positions_limit(self, setup_arb):
        arb, _, _, _ = setup_arb

        # Fill up 5 positions
        for i in range(5):
            arb._positions[f"PAIR{i}/USDT"] = FundingPosition(
                position_id=f"fund_{i}", symbol=f"PAIR{i}/USDT",
                short_exchange="mexc", long_exchange="binance",
                quantity=Decimal('2000'), entry_differential=Decimal('0.001'),
                entry_time=datetime.utcnow(),
            )

        opp = FundingOpportunity(
            symbol="NEW/USDT", mexc_rate=Decimal('0.001'),
            binance_rate=Decimal('0.0001'), differential=Decimal('0.0009'),
            annual_apr=Decimal('98.55'), short_exchange="mexc",
            long_exchange="binance",
        )

        await arb._evaluate_and_open(opp)
        assert "NEW/USDT" not in arb._positions

    @pytest.mark.asyncio
    async def test_risk_rejection_prevents_opening(self, setup_arb):
        arb, _, _, risk_engine = setup_arb
        risk_engine.approve_funding_arb.return_value = False

        opp = FundingOpportunity(
            symbol="BTC/USDT", mexc_rate=Decimal('0.001'),
            binance_rate=Decimal('0.0001'), differential=Decimal('0.0009'),
            annual_apr=Decimal('98.55'), short_exchange="mexc",
            long_exchange="binance",
        )

        await arb._evaluate_and_open(opp)
        assert "BTC/USDT" not in arb._positions

    @pytest.mark.asyncio
    async def test_monitor_closes_position_on_apr_collapse(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        # Open position
        arb._positions["BTC/USDT"] = FundingPosition(
            position_id="fund_btc", symbol="BTC/USDT",
            short_exchange="mexc", long_exchange="binance",
            quantity=Decimal('2000'), entry_differential=Decimal('0.0009'),
            entry_time=datetime.utcnow(),
        )

        # Current differential very small => APR < CLOSE_DIFFERENTIAL_APR
        mexc.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("mexc", "BTC/USDT", "0.0001")
        )
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.00009")
        )

        await arb._monitor_positions()

        assert arb._positions["BTC/USDT"].is_open is False

    @pytest.mark.asyncio
    async def test_monitor_keeps_position_open_when_profitable(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        arb._positions["BTC/USDT"] = FundingPosition(
            position_id="fund_btc", symbol="BTC/USDT",
            short_exchange="mexc", long_exchange="binance",
            quantity=Decimal('2000'), entry_differential=Decimal('0.0009'),
            entry_time=datetime.utcnow(),
        )

        # Still highly profitable
        mexc.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("mexc", "BTC/USDT", "0.001")
        )
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.0001")
        )

        await arb._monitor_positions()

        assert arb._positions["BTC/USDT"].is_open is True
        assert arb._positions["BTC/USDT"].cumulative_funding_collected > 0

    def test_get_stats_empty(self, setup_arb):
        arb, _, _, _ = setup_arb
        stats = arb.get_stats()
        assert stats["opportunities_found"] == 0
        assert stats["open_positions"] == 0
        assert stats["closed_positions"] == 0

    def test_get_stats_with_positions(self, setup_arb):
        arb, _, _, _ = setup_arb
        arb._opportunities_found = 5
        arb._positions["BTC/USDT"] = FundingPosition(
            position_id="fund_btc", symbol="BTC/USDT",
            short_exchange="mexc", long_exchange="binance",
            quantity=Decimal('2000'), entry_differential=Decimal('0.0009'),
            entry_time=datetime.utcnow(),
            cumulative_funding_collected=Decimal('1.50'),
        )
        stats = arb.get_stats()
        assert stats["open_positions"] == 1
        assert stats["total_funding_collected_usd"] == 1.50

    def test_stop(self, setup_arb):
        arb, _, _, _ = setup_arb
        arb._running = True
        arb.stop()
        assert arb._running is False

    @pytest.mark.asyncio
    async def test_scan_handles_fetch_error_gracefully(self, setup_arb):
        arb, mexc, binance, _ = setup_arb

        mexc.get_funding_rate = AsyncMock(side_effect=Exception("API error"))
        binance.get_funding_rate = AsyncMock(
            return_value=_make_funding_rate("binance", "BTC/USDT", "0.0001")
        )

        opportunities = await arb.scan_opportunities()
        # Should not crash, just skip errored pairs
        assert isinstance(opportunities, list)
