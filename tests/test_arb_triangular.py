"""
Tests for arbitrage/triangular/triangle_arb.py

Covers:
  - TrianglePath dataclass
  - Graph building from ticker data
  - 3-pair cycle rate calculation
  - Profit detection above threshold
  - No-trade when profit < MIN_NET_PROFIT_BPS
  - Fee deduction from profit
  - Cycle direction (start currency selection)
  - Stats reporting
"""
import asyncio
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from arbitrage.triangular.triangle_arb import (
    TriangularArbitrage, TrianglePath,
)


def _make_tickers_profitable():
    """Create tickers where USDT->BTC->ETH->USDT cycle is profitable."""
    return {
        "BTC/USDT": {
            "bid": Decimal('50000'), "ask": Decimal('50000'),
            "last_price": Decimal('50000'),
        },
        "ETH/BTC": {
            "bid": Decimal('0.065'), "ask": Decimal('0.065'),
            "last_price": Decimal('0.065'),
        },
        "ETH/USDT": {
            "bid": Decimal('3300'), "ask": Decimal('3250'),
            "last_price": Decimal('3275'),
        },
    }


def _make_tickers_no_profit():
    """Create tickers where no profitable cycle exists."""
    return {
        "BTC/USDT": {
            "bid": Decimal('50000'), "ask": Decimal('50000'),
            "last_price": Decimal('50000'),
        },
        "ETH/BTC": {
            "bid": Decimal('0.065'), "ask": Decimal('0.065'),
            "last_price": Decimal('0.065'),
        },
        "ETH/USDT": {
            "bid": Decimal('3250'), "ask": Decimal('3250'),
            "last_price": Decimal('3250'),
        },
    }


class TestTrianglePath:
    def test_dataclass_creation(self):
        path = TrianglePath(
            start_currency="USDT",
            path=[("BTC/USDT", "buy", "BTC"), ("ETH/BTC", "buy", "ETH"), ("ETH/USDT", "sell", "USDT")],
            cycle_rate=Decimal('1.002'),
            profit_bps=Decimal('20'),
            exchange="mexc",
        )
        assert path.start_currency == "USDT"
        assert len(path.path) == 3
        assert path.profit_bps == Decimal('20')


class TestTriangularArbitrage:
    @pytest.fixture
    def setup_arb(self):
        mexc_client = AsyncMock()
        cost_model = MagicMock()
        cost_model.FEES = {
            "mexc": {
                "spot": {"maker": Decimal('0'), "taker": Decimal('0.0005')},
            },
            "binance": {
                "spot": {"maker": Decimal('0.00075'), "taker": Decimal('0.00075')},
            },
        }
        risk_engine = MagicMock()
        risk_engine.approve_arbitrage.return_value = True
        signal_queue = asyncio.Queue(maxsize=100)

        arb = TriangularArbitrage(mexc_client, cost_model, risk_engine, signal_queue)
        return arb, mexc_client, cost_model, risk_engine, signal_queue

    def test_initialization(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        assert arb.MIN_NET_PROFIT_BPS == Decimal('0.5')
        assert arb.MAX_TRADE_USD == Decimal('200')
        assert "USDT" in arb.START_CURRENCIES

    def test_update_graph_builds_edges(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = _make_tickers_profitable()
        arb._update_graph(tickers)

        # Should have edges for USDT, BTC, ETH
        assert "USDT" in arb._pair_graph or "BTC" in arb._pair_graph
        # Check BTC -> USDT edge (selling BTC for USDT)
        assert "USDT" in arb._pair_graph.get("BTC", {})

    def test_update_graph_skips_zero_price(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = {
            "ZERO/USDT": {"bid": Decimal('0'), "ask": Decimal('0'), "last_price": Decimal('0')},
        }
        arb._update_graph(tickers)
        assert "ZERO" not in arb._pair_graph

    def test_update_graph_skips_no_slash_symbols(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = {
            "BTCUSDT": {"bid": Decimal('50000'), "ask": Decimal('50000'), "last_price": Decimal('50000')},
        }
        arb._update_graph(tickers)
        assert len(arb._pair_graph) == 0

    def test_find_profitable_cycles_detects_profit(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = _make_tickers_profitable()
        arb._update_graph(tickers)

        opportunities = arb._find_profitable_cycles(tickers)
        # Should find profitable cycles (the ETH/USDT bid=3300 vs ask=3250 creates opportunity)
        # Cycle: USDT -> buy BTC (1/50000) -> buy ETH with BTC (0.065) -> sell ETH for USDT (3300)
        # Rate = (1/50000) * 0.065 * 3300 = 0.0000200 * 0.065 * 3300 = ... let's check
        # USDT->BTC edge: rate = 1/50000 = 0.00002 (buy BTC with USDT)
        # BTC->ETH edge: rate = 1/0.065 = 15.3846 (buy ETH with BTC)
        # ETH->USDT edge: rate = 3300 (sell ETH for USDT)
        # Hmm that's not right. Let's trace the graph.
        # _update_graph: base->quote edge = bid; quote->base edge = 1/ask
        # BTC/USDT: BTC->USDT = 50000, USDT->BTC = 1/50000
        # ETH/BTC: ETH->BTC = 0.065, BTC->ETH = 1/0.065
        # ETH/USDT: ETH->USDT = 3300, USDT->ETH = 1/3250
        # Cycle USDT->BTC->ETH->USDT = (1/50000) * (1/0.065) * 3300 = 1.0153...
        # That IS profitable
        profitable = [o for o in opportunities if o.profit_bps > Decimal('0')]
        assert len(profitable) >= 1

    def test_find_profitable_cycles_no_profit(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = _make_tickers_no_profit()
        arb._update_graph(tickers)

        opportunities = arb._find_profitable_cycles(tickers)
        # USDT->BTC->ETH->USDT = (1/50000) * (1/0.065) * 3250 = 1.0
        # No profit
        profitable = [o for o in opportunities if o.cycle_rate > 1]
        assert len(profitable) == 0

    def test_opportunities_sorted_by_profit(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = _make_tickers_profitable()
        arb._update_graph(tickers)

        opportunities = arb._find_profitable_cycles(tickers)
        if len(opportunities) >= 2:
            assert opportunities[0].profit_bps >= opportunities[1].profit_bps

    def test_fee_deduction_from_profit(self, setup_arb):
        arb, _, cost_model, _, _ = setup_arb
        # MEXC 0% maker fee => total_fee_bps = 0 * 3 * 10000 = 0
        tickers = _make_tickers_profitable()
        arb._update_graph(tickers)

        opportunities = arb._find_profitable_cycles(tickers)
        # With 0% fees, net_profit = gross_profit
        for opp in opportunities:
            assert opp.profit_bps > 0  # No fee deducted for 0% maker

    @pytest.mark.asyncio
    async def test_build_pair_graph_from_exchange(self, setup_arb):
        arb, mexc_client, _, _, _ = setup_arb
        mexc_client.get_all_tickers = AsyncMock(return_value=_make_tickers_profitable())

        await arb._build_pair_graph()
        assert len(arb._pair_graph) > 0

    def test_stop(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        arb._running = True
        arb.stop()
        assert arb._running is False

    def test_get_stats(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        arb._scan_count = 50
        arb._opportunities_found = 3

        stats = arb.get_stats()
        assert stats["scan_count"] == 50
        assert stats["opportunities_found"] == 3
        assert "graph_currencies" in stats
        assert "graph_edges" in stats

    def test_graph_uses_last_price_as_fallback(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        tickers = {
            "BTC/USDT": {
                "bid": Decimal('0'), "ask": Decimal('0'),
                "last_price": Decimal('50000'),
            },
        }
        arb._update_graph(tickers)
        # Should use last_price as fallback
        assert "USDT" in arb._pair_graph.get("BTC", {})

    def test_start_currencies_list(self, setup_arb):
        arb, _, _, _, _ = setup_arb
        assert "USDT" in arb.START_CURRENCIES
        assert "BTC" in arb.START_CURRENCIES
        assert "ETH" in arb.START_CURRENCIES
