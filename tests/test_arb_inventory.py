"""
Tests for arbitrage/inventory/manager.py

Covers:
  - InventoryManager initialization
  - Balance tracking across exchanges
  - Imbalance detection (70% threshold)
  - Rebalance plan generation
  - Target allocation (50/50)
  - Preferred networks for transfer
  - Summary reporting
  - Edge cases: zero balances, missing currencies
"""
import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

from arbitrage.inventory.manager import (
    InventoryManager, InventorySnapshot, RebalanceRequest,
)
from arbitrage.exchanges.base import Balance


def _make_balances(exchange, overrides=None):
    """Create a balance dict for an exchange. Override specific currencies."""
    defaults = {
        "USDT": Decimal('5000'), "BTC": Decimal('0.15'),
        "ETH": Decimal('2.5'), "SOL": Decimal('50'),
    }
    if overrides:
        defaults.update(overrides)

    balances = {}
    for currency, amount in defaults.items():
        balances[currency] = Balance(
            exchange=exchange, currency=currency,
            free=amount, locked=Decimal('0'), total=amount,
        )
    return balances


class TestInventorySnapshot:
    def test_dataclass_creation(self):
        snap = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances={"USDT": Decimal('5000')},
            binance_balances={"USDT": Decimal('5000')},
            imbalances={},
        )
        assert snap.mexc_balances["USDT"] == Decimal('5000')


class TestRebalanceRequest:
    def test_dataclass_creation(self):
        req = RebalanceRequest(
            currency="USDT", from_exchange="mexc", to_exchange="binance",
            amount=Decimal('1000'), network="TRC20",
            estimated_fee=Decimal('1'), estimated_time_minutes=3,
            reason="Imbalance: mexc has 80%",
        )
        assert req.currency == "USDT"
        assert req.network == "TRC20"


class TestInventoryManager:
    @pytest.fixture
    def setup_manager(self):
        mexc = AsyncMock()
        binance = AsyncMock()
        manager = InventoryManager(mexc, binance)
        return manager, mexc, binance

    def test_initialization(self, setup_manager):
        mgr, _, _ = setup_manager
        assert mgr.TARGET_ALLOCATION == Decimal('0.50')
        assert mgr.REBALANCE_THRESHOLD == Decimal('0.70')

    @pytest.mark.asyncio
    async def test_check_inventory_balanced(self, setup_manager):
        mgr, mexc, binance = setup_manager

        # Balanced: 50/50 split
        mexc.get_balances = AsyncMock(return_value=_make_balances("mexc"))
        binance.get_balances = AsyncMock(return_value=_make_balances("binance"))

        snapshot = await mgr.check_inventory()

        for currency, info in snapshot.imbalances.items():
            assert info['needs_rebalance'] is False

    @pytest.mark.asyncio
    async def test_check_inventory_detects_imbalance(self, setup_manager):
        mgr, mexc, binance = setup_manager

        # MEXC has 9000 USDT, Binance has 1000 => 90% / 10%
        mexc.get_balances = AsyncMock(return_value=_make_balances(
            "mexc", {"USDT": Decimal('9000')}
        ))
        binance.get_balances = AsyncMock(return_value=_make_balances(
            "binance", {"USDT": Decimal('1000')}
        ))

        snapshot = await mgr.check_inventory()

        usdt_info = snapshot.imbalances.get("USDT", {})
        assert usdt_info.get('needs_rebalance') is True
        assert usdt_info['mexc_pct'] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_check_inventory_zero_total_skipped(self, setup_manager):
        mgr, mexc, binance = setup_manager

        # Both exchanges have 0 BTC
        mexc.get_balances = AsyncMock(return_value=_make_balances(
            "mexc", {"BTC": Decimal('0')}
        ))
        binance.get_balances = AsyncMock(return_value=_make_balances(
            "binance", {"BTC": Decimal('0')}
        ))

        snapshot = await mgr.check_inventory()
        assert "BTC" not in snapshot.imbalances

    def test_generate_rebalance_plan_imbalanced(self, setup_manager):
        mgr, _, _ = setup_manager

        snapshot = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances={"USDT": Decimal('9000')},
            binance_balances={"USDT": Decimal('1000')},
            imbalances={
                "USDT": {
                    'mexc_amount': 9000.0, 'binance_amount': 1000.0,
                    'total': 10000.0, 'mexc_pct': 0.9, 'binance_pct': 0.1,
                    'needs_rebalance': True,
                },
            },
        )

        requests = mgr.generate_rebalance_plan(snapshot)

        assert len(requests) == 1
        req = requests[0]
        assert req.currency == "USDT"
        assert req.from_exchange == "mexc"
        assert req.to_exchange == "binance"
        # Transfer amount = (0.9 - 0.5) * 10000 = 4000
        assert req.amount == Decimal('0.4') * Decimal('10000')
        assert req.network == "TRC20"

    def test_generate_rebalance_plan_binance_heavy(self, setup_manager):
        mgr, _, _ = setup_manager

        snapshot = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances={"USDT": Decimal('1000')},
            binance_balances={"USDT": Decimal('9000')},
            imbalances={
                "USDT": {
                    'mexc_amount': 1000.0, 'binance_amount': 9000.0,
                    'total': 10000.0, 'mexc_pct': 0.1, 'binance_pct': 0.9,
                    'needs_rebalance': True,
                },
            },
        )

        requests = mgr.generate_rebalance_plan(snapshot)
        assert len(requests) == 1
        assert requests[0].from_exchange == "binance"
        assert requests[0].to_exchange == "mexc"

    def test_generate_rebalance_plan_no_rebalance_needed(self, setup_manager):
        mgr, _, _ = setup_manager

        snapshot = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances={"USDT": Decimal('5000')},
            binance_balances={"USDT": Decimal('5000')},
            imbalances={
                "USDT": {
                    'mexc_amount': 5000.0, 'binance_amount': 5000.0,
                    'total': 10000.0, 'mexc_pct': 0.5, 'binance_pct': 0.5,
                    'needs_rebalance': False,
                },
            },
        )

        requests = mgr.generate_rebalance_plan(snapshot)
        assert len(requests) == 0

    def test_preferred_networks(self, setup_manager):
        mgr, _, _ = setup_manager
        assert mgr.PREFERRED_NETWORKS["USDT"][0] == "TRC20"
        assert mgr.PREFERRED_NETWORKS["SOL"][0] == "SOL"
        assert mgr.PREFERRED_NETWORKS["BTC"][0] == "BTC"

    def test_get_summary_no_snapshots(self, setup_manager):
        mgr, _, _ = setup_manager
        summary = mgr.get_summary()
        assert summary["status"] == "no_snapshots"

    @pytest.mark.asyncio
    async def test_get_summary_with_snapshots(self, setup_manager):
        mgr, mexc, binance = setup_manager

        mexc.get_balances = AsyncMock(return_value=_make_balances("mexc"))
        binance.get_balances = AsyncMock(return_value=_make_balances("binance"))

        await mgr.check_inventory()

        summary = mgr.get_summary()
        assert summary["snapshot_count"] == 1
        assert "last_check" in summary
        assert "currencies_monitored" in summary

    def test_generate_rebalance_skips_unknown_network(self, setup_manager):
        """Currency not in PREFERRED_NETWORKS should be skipped."""
        mgr, _, _ = setup_manager

        snapshot = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances={"UNKNOWN": Decimal('900')},
            binance_balances={"UNKNOWN": Decimal('100')},
            imbalances={
                "UNKNOWN": {
                    'mexc_amount': 900.0, 'binance_amount': 100.0,
                    'total': 1000.0, 'mexc_pct': 0.9, 'binance_pct': 0.1,
                    'needs_rebalance': True,
                },
            },
        )

        requests = mgr.generate_rebalance_plan(snapshot)
        assert len(requests) == 0

    @pytest.mark.asyncio
    async def test_snapshot_stored_in_history(self, setup_manager):
        mgr, mexc, binance = setup_manager

        mexc.get_balances = AsyncMock(return_value=_make_balances("mexc"))
        binance.get_balances = AsyncMock(return_value=_make_balances("binance"))

        await mgr.check_inventory()
        await mgr.check_inventory()

        assert len(mgr._snapshots) == 2
