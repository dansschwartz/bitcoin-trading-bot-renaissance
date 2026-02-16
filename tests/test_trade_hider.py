"""
Tests for execution/trade_hider.py — TradeHider.

Covers timing jitter bounds, size variance bounds, order type randomisation,
order splitting, and the should_split threshold logic.
"""

import asyncio
import random

import pytest

from execution.trade_hider import TradeHider, _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> dict:
    cfg = {"trade_hider": {}}
    cfg["trade_hider"].update(overrides)
    return cfg


def _run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Tests — Initialization
# ---------------------------------------------------------------------------

class TestTradeHiderInit:
    def test_default_config(self):
        th = TradeHider({"trade_hider": {}})
        assert th._enabled is True
        assert th._max_jitter == 3.0
        assert th._size_var_pct == 10.0
        assert th._chunk_threshold == 500.0

    def test_custom_config(self):
        th = TradeHider(_make_config(
            enabled=False,
            max_jitter_seconds=1.0,
            size_variance_pct=5.0,
            chunk_orders_above_usd=1000,
        ))
        assert th._enabled is False
        assert th._max_jitter == 1.0
        assert th._size_var_pct == 5.0
        assert th._chunk_threshold == 1000.0

    def test_config_merges_with_defaults(self):
        th = TradeHider(_make_config(max_jitter_seconds=2.0))
        # Overridden value
        assert th._max_jitter == 2.0
        # Default values preserved
        assert th._size_var_pct == 10.0


# ---------------------------------------------------------------------------
# Tests — apply_jitter
# ---------------------------------------------------------------------------

class TestApplyJitter:
    def test_disabled_passes_through(self):
        th = TradeHider(_make_config(enabled=False))
        signal = {"quantity": 1.0, "side": "buy", "price": 100.0}
        result = _run_async(th.apply_jitter(signal))
        assert result["jitter_delay_ms"] == 0.0
        assert result["quantity"] == 1.0  # Unchanged

    def test_original_signal_not_mutated(self):
        th = TradeHider(_make_config(max_jitter_seconds=0.001))
        signal = {"quantity": 1.0, "side": "buy", "price": 100.0}
        original_qty = signal["quantity"]
        _run_async(th.apply_jitter(signal))
        assert signal["quantity"] == original_qty

    def test_jitter_delay_within_bounds(self):
        th = TradeHider(_make_config(max_jitter_seconds=0.01))  # Small jitter for speed
        signal = {"quantity": 1.0, "side": "buy"}
        result = _run_async(th.apply_jitter(signal))
        assert 0.0 <= result["jitter_delay_ms"] <= 10.0  # 0.01s = 10ms

    def test_size_variance_within_bounds(self):
        """Quantity should be within +/- size_variance_pct of original."""
        th = TradeHider(_make_config(
            max_jitter_seconds=0.001,
            size_variance_pct=10.0,
        ))
        original_qty = 1.0
        for _ in range(50):
            signal = {"quantity": original_qty, "side": "buy"}
            result = _run_async(th.apply_jitter(signal))
            # Should be within 10% of original
            assert 0.9 * original_qty <= result["quantity"] <= 1.1 * original_qty

    def test_order_type_set(self):
        th = TradeHider(_make_config(max_jitter_seconds=0.001))
        signal = {"quantity": 1.0, "side": "buy"}
        result = _run_async(th.apply_jitter(signal))
        assert result["order_type"] in ("limit", "market")

    def test_order_type_distribution(self):
        """Over many runs, both order types should appear."""
        th = TradeHider(_make_config(
            max_jitter_seconds=0.001,
            order_type_weights={"limit": 0.5, "market": 0.5},
        ))
        types_seen = set()
        for _ in range(100):
            signal = {"quantity": 1.0, "side": "buy"}
            result = _run_async(th.apply_jitter(signal))
            types_seen.add(result["order_type"])
        assert "limit" in types_seen
        assert "market" in types_seen

    def test_no_quantity_field_no_crash(self):
        th = TradeHider(_make_config(max_jitter_seconds=0.001))
        signal = {"side": "buy", "price": 100.0}
        result = _run_async(th.apply_jitter(signal))
        assert "quantity" not in result or result.get("quantity") is None

    def test_none_quantity_not_modified(self):
        th = TradeHider(_make_config(max_jitter_seconds=0.001))
        signal = {"quantity": None, "side": "buy"}
        result = _run_async(th.apply_jitter(signal))
        # None quantity should pass through without variance
        assert result["quantity"] is None


# ---------------------------------------------------------------------------
# Tests — split_order
# ---------------------------------------------------------------------------

class TestSplitOrder:
    def test_sum_equals_total(self):
        th = TradeHider({"trade_hider": {}})
        total = 10.0
        for _ in range(20):
            chunks = th.split_order(total, min_chunks=2, max_chunks=5)
            assert abs(sum(chunks) - total) < 1e-6

    def test_chunk_count_within_bounds(self):
        th = TradeHider({"trade_hider": {}})
        for _ in range(50):
            chunks = th.split_order(10.0, min_chunks=3, max_chunks=5)
            assert 3 <= len(chunks) <= 5

    def test_single_chunk_when_min_is_one(self):
        th = TradeHider({"trade_hider": {}})
        random.seed(42)
        # With min_chunks=1, max_chunks=1, should return exactly 1 chunk
        chunks = th.split_order(10.0, min_chunks=1, max_chunks=1)
        assert len(chunks) == 1
        assert chunks[0] == 10.0

    def test_all_chunks_positive(self):
        th = TradeHider({"trade_hider": {}})
        for _ in range(50):
            chunks = th.split_order(10.0, min_chunks=2, max_chunks=5)
            for c in chunks:
                assert c >= 0.0

    def test_default_min_max_from_config(self):
        th = TradeHider(_make_config(min_chunks=2, max_chunks=3))
        for _ in range(30):
            chunks = th.split_order(5.0)
            assert 2 <= len(chunks) <= 3

    def test_min_clamped_to_one(self):
        th = TradeHider({"trade_hider": {}})
        chunks = th.split_order(10.0, min_chunks=0, max_chunks=2)
        assert len(chunks) >= 1

    def test_max_at_least_min(self):
        th = TradeHider({"trade_hider": {}})
        chunks = th.split_order(10.0, min_chunks=5, max_chunks=3)
        # max is clamped to at least min
        assert len(chunks) >= 5


# ---------------------------------------------------------------------------
# Tests — should_split
# ---------------------------------------------------------------------------

class TestShouldSplit:
    def test_above_threshold(self):
        th = TradeHider(_make_config(chunk_orders_above_usd=500))
        assert th.should_split(600.0) is True

    def test_below_threshold(self):
        th = TradeHider(_make_config(chunk_orders_above_usd=500))
        assert th.should_split(300.0) is False

    def test_at_threshold(self):
        th = TradeHider(_make_config(chunk_orders_above_usd=500))
        assert th.should_split(500.0) is True

    def test_zero_amount(self):
        th = TradeHider(_make_config(chunk_orders_above_usd=500))
        assert th.should_split(0.0) is False
