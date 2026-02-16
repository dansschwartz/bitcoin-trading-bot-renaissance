"""
Tests for signals/liquidation_detector.py — LiquidationCascadeDetector
======================================================================
Covers: initialization, risk scoring, direction inference, signal building,
history management, scan loop, fast eval, on_price_update, lifecycle, and
edge cases. All Binance API calls are mocked.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Module under test
from signals.liquidation_detector import (
    CascadeRiskSignal,
    LiquidationCascadeDetector,
    _SymbolHistory,
    _SymbolSnapshot,
    _DEFAULT_CONFIG,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    """Default detector with default config."""
    return LiquidationCascadeDetector()


@pytest.fixture
def custom_detector():
    """Detector with lowered threshold for easier signal emission."""
    return LiquidationCascadeDetector({
        "symbols": ["BTCUSDT"],
        "risk_threshold": 0.2,
        "scan_interval_seconds": 1,
        "extreme_funding_percentile": 0.80,
        "high_oi_change_pct": 5.0,
        "extreme_ls_ratio": 1.5,
    })


@pytest.fixture
def snapshot_high_risk():
    """A snapshot that should produce a high risk score."""
    return _SymbolSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        funding_rate=0.005,      # extremely high
        open_interest=150_000.0,
        long_short_ratio=2.5,    # extremely long-biased
        top_trader_ls_ratio=2.0,
    )


@pytest.fixture
def snapshot_neutral():
    """A snapshot that should produce a low risk score."""
    return _SymbolSnapshot(
        symbol="BTCUSDT",
        timestamp=datetime.now(timezone.utc),
        funding_rate=0.0001,
        open_interest=100_000.0,
        long_short_ratio=1.0,
        top_trader_ls_ratio=1.0,
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config(self, detector):
        assert detector._symbols == _DEFAULT_CONFIG["symbols"]
        assert detector._risk_threshold == _DEFAULT_CONFIG["risk_threshold"]
        assert detector._scan_interval == _DEFAULT_CONFIG["scan_interval_seconds"]
        assert detector._enabled is True

    def test_custom_config_overrides(self, custom_detector):
        assert custom_detector._symbols == ["BTCUSDT"]
        assert custom_detector._risk_threshold == 0.2

    def test_history_initialized_for_all_symbols(self, detector):
        for sym in detector._symbols:
            assert sym in detector._history
            assert isinstance(detector._history[sym], _SymbolHistory)

    def test_disabled_config(self):
        det = LiquidationCascadeDetector({"enabled": False})
        assert det.enabled is False

    def test_enabled_property_setter(self, detector):
        detector.enabled = False
        assert detector.enabled is False
        detector.enabled = True
        assert detector.enabled is True


# ---------------------------------------------------------------------------
# CascadeRiskSignal tests
# ---------------------------------------------------------------------------

class TestCascadeRiskSignal:
    def test_to_dict_contains_all_fields(self):
        sig = CascadeRiskSignal(
            signal_id="abc-123",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            symbol="BTCUSDT",
            direction="long_liquidation",
            risk_score=0.85,
            funding_rate=0.001,
            funding_rate_percentile=0.95,
            open_interest_change_24h=15.0,
            long_short_ratio=2.5,
            estimated_liquidation_usd=5000.0,
            recommended_action="open_short",
            entry_trigger="immediate_market_order",
            expected_move_pct=6.8,
        )
        d = sig.to_dict()
        assert d["signal_id"] == "abc-123"
        assert d["direction"] == "long_liquidation"
        assert d["risk_score"] == 0.85
        assert d["recommended_action"] == "open_short"


# ---------------------------------------------------------------------------
# Direction inference tests
# ---------------------------------------------------------------------------

class TestInferDirection:
    def test_positive_funding_long_ls(self):
        snap = _SymbolSnapshot(
            symbol="X", timestamp=datetime.now(timezone.utc),
            funding_rate=0.001, long_short_ratio=1.5,
        )
        assert LiquidationCascadeDetector._infer_direction(snap) == "long_liquidation"

    def test_negative_funding_short_ls(self):
        snap = _SymbolSnapshot(
            symbol="X", timestamp=datetime.now(timezone.utc),
            funding_rate=-0.001, long_short_ratio=0.4,
        )
        assert LiquidationCascadeDetector._infer_direction(snap) == "short_squeeze"

    def test_zero_funding_defaults_to_long_liquidation(self):
        snap = _SymbolSnapshot(
            symbol="X", timestamp=datetime.now(timezone.utc),
            funding_rate=0.0, long_short_ratio=1.0,
        )
        assert LiquidationCascadeDetector._infer_direction(snap) == "long_liquidation"

    def test_none_values_fallback(self):
        snap = _SymbolSnapshot(
            symbol="X", timestamp=datetime.now(timezone.utc),
            funding_rate=None, long_short_ratio=None,
        )
        # funding=0.0, ls=1.0 -> long_liquidation
        assert LiquidationCascadeDetector._infer_direction(snap) == "long_liquidation"


# ---------------------------------------------------------------------------
# History management tests
# ---------------------------------------------------------------------------

class TestHistoryManagement:
    def test_update_history_appends_values(self, detector):
        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=0.0001,
            open_interest=50_000.0,
        )
        detector._update_history(snap)
        hist = detector._history["BTCUSDT"]
        assert len(hist.funding_rates) == 1
        assert hist.funding_rates[-1] == 0.0001
        assert len(hist.open_interests) == 1

    def test_none_values_not_appended(self, detector):
        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=None,
            open_interest=None,
        )
        detector._update_history(snap)
        hist = detector._history["BTCUSDT"]
        assert len(hist.funding_rates) == 0
        assert len(hist.open_interests) == 0
        # timestamp is always appended
        assert len(hist.timestamps) == 1

    def test_history_for_unknown_symbol_created(self, detector):
        snap = _SymbolSnapshot(
            symbol="NEWCOIN",
            timestamp=datetime.now(timezone.utc),
            funding_rate=0.0002,
            open_interest=1000.0,
        )
        detector._update_history(snap)
        assert "NEWCOIN" in detector._history
        assert len(detector._history["NEWCOIN"].funding_rates) == 1


# ---------------------------------------------------------------------------
# Risk scoring tests
# ---------------------------------------------------------------------------

class TestScoreRisk:
    def _build_history(self, detector, symbol, count=30, fr=0.0001, oi=100_000.0):
        """Populate rolling history for a symbol."""
        hist = detector._history.setdefault(symbol, _SymbolHistory())
        for _ in range(count):
            hist.funding_rates.append(fr)
            hist.open_interests.append(oi)
            hist.timestamps.append(datetime.now(timezone.utc))

    def test_score_zero_with_neutral_data(self, detector, snapshot_neutral):
        self._build_history(detector, "BTCUSDT", count=30)
        risk = detector._score_risk("BTCUSDT", snapshot_neutral)
        assert risk["risk_score"] <= 0.15  # should be low

    def test_extreme_funding_triggers_score(self, custom_detector, snapshot_high_risk):
        """Extreme funding rate should add 0.35 to score."""
        self._build_history(custom_detector, "BTCUSDT", count=30, fr=0.0001)
        risk = custom_detector._score_risk("BTCUSDT", snapshot_high_risk)
        # funding 0.005 is well above 90th percentile of 0.0001 history
        assert risk["risk_score"] >= 0.35

    def test_extreme_ls_ratio_triggers_score(self, custom_detector):
        """LS ratio > extreme threshold adds 0.25."""
        self._build_history(custom_detector, "BTCUSDT", count=5)
        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=None,
            open_interest=None,
            long_short_ratio=3.0,  # > extreme_ls_ratio of 1.5
        )
        risk = custom_detector._score_risk("BTCUSDT", snap)
        assert risk["risk_score"] >= 0.25

    def test_high_oi_change_triggers_score(self, custom_detector):
        """OI change > threshold adds 0.25."""
        hist = custom_detector._history.setdefault("BTCUSDT", _SymbolHistory())
        # Build enough history with OI at 100k, then snapshot at 120k (20% change)
        for _ in range(200):
            hist.open_interests.append(100_000.0)
            hist.funding_rates.append(0.0001)
            hist.timestamps.append(datetime.now(timezone.utc))

        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=None,
            open_interest=120_000.0,  # 20% above baseline
            long_short_ratio=1.0,
        )
        risk = custom_detector._score_risk("BTCUSDT", snap)
        assert risk["risk_score"] >= 0.25

    def test_score_capped_at_one(self, custom_detector, snapshot_high_risk):
        """Risk score must not exceed 1.0."""
        self._build_history(custom_detector, "BTCUSDT", count=30, fr=0.0001)
        risk = custom_detector._score_risk("BTCUSDT", snapshot_high_risk)
        assert risk["risk_score"] <= 1.0

    def test_risk_returns_correct_keys(self, detector, snapshot_neutral):
        risk = detector._score_risk("BTCUSDT", snapshot_neutral)
        for key in ("risk_score", "direction", "funding_rate",
                     "funding_rate_percentile", "open_interest_change_24h",
                     "long_short_ratio", "timestamp"):
            assert key in risk

    def test_funding_divergence_triggers_score(self, detector):
        """Funding rate divergence (recent vs older window) adds 0.15."""
        hist = detector._history.setdefault("BTCUSDT", _SymbolHistory())
        # Older window: 10 readings at 0.0001
        for _ in range(10):
            hist.funding_rates.append(0.0001)
            hist.timestamps.append(datetime.now(timezone.utc))
        # Recent window: 10 readings at 0.001 (10x increase)
        for _ in range(10):
            hist.funding_rates.append(0.001)
            hist.timestamps.append(datetime.now(timezone.utc))

        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=0.001,
            open_interest=None,
            long_short_ratio=1.0,
        )
        risk = detector._score_risk("BTCUSDT", snap)
        # Should have at least divergence component
        assert risk["risk_score"] >= 0.15

    def test_short_history_extreme_funding_fallback(self, detector):
        """If history < 10 but |funding| > 0.001, add 0.20 as fallback."""
        hist = detector._history.setdefault("BTCUSDT", _SymbolHistory())
        for _ in range(5):  # not enough for percentile calc
            hist.funding_rates.append(0.0001)
            hist.timestamps.append(datetime.now(timezone.utc))

        snap = _SymbolSnapshot(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            funding_rate=0.002,  # > 0.001
            open_interest=None,
            long_short_ratio=1.0,
        )
        risk = detector._score_risk("BTCUSDT", snap)
        assert risk["risk_score"] >= 0.20


# ---------------------------------------------------------------------------
# Signal building tests
# ---------------------------------------------------------------------------

class TestBuildSignal:
    def test_high_risk_gets_immediate_action(self, detector, snapshot_high_risk):
        risk = {
            "risk_score": 0.85,
            "direction": "long_liquidation",
            "funding_rate": 0.005,
            "funding_rate_percentile": 0.95,
            "open_interest_change_24h": 12.0,
            "long_short_ratio": 2.5,
        }
        sig = detector._build_signal("BTCUSDT", snapshot_high_risk, risk)
        assert isinstance(sig, CascadeRiskSignal)
        assert sig.recommended_action == "open_short"
        assert sig.entry_trigger == "immediate_market_order"
        assert sig.expected_move_pct == round(0.85 * 8.0, 2)

    def test_medium_risk_gets_prepare_action(self, detector, snapshot_high_risk):
        risk = {
            "risk_score": 0.65,
            "direction": "short_squeeze",
            "funding_rate": -0.002,
            "funding_rate_percentile": 0.80,
            "open_interest_change_24h": 5.0,
            "long_short_ratio": 0.4,
        }
        sig = detector._build_signal("BTCUSDT", snapshot_high_risk, risk)
        assert sig.recommended_action == "prepare_long"
        assert sig.entry_trigger == "limit_order_at_key_level"

    def test_low_risk_gets_monitor_action(self, detector, snapshot_neutral):
        risk = {
            "risk_score": 0.55,
            "direction": "long_liquidation",
            "funding_rate": 0.0001,
        }
        sig = detector._build_signal("BTCUSDT", snapshot_neutral, risk)
        assert sig.recommended_action == "monitor_closely"
        assert sig.entry_trigger == "wait_for_confirmation"

    def test_signal_has_uuid(self, detector, snapshot_neutral):
        risk = {"risk_score": 0.9, "direction": "long_liquidation"}
        sig = detector._build_signal("BTCUSDT", snapshot_neutral, risk)
        assert len(sig.signal_id) > 0  # non-empty UUID string


# ---------------------------------------------------------------------------
# on_price_update / fast eval tests
# ---------------------------------------------------------------------------

class TestOnPriceUpdate:
    def test_no_op_when_fast_eval_disabled(self, detector):
        """Default config has fast_eval disabled, so updates are ignored."""
        detector.on_price_update("BTCUSDT", 50000.0, 1.0, 2.0, 1000.0)
        state = detector._realtime_state.get("BTCUSDT")
        assert len(state.price_window) == 0

    def test_feed_data_when_fast_eval_enabled(self):
        det = LiquidationCascadeDetector({
            "fast_eval_enabled": True,
            "symbols": ["BTCUSDT"],
        })
        det.on_price_update("BTCUSDT", 50000.0, 1.0, 2.0, 1000.0)
        state = det._realtime_state["BTCUSDT"]
        assert len(state.price_window) == 1
        assert state.price_window[-1] == (1000.0, 50000.0)

    def test_ignore_untracked_symbol(self):
        det = LiquidationCascadeDetector({
            "fast_eval_enabled": True,
            "symbols": ["BTCUSDT"],
        })
        det.on_price_update("RANDOMCOIN", 1.0, 1.0, 1.0, 1000.0)
        assert "RANDOMCOIN" not in det._realtime_state or \
            len(det._realtime_state.get("RANDOMCOIN", MagicMock(price_window=[])).price_window) == 0


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------

class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, detector):
        await detector.start()
        assert detector._running is True
        assert detector._session is not None
        await detector.stop()
        assert detector._running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, detector):
        await detector.start()
        task1 = detector._scan_task
        await detector.start()  # should warn and return
        assert detector._scan_task is task1
        await detector.stop()

    @pytest.mark.asyncio
    async def test_get_signals_empty_queue(self, detector):
        signals = await detector.get_signals()
        assert signals == []

    @pytest.mark.asyncio
    async def test_get_current_risk_empty(self, detector):
        risk = await detector.get_current_risk()
        assert risk == {}


# ---------------------------------------------------------------------------
# API fetch tests (mocked)
# ---------------------------------------------------------------------------

class TestApiFetching:
    @pytest.mark.asyncio
    async def test_fetch_funding_rate_success(self, detector):
        await detector.start()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[{"fundingRate": "0.0001"}])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(detector._session, "get", return_value=mock_resp):
            rate = await detector._fetch_funding_rate("BTCUSDT")
            assert rate == 0.0001
        await detector.stop()

    @pytest.mark.asyncio
    async def test_fetch_funding_rate_empty(self, detector):
        await detector.start()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=[])
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(detector._session, "get", return_value=mock_resp):
            rate = await detector._fetch_funding_rate("BTCUSDT")
            assert rate is None
        await detector.stop()

    @pytest.mark.asyncio
    async def test_api_get_rate_limit_429(self, detector):
        await detector.start()
        mock_resp = AsyncMock()
        mock_resp.status = 429
        mock_resp.headers = {"Retry-After": "30"}
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(detector._session, "get", return_value=mock_resp):
            result = await detector._api_get("/test", {})
            assert result is None
            assert detector._backoff_until > 0
        await detector.stop()

    @pytest.mark.asyncio
    async def test_api_get_ip_ban_418(self, detector):
        await detector.start()
        mock_resp = AsyncMock()
        mock_resp.status = 418
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        with patch.object(detector._session, "get", return_value=mock_resp):
            result = await detector._api_get("/test", {})
            assert result is None
        await detector.stop()

    @pytest.mark.asyncio
    async def test_api_get_no_session(self, detector):
        """When session is None, _api_get returns None."""
        result = await detector._api_get("/test", {})
        assert result is None
