"""Tests for TokenSprayEngine — rapid-fire micro-position engine.

Tests cover:
  - Token spray creates position at min signal strength
  - Blacklisted pairs are rejected
  - Max budget cap prevents over-spending
  - Max tokens per pair enforced
  - Stop loss exit at configured BPS
  - Trail activation and distance work correctly
  - Max hold time forces close
  - Cooldown prevents immediate re-entry
  - Direction rule classification
  - Vol regime classification
  - PnL calculation for LONG and SHORT
"""

import time
import asyncio
import pytest
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from strategies.token_spray_engine import (
    TokenSprayEngine,
    SprayToken,
    Wallet,
    _get_pair_exit_config,
    DEFAULT_PAIR_EXIT,
    PAIR_EXIT_DEFAULTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    """Create temp DB path and ensure table exists."""
    path = str(tmp_path / "test_spray.db")
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS token_spray_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            token_id TEXT UNIQUE,
            pair TEXT,
            direction TEXT,
            direction_rule TEXT,
            token_size_usd REAL,
            entry_price REAL,
            vol_regime TEXT,
            expected_move_bps REAL,
            target_bps REAL,
            stop_bps REAL,
            max_hold_seconds REAL,
            observation_mode INTEGER,
            weighted_signal REAL,
            confidence REAL,
            wallet_id TEXT,
            exit_price REAL,
            exit_time TEXT,
            exit_reason TEXT,
            exit_pnl_bps REAL,
            exit_pnl_usd REAL,
            hold_time_seconds REAL,
            peak_pnl_bps REAL
        )
    """)
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def default_config():
    """Default spray engine config."""
    return {
        "token_size_usd": 20.0,
        "max_budget_usd": 2000.0,
        "max_open_tokens": 100,
        "max_tokens_per_pair": 5,
        "min_signal_strength": 0.02,
        "min_confidence": 0.42,
        "cooldown_seconds": 0.0,  # No cooldown for tests
        "exit_check_interval_seconds": 5.0,
        "observation_mode": True,
        "exit_config": {
            "stop_loss_bps": 12.0,
            "trail_activation_bps": 3.0,
            "trail_distance_bps": 5.0,
            "max_hold_seconds": 600.0,
            "min_move_bps": 3.0,
        },
        "blacklisted_pairs": ["SCAM-USD"],
        "vol_scaling": {"low": 1.0, "medium": 0.7, "high": 0.4, "extreme": 0.2},
    }


@pytest.fixture
def engine(default_config, db_path):
    """Create a TokenSprayEngine for tests."""
    return TokenSprayEngine(config=default_config, db_path=db_path)


@pytest.fixture
def market_data():
    """Standard market data dict for tests."""
    return {
        "ticker": {"price": 50000.0, "bid": 49999.0, "ask": 50001.0},
        "garch_forecast": {"forecast_vol": 0.01},  # "low" vol
    }


@pytest.fixture
def ml_package():
    """Mock ML package."""
    mock = MagicMock()
    mock.confidence_score = 0.6
    return mock


# ── Test: Token spray creates position ──────────────────────────────────────

class TestSprayCreation:
    @pytest.mark.asyncio
    async def test_spray_creates_token(self, engine, market_data, ml_package):
        """spray() with valid inputs should create and return a SprayToken."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.05,
            contributions={"rsi": 0.03, "macd": 0.02},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        assert isinstance(token, SprayToken)
        assert token.pair == "BTC-USD"
        assert token.side == "LONG"
        assert token.entry_price == 50000.0
        assert token.status == "open"

    @pytest.mark.asyncio
    async def test_spray_short_on_negative_signal(self, engine, market_data, ml_package):
        """Negative weighted_signal should create a SHORT token."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=-0.05,
            contributions={"rsi": -0.03},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        assert token.side == "SHORT"

    @pytest.mark.asyncio
    async def test_spray_uses_fresh_price(self, engine, market_data, ml_package):
        """When fresh_price is provided, it should be used over ticker price."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.05,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
            fresh_price=51000.0,
        )
        assert token is not None
        assert token.entry_price == 51000.0

    @pytest.mark.asyncio
    async def test_spray_registers_token(self, engine, market_data, ml_package):
        """Spray should register the token in open_tokens and update budget."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.05,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        assert token.token_id in engine.open_tokens
        assert engine.budget_deployed_usd > 0
        assert engine._total_sprayed == 1


# ── Test: Signal strength gate ──────────────────────────────────────────────

class TestSignalStrengthGate:
    @pytest.mark.asyncio
    async def test_weak_signal_rejected(self, engine, market_data, ml_package):
        """Signal below min_signal_strength should be rejected."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.01,  # Below 0.02 min
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is None

    @pytest.mark.asyncio
    async def test_zero_signal_rejected(self, engine, market_data, ml_package):
        """Zero signal should be rejected."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.0,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is None


# ── Test: Confidence gate ───────────────────────────────────────────────────

class TestConfidenceGate:
    @pytest.mark.asyncio
    async def test_low_confidence_rejected(self, engine, market_data, ml_package):
        """Confidence below min_confidence should be rejected."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.30,  # Below 0.42 min
        )
        assert token is None


# ── Test: Blacklisted pairs ────────────────────────────────────────────────

class TestBlacklist:
    @pytest.mark.asyncio
    async def test_blacklisted_pair_rejected(self, engine, market_data, ml_package):
        """Blacklisted pairs should be rejected."""
        token = await engine.spray(
            pair="SCAM-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is None

    @pytest.mark.asyncio
    async def test_non_blacklisted_pair_accepted(self, engine, market_data, ml_package):
        """Non-blacklisted pairs should be accepted."""
        token = await engine.spray(
            pair="ETH-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None


# ── Test: Max budget cap ───────────────────────────────────────────────────

class TestBudgetCap:
    @pytest.mark.asyncio
    async def test_budget_cap_blocks_new_tokens(self, engine, market_data, ml_package):
        """Cannot spray more tokens when budget is exhausted."""
        engine.budget_deployed_usd = 1990.0  # Near $2000 cap
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is None  # $20 + $1990 > $2000


# ── Test: Max tokens per pair ──────────────────────────────────────────────

class TestMaxTokensPerPair:
    @pytest.mark.asyncio
    async def test_max_tokens_per_pair_enforced(self, default_config, db_path, market_data, ml_package):
        """Cannot open more than max_tokens_per_pair on same pair."""
        default_config["max_tokens_per_pair"] = 2
        default_config["cooldown_seconds"] = 0.0
        engine = TokenSprayEngine(config=default_config, db_path=db_path)

        tokens = []
        for i in range(5):
            t = await engine.spray(
                pair="BTC-USD",
                weighted_signal=0.10,
                contributions={},
                ml_package=ml_package,
                market_data=market_data,
                confidence=0.6,
            )
            tokens.append(t)

        opened = [t for t in tokens if t is not None]
        assert len(opened) == 2

    @pytest.mark.asyncio
    async def test_different_pairs_have_independent_limits(self, default_config, db_path, market_data, ml_package):
        """Max per-pair limit is per pair, not global."""
        default_config["max_tokens_per_pair"] = 1
        default_config["cooldown_seconds"] = 0.0
        engine = TokenSprayEngine(config=default_config, db_path=db_path)

        t1 = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        t2 = await engine.spray(
            pair="ETH-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert t1 is not None
        assert t2 is not None


# ── Test: Stop loss exit ───────────────────────────────────────────────────

class TestStopLoss:
    @pytest.mark.asyncio
    async def test_stop_loss_closes_token(self, engine, market_data, ml_package):
        """Hard stop should close token when loss exceeds stop_loss_bps."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        entry = token.entry_price

        # Move price down beyond BTC stop (7 bps default for BTC)
        # Use the per-pair stop_loss_bps
        stop_bps = token.stop_loss_bps
        drop_price = entry * (1 - (stop_bps + 1) / 10000)

        async def price_fetcher(pairs):
            return {p: drop_price for p in pairs}

        closed = await engine.check_exits(price_fetcher)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop"

    @pytest.mark.asyncio
    async def test_short_stop_loss(self, engine, market_data, ml_package):
        """SHORT token stop loss should trigger on price rise."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=-0.10,  # SHORT
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        entry = token.entry_price
        stop_bps = token.stop_loss_bps

        # Price rises beyond stop
        rise_price = entry * (1 + (stop_bps + 1) / 10000)

        async def price_fetcher(pairs):
            return {p: rise_price for p in pairs}

        closed = await engine.check_exits(price_fetcher)
        assert len(closed) == 1
        assert closed[0].exit_reason == "stop"


# ── Test: Trailing stop ────────────────────────────────────────────────────

class TestTrailingStop:
    @pytest.mark.asyncio
    async def test_trail_activates_and_triggers(self, engine, market_data, ml_package):
        """Trail should activate after activation_bps and trigger on giveback."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        entry = token.entry_price
        trail_act = token.trail_activation_bps
        trail_dist = token.trail_distance_bps

        # Hack: set entry time far enough back to pass min_hold_for_trail
        token.entry_time = time.time() - 60

        # Step 1: Move price up to activate trail
        high_price = entry * (1 + (trail_act + 5) / 10000)

        async def price_high(pairs):
            return {p: high_price for p in pairs}

        await engine.check_exits(price_high)
        assert token.trail_active is True

        # Step 2: Price drops back by trail_distance from peak
        giveback_price = entry * (1 + (trail_act + 5 - trail_dist - 1) / 10000)

        async def price_drop(pairs):
            return {p: giveback_price for p in pairs}

        closed = await engine.check_exits(price_drop)
        assert len(closed) == 1
        assert closed[0].exit_reason == "trail_stop"


# ── Test: Max hold time ────────────────────────────────────────────────────

class TestMaxHoldTime:
    @pytest.mark.asyncio
    async def test_timeout_forces_close(self, engine, market_data, ml_package):
        """Tokens held past max_hold_seconds should be force-closed."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None

        # Make the token appear old
        token.entry_time = time.time() - 700  # 700s > 600s max_hold

        async def price_fetcher(pairs):
            return {p: 50000.0 for p in pairs}

        closed = await engine.check_exits(price_fetcher)
        assert len(closed) == 1
        assert "timeout" in closed[0].exit_reason

    @pytest.mark.asyncio
    async def test_timeout_classifies_flat(self, engine, market_data, ml_package):
        """Timeout with small P&L should classify as timeout_flat."""
        token = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert token is not None
        token.entry_time = time.time() - 700

        # Price barely moved (< min_move_bps=3)
        async def price_fetcher(pairs):
            return {p: token.entry_price * 1.0001 for p in pairs}  # +1bps

        closed = await engine.check_exits(price_fetcher)
        assert closed[0].exit_reason == "timeout_flat"


# ── Test: Cooldown ──────────────────────────────────────────────────────────

class TestCooldown:
    @pytest.mark.asyncio
    async def test_cooldown_blocks_rapid_spray(self, default_config, db_path, market_data, ml_package):
        """Cannot spray the same pair within cooldown_seconds."""
        default_config["cooldown_seconds"] = 30.0
        engine = TokenSprayEngine(config=default_config, db_path=db_path)

        t1 = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert t1 is not None

        # Immediately try again
        t2 = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert t2 is None  # Blocked by cooldown

    @pytest.mark.asyncio
    async def test_cooldown_per_pair(self, default_config, db_path, market_data, ml_package):
        """Cooldown is per-pair — different pair should work."""
        default_config["cooldown_seconds"] = 30.0
        engine = TokenSprayEngine(config=default_config, db_path=db_path)

        t1 = await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert t1 is not None

        # Different pair should work
        t2 = await engine.spray(
            pair="ETH-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        assert t2 is not None


# ── Test: PnL calculation ──────────────────────────────────────────────────

class TestPnlCalc:
    def test_long_pnl_positive_on_price_rise(self):
        """LONG token P&L should be positive when price rises."""
        token = SprayToken(
            token_id="test1", pair="BTC-USD", side="LONG",
            entry_price=50000.0, entry_time=time.time(),
            size_usd=20.0, size_units=0.0004,
            direction_rule="rsi", vol_regime="low",
            stop_loss_bps=8.0, trail_activation_bps=3.0,
            trail_distance_bps=5.0, min_hold_for_trail=30.0,
            max_hold_seconds=600.0, observation_mode=True,
        )
        pnl = TokenSprayEngine._calc_pnl_bps(token, 50010.0)
        assert pnl > 0  # +2 bps

    def test_short_pnl_positive_on_price_drop(self):
        """SHORT token P&L should be positive when price drops."""
        token = SprayToken(
            token_id="test2", pair="BTC-USD", side="SHORT",
            entry_price=50000.0, entry_time=time.time(),
            size_usd=20.0, size_units=0.0004,
            direction_rule="rsi", vol_regime="low",
            stop_loss_bps=8.0, trail_activation_bps=3.0,
            trail_distance_bps=5.0, min_hold_for_trail=30.0,
            max_hold_seconds=600.0, observation_mode=True,
        )
        pnl = TokenSprayEngine._calc_pnl_bps(token, 49990.0)
        assert pnl > 0  # +2 bps

    def test_zero_entry_price_returns_zero(self):
        """Edge case: zero entry price should return 0 bps."""
        token = SprayToken(
            token_id="test3", pair="BTC-USD", side="LONG",
            entry_price=0.0, entry_time=time.time(),
            size_usd=20.0, size_units=0.0,
            direction_rule="mixed", vol_regime="low",
            stop_loss_bps=8.0, trail_activation_bps=3.0,
            trail_distance_bps=5.0, min_hold_for_trail=30.0,
            max_hold_seconds=600.0, observation_mode=True,
        )
        pnl = TokenSprayEngine._calc_pnl_bps(token, 50000.0)
        assert pnl == 0.0


# ── Test: Direction rule classification ─────────────────────────────────────

class TestDirectionRule:
    def test_dominant_contributor_detected(self, engine):
        """When one signal contributes >30%, its key name is returned."""
        rule = engine._classify_direction_rule(
            {"rsi": 0.5, "macd": 0.1, "entropy": 0.1},
            weighted_signal=0.7,
        )
        assert rule == "rsi"

    def test_mixed_when_no_dominant(self, engine):
        """When no signal contributes >30%, 'mixed' is returned."""
        rule = engine._classify_direction_rule(
            {"rsi": 0.25, "macd": 0.25, "entropy": 0.25, "ml": 0.25},
            weighted_signal=1.0,
        )
        assert rule == "mixed"

    def test_empty_contributions(self, engine):
        """Empty contributions should return 'mixed'."""
        rule = engine._classify_direction_rule({}, weighted_signal=0.1)
        assert rule == "mixed"


# ── Test: Vol regime classification ─────────────────────────────────────────

class TestVolRegime:
    def test_low_vol_from_garch(self, engine):
        """Low GARCH forecast should return 'low'."""
        regime = engine._calc_vol_regime({"garch_forecast": {"forecast_vol": 0.01}})
        assert regime == "low"

    def test_high_vol_from_garch(self, engine):
        """High GARCH forecast should return 'high'."""
        regime = engine._calc_vol_regime({"garch_forecast": {"forecast_vol": 0.05}})
        assert regime == "high"

    def test_extreme_vol_from_garch(self, engine):
        """Very high GARCH forecast should return 'extreme'."""
        regime = engine._calc_vol_regime({"garch_forecast": {"forecast_vol": 0.10}})
        assert regime == "extreme"

    def test_fallback_to_medium(self, engine):
        """No vol data should default to 'medium'."""
        regime = engine._calc_vol_regime({})
        assert regime == "medium"


# ── Test: Per-pair exit config ──────────────────────────────────────────────

class TestPairExitConfig:
    def test_btc_has_custom_config(self):
        """BTC should have its own exit parameters."""
        cfg = _get_pair_exit_config("BTC-USD", {})
        assert cfg["stop_loss_bps"] == 7  # BTC-specific

    def test_unknown_pair_uses_default(self):
        """Unknown pairs should use DEFAULT_PAIR_EXIT."""
        cfg = _get_pair_exit_config("UNKNOWN-USD", {})
        assert cfg["stop_loss_bps"] == DEFAULT_PAIR_EXIT["stop_loss_bps"]

    def test_config_overrides_take_priority(self):
        """Config overrides should beat built-in defaults."""
        overrides = {"BTC": {"stop_loss_bps": 99, "trail_activation_bps": 50,
                             "trail_distance_bps": 25}}
        cfg = _get_pair_exit_config("BTC-USD", overrides)
        assert cfg["stop_loss_bps"] == 99


# ── Test: Engine status ────────────────────────────────────────────────────

class TestEngineStatus:
    def test_status_has_required_fields(self, engine):
        """get_status() should return all required dashboard fields."""
        status = engine.get_status()
        assert "active" in status
        assert "observation_mode" in status
        assert "total_sprayed" in status
        assert "total_open" in status
        assert "budget" in status
        assert "exit_config" in status

    @pytest.mark.asyncio
    async def test_status_tracks_opens(self, engine, market_data, ml_package):
        """Status should reflect sprayed tokens."""
        await engine.spray(
            pair="BTC-USD",
            weighted_signal=0.10,
            contributions={},
            ml_package=ml_package,
            market_data=market_data,
            confidence=0.6,
        )
        status = engine.get_status()
        assert status["total_sprayed"] == 1
        assert status["total_open"] == 1


# ── Test: Max open tokens cap ──────────────────────────────────────────────

class TestMaxOpenTokens:
    @pytest.mark.asyncio
    async def test_max_open_tokens_enforced(self, default_config, db_path, market_data, ml_package):
        """Cannot open more than max_open_tokens globally."""
        default_config["max_open_tokens"] = 3
        default_config["max_tokens_per_pair"] = 100
        default_config["cooldown_seconds"] = 0.0
        engine = TokenSprayEngine(config=default_config, db_path=db_path)

        pairs = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD"]
        tokens = []
        for pair in pairs:
            t = await engine.spray(
                pair=pair,
                weighted_signal=0.10,
                contributions={},
                ml_package=ml_package,
                market_data=market_data,
                confidence=0.6,
            )
            tokens.append(t)

        opened = [t for t in tokens if t is not None]
        assert len(opened) == 3
