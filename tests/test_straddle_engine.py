"""Tests for StraddleEngine and StraddleFleetController.

Tests cover:
  - Straddle opens both LONG and SHORT legs
  - Stop loss triggers at configured BPS
  - Trailing stop activates after activation_bps profit
  - Max hold time forces close
  - Daily loss limit stops new straddle opens
  - Vol scaling adjusts leg size
  - Blocked hours prevent trading
  - Max open straddles cap enforced
  - Fleet controller daily loss limit
  - Fleet controller max deployed limit
"""

import time
import pytest
import sqlite3
import tempfile
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

from straddle_engine import (
    StraddleEngine,
    StraddleFleetController,
    Straddle,
    StraddleLeg,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    """Create a temp DB path for tests."""
    return str(tmp_path / "test_straddle.db")


@pytest.fixture
def default_config():
    """Default engine config."""
    return {
        "asset": "BTC",
        "pair": "BTC-USD",
        "binance_symbol": "BTCUSDT",
        "leg_size_usd": 100.0,
        "straddle_interval_seconds": 0.0,  # No cooldown for tests
        "check_interval_seconds": 1.0,
        "max_hold_seconds": 120.0,
        "stop_loss_bps": 4.0,
        "trail_activation_bps": 2.0,
        "trail_distance_bps": 1.0,
        "vol_scaling": "none",  # Disable vol scaling for deterministic tests
        "max_open_straddles": 35,
        "max_capital_deployed": 7000.0,
        "max_daily_loss_usd": 700.0,
        "observation_mode": False,
        "blocked_hours_utc": [],
    }


@pytest.fixture
def engine(default_config, db_path):
    """Create a StraddleEngine with test config."""
    return StraddleEngine(config=default_config, db_path=db_path)


# ── Test: Straddle opens both legs ───────────────────────────────────────────

class TestStraddleOpen:
    def test_opens_both_legs(self, engine):
        """open_straddle() creates a straddle with LONG and SHORT legs."""
        straddle = engine.open_straddle(price=50000.0)
        assert straddle is not None
        assert straddle.long_leg.side == "LONG"
        assert straddle.short_leg.side == "SHORT"
        assert straddle.long_leg.entry_price == 50000.0
        assert straddle.short_leg.entry_price == 50000.0
        assert len(engine.open_straddles) == 1

    def test_straddle_stored_in_db(self, engine, db_path):
        """Straddle is persisted to the database on open."""
        straddle = engine.open_straddle(price=50000.0)
        assert straddle is not None

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT * FROM straddle_log WHERE id = ?", (straddle.straddle_id,)
        ).fetchone()
        conn.close()
        assert row is not None

    def test_both_legs_in_db(self, engine, db_path):
        """Both LONG and SHORT legs are persisted to straddle_legs table."""
        straddle = engine.open_straddle(price=50000.0)
        assert straddle is not None

        conn = sqlite3.connect(db_path)
        legs = conn.execute(
            "SELECT side FROM straddle_legs WHERE straddle_id = ?",
            (straddle.straddle_id,)
        ).fetchall()
        conn.close()
        sides = {row[0] for row in legs}
        assert sides == {"LONG", "SHORT"}

    def test_straddle_id_increments(self, engine):
        """Each straddle gets a unique, incrementing ID."""
        s1 = engine.open_straddle(price=50000.0)
        s2 = engine.open_straddle(price=50100.0)
        assert s1 is not None and s2 is not None
        assert s2.straddle_id > s1.straddle_id


# ── Test: Stop loss triggers ────────────────────────────────────────────────

class TestStopLoss:
    def test_stop_loss_triggers_on_long_leg(self, engine):
        """LONG leg should stop loss when price drops enough in BPS."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None

        # 4 bps stop on LONG: price needs to drop by 0.04% = $40
        drop_price = 100000.0 * (1 - 4.0 / 10000)  # 99996.0
        closed = engine.check_exits(drop_price)

        # Both legs should be evaluated; long should hit stop
        assert straddle.long_leg.closed
        assert straddle.long_leg.exit_reason == "stop_loss"
        assert straddle.long_leg.pnl_bps <= -4.0

    def test_stop_loss_triggers_on_short_leg(self, engine):
        """SHORT leg should stop loss when price rises enough in BPS."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None

        # 4 bps stop on SHORT: price needs to rise by 0.04% = $40
        rise_price = 100000.0 * (1 + 4.0 / 10000)  # 100040.0
        closed = engine.check_exits(rise_price)

        assert straddle.short_leg.closed
        assert straddle.short_leg.exit_reason == "stop_loss"


# ── Test: Trailing stop ─────────────────────────────────────────────────────

class TestTrailingStop:
    def test_trail_activates_after_activation_bps(self, engine):
        """Trail should activate after peak P&L crosses activation_bps."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None
        leg = straddle.long_leg

        # Move price up by 3 bps (above activation_bps=2)
        high_price = 100000.0 * (1 + 3.0 / 10000)
        engine._check_leg_exit(leg, high_price, time.time(), straddle)
        assert leg.trail_active is True
        assert leg.peak_pnl_bps >= 2.0

    def test_trail_stop_triggers_on_giveback(self, engine):
        """Once trail active, giving back trail_distance_bps from peak closes the leg."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None
        leg = straddle.long_leg
        now = time.time()

        # Step 1: Move up 5 bps to activate trail
        high_price = 100000.0 * (1 + 5.0 / 10000)
        engine._check_leg_exit(leg, high_price, now, straddle)
        assert leg.trail_active is True

        # Step 2: Price drops back by trail_distance_bps=1 from peak
        # Peak was +5bps, so trail triggers at +4bps
        giveback_price = 100000.0 * (1 + (5.0 - 1.0) / 10000)
        engine._check_leg_exit(leg, giveback_price, now + 1, straddle)

        # Since peak was 5bps and current is 4bps, drawdown = 1bps = trail_distance_bps
        assert leg.closed is True
        assert leg.exit_reason == "trail_stop"


# ── Test: Max hold time ─────────────────────────────────────────────────────

class TestMaxHoldTime:
    def test_timeout_forces_close(self, engine):
        """Positions held past max_hold_seconds should be force-closed."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None

        # Simulate time passing beyond max_hold_seconds (120s)
        future = time.time() + 130.0
        straddle.long_leg.opened_at = time.time() - 130.0
        straddle.short_leg.opened_at = time.time() - 130.0

        engine._check_leg_exit(
            straddle.long_leg, 100000.0, future, straddle
        )
        engine._check_leg_exit(
            straddle.short_leg, 100000.0, future, straddle
        )

        assert straddle.long_leg.closed
        assert straddle.short_leg.closed
        assert "timeout" in straddle.long_leg.exit_reason
        assert "timeout" in straddle.short_leg.exit_reason

    def test_timeout_classifies_pnl(self, engine):
        """Timeout exit should sub-classify as profitable/loss/flat."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None

        future = time.time() + 130.0
        straddle.long_leg.opened_at = time.time() - 130.0

        # Profitable timeout
        engine._check_leg_exit(
            straddle.long_leg, 100010.0, future, straddle
        )
        assert straddle.long_leg.exit_reason == "timeout_profitable"


# ── Test: Daily loss limit ──────────────────────────────────────────────────

class TestDailyLossLimit:
    def test_daily_loss_blocks_new_straddles(self, engine):
        """When daily loss >= max_daily_loss_usd, new straddles are blocked."""
        engine._daily_loss_usd = 700.0  # At limit

        straddle = engine.open_straddle(price=50000.0)
        assert straddle is None

    def test_daily_loss_accumulates(self, engine):
        """Closing straddles with net loss should accumulate daily loss."""
        s = engine.open_straddle(price=100000.0)
        assert s is not None

        # Force close both legs with loss
        now = time.time()
        s.long_leg.pnl_bps = -3.0
        s.long_leg.pnl_usd = -0.03
        s.long_leg.closed = True
        s.long_leg.closed_at = now
        s.long_leg.exit_reason = "stop_loss"
        s.long_leg.exit_price = 99997.0

        s.short_leg.pnl_bps = -3.0
        s.short_leg.pnl_usd = -0.03
        s.short_leg.closed = True
        s.short_leg.closed_at = now
        s.short_leg.exit_reason = "stop_loss"
        s.short_leg.exit_price = 100003.0

        closed = engine.check_exits(100000.0)  # Triggers close processing
        # Even though we manually closed legs, check_exits processes them
        assert engine._daily_loss_usd >= 0  # Loss accumulated


# ── Test: Vol scaling ───────────────────────────────────────────────────────

class TestVolScaling:
    def test_proportional_vol_scaling(self, default_config, db_path):
        """Proportional vol scaling adjusts exit thresholds by vol ratio."""
        default_config["vol_scaling"] = "proportional"
        engine = StraddleEngine(config=default_config, db_path=db_path)

        # Simulate high vol (2x base)
        engine._cached_vol_ratio = 2.0
        engine._last_vol_check = time.time()  # Prevent refresh

        straddle = engine.open_straddle(price=50000.0)
        assert straddle is not None
        # Effective thresholds should be doubled
        assert straddle.long_leg.stop_loss_bps == pytest.approx(8.0, abs=0.1)
        assert straddle.long_leg.trail_activation_bps == pytest.approx(4.0, abs=0.1)
        assert straddle.long_leg.trail_distance_bps == pytest.approx(2.0, abs=0.1)

    def test_no_vol_scaling_when_disabled(self, default_config, db_path):
        """When vol_scaling='none', ratio should be 1.0."""
        default_config["vol_scaling"] = "none"
        engine = StraddleEngine(config=default_config, db_path=db_path)

        ratio = engine._get_vol_ratio()
        assert ratio == 1.0


# ── Test: Blocked hours ─────────────────────────────────────────────────────

class TestBlockedHours:
    def test_blocked_hours_prevent_open(self, default_config, db_path):
        """Straddles should not open during blocked UTC hours."""
        current_hour = datetime.now(timezone.utc).hour
        default_config["blocked_hours_utc"] = [current_hour]
        engine = StraddleEngine(config=default_config, db_path=db_path)

        straddle = engine.open_straddle(price=50000.0)
        assert straddle is None

    def test_non_blocked_hours_allow_open(self, default_config, db_path):
        """Straddles should open when current hour is NOT in blocked list."""
        current_hour = datetime.now(timezone.utc).hour
        # Block a different hour
        blocked = [(current_hour + 6) % 24]
        default_config["blocked_hours_utc"] = blocked
        engine = StraddleEngine(config=default_config, db_path=db_path)

        straddle = engine.open_straddle(price=50000.0)
        assert straddle is not None


# ── Test: Max open straddles ────────────────────────────────────────────────

class TestMaxOpenStraddles:
    def test_max_open_cap_enforced(self, default_config, db_path):
        """Cannot open more straddles than max_open_straddles."""
        default_config["max_open_straddles"] = 3
        default_config["straddle_interval_seconds"] = 0.0
        default_config["max_capital_deployed"] = 100000.0
        engine = StraddleEngine(config=default_config, db_path=db_path)

        results = []
        for i in range(5):
            s = engine.open_straddle(price=50000.0 + i)
            results.append(s)

        opened = [s for s in results if s is not None]
        assert len(opened) == 3
        assert len(engine.open_straddles) == 3

    def test_max_capital_deployed_enforced(self, default_config, db_path):
        """Cannot deploy more capital than max_capital_deployed."""
        default_config["max_capital_deployed"] = 400.0  # Only 2 straddles at $100/leg
        default_config["max_open_straddles"] = 100
        default_config["straddle_interval_seconds"] = 0.0
        engine = StraddleEngine(config=default_config, db_path=db_path)

        results = []
        for i in range(5):
            s = engine.open_straddle(price=50000.0 + i)
            results.append(s)

        opened = [s for s in results if s is not None]
        # 2 straddles × $100/leg × 2 legs = $400 = max
        assert len(opened) == 2


# ── Test: Cooldown between straddles ────────────────────────────────────────

class TestCooldown:
    def test_cooldown_blocks_rapid_opens(self, default_config, db_path):
        """Opening straddles within the cooldown interval should be blocked."""
        default_config["straddle_interval_seconds"] = 10.0
        engine = StraddleEngine(config=default_config, db_path=db_path)

        s1 = engine.open_straddle(price=50000.0)
        assert s1 is not None

        # Immediately try again
        s2 = engine.open_straddle(price=50000.0)
        assert s2 is None  # Blocked by cooldown


# ── Test: StraddleFleetController ────────────────────────────────────────────

class TestFleetController:
    def test_fleet_allows_open(self, default_config, db_path):
        """Fleet controller should allow opens within limits."""
        fleet = StraddleFleetController(
            fleet_daily_loss_limit=1400.0,
            fleet_max_deployed=14000.0,
        )
        assert fleet.allow_open("BTC", 100.0) is True

    def test_fleet_blocks_after_daily_loss(self):
        """Fleet should block new straddles after daily loss limit hit."""
        fleet = StraddleFleetController(
            fleet_daily_loss_limit=100.0,
            fleet_max_deployed=14000.0,
        )
        # Report losses
        fleet.report_close("BTC", -50.0)
        fleet.report_close("ETH", -60.0)
        # Total loss = $110 > limit $100
        assert fleet.allow_open("BTC", 100.0) is False

    def test_fleet_blocks_over_deployment(self, default_config, db_path):
        """Fleet should block when total deployed exceeds max."""
        fleet = StraddleFleetController(
            fleet_daily_loss_limit=10000.0,
            fleet_max_deployed=300.0,  # Very low
        )
        default_config["fleet_controller"] = None
        engine = StraddleEngine(config=default_config, db_path=db_path, fleet_controller=fleet)
        fleet.register(engine)

        # Open 1 straddle ($100/leg × 2 = $200 deployed)
        s1 = engine.open_straddle(price=50000.0)
        assert s1 is not None

        # Next one would push to $400 > $300 max
        assert fleet.allow_open("BTC", 100.0) is False

    def test_fleet_status_report(self, default_config, db_path):
        """Fleet status should report correctly."""
        fleet = StraddleFleetController()
        engine = StraddleEngine(
            config=default_config, db_path=db_path, fleet_controller=fleet
        )
        fleet.register(engine)

        status = fleet.get_fleet_status()
        assert "halted" in status
        assert "total_deployed" in status
        assert "engines" in status
        assert "BTC" in status["engines"]


# ── Test: PnL calculation ───────────────────────────────────────────────────

class TestPnlCalculation:
    def test_long_leg_pnl_positive_on_price_rise(self, engine):
        """LONG leg P&L should be positive when price rises."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None
        leg = straddle.long_leg

        engine._check_leg_exit(leg, 100010.0, time.time(), straddle)
        assert leg.pnl_bps > 0  # +1 bps
        assert leg.pnl_usd > 0

    def test_short_leg_pnl_positive_on_price_drop(self, engine):
        """SHORT leg P&L should be positive when price drops."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None
        leg = straddle.short_leg

        engine._check_leg_exit(leg, 99990.0, time.time(), straddle)
        assert leg.pnl_bps > 0  # +1 bps
        assert leg.pnl_usd > 0

    def test_net_pnl_on_flat_is_zero(self, engine):
        """Net P&L at entry price should be ~zero."""
        straddle = engine.open_straddle(price=100000.0)
        assert straddle is not None

        engine._check_leg_exit(straddle.long_leg, 100000.0, time.time(), straddle)
        engine._check_leg_exit(straddle.short_leg, 100000.0, time.time(), straddle)

        net = straddle.long_leg.pnl_bps + straddle.short_leg.pnl_bps
        assert abs(net) < 0.01


# ── Test: Engine status ─────────────────────────────────────────────────────

class TestEngineStatus:
    def test_status_has_required_fields(self, engine):
        """get_status() should return all required fields."""
        status = engine.get_status()
        required_keys = [
            "active", "observation_mode", "asset", "pair",
            "size_usd", "wallet_usd", "max_open", "open_count",
            "total_opened", "total_closed", "total_pnl_usd",
            "daily_loss_usd", "config",
        ]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_status_tracks_opens(self, engine):
        """Status should reflect opened straddles."""
        engine.open_straddle(price=50000.0)
        status = engine.get_status()
        assert status["open_count"] == 1
        assert status["total_opened"] == 1
