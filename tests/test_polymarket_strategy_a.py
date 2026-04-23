"""Tests for StrategyAExecutor — Polymarket multi-asset crash model betting.

Tests cover:
  - Kelly criterion bet sizing (half-Kelly, floors, ceilings)
  - window_start_price used for resolution (not entry_asset_price)
  - Bankroll reconciliation logic (load, periodic, auto-correct)
  - Per-asset daily loss limits (global + 5m-specific)
  - Max positions cap (MAX_OPEN_BETS)
  - Confidence thresholds (entry floor, cap, exit, add)
  - Rate limiting and cooldown after loss
  - Asset concentration check
  - Minimum bankroll gate
"""

import sys
import os
import sqlite3
import time
import logging
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import pytest

# Archived module — add archive path so import resolves
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "archive", "polymarket"))

# Mock external deps that may not be installed in test env
if "requests" not in sys.modules:
    sys.modules["requests"] = MagicMock()
# polymarket_timing_features is also in archive — mock if import fails
if "polymarket_timing_features" not in sys.modules:
    _mock_timing_mod = MagicMock()
    _mock_timing_mod.TimingFeatureEngine = MagicMock
    sys.modules["polymarket_timing_features"] = _mock_timing_mod


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_db(tmp_path) -> str:
    """Return path to an empty DB; let StrategyAExecutor._ensure_tables() create schema."""
    db_path = str(tmp_path / "test.db")
    # Create the file so sqlite3.connect works
    sqlite3.connect(db_path).close()
    return db_path


def _make_config(bankroll: float = 500.0) -> dict:
    return {
        "polymarket": {
            "executor_enabled": True,
            "initial_bankroll": bankroll,
        }
    }


@pytest.fixture
def db_path(tmp_path):
    return _make_db(tmp_path)


@pytest.fixture
def executor(db_path):
    """Create a StrategyAExecutor with mocked TimingFeatureEngine."""
    with patch("polymarket_strategy_a.TimingFeatureEngine") as mock_timing:
        mock_timing.return_value.is_follower.return_value = False
        from polymarket_strategy_a import StrategyAExecutor
        exe = StrategyAExecutor(
            config=_make_config(500.0),
            db_path=db_path,
            logger=logging.getLogger("test"),
        )
    return exe


# ═══════════════════════════════════════════════════════════════
# KELLY CRITERION BET SIZING
# ═══════════════════════════════════════════════════════════════

class TestKellyBetSizing:
    """Tests for _compute_kelly_bet()."""

    def test_basic_kelly_returns_positive(self, executor):
        """52% probability, 0.45 token cost → positive bet."""
        bet = executor._compute_kelly_bet(0.52, 0.45)
        assert bet > 0, "Should return positive bet for edge case"

    def test_no_edge_returns_zero(self, executor):
        """50% probability → no edge → 0 bet."""
        bet = executor._compute_kelly_bet(0.50, 0.50)
        assert bet == 0.0, "50/50 with 50c token = no edge"

    def test_floor_at_min_bet(self, executor):
        """Tiny edge should still return at least MIN_BET if any edge exists."""
        # 51% prob, 0.49 token cost → very small edge
        bet = executor._compute_kelly_bet(0.51, 0.49)
        if bet > 0:
            assert bet >= executor.MIN_BET, f"Should be >= MIN_BET ({executor.MIN_BET})"

    def test_ceiling_at_max_bet(self, executor):
        """High probability should be capped at MAX_BET_USD."""
        bet = executor._compute_kelly_bet(0.99, 0.10)
        assert bet <= executor.MAX_BET_USD, (
            f"Bet ${bet} exceeds MAX_BET_USD ${executor.MAX_BET_USD}"
        )

    def test_ceiling_respects_per_instrument_cap(self, executor):
        """Per-instrument max_bet_usd should further limit sizing."""
        bet = executor._compute_kelly_bet(0.99, 0.10, max_bet_usd=10.0)
        assert bet <= 10.0, "Per-instrument cap should be respected"

    def test_half_kelly_smaller_than_full(self, executor):
        """Half-Kelly should produce smaller bet than full Kelly."""
        half = executor._compute_kelly_bet(0.60, 0.40, kelly_fraction=0.5)
        full = executor._compute_kelly_bet(0.60, 0.40, kelly_fraction=1.0)
        # Both capped at same ceiling, but if not hitting ceiling, half < full
        assert half <= full, "Half-Kelly should be <= full Kelly"

    def test_probability_clamped_to_50_min(self, executor):
        """Probability below 0.5 should be clamped (no negative Kelly)."""
        bet = executor._compute_kelly_bet(0.40, 0.50)
        assert bet == 0.0, "Below 50% probability should yield 0 bet"

    def test_bankroll_sizing_cap(self, executor):
        """Effective bankroll capped at MAX_SIZING_BANKROLL."""
        executor.bankroll = 5000.0
        bet = executor._compute_kelly_bet(0.60, 0.30)
        # Ceiling = min(1000 * 0.05, 20) = min(50, 20) = 20
        assert bet <= executor.MAX_BET_USD

    def test_kelly_formula_correctness(self, executor):
        """Verify Kelly formula: f = (p*b - q) / b."""
        # p=0.60, token_cost=0.40 → b = 0.60/0.40 = 1.5
        # Full Kelly = (0.60*1.5 - 0.40) / 1.5 = (0.90-0.40)/1.5 = 0.333
        # Half-Kelly = 0.1667
        # bet = min(bankroll, 1000) * 0.1667 = 500 * 0.1667 = 83.33
        # ceiling = min(500*0.05, 20) = min(25, 20) = 20
        # final = max(5, min(83.33, 20)) = 20
        executor.bankroll = 500.0
        bet = executor._compute_kelly_bet(0.60, 0.40, kelly_fraction=0.5)
        assert bet == 20.0, f"Expected $20.00 (ceiling), got ${bet}"


# ═══════════════════════════════════════════════════════════════
# WINDOW START PRICE FOR RESOLUTION
# ═══════════════════════════════════════════════════════════════

class TestWindowStartPriceResolution:
    """Verify window_start_price (not entry_asset_price) is used for resolution."""

    def test_close_bet_uses_window_start_price(self, executor, db_path):
        """When closing a bet, resolution should compare exit price to window_start_price."""
        # Insert an open bet where window_start_price != entry_asset_price
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, total_invested, total_tokens, avg_cost, status,
             regime, entry_asset_price, window_start_price, question, timeframe, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, datetime('now'))
        """, (
            "btc-updown-15m-1700000000", "BTC", "YES", 0.45, 10.0, 22.22,
            52.0, 10.0, 22.22, 0.45, "trending",
            95000.0,  # entry_asset_price (mid-window)
            94500.0,  # window_start_price (t=0)
            "Will BTC go up?", 15,
        ))
        conn.commit()

        # Fetch the bet
        conn.row_factory = sqlite3.Row
        bet = dict(conn.execute(
            "SELECT * FROM polymarket_bets WHERE status='OPEN'"
        ).fetchone())
        conn.close()

        # Mock market fetch and lifecycle update
        with patch.object(executor, "_fetch_market_by_slug", return_value=None), \
             patch.object(executor, "_update_lifecycle_resolution") as mock_lifecycle, \
             patch.object(executor, "_get_exit_asset_price", return_value=94800.0), \
             patch.object(executor, "_log_bankroll"):
            executor._close_bet(bet, "low_confidence", {"BTC-USD": 94800.0})

        # exit_asset_price=94800 vs window_start_price=94500 → UP
        # If it used entry_asset_price=95000, it would be DOWN
        mock_lifecycle.assert_called_once()
        call_kwargs = mock_lifecycle.call_args
        assert call_kwargs[1]["final_result"] == "UP", (
            "Should resolve as UP using window_start_price=94500, not entry_asset_price=95000"
        )

    def test_close_bet_falls_back_to_entry_price(self, executor, db_path):
        """When window_start_price is None, falls back to entry_asset_price."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, total_invested, total_tokens, avg_cost, status,
             regime, entry_asset_price, window_start_price, question, timeframe, opened_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, ?, datetime('now'))
        """, (
            "btc-updown-15m-1700000000", "BTC", "YES", 0.45, 10.0, 22.22,
            52.0, 10.0, 22.22, 0.45, "trending",
            95000.0,  # entry_asset_price
            None,     # window_start_price missing
            "Will BTC go up?", 15,
        ))
        conn.commit()

        conn.row_factory = sqlite3.Row
        bet = dict(conn.execute(
            "SELECT * FROM polymarket_bets WHERE status='OPEN'"
        ).fetchone())
        conn.close()

        with patch.object(executor, "_fetch_market_by_slug", return_value=None), \
             patch.object(executor, "_update_lifecycle_resolution") as mock_lifecycle, \
             patch.object(executor, "_get_exit_asset_price", return_value=95100.0), \
             patch.object(executor, "_log_bankroll"):
            executor._close_bet(bet, "direction_flip", {"BTC-USD": 95100.0})

        # exit=95100 vs entry=95000 → UP (fallback)
        call_kwargs = mock_lifecycle.call_args
        assert call_kwargs[1]["final_result"] == "UP"


# ═══════════════════════════════════════════════════════════════
# BANKROLL RECONCILIATION
# ═══════════════════════════════════════════════════════════════

class TestBankrollReconciliation:

    def test_load_bankroll_from_pnl(self, db_path):
        """Bankroll = initial + resolved_pnl - open_exposure."""
        from polymarket_strategy_a import StrategyAExecutor

        # First: create an executor to set up the DB tables
        with patch("polymarket_strategy_a.TimingFeatureEngine") as mt:
            mt.return_value.is_follower.return_value = False
            StrategyAExecutor(config=_make_config(500.0), db_path=db_path,
                              logger=logging.getLogger("test"))

        # Now insert test data into the tables
        conn = sqlite3.connect(db_path)
        for pnl_val in [5.0, -2.0, 3.0]:  # net = +6.0
            conn.execute("""
                INSERT INTO polymarket_bets
                (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
                 entry_confidence, total_invested, total_tokens, avg_cost,
                 status, pnl, opened_at)
                VALUES ('s', 'BTC', 'YES', 0.5, 10.0, 20.0, 52.0, 10.0, 20.0, 0.5,
                        'WON', ?, datetime('now'))
            """, (pnl_val,))
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, total_invested, total_tokens, avg_cost,
             status, opened_at)
            VALUES ('open-s', 'ETH', 'NO', 0.4, 15.0, 37.5, 52.0, 15.0, 37.5, 0.4,
                    'OPEN', datetime('now'))
        """)
        conn.commit()
        conn.close()

        # Create fresh executor — it will load bankroll from the seeded data
        with patch("polymarket_strategy_a.TimingFeatureEngine") as mt:
            mt.return_value.is_follower.return_value = False
            exe = StrategyAExecutor(config=_make_config(500.0), db_path=db_path,
                                    logger=logging.getLogger("test"))

        # Expected: min(500 + 6.0 - 15.0, 1000) = min(491.0, 1000) = 491.0
        assert abs(exe.bankroll - 491.0) < 0.01, f"Expected ~491.0, got {exe.bankroll}"

    def test_reconcile_auto_corrects_large_discrepancy(self, executor, db_path):
        """Discrepancy > $5 triggers auto-correction."""
        # Manually set bankroll to a wrong value
        executor.bankroll = 600.0  # No bets in DB → expected = min(500, 1000) = 500.0

        executor._reconcile_bankroll()

        assert abs(executor.bankroll - 500.0) < 0.01, (
            f"Should auto-correct to 500.0, got {executor.bankroll}"
        )

    def test_reconcile_no_correction_for_small_discrepancy(self, executor, db_path):
        """Discrepancy < $1 is ignored."""
        executor.bankroll = 500.5  # Only 0.5 off

        executor._reconcile_bankroll()

        assert abs(executor.bankroll - 500.5) < 0.01, (
            "Small discrepancy should not trigger correction"
        )

    def test_bankroll_capped_at_max_sizing(self, db_path):
        """Bankroll should be capped at MAX_SIZING_BANKROLL."""
        from polymarket_strategy_a import StrategyAExecutor

        # First: create an executor to set up the DB tables
        with patch("polymarket_strategy_a.TimingFeatureEngine") as mt:
            mt.return_value.is_follower.return_value = False
            StrategyAExecutor(config=_make_config(500.0), db_path=db_path,
                              logger=logging.getLogger("test"))

        # Insert lots of wins
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, total_invested, total_tokens, avg_cost,
             status, pnl, opened_at)
            VALUES ('s', 'BTC', 'YES', 0.5, 10.0, 20.0, 52.0, 10.0, 20.0, 0.5,
                    'WON', 800.0, datetime('now'))
        """)
        conn.commit()
        conn.close()

        # Create fresh executor — should load with capped bankroll
        with patch("polymarket_strategy_a.TimingFeatureEngine") as mt:
            mt.return_value.is_follower.return_value = False
            exe = StrategyAExecutor(config=_make_config(500.0), db_path=db_path,
                                    logger=logging.getLogger("test"))

        # raw = 500 + 800 = 1300, capped at 1000
        assert exe.bankroll == 1000.0, f"Expected 1000.0 (capped), got {exe.bankroll}"


# ═══════════════════════════════════════════════════════════════
# DAILY LOSS LIMITS
# ═══════════════════════════════════════════════════════════════

class TestDailyLossLimits:

    def test_global_daily_loss_blocks_trading(self, executor):
        """When daily PnL exceeds 10% loss, trading is blocked."""
        executor._daily_start_bankroll = 500.0
        executor._daily_pnl = -51.0  # 10.2% loss > 10% limit
        executor._last_trading_day = datetime.now(timezone.utc).date()

        assert executor._check_daily_loss_limit(timeframe=15) is False

    def test_global_daily_loss_allows_within_limit(self, executor):
        """When daily PnL is within limit, trading allowed."""
        executor._daily_start_bankroll = 500.0
        executor._daily_pnl = -40.0  # 8% loss < 10% limit
        executor._last_trading_day = datetime.now(timezone.utc).date()

        assert executor._check_daily_loss_limit(timeframe=15) is True

    def test_5m_specific_loss_limit(self, executor):
        """5m markets have a tighter 5% daily loss limit."""
        executor._daily_start_bankroll = 500.0
        executor._daily_pnl = -20.0       # Global: 4% < 10% → OK
        executor._daily_pnl_5m = -26.0    # 5m: 5.2% > 5% → BLOCKED
        executor._last_trading_day = datetime.now(timezone.utc).date()

        assert executor._check_daily_loss_limit(timeframe=5) is False

    def test_5m_allowed_if_within_both_limits(self, executor):
        """5m trade allowed when both global and 5m limits are OK."""
        executor._daily_start_bankroll = 500.0
        executor._daily_pnl = -10.0
        executor._daily_pnl_5m = -10.0
        executor._last_trading_day = datetime.now(timezone.utc).date()

        assert executor._check_daily_loss_limit(timeframe=5) is True

    def test_15m_ignores_5m_limit(self, executor):
        """15m trades should not be blocked by the 5m-specific cap."""
        executor._daily_start_bankroll = 500.0
        executor._daily_pnl = -10.0       # Global: 2% → OK
        executor._daily_pnl_5m = -30.0    # 5m: 6% > 5% — but irrelevant for 15m
        executor._last_trading_day = datetime.now(timezone.utc).date()

        assert executor._check_daily_loss_limit(timeframe=15) is True

    def test_daily_reset_on_new_day(self, executor):
        """Daily counters should reset when a new UTC day starts."""
        import datetime as dt
        yesterday = (datetime.now(timezone.utc) - dt.timedelta(days=1)).date()
        executor._last_trading_day = yesterday
        executor._daily_pnl = -100.0
        executor._daily_pnl_5m = -100.0

        executor._reset_daily_if_needed()

        assert executor._daily_pnl == 0.0
        assert executor._daily_pnl_5m == 0.0
        assert executor._last_trading_day == datetime.now(timezone.utc).date()


# ═══════════════════════════════════════════════════════════════
# MAX POSITIONS CAP
# ═══════════════════════════════════════════════════════════════

class TestMaxPositionsCap:

    def test_max_open_bets_constant(self, executor):
        """MAX_OPEN_BETS should be 8."""
        assert executor.MAX_OPEN_BETS == 8

    def test_max_position_per_market(self, executor):
        """MAX_POSITION_PER_MARKET should be $150."""
        assert executor.MAX_POSITION_PER_MARKET == 150.0


# ═══════════════════════════════════════════════════════════════
# CONFIDENCE THRESHOLDS
# ═══════════════════════════════════════════════════════════════

class TestConfidenceThresholds:

    def test_entry_threshold(self, executor):
        assert executor.CONFIDENCE_THRESHOLD == 52.0

    def test_confidence_cap(self, executor):
        assert executor.CONFIDENCE_CAP == 52.5

    def test_exit_confidence(self, executor):
        assert executor.EXIT_CONFIDENCE == 50.0

    def test_add_confidence(self, executor):
        assert executor.ADD_CONFIDENCE == 52.0


# ═══════════════════════════════════════════════════════════════
# RATE LIMITING & COOLDOWN
# ═══════════════════════════════════════════════════════════════

class TestRateLimiting:

    def test_under_rate_limit_allows(self, executor):
        """No bets placed → should be under limit."""
        assert executor._check_rate_limit() is True

    def test_at_rate_limit_blocks(self, executor):
        """At MAX_BETS_PER_HOUR, should block."""
        now = time.time()
        executor._bets_this_hour = [now] * executor.MAX_BETS_PER_HOUR
        assert executor._check_rate_limit() is False

    def test_expired_bets_pruned(self, executor):
        """Bets older than 1 hour should be pruned."""
        old_time = time.time() - 3700
        executor._bets_this_hour = [old_time] * 20
        assert executor._check_rate_limit() is True


class TestCooldown:

    def test_no_loss_allows_immediately(self, executor):
        """With no prior loss, cooldown passes."""
        executor._last_loss_time = 0.0
        assert executor._check_cooldown() is True

    def test_recent_loss_blocks(self, executor):
        """Loss within COOLDOWN_AFTER_LOSS blocks."""
        executor._last_loss_time = time.time() - 10  # 10s ago
        # COOLDOWN_AFTER_LOSS = 120s
        assert executor._check_cooldown() is False

    def test_old_loss_allows(self, executor):
        """Loss older than COOLDOWN_AFTER_LOSS allows."""
        executor._last_loss_time = time.time() - 200  # 200s ago > 120s
        assert executor._check_cooldown() is True


# ═══════════════════════════════════════════════════════════════
# ASSET CONCENTRATION
# ═══════════════════════════════════════════════════════════════

class TestAssetConcentration:

    def test_fewer_than_3_assets_always_passes(self, executor, db_path):
        """With < 3 distinct assets, concentration check is skipped."""
        conn = sqlite3.connect(db_path)
        conn.execute("""
            INSERT INTO polymarket_bets
            (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
             entry_confidence, total_invested, total_tokens, avg_cost,
             status, opened_at)
            VALUES ('s1', 'BTC', 'YES', 0.5, 100.0, 200.0, 52.0, 100.0, 200.0, 0.5,
                    'OPEN', datetime('now'))
        """)
        conn.commit()
        conn.close()

        # Only 1 asset open → should pass even though 100% concentrated
        assert executor._check_asset_concentration("BTC", 50.0) is True

    def test_blocks_over_50_pct_concentration(self, executor, db_path):
        """With 3+ assets, > 50% concentration is blocked."""
        conn = sqlite3.connect(db_path)
        # 3 assets: BTC=$100, ETH=$50, SOL=$50
        for asset, amt in [("BTC", 100.0), ("ETH", 50.0), ("SOL", 50.0)]:
            conn.execute("""
                INSERT INTO polymarket_bets
                (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
                 entry_confidence, total_invested, total_tokens, avg_cost,
                 status, opened_at)
                VALUES (?, ?, 'YES', 0.5, ?, ?, 52.0, ?, ?, 0.5, 'OPEN', datetime('now'))
            """, (f"s-{asset}", asset, amt, amt*2, amt, amt*2))
        conn.commit()
        conn.close()

        # BTC has $100 already. Adding $60 → BTC=$160 / total=$260 = 61.5% > 50%
        assert executor._check_asset_concentration("BTC", 60.0) is False

    def test_allows_under_50_pct_concentration(self, executor, db_path):
        """With 3+ assets, under 50% is allowed."""
        conn = sqlite3.connect(db_path)
        for asset, amt in [("BTC", 50.0), ("ETH", 50.0), ("SOL", 50.0)]:
            conn.execute("""
                INSERT INTO polymarket_bets
                (slug, asset, entry_side, entry_token_cost, entry_amount, entry_tokens,
                 entry_confidence, total_invested, total_tokens, avg_cost,
                 status, opened_at)
                VALUES (?, ?, 'YES', 0.5, ?, ?, 52.0, ?, ?, 0.5, 'OPEN', datetime('now'))
            """, (f"s-{asset}", asset, amt, amt*2, amt, amt*2))
        conn.commit()
        conn.close()

        # BTC has $50. Adding $10 → BTC=$60 / total=$160 = 37.5% < 50%
        assert executor._check_asset_concentration("BTC", 10.0) is True


# ═══════════════════════════════════════════════════════════════
# MINIMUM BANKROLL GATE
# ═══════════════════════════════════════════════════════════════

class TestMinBankroll:

    def test_above_min_allows(self, executor):
        executor.bankroll = 100.0
        assert executor._check_min_bankroll() is True

    def test_below_min_blocks(self, executor):
        executor.bankroll = 30.0  # MIN_BANKROLL_TO_TRADE = 50
        assert executor._check_min_bankroll() is False

    def test_at_min_allows(self, executor):
        executor.bankroll = 50.0
        assert executor._check_min_bankroll() is True


# ═══════════════════════════════════════════════════════════════
# PARSE CROWD UP
# ═══════════════════════════════════════════════════════════════

class TestParseCrowdUp:

    def test_parse_from_outcome_prices_json(self, executor):
        """outcomePrices is a JSON array string."""
        market = {"outcomePrices": '["0.55", "0.45"]'}
        result = executor._parse_crowd_up(market)
        assert abs(result - 0.55) < 0.01

    def test_parse_fallback_to_best_ask(self, executor):
        """Falls back to bestAsk if outcomePrices is missing."""
        market = {"bestAsk": "0.60"}
        result = executor._parse_crowd_up(market)
        assert abs(result - 0.60) < 0.01

    def test_parse_returns_default_on_garbage(self, executor):
        """Returns 0.50 on unparseable data."""
        market = {"outcomePrices": "garbage"}
        result = executor._parse_crowd_up(market)
        assert abs(result - 0.50) < 0.01
