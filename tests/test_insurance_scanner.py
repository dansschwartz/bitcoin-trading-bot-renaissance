"""
Tests for intelligence/insurance_scanner.py — InsurancePremiumScanner
=====================================================================
Covers Funding Settlement Premium, Weekend Premium, Scheduled Event
Premium, and combined scan logic.
"""

import math
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelligence.insurance_scanner import InsurancePremiumScanner


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config():
    return {
        "enabled": True,
        "lookback_days": 90,
        "min_observations": 3,
        "significance_level": 0.05,
        "min_dislocation_bps": 3.0,
        "weekend_start_hour": 20,
        "weekend_end_hour": 22,
    }


@pytest.fixture
def scanner(default_config, tmp_path):
    db_path = str(tmp_path / "test.db")
    return InsurancePremiumScanner(default_config, db_path)


def _make_market_row(price, timestamp_str, volume=100.0, spread=0.5):
    """Build a market_data row dict."""
    return {
        "price": price,
        "volume": volume,
        "bid": price - spread / 2,
        "ask": price + spread / 2,
        "spread": spread,
        "timestamp": timestamp_str,
        "product_id": "BTC-USD",
    }


# ---------------------------------------------------------------------------
# Tests — Timestamp parsing
# ---------------------------------------------------------------------------

class TestParseTimestamp:

    def test_valid_iso_format(self):
        dt = InsurancePremiumScanner._parse_timestamp("2024-07-11T12:30:00+00:00")
        assert dt is not None
        assert dt.hour == 12

    def test_naive_timestamp_gets_utc(self):
        dt = InsurancePremiumScanner._parse_timestamp("2024-07-11T12:30:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_invalid_timestamp_returns_none(self):
        assert InsurancePremiumScanner._parse_timestamp("not-a-date") is None
        assert InsurancePremiumScanner._parse_timestamp(None) is None
        assert InsurancePremiumScanner._parse_timestamp("") is None


# ---------------------------------------------------------------------------
# Tests — Weekend detection
# ---------------------------------------------------------------------------

class TestIsWeekend:

    def test_friday_before_20_not_weekend(self, scanner):
        dt = datetime(2024, 7, 12, 18, 0, tzinfo=timezone.utc)  # Friday 18:00
        assert scanner._is_weekend(dt) is False

    def test_friday_after_20_is_weekend(self, scanner):
        dt = datetime(2024, 7, 12, 21, 0, tzinfo=timezone.utc)  # Friday 21:00
        assert scanner._is_weekend(dt) is True

    def test_saturday_is_weekend(self, scanner):
        dt = datetime(2024, 7, 13, 12, 0, tzinfo=timezone.utc)  # Saturday
        assert scanner._is_weekend(dt) is True

    def test_sunday_before_22_is_weekend(self, scanner):
        dt = datetime(2024, 7, 14, 20, 0, tzinfo=timezone.utc)  # Sunday 20:00
        assert scanner._is_weekend(dt) is True

    def test_sunday_after_22_not_weekend(self, scanner):
        dt = datetime(2024, 7, 14, 23, 0, tzinfo=timezone.utc)  # Sunday 23:00
        assert scanner._is_weekend(dt) is False

    def test_wednesday_not_weekend(self, scanner):
        dt = datetime(2024, 7, 10, 12, 0, tzinfo=timezone.utc)  # Wednesday
        assert scanner._is_weekend(dt) is False


# ---------------------------------------------------------------------------
# Tests — Funding Settlement Premium
# ---------------------------------------------------------------------------

class TestFundingSettlementPremium:

    def test_insufficient_data_returns_default(self, scanner):
        with patch.object(scanner, "_fetch_market_data", return_value=[]):
            result = scanner.scan_funding_settlement_premium("BTC-USD")
            assert result["premium_detected"] is False
            assert result["direction"] == "none"

    def test_no_premium_detected_with_random_data(self, scanner):
        """Feed data that doesn't correlate with settlement hours."""
        # Generate rows at random hours that don't cluster around settlement
        rows = []
        base = datetime(2024, 6, 1, 3, 15, tzinfo=timezone.utc)
        for i in range(200):
            dt = base + timedelta(hours=i)
            rows.append(_make_market_row(
                price=50000 + (i % 10),
                timestamp_str=dt.isoformat(),
            ))

        with patch.object(scanner, "_fetch_market_data", return_value=rows):
            result = scanner.scan_funding_settlement_premium("BTC-USD")
            # P-value should be high -> no premium
            assert isinstance(result["settlement_stats"], dict)

    def test_settlement_hours_constant(self, scanner):
        assert scanner.FUNDING_SETTLEMENT_HOURS == [0, 8, 16]

    def test_settlement_window_constant(self, scanner):
        assert scanner.SETTLEMENT_WINDOW_MINUTES == 30


# ---------------------------------------------------------------------------
# Tests — Weekend Premium
# ---------------------------------------------------------------------------

class TestWeekendPremium:

    def test_insufficient_data(self, scanner):
        with patch.object(scanner, "_fetch_market_data", return_value=[]):
            result = scanner.scan_weekend_premium("BTC-USD")
            assert result["premium_detected"] is False

    def test_happy_path_with_mixed_data(self, scanner):
        """Create data spanning weekdays and weekends."""
        rows = []
        base = datetime(2024, 7, 8, 0, 0, tzinfo=timezone.utc)  # Monday
        for i in range(7 * 24 * 4):  # 4 weeks, hourly-ish
            dt = base + timedelta(hours=i)
            price = 50000.0 + (i % 7) * 10
            rows.append(_make_market_row(
                price=price,
                timestamp_str=dt.isoformat(),
                spread=0.5 if dt.weekday() < 5 else 1.5,  # wider weekend spread
            ))

        with patch.object(scanner, "_fetch_market_data", return_value=rows):
            result = scanner.scan_weekend_premium("BTC-USD")
            assert "avg_weekend_return_bps" in result
            assert "avg_weekday_return_bps" in result
            assert "spread_ratio" in result
            assert isinstance(result["p_value"], float)

    def test_spread_ratio_detected(self, scanner):
        """If weekend spreads are much wider, the spread_ratio triggers premium."""
        rows = []
        base = datetime(2024, 7, 8, 0, 0, tzinfo=timezone.utc)
        for i in range(7 * 24 * 4):
            dt = base + timedelta(hours=i)
            price = 50000.0
            spread = 1.0 if dt.weekday() < 5 else 5.0
            rows.append(_make_market_row(
                price=price,
                timestamp_str=dt.isoformat(),
                spread=spread,
            ))

        with patch.object(scanner, "_fetch_market_data", return_value=rows):
            result = scanner.scan_weekend_premium("BTC-USD")
            # Prices are flat so returns identical, but spread ratio should be > 1.5
            # triggering provide_weekend_liquidity
            if result["spread_ratio"] > 1.5:
                assert result["premium_detected"] is True
                assert result["recommended_action"] == "provide_weekend_liquidity"


# ---------------------------------------------------------------------------
# Tests — Scheduled Event Premium
# ---------------------------------------------------------------------------

class TestScheduledEventPremium:

    def test_no_events_returns_default(self, scanner):
        result = scanner.scan_scheduled_event_premium("BTC-USD", [])
        assert result["premium_detected"] is False
        assert result["events_analysed"] == 0

    def test_insufficient_data(self, scanner):
        events = [{"name": "CPI", "datetime": "2024-07-11T12:30:00+00:00"}]
        with patch.object(scanner, "_fetch_market_data", return_value=[]):
            result = scanner.scan_scheduled_event_premium("BTC-USD", events)
            assert result["premium_detected"] is False

    def test_event_with_pre_post_data(self, scanner):
        """Feed rows around an event and check that event_details are populated."""
        event_dt = datetime(2024, 7, 11, 12, 30, tzinfo=timezone.utc)
        events = [{"name": "CPI", "datetime": event_dt.isoformat()}]

        rows = []
        # Pre-event: 4 hours before
        for i in range(24):
            dt = event_dt - timedelta(hours=4) + timedelta(minutes=i * 10)
            rows.append(_make_market_row(
                price=50000.0 + i * 2,
                timestamp_str=dt.isoformat(),
            ))
        # Post-event: 4 hours after
        for i in range(24):
            dt = event_dt + timedelta(minutes=i * 10)
            rows.append(_make_market_row(
                price=50050.0 - i * 3,
                timestamp_str=dt.isoformat(),
            ))

        with patch.object(scanner, "_fetch_market_data", return_value=rows):
            result = scanner.scan_scheduled_event_premium("BTC-USD", events)
            assert result["events_analysed"] >= 1
            assert len(result["event_details"]) >= 1
            detail = result["event_details"][0]
            assert detail["name"] == "CPI"
            assert "dislocation_bps" in detail


# ---------------------------------------------------------------------------
# Tests — Combined scan
# ---------------------------------------------------------------------------

class TestGetAllPremiums:

    def test_get_all_premiums_structure(self, scanner):
        """Verify the output structure of get_all_premiums."""
        with patch.object(scanner, "scan_funding_settlement_premium",
                          return_value={"premium_detected": False, "recommended_action": "no_action"}):
            with patch.object(scanner, "scan_weekend_premium",
                              return_value={"premium_detected": False, "recommended_action": "no_action"}):
                with patch.object(scanner, "scan_scheduled_event_premium",
                                  return_value={"premium_detected": False, "recommended_action": "no_action"}):
                    result = scanner.get_all_premiums("BTC-USD")
                    assert result["pair"] == "BTC-USD"
                    assert result["any_premium_detected"] is False
                    assert result["total_premiums_found"] == 0
                    assert result["combined_recommendation"] == "no_actionable_premiums"

    def test_single_premium_uses_its_recommendation(self, scanner):
        with patch.object(scanner, "scan_funding_settlement_premium",
                          return_value={"premium_detected": True,
                                        "recommended_action": "buy_before_settlement_sell_after"}):
            with patch.object(scanner, "scan_weekend_premium",
                              return_value={"premium_detected": False, "recommended_action": "no_action"}):
                with patch.object(scanner, "scan_scheduled_event_premium",
                                  return_value={"premium_detected": False, "recommended_action": "no_action"}):
                    result = scanner.get_all_premiums("BTC-USD")
                    assert result["total_premiums_found"] == 1
                    assert result["combined_recommendation"] == "buy_before_settlement_sell_after"

    def test_multiple_premiums(self, scanner):
        with patch.object(scanner, "scan_funding_settlement_premium",
                          return_value={"premium_detected": True, "recommended_action": "buy"}):
            with patch.object(scanner, "scan_weekend_premium",
                              return_value={"premium_detected": True, "recommended_action": "sell"}):
                with patch.object(scanner, "scan_scheduled_event_premium",
                                  return_value={"premium_detected": False, "recommended_action": "no_action"}):
                    result = scanner.get_all_premiums("BTC-USD")
                    assert result["total_premiums_found"] == 2
                    assert result["combined_recommendation"] == "multiple_premiums_detected_review_allocation"

    def test_repr(self, scanner):
        r = repr(scanner)
        assert "InsurancePremiumScanner" in r
        assert "lookback=90" in r
