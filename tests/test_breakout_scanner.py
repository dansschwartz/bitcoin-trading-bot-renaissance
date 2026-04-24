"""
Tests for breakout_scanner.py — composite breakout score calculation.

Exercises the scoring engine (_score_ticker, volume/price/momentum/volatility/
divergence scoring) without making real API calls.
"""

import sys
from collections import deque
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from strategies.breakout_scanner import BreakoutScanner, BreakoutSignal, ALWAYS_SCAN, EXCLUDED_BASES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ticker(
    symbol: str = "TESTUSDT",
    last_price: float = 100.0,
    price_change_pct: float = 5.0,
    high_price: float = 110.0,
    low_price: float = 90.0,
    quote_volume: float = 10_000_000,
) -> dict:
    """Build a minimal Binance 24hr ticker dict."""
    return {
        "symbol": symbol,
        "lastPrice": str(last_price),
        "priceChangePercent": str(price_change_pct),
        "highPrice": str(high_price),
        "lowPrice": str(low_price),
        "quoteVolume": str(quote_volume),
    }


# ---------------------------------------------------------------------------
# Tests: Composite Score
# ---------------------------------------------------------------------------

class TestScoreTicker:
    """Test the _score_ticker method that computes the composite breakout score."""

    def test_basic_score_calculation(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=10.0, quote_volume=50_000_000)
        signal = scanner._score_ticker(ticker)

        assert signal is not None
        assert isinstance(signal, BreakoutSignal)
        assert signal.symbol == "TESTUSDT"
        assert signal.product_id == "TEST-USD"
        assert signal.breakout_score > 0

    def test_score_components_are_non_negative(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=15.0)
        signal = scanner._score_ticker(ticker)

        assert signal.volume_score >= 0
        assert signal.price_score >= 0
        assert signal.momentum_score >= 0
        assert signal.volatility_score >= 0
        assert signal.divergence_score >= 0

    def test_composite_equals_sum_of_components(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=8.0)
        signal = scanner._score_ticker(ticker)

        expected = (signal.volume_score + signal.price_score +
                    signal.momentum_score + signal.volatility_score +
                    signal.divergence_score)
        assert signal.breakout_score == pytest.approx(expected)

    def test_zero_price_returns_none(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(last_price=0.0)
        signal = scanner._score_ticker(ticker)
        assert signal is None

    def test_bullish_direction_on_positive_change(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=5.0)
        signal = scanner._score_ticker(ticker)
        assert signal.direction == "bullish"

    def test_bearish_direction_on_negative_change(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=-5.0)
        signal = scanner._score_ticker(ticker)
        assert signal.direction == "bearish"

    def test_details_contains_benchmarks(self):
        scanner = BreakoutScanner()
        scanner._btc_change = 2.0
        scanner._eth_change = 3.0
        ticker = _make_ticker()
        signal = scanner._score_ticker(ticker)

        assert signal.details["btc_change"] == 2.0
        assert signal.details["eth_change"] == 3.0


# ---------------------------------------------------------------------------
# Tests: Momentum Score
# ---------------------------------------------------------------------------

class TestMomentumScore:
    """Verify momentum score tiers based on |price_change_pct|."""

    @pytest.mark.parametrize("change,expected_min,expected_max", [
        (25.0, 25.0, 25.0),  # >20%
        (12.0, 20.0, 20.0),  # >10%
        (7.0, 15.0, 15.0),   # >5%
        (4.0, 10.0, 10.0),   # >3%
        (2.0, 5.0, 5.0),     # >1.5%
        (0.5, 0.0, 5.0),     # <1.5%
    ])
    def test_momentum_tiers(self, change, expected_min, expected_max):
        scanner = BreakoutScanner()
        ticker = _make_ticker(price_change_pct=change)
        signal = scanner._score_ticker(ticker)
        assert expected_min <= signal.momentum_score <= expected_max


# ---------------------------------------------------------------------------
# Tests: Volume Score
# ---------------------------------------------------------------------------

class TestVolumeScore:
    """Verify volume spike scoring relative to historical average."""

    def test_no_history_high_volume_scores_15(self):
        """Without history, volume > 100M should score 15."""
        scanner = BreakoutScanner()
        score = scanner._calc_volume_score("NEWUSDT", 150_000_000)
        assert score == 15.0

    def test_no_history_medium_volume_scores_10(self):
        scanner = BreakoutScanner()
        score = scanner._calc_volume_score("NEWUSDT", 60_000_000)
        assert score == 10.0

    def test_no_history_low_volume_scores_5(self):
        scanner = BreakoutScanner()
        score = scanner._calc_volume_score("NEWUSDT", 20_000_000)
        assert score == 5.0

    def test_no_history_tiny_volume_scores_0(self):
        scanner = BreakoutScanner()
        score = scanner._calc_volume_score("NEWUSDT", 500_000)
        assert score == 0.0

    def test_5x_volume_spike_scores_30(self):
        """Volume spike > 5x average should score max 30."""
        scanner = BreakoutScanner()
        scanner._history["SPIKEUSDT"] = deque([
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
        ])
        score = scanner._calc_volume_score("SPIKEUSDT", 6_000_000)
        assert score == 30.0

    def test_3x_volume_spike_scores_22(self):
        scanner = BreakoutScanner()
        scanner._history["MIDUSDT"] = deque([
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
        ])
        score = scanner._calc_volume_score("MIDUSDT", 3_500_000)
        assert score == 22.0

    def test_normal_volume_scores_0(self):
        scanner = BreakoutScanner()
        scanner._history["FLATUSDT"] = deque([
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
            {"volume": 1_000_000, "price": 10, "high": 11, "low": 9, "range_pct": 20},
        ])
        score = scanner._calc_volume_score("FLATUSDT", 1_000_000)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Tests: Price Breakout Score
# ---------------------------------------------------------------------------

class TestPriceBreakoutScore:
    """Verify price breakout scoring based on proximity to 24h extremes."""

    def test_at_24h_high_scores_max(self):
        """Price exactly at 24h high should score 25."""
        scanner = BreakoutScanner()
        ticker = _make_ticker(last_price=110.0, high_price=110.0, low_price=90.0)
        signal = scanner._score_ticker(ticker)
        assert signal.price_score == pytest.approx(25.0)

    def test_at_24h_low_scores_max(self):
        """Price exactly at 24h low should score 25."""
        scanner = BreakoutScanner()
        ticker = _make_ticker(last_price=90.0, high_price=110.0, low_price=90.0)
        signal = scanner._score_ticker(ticker)
        assert signal.price_score == pytest.approx(25.0)

    def test_midrange_scores_zero(self):
        """Price in the middle of the range should score 0."""
        scanner = BreakoutScanner()
        ticker = _make_ticker(last_price=100.0, high_price=110.0, low_price=90.0)
        signal = scanner._score_ticker(ticker)
        assert signal.price_score == 0.0


# ---------------------------------------------------------------------------
# Tests: Volatility Score
# ---------------------------------------------------------------------------

class TestVolatilityScore:
    def test_no_history_high_range_scores_10(self):
        scanner = BreakoutScanner()
        # current_range_pct = (110-90)/100 * 100 = 20%
        score = scanner._calc_volatility_score("NEWUSDT", 110, 90, 100)
        assert score == 10.0

    def test_no_history_low_range_scores_0(self):
        scanner = BreakoutScanner()
        # current_range_pct = (101-99)/100 * 100 = 2%
        score = scanner._calc_volatility_score("NEWUSDT", 101, 99, 100)
        assert score == 0.0

    def test_invalid_price_scores_0(self):
        scanner = BreakoutScanner()
        assert scanner._calc_volatility_score("X", 100, 90, 0) == 0.0
        assert scanner._calc_volatility_score("X", 90, 100, 100) == 0.0


# ---------------------------------------------------------------------------
# Tests: Divergence Score
# ---------------------------------------------------------------------------

class TestDivergenceScore:
    def test_high_divergence_scores_10(self):
        """Pair moving 5x+ more than BTC should get max divergence score."""
        scanner = BreakoutScanner()
        scanner._btc_change = 1.0  # BTC moved 1%
        # Pair moved 15% -> divergence ratio = 15 / 1.1 ≈ 13.6
        ticker = _make_ticker(price_change_pct=15.0)
        signal = scanner._score_ticker(ticker)
        assert signal.divergence_score == 10.0

    def test_no_divergence_scores_0(self):
        """Pair moving same as BTC should get 0 divergence."""
        scanner = BreakoutScanner()
        scanner._btc_change = 5.0
        ticker = _make_ticker(price_change_pct=5.0)
        signal = scanner._score_ticker(ticker)
        # ratio = 5 / 5.1 ≈ 0.98 -> 0 points
        assert signal.divergence_score == 0.0


# ---------------------------------------------------------------------------
# Tests: History tracking
# ---------------------------------------------------------------------------

class TestHistoryTracking:
    def test_update_history_stores_data(self):
        scanner = BreakoutScanner()
        ticker = _make_ticker(symbol="AAAUSDT", last_price=50.0,
                              high_price=55.0, low_price=45.0,
                              quote_volume=5_000_000)
        scanner._update_history(ticker)

        assert "AAAUSDT" in scanner._history
        assert len(scanner._history["AAAUSDT"]) == 1
        entry = scanner._history["AAAUSDT"][0]
        assert entry["price"] == 50.0
        assert entry["volume"] == 5_000_000
        assert entry["range_pct"] == pytest.approx(20.0)

    def test_history_respects_maxlen(self):
        scanner = BreakoutScanner(history_size=3)
        for i in range(5):
            ticker = _make_ticker(symbol="LIMUSDT", last_price=10.0 + i,
                                  quote_volume=1_000_000 * (i + 1))
            scanner._update_history(ticker)

        assert len(scanner._history["LIMUSDT"]) == 3


# ---------------------------------------------------------------------------
# Tests: Utility methods
# ---------------------------------------------------------------------------

class TestUtilityMethods:
    def test_get_stats_initial(self):
        scanner = BreakoutScanner()
        stats = scanner.get_stats()
        assert stats["total_scans"] == 0
        assert stats["total_flagged"] == 0
        assert stats["pairs_tracked"] == 0

    def test_get_always_scan_pairs_format(self):
        scanner = BreakoutScanner()
        pairs = scanner.get_always_scan_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) == len(ALWAYS_SCAN)
        # All should be in "XXX-USD" format
        for p in pairs:
            assert p.endswith("-USD"), f"{p} doesn't end with -USD"
            assert "USDT" not in p

    def test_excluded_bases_contains_stablecoins(self):
        """Sanity check that known stablecoins are excluded."""
        assert "USDT" in EXCLUDED_BASES
        assert "USDC" in EXCLUDED_BASES
        assert "DAI" in EXCLUDED_BASES
