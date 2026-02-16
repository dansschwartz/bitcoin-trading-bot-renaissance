"""
Statistical Predictor — Medium-Term (30s to 5m)
=================================================
Predicts price direction and magnitude over 30 seconds to 5 minutes
using statistical patterns: rolling VWAP deviation, volatility regime,
volume profiles, bar momentum, and return autocorrelation.

At these timescales microstructure data is mostly noise.  The mean
reversion signal is the star: if price deviated from rolling VWAP by
2 standard deviations, the probability of reversion over 1-5 minutes
is historically 55-60%.

Performance: < 2ms per position per horizon.
"""

from __future__ import annotations

import math
import time
from collections import deque

from core.data_structures import HorizonEstimate


class StatisticalPredictor:
    """
    Medium-term price predictor using statistical patterns.
    Called by MHPE for horizons: 30s, 2m (120s), 5m (300s).
    """

    def __init__(self, config: dict):
        self.config = config

        # Rolling statistics per pair
        self.vwap_data: dict = {}       # pair -> rolling VWAP components
        self.bar_momentum: dict = {}    # pair -> latest 5-min bar stats

    # ── FEED HANDLERS ──

    def on_trade(self, pair: str, price: float, size: float, timestamp: float) -> None:
        """Update rolling VWAP and volume statistics from trade stream."""
        if pair not in self.vwap_data:
            self.vwap_data[pair] = {
                "price_volume_sum": 0.0,
                "volume_sum": 0.0,
                "prices": deque(maxlen=self.config.get("vwap_window_trades", 3000)),
                "returns": deque(maxlen=self.config.get("vwap_window_trades", 3000)),
                "last_price": price,
            }

        data = self.vwap_data[pair]
        data["price_volume_sum"] += price * size
        data["volume_sum"] += size
        data["prices"].append({"price": price, "timestamp": timestamp, "size": size})

        if data["last_price"] > 0:
            ret = (price - data["last_price"]) / data["last_price"] * 10000
            data["returns"].append(ret)
        data["last_price"] = price

    def on_bar_close(self, pair: str, bar: dict) -> None:
        """Update bar-based momentum from 5-minute bar close."""
        open_ = bar.get("open", 0)
        self.bar_momentum[pair] = {
            "close": bar.get("close", 0),
            "open": open_,
            "high": bar.get("high", 0),
            "low": bar.get("low", 0),
            "volume": bar.get("volume", 0),
            "bar_return_bps": ((bar.get("close", 0) - open_) / open_ * 10000) if open_ else 0,
            "timestamp": bar.get("timestamp", time.time()),
        }

    # ── PREDICTION ──

    def predict(self, pair: str, side: str, horizon_seconds: int) -> HorizonEstimate:
        now = time.time()

        vwap_signal = self._compute_vwap_deviation(pair)
        vol_state = self._compute_volatility_state(pair)
        volume_signal = self._compute_volume_anomaly(pair)
        bar_signal = self._compute_bar_momentum(pair)
        autocorr = self._compute_autocorrelation(pair)

        weights = self._get_weights(horizon_seconds)

        raw_signal = (
            weights["vwap"] * vwap_signal
            + weights["vol"] * vol_state["direction_signal"]
            + weights["volume"] * volume_signal
            + weights["bar"] * bar_signal
            + weights["autocorr"] * autocorr
        )

        steepness = {30: 0.6, 120: 0.5, 300: 0.4}.get(horizon_seconds, 0.4)
        clamped = max(-3, min(3, raw_signal))
        p_up = 1.0 / (1.0 + math.exp(-steepness * clamped))
        lo, hi = self.config.get("probability_clamp", [0.42, 0.62])
        p_up = max(lo, min(hi, p_up))

        p_favorable = p_up if side == "long" else (1.0 - p_up)

        per_second_vol = vol_state["per_second_vol_bps"]
        base_vol_bps = per_second_vol * math.sqrt(horizon_seconds)

        if vol_state["regime"] == "expanding":
            base_vol_bps *= 1.3
        elif vol_state["regime"] == "contracting":
            base_vol_bps *= 0.8

        e_favorable = base_vol_bps * p_favorable * 0.8
        e_adverse = base_vol_bps * (1.0 - p_favorable) * 0.8
        e_net = p_favorable * e_favorable - (1.0 - p_favorable) * e_adverse

        p_adv_10 = self._tail_prob(base_vol_bps, 10, 1.0 - p_favorable)
        p_adv_25 = self._tail_prob(base_vol_bps, 25, 1.0 - p_favorable)
        p_adv_50 = self._tail_prob(base_vol_bps, 50, 1.0 - p_favorable)

        data = self.vwap_data.get(pair, {})
        returns_count = len(data.get("returns", []))
        data_quality = min(1.0, returns_count / 200) * (1.0 / (1.0 + 0.01 * horizon_seconds))

        dominant = max(weights, key=weights.get)

        return HorizonEstimate(
            horizon_seconds=horizon_seconds,
            horizon_label=f"{horizon_seconds}s" if horizon_seconds < 60 else f"{horizon_seconds // 60}m",
            p_profit=round(p_favorable, 4),
            p_loss=round(1.0 - p_favorable, 4),
            e_favorable_bps=round(e_favorable, 2),
            e_adverse_bps=round(e_adverse, 2),
            e_net_bps=round(e_net, 2),
            sigma_bps=round(base_vol_bps, 2),
            p_adverse_10bps=round(p_adv_10, 4),
            p_adverse_25bps=round(p_adv_25, 4),
            p_adverse_50bps=round(p_adv_50, 4),
            estimate_confidence=round(data_quality, 3),
            dominant_signal=dominant,
        )

    # ── FEATURE COMPUTATION ──

    def _compute_vwap_deviation(self, pair: str) -> float:
        """Z-score of current price from rolling VWAP, inverted for mean reversion."""
        data = self.vwap_data.get(pair)
        if not data or data["volume_sum"] == 0:
            return 0.0

        vwap = data["price_volume_sum"] / data["volume_sum"]
        current = data["last_price"]
        returns = list(data["returns"])

        if not returns or vwap == 0:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        stdev = math.sqrt(variance) if variance > 0 else 1.0

        deviation_bps = (current - vwap) / vwap * 10000
        z_score = deviation_bps / stdev if stdev > 0 else 0

        # Invert: below VWAP → positive (reversion upward expected)
        signal = -z_score / 3
        return max(-1.0, min(1.0, signal))

    def _compute_volatility_state(self, pair: str) -> dict:
        data = self.vwap_data.get(pair)
        if not data:
            return {"regime": "stable", "per_second_vol_bps": 2.0, "direction_signal": 0.0}

        returns = list(data["returns"])
        if len(returns) < 20:
            return {"regime": "stable", "per_second_vol_bps": 2.0, "direction_signal": 0.0}

        recent = returns[-30:]
        older = returns[-100:-30] if len(returns) >= 100 else returns[: len(returns) - 30]

        recent_var = sum(r ** 2 for r in recent) / len(recent)
        recent_vol = math.sqrt(recent_var) if recent_var > 0 else 1.0

        older_var = sum(r ** 2 for r in older) / len(older) if older else recent_var
        older_vol = math.sqrt(older_var) if older_var > 0 else 1.0

        ratio = recent_vol / older_vol if older_vol > 0 else 1.0

        if ratio > 1.5:
            return {"regime": "expanding", "per_second_vol_bps": recent_vol, "direction_signal": -0.3}
        elif ratio < 0.7:
            return {"regime": "contracting", "per_second_vol_bps": recent_vol, "direction_signal": 0.1}
        return {"regime": "stable", "per_second_vol_bps": recent_vol, "direction_signal": 0.0}

    def _compute_volume_anomaly(self, pair: str) -> float:
        data = self.vwap_data.get(pair)
        if not data:
            return 0.0
        prices = list(data["prices"])
        if len(prices) < 50:
            return 0.0
        recent_vol = sum(p["size"] for p in prices[-20:])
        avg_vol = sum(p["size"] for p in prices) / len(prices) * 20
        if avg_vol == 0:
            return 0.0
        ratio = recent_vol / avg_vol
        recent_returns = list(data["returns"])[-20:]
        avg_return = sum(recent_returns) / len(recent_returns) if recent_returns else 0
        if ratio > 1.5 and avg_return > 0:
            return 0.5
        elif ratio > 1.5 and avg_return < 0:
            return -0.5
        return 0.0

    def _compute_bar_momentum(self, pair: str) -> float:
        bar = self.bar_momentum.get(pair)
        if not bar:
            return 0.0
        return max(-1.0, min(1.0, bar["bar_return_bps"] / 20))

    def _compute_autocorrelation(self, pair: str) -> float:
        data = self.vwap_data.get(pair)
        if not data:
            return 0.0
        returns = list(data["returns"])
        if len(returns) < 30:
            return 0.0
        recent = returns[-30:]
        n = len(recent)
        mean = sum(recent) / n
        numerator = sum((recent[i] - mean) * (recent[i - 1] - mean) for i in range(1, n))
        denominator = sum((r - mean) ** 2 for r in recent)
        if denominator == 0:
            return 0.0
        autocorr = numerator / denominator
        return max(-1.0, min(1.0, autocorr * 3))

    def _get_weights(self, horizon_seconds: int) -> dict:
        if horizon_seconds <= 30:
            return {"vwap": 0.35, "vol": 0.15, "volume": 0.15, "bar": 0.15, "autocorr": 0.20}
        elif horizon_seconds <= 120:
            return {"vwap": 0.30, "vol": 0.20, "volume": 0.15, "bar": 0.20, "autocorr": 0.15}
        return {"vwap": 0.25, "vol": 0.25, "volume": 0.10, "bar": 0.25, "autocorr": 0.15}

    @staticmethod
    def _tail_prob(sigma: float, threshold: float, p_adv: float) -> float:
        if sigma <= 0:
            return 0.0
        z = threshold / sigma
        tail = 0.5 * math.erfc(z / math.sqrt(2))
        return min(1.0, tail * p_adv * 2)
