"""
Microstructure Predictor — Ultra-Short-Term (1s to 30s)
========================================================
Predicts price direction and magnitude over 1 to 30 seconds using
real-time orderbook and trade flow data.

At 1 second, orderbook imbalance alone predicts direction with 55-65%
accuracy in liquid crypto pairs.  Predictive power decays rapidly: by
30 seconds, microstructure signals are mostly noise.

Data sources (all in-memory, no API calls):
  1. Orderbook imbalance — bid depth vs ask depth
  2. Trade flow — buyer-initiated vs seller-initiated volume
  3. Spread dynamics — widening or narrowing
  4. Tick momentum — direction and size of last N trades

Performance: < 1ms per position per horizon.
"""

from __future__ import annotations

import math
import time
from collections import deque
from decimal import Decimal
from typing import Optional

from core.data_structures import HorizonEstimate


class MicrostructurePredictor:
    """
    Ultra-short-term price direction and magnitude predictor.

    Maintains rolling statistics per pair from WebSocket feeds.
    Called by MHPE for horizons: 1s, 5s, 30s.
    """

    def __init__(self, config: dict):
        self.config = config

        # Rolling data per pair — fed by WebSocket callbacks
        self.orderbook_snapshots: dict = {}   # pair -> deque of snapshots
        self.trade_flow: dict = {}            # pair -> deque of trades
        self.spread_history: dict = {}        # pair -> deque of spreads
        self.tick_history: dict = {}          # pair -> deque of ticks

        # History window sizes
        self.max_history_seconds = config.get("max_history_seconds", 120)
        self.max_ticks = config.get("max_ticks", 2000)

    # ── WEBSOCKET FEED HANDLERS ──

    def on_orderbook_update(
        self, pair: str, bids: list, asks: list, timestamp: float
    ) -> None:
        """Called on every orderbook update (~100ms)."""
        if pair not in self.orderbook_snapshots:
            self.orderbook_snapshots[pair] = deque(maxlen=600)

        levels = self.config.get("orderbook_levels", 10)
        bid_depth = sum(
            float(level[1]) if not isinstance(level[1], Decimal)
            else float(level[1])
            for level in bids[:levels]
        )
        ask_depth = sum(
            float(level[1]) if not isinstance(level[1], Decimal)
            else float(level[1])
            for level in asks[:levels]
        )
        total = bid_depth + ask_depth
        imbalance = bid_depth / total if total > 0 else 0.5

        self.orderbook_snapshots[pair].append({
            "timestamp": timestamp,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "imbalance": imbalance,
        })

    def on_trade(
        self, pair: str, price: float, size: float, side: str, timestamp: float
    ) -> None:
        """Called on every trade.  side: 'buy' or 'sell'."""
        if pair not in self.trade_flow:
            self.trade_flow[pair] = deque(maxlen=self.max_ticks)
        if pair not in self.tick_history:
            self.tick_history[pair] = deque(maxlen=self.max_ticks)

        self.trade_flow[pair].append({
            "timestamp": timestamp,
            "side": side,
            "size": size,
            "price": price,
        })

        direction = 1 if side == "buy" else -1
        self.tick_history[pair].append({
            "timestamp": timestamp,
            "price": price,
            "size": size,
            "direction": direction,
        })

    def on_spread_update(
        self, pair: str, best_bid: float, best_ask: float, timestamp: float
    ) -> None:
        """Track spread evolution."""
        if pair not in self.spread_history:
            self.spread_history[pair] = deque(maxlen=600)

        mid = (best_bid + best_ask) / 2
        spread_bps = (best_ask - best_bid) / mid * 10000 if mid > 0 else 0

        self.spread_history[pair].append({
            "timestamp": timestamp,
            "spread_bps": spread_bps,
            "mid": mid,
        })

    # ── PREDICTION ──

    def predict(self, pair: str, side: str, horizon_seconds: int) -> HorizonEstimate:
        """
        Predict price movement for a specific pair and horizon.

        Args:
            pair: e.g. "BTC-USD"
            side: "long" or "short"
            horizon_seconds: 1, 5, or 30
        """
        now = time.time()

        # Features
        imbalance = self._compute_orderbook_imbalance(pair)
        flow = self._compute_trade_flow(pair, now, min(horizon_seconds * 3, 30))
        spread_trend = self._compute_spread_trend(pair, now)
        momentum = self._compute_tick_momentum(pair, now, min(horizon_seconds * 2, 20))
        recent_vol = self._compute_recent_volatility(pair, now, 60)

        # Weights decay with horizon
        weights = self._get_feature_weights(horizon_seconds)

        raw_signal = (
            weights["imbalance"] * self._imbalance_to_signal(imbalance)
            + weights["flow"] * self._flow_to_signal(flow)
            + weights["spread"] * self._spread_to_signal(spread_trend)
            + weights["momentum"] * self._momentum_to_signal(momentum)
        )

        # Convert to probability
        p_up = self._signal_to_probability(raw_signal, horizon_seconds)
        p_favorable = p_up if side == "long" else (1.0 - p_up)

        # Magnitude estimation — volatility scales with sqrt(time)
        base_vol_bps = recent_vol * math.sqrt(horizon_seconds)

        e_favorable = base_vol_bps * self._expected_positive_move(p_favorable)
        e_adverse = base_vol_bps * self._expected_positive_move(1.0 - p_favorable)
        e_net = p_favorable * e_favorable - (1.0 - p_favorable) * e_adverse

        # Tail risk
        p_adv_10 = self._tail_probability(base_vol_bps, 10, 1.0 - p_favorable)
        p_adv_25 = self._tail_probability(base_vol_bps, 25, 1.0 - p_favorable)
        p_adv_50 = self._tail_probability(base_vol_bps, 50, 1.0 - p_favorable)

        # Estimate confidence from data quality
        data_quality = self._assess_data_quality(pair)
        horizon_decay = 1.0 / (1.0 + 0.05 * horizon_seconds)
        estimate_confidence = data_quality * horizon_decay

        return HorizonEstimate(
            horizon_seconds=horizon_seconds,
            horizon_label=self._format_horizon(horizon_seconds),
            p_profit=round(p_favorable, 4),
            p_loss=round(1.0 - p_favorable, 4),
            e_favorable_bps=round(e_favorable, 2),
            e_adverse_bps=round(e_adverse, 2),
            e_net_bps=round(e_net, 2),
            sigma_bps=round(base_vol_bps, 2),
            p_adverse_10bps=round(p_adv_10, 4),
            p_adverse_25bps=round(p_adv_25, 4),
            p_adverse_50bps=round(p_adv_50, 4),
            estimate_confidence=round(estimate_confidence, 3),
            dominant_signal=self._dominant_signal_name(weights),
        )

    # ── FEATURE COMPUTATION ──

    def _compute_orderbook_imbalance(self, pair: str) -> float:
        snapshots = self.orderbook_snapshots.get(pair)
        if not snapshots:
            return 0.5
        return snapshots[-1]["imbalance"]

    def _compute_trade_flow(self, pair: str, now: float, window_seconds: int) -> float:
        trades = self.trade_flow.get(pair)
        if not trades:
            return 0.5
        cutoff = now - window_seconds
        buy_vol = 0.0
        sell_vol = 0.0
        for t in reversed(trades):
            if t["timestamp"] < cutoff:
                break
            if t["side"] == "buy":
                buy_vol += t["size"]
            else:
                sell_vol += t["size"]
        total = buy_vol + sell_vol
        return buy_vol / total if total > 0 else 0.5

    def _compute_spread_trend(self, pair: str, now: float) -> float:
        history = self.spread_history.get(pair)
        if not history or len(history) < 10:
            return 0.0
        recent = list(history)[-5:]
        recent_avg = sum(s["spread_bps"] for s in recent) / len(recent)
        cutoff = now - 30
        older = [s for s in history if s["timestamp"] < cutoff]
        if not older:
            return 0.0
        older_avg = sum(s["spread_bps"] for s in older[-5:]) / min(5, len(older))
        if older_avg == 0:
            return 0.0
        change_pct = (recent_avg - older_avg) / older_avg
        return max(-1.0, min(1.0, -change_pct * 5))

    def _compute_tick_momentum(self, pair: str, now: float, window_seconds: int) -> float:
        ticks = self.tick_history.get(pair)
        if not ticks:
            return 0.0
        cutoff = now - window_seconds
        weighted_sum = 0.0
        total_weight = 0.0
        for t in reversed(ticks):
            if t["timestamp"] < cutoff:
                break
            age = now - t["timestamp"]
            weight = math.exp(-age / max(window_seconds, 1)) * t["size"]
            weighted_sum += t["direction"] * weight
            total_weight += weight
        if total_weight == 0:
            return 0.0
        return max(-1.0, min(1.0, weighted_sum / total_weight))

    def _compute_recent_volatility(self, pair: str, now: float, window_seconds: int) -> float:
        ticks = self.tick_history.get(pair)
        if not ticks or len(ticks) < 10:
            return 2.0  # default 2 bps per second for BTC
        cutoff = now - window_seconds
        prices = []
        for t in reversed(ticks):
            if t["timestamp"] < cutoff:
                break
            prices.append(t["price"])
        if len(prices) < 5:
            return 2.0
        returns = []
        for i in range(1, len(prices)):
            if prices[i] > 0:
                returns.append(abs((prices[i - 1] - prices[i]) / prices[i] * 10000))
        if not returns:
            return 2.0
        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance) if variance > 0 else 1.0

    # ── SIGNAL CONVERSION ──

    def _get_feature_weights(self, horizon_seconds: int) -> dict:
        if horizon_seconds <= 1:
            return {"imbalance": 0.40, "flow": 0.30, "spread": 0.10, "momentum": 0.20}
        elif horizon_seconds <= 5:
            return {"imbalance": 0.25, "flow": 0.35, "spread": 0.10, "momentum": 0.30}
        elif horizon_seconds <= 30:
            return {"imbalance": 0.10, "flow": 0.25, "spread": 0.15, "momentum": 0.50}
        return {"imbalance": 0.05, "flow": 0.15, "spread": 0.20, "momentum": 0.60}

    @staticmethod
    def _imbalance_to_signal(imbalance: float) -> float:
        return (imbalance - 0.5) * 2

    @staticmethod
    def _flow_to_signal(flow: float) -> float:
        return (flow - 0.5) * 2

    @staticmethod
    def _spread_to_signal(spread_trend: float) -> float:
        return spread_trend

    @staticmethod
    def _momentum_to_signal(momentum: float) -> float:
        return momentum

    def _signal_to_probability(self, raw_signal: float, horizon_seconds: int) -> float:
        steepness_map = self.config.get("signal_steepness", {})
        steepness = steepness_map.get(str(horizon_seconds),
                                      {1: 1.2, 5: 1.0, 30: 0.7}.get(horizon_seconds, 0.5))
        clamped = max(-3, min(3, raw_signal))
        prob = 1.0 / (1.0 + math.exp(-steepness * clamped))
        lo, hi = self.config.get("probability_clamp", [0.40, 0.70])
        return max(lo, min(hi, prob))

    @staticmethod
    def _expected_positive_move(p: float) -> float:
        return p * 0.8

    @staticmethod
    def _tail_probability(sigma_bps: float, threshold_bps: float, p_adverse: float) -> float:
        if sigma_bps <= 0:
            return 0.0
        z = threshold_bps / sigma_bps
        tail = 0.5 * math.erfc(z / math.sqrt(2))
        return min(1.0, tail * p_adverse * 2)

    def _assess_data_quality(self, pair: str) -> float:
        tick_count = len(self.tick_history.get(pair, []))
        book_count = len(self.orderbook_snapshots.get(pair, []))
        tick_score = min(1.0, tick_count / 50)
        book_score = min(1.0, book_count / 30)
        return tick_score * 0.6 + book_score * 0.4

    @staticmethod
    def _dominant_signal_name(weights: dict) -> str:
        return max(weights, key=weights.get)

    @staticmethod
    def _format_horizon(seconds: int) -> str:
        return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"
