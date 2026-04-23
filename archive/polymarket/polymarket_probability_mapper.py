"""
polymarket_probability_mapper.py — Production probability mapping.

Loads calibration from disk. Provides edge computation for the executor.

Usage:
    mapper = ProbabilityMapper()
    mapper.load()

    edge = mapper.get_edge(
        asset="BTC",
        raw_prediction=0.087,
        crowd_yes_price=0.54,
    )

    if edge["has_edge"]:
        direction = edge["direction"]
        kelly = edge["kelly_fraction"]
"""

import json
import logging
import os
import pickle
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ProbabilityMapper:

    # Minimum edge to consider a bet (after fees)
    MIN_EDGE_PCT = 0.03  # 3%

    # Polymarket fees (conservative estimate)
    TAKER_FEE = 0.01  # 1% of winnings

    def __init__(self, calibration_path: str = "data/calibration/calibration_model.json"):
        self.calibration_path = calibration_path
        self.calibration_curve = None
        self.calibration_method = None
        self.diagnostics = {}
        self.loaded = False

    def load(self) -> bool:
        """Load calibration from disk. Returns True if successful."""
        model_path = self.calibration_path.replace(".json", ".pkl")

        if not os.path.exists(model_path):
            logger.warning(
                f"No calibration model at {model_path} — using linear mapping"
            )
            return False

        try:
            with open(model_path, "rb") as f:
                data = pickle.load(f)
                self.calibration_curve = data["curve"]
                self.calibration_method = data["method"]

            if os.path.exists(self.calibration_path):
                with open(self.calibration_path, "r") as f:
                    self.diagnostics = json.load(f)

            self.loaded = True
            logger.info(
                f"ProbabilityMapper loaded: method={self.calibration_method} "
                f"n_samples={self.diagnostics.get('n_samples', '?')}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
            return False

    def raw_to_probability(self, raw_prediction: float) -> float:
        """
        Convert raw ML prediction [-1, +1] to calibrated P(UP) [0.01, 0.99].
        """
        if not self.loaded or self.calibration_curve is None:
            return max(0.01, min(0.99, (raw_prediction + 1) / 2))

        try:
            if self.calibration_method == "isotonic":
                return float(
                    self.calibration_curve.predict([raw_prediction])[0]
                )
            elif self.calibration_method == "platt":
                return float(
                    self.calibration_curve.predict_proba(
                        np.array([[raw_prediction]])
                    )[0, 1]
                )
        except Exception as e:
            logger.debug(f"Calibration prediction failed: {e}")

        return max(0.01, min(0.99, (raw_prediction + 1) / 2))

    def get_edge(
        self,
        asset: str,
        raw_prediction: float,
        crowd_yes_price: float,
        regime: str = "unknown",
    ) -> Dict:
        """
        Compute the trading edge for a 5-minute direction market.

        Args:
            asset: "BTC", "ETH", etc.
            raw_prediction: float in [-1, +1] from ML ensemble
            crowd_yes_price: current Polymarket YES price (0 to 1)
            regime: current market regime

        Returns dict with edge analysis.
        """
        our_p_up = self.raw_to_probability(raw_prediction)
        our_p_down = 1.0 - our_p_up

        crowd_p_up = max(0.01, min(0.99, crowd_yes_price))
        crowd_p_down = 1.0 - crowd_p_up

        # Determine which side has edge
        up_edge = our_p_up - crowd_p_up
        down_edge = our_p_down - crowd_p_down

        if abs(up_edge) >= abs(down_edge):
            direction = "UP"
            edge = up_edge
            our_prob = our_p_up
            entry_price = crowd_p_up  # Buy YES
        else:
            direction = "DOWN"
            edge = down_edge
            our_prob = our_p_down
            entry_price = crowd_p_down  # Buy NO

        # Payout structure: pay entry_price, get $1 if right
        profit_if_win = (
            1.0 - entry_price - (self.TAKER_FEE * (1.0 - entry_price))
        )
        loss_if_wrong = entry_price

        # Expected value per dollar bet
        ev = our_prob * profit_if_win - (1 - our_prob) * loss_if_wrong

        # Kelly: f* = (p*b - q) / b
        b = profit_if_win / loss_if_wrong if loss_if_wrong > 0 else 0
        kelly = 0.0
        if b > 0:
            kelly = (our_prob * b - (1 - our_prob)) / b
            kelly = max(0.0, min(0.25, kelly))  # Cap at 25%
            kelly *= 0.5  # Half-Kelly for safety

        has_edge = abs(edge) >= self.MIN_EDGE_PCT and ev > 0.005

        return {
            "has_edge": has_edge,
            "direction": direction,
            "our_probability": round(our_prob, 4),
            "crowd_probability": round(
                crowd_p_up if direction == "UP" else crowd_p_down, 4
            ),
            "edge_pct": round(abs(edge), 4),
            "entry_price": round(entry_price, 4),
            "payout": 1.0,
            "profit_if_win": round(profit_if_win, 4),
            "loss_if_wrong": round(loss_if_wrong, 4),
            "expected_value": round(ev, 4),
            "kelly_fraction": round(kelly, 4),
            "raw_prediction": round(raw_prediction, 6),
            "calibration_method": self.calibration_method or "linear",
            "asset": asset,
            "regime": regime,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mapper = ProbabilityMapper()
    loaded = mapper.load()
    print(f"Calibration loaded: {loaded}")

    test_cases = [
        ("BTC", 0.087, 0.54),
        ("BTC", 0.200, 0.52),
        ("ETH", -0.150, 0.48),
        ("SOL", 0.050, 0.55),
        ("BTC", 0.350, 0.50),
        ("DOGE", -0.300, 0.52),
    ]

    hdr = (
        f"{'Asset':>5} | {'Pred':>6} | {'Crowd':>6} | {'Our P':>6} | "
        f"{'Edge':>6} | {'Dir':>4} | {'EV':>7} | {'Kelly':>6} | {'Bet?':>4}"
    )
    print(f"\n{hdr}")
    print("-" * 75)

    for asset, pred, crowd in test_cases:
        e = mapper.get_edge(
            asset=asset, raw_prediction=pred, crowd_yes_price=crowd
        )
        print(
            f"{asset:>5} | {pred:+.3f} | {crowd:.2f}   | "
            f"{e['our_probability']:.2f}   | {e['edge_pct']:.1%}  | "
            f"{e['direction']:>4} | {e['expected_value']:+.4f} | "
            f"{e['kelly_fraction']:.3f}  | "
            f"{'YES' if e['has_edge'] else 'no':>4}"
        )
