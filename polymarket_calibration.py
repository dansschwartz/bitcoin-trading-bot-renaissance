"""
polymarket_calibration.py — ML prediction calibration for Polymarket edge.

Answers: "When our model predicts +0.087, what is the TRUE probability
the price goes up in 5 minutes?"

Uses historical outcomes + ML predictions to fit a calibration curve.
The output is a callable function: raw_prediction → calibrated_probability.

Usage:
    analyzer = CalibrationAnalyzer(db_path="data/renaissance_bot.db")
    predictions, outcomes, assets = analyzer.load_matched_data()
    diag = analyzer.fit_calibration(predictions, outcomes)
    analyzer.save()
    analyzer.generate_report()
"""

import sqlite3
import numpy as np
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CalibrationAnalyzer:

    def __init__(self, db_path: str = "data/renaissance_bot.db"):
        self.db_path = db_path
        self.calibration_curve = None
        self.calibration_method = None
        self.diagnostics = {}

    def load_matched_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Match ML ensemble predictions to 5-minute outcomes.

        Joins ml_predictions (model_name='meta_ensemble') with
        polymarket_5m_history by matching:
          - product_id ↔ asset
          - prediction timestamp within the 5-minute window

        Returns: (predictions, outcomes, assets)
        """
        conn = sqlite3.connect(self.db_path)

        # Asset mapping: ml_predictions uses "BTC-USD", history uses "BTC"
        asset_to_product = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",
            "XRP": "XRP-USD",
            "DOGE": "DOGE-USD",
        }

        predictions_data = []
        outcomes_data = []
        assets_data = []

        # Build prediction lookup: load all meta_ensemble predictions,
        # bucket by (product_id, 5min_slot) for fast matching
        logger.info("Loading meta_ensemble predictions into memory...")
        pred_rows = conn.execute("""
            SELECT product_id, timestamp, prediction
            FROM ml_predictions
            WHERE model_name = 'meta_ensemble'
              AND prediction IS NOT NULL
            ORDER BY timestamp
        """).fetchall()

        # Index by (product_id, 5-min slot)
        pred_lookup = {}  # (product_id, slot) -> prediction
        for product_id, ts_str, prediction in pred_rows:
            # Parse ISO timestamp to unix
            try:
                dt = datetime.fromisoformat(ts_str.replace("+00:00", ""))
                unix_ts = dt.timestamp()
            except (ValueError, AttributeError):
                continue
            slot = int(unix_ts // 300) * 300
            key = (product_id, slot)
            if key not in pred_lookup:
                pred_lookup[key] = float(prediction)

        logger.info(f"Loaded {len(pred_lookup)} prediction slots")

        # Get all resolved history rows
        history_rows = conn.execute("""
            SELECT asset, window_start, window_end, resolved, price_change_pct
            FROM polymarket_5m_history
            WHERE resolved IS NOT NULL
            ORDER BY window_start
        """).fetchall()

        logger.info(f"Found {len(history_rows)} historical 5m outcomes")

        # Match by (product_id, slot) — instant lookup
        for h_asset, h_start, h_end, h_resolved, h_pct in history_rows:
            product_id = asset_to_product.get(h_asset)
            if not product_id:
                continue

            slot = int(h_start)
            # Try exact slot, then ±1 slot for edge cases
            pred = pred_lookup.get((product_id, slot))
            if pred is None:
                pred = pred_lookup.get((product_id, slot - 300))
            if pred is None:
                pred = pred_lookup.get((product_id, slot + 300))

            if pred is not None:
                predictions_data.append(pred)
                outcomes_data.append(int(h_resolved))
                assets_data.append(h_asset)

        conn.close()

        n_matched = len(predictions_data)
        logger.info(
            f"CALIBRATION: Matched {n_matched} prediction-outcome pairs "
            f"from {len(history_rows)} outcomes"
        )

        if n_matched > 0:
            preds = np.array(predictions_data)
            outs = np.array(outcomes_data)
            logger.info(
                f"  Predictions: [{preds.min():.4f}, {preds.max():.4f}] "
                f"mean={preds.mean():.4f}"
            )
            logger.info(
                f"  Outcomes: {outs.sum()}/{len(outs)} UP ({outs.mean():.1%})"
            )
            logger.info(f"  Assets: {dict(zip(*np.unique(assets_data, return_counts=True)))}")

        return (
            np.array(predictions_data) if predictions_data else np.array([]),
            np.array(outcomes_data) if outcomes_data else np.array([]),
            assets_data,
        )

    def fit_calibration(self, predictions: np.ndarray,
                        outcomes: np.ndarray) -> Dict:
        """
        Fit calibration curves using both Platt scaling and isotonic regression.

        Returns diagnostics dict with accuracy, Brier scores, bin analysis.
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        n = len(predictions)
        if n < 10:
            logger.warning(f"Only {n} samples — too few for calibration")
            self.diagnostics = {
                "n_samples": n,
                "error": "insufficient_data",
            }
            return self.diagnostics

        # Convert predictions to rough probabilities [0, 1]
        raw_probs = (predictions + 1) / 2  # -1→0.0, 0→0.5, +1→1.0

        # Raw accuracy
        raw_correct = np.sum((predictions > 0) == (outcomes == 1))
        raw_accuracy = raw_correct / n

        # Brier score (raw)
        raw_brier = np.mean((raw_probs - outcomes) ** 2)

        # --- Isotonic Regression ---
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds='clip')
        iso.fit(predictions, outcomes)
        iso_probs = iso.predict(predictions)
        iso_brier = np.mean((iso_probs - outcomes) ** 2)

        # --- Platt Scaling ---
        lr = LogisticRegression(C=1.0)
        lr.fit(predictions.reshape(-1, 1), outcomes)
        platt_probs = lr.predict_proba(predictions.reshape(-1, 1))[:, 1]
        platt_brier = np.mean((platt_probs - outcomes) ** 2)

        # Choose the better calibration
        if iso_brier <= platt_brier:
            self.calibration_curve = iso
            self.calibration_method = "isotonic"
            cal_probs = iso_probs
            cal_brier = iso_brier
        else:
            self.calibration_curve = lr
            self.calibration_method = "platt"
            cal_probs = platt_probs
            cal_brier = platt_brier

        # Bin analysis (reliability diagram data)
        n_bins = min(10, max(5, n // 30))
        bin_edges = np.linspace(predictions.min(), predictions.max(), n_bins + 1)
        bin_stats = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
            else:
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

            if mask.sum() > 0:
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                actual_freq = outcomes[mask].mean()
                calibrated_prob = cal_probs[mask].mean()
                bin_stats.append({
                    "bin_center": float(bin_center),
                    "raw_prediction_range": [float(bin_edges[i]),
                                             float(bin_edges[i + 1])],
                    "actual_up_frequency": float(actual_freq),
                    "calibrated_probability": float(calibrated_prob),
                    "count": int(mask.sum()),
                })

        # Per-asset accuracy
        per_asset = {}
        if hasattr(self, '_last_assets'):
            for asset in set(self._last_assets):
                mask = np.array([a == asset for a in self._last_assets])
                if mask.sum() > 0:
                    correct = np.sum(
                        (predictions[mask] > 0) == (outcomes[mask] == 1)
                    )
                    per_asset[asset] = {
                        "n": int(mask.sum()),
                        "accuracy": float(correct / mask.sum()),
                        "up_rate": float(outcomes[mask].mean()),
                    }

        self.diagnostics = {
            "n_samples": n,
            "raw_accuracy": float(raw_accuracy),
            "raw_brier_score": float(raw_brier),
            "isotonic_brier": float(iso_brier),
            "platt_brier": float(platt_brier),
            "chosen_method": self.calibration_method,
            "calibrated_brier": float(cal_brier),
            "brier_improvement_pct": float(
                (raw_brier - cal_brier) / raw_brier * 100
            ) if raw_brier > 0 else 0,
            "prediction_range": [float(predictions.min()),
                                 float(predictions.max())],
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std()),
            "outcome_balance": float(outcomes.mean()),
            "bin_stats": bin_stats,
            "per_asset": per_asset,
            "fitted_at": datetime.utcnow().isoformat(),
        }

        return self.diagnostics

    def predict_probability(self, raw_prediction: float) -> float:
        """
        Convert a raw ML prediction to a calibrated probability.

        Args:
            raw_prediction: float in [-1, +1] from ML ensemble

        Returns:
            float in [0.01, 0.99] representing P(price goes UP)
        """
        if self.calibration_curve is None:
            return max(0.01, min(0.99, (raw_prediction + 1) / 2))

        try:
            if self.calibration_method == "isotonic":
                return float(self.calibration_curve.predict([raw_prediction])[0])
            elif self.calibration_method == "platt":
                return float(self.calibration_curve.predict_proba(
                    np.array([[raw_prediction]])
                )[0, 1])
        except Exception as e:
            logger.debug(f"Calibration predict failed: {e}")

        return max(0.01, min(0.99, (raw_prediction + 1) / 2))

    def save(self, path: str = "data/calibration/calibration_model.json"):
        """Save calibration to disk for production use."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model_path = path.replace(".json", ".pkl")
        with open(model_path, "wb") as f:
            pickle.dump({
                "curve": self.calibration_curve,
                "method": self.calibration_method,
            }, f)

        with open(path, "w") as f:
            json.dump(self.diagnostics, f, indent=2)

        logger.info(f"Calibration saved: model={model_path} diagnostics={path}")

    def load(self, path: str = "data/calibration/calibration_model.json"):
        """Load saved calibration."""
        model_path = path.replace(".json", ".pkl")

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.calibration_curve = data["curve"]
            self.calibration_method = data["method"]

        with open(path, "r") as f:
            self.diagnostics = json.load(f)

        logger.info(f"Calibration loaded: method={self.calibration_method}")

    def generate_report(self, save_dir: str = "data/calibration/"):
        """Generate a human-readable calibration report + reliability plot."""
        os.makedirs(save_dir, exist_ok=True)

        d = self.diagnostics
        if not d or d.get("error"):
            logger.warning("No diagnostics available. Run fit_calibration first.")
            return

        report_lines = [
            "=" * 60,
            "POLYMARKET 5M CALIBRATION REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 60,
            "",
            f"Data points: {d['n_samples']}",
            f"Outcome balance: {d['outcome_balance']:.1%} UP",
            f"Prediction range: [{d['prediction_range'][0]:.4f}, "
            f"{d['prediction_range'][1]:.4f}]",
            f"Prediction mean: {d['prediction_mean']:.4f}, "
            f"std: {d['prediction_std']:.4f}",
            "",
            "--- RAW MODEL PERFORMANCE ---",
            f"Directional accuracy: {d['raw_accuracy']:.1%}",
            f"Brier score (raw): {d['raw_brier_score']:.4f}",
            "",
            "--- CALIBRATION ---",
            f"Method chosen: {d['chosen_method']}",
            f"Brier score (isotonic): {d['isotonic_brier']:.4f}",
            f"Brier score (Platt): {d['platt_brier']:.4f}",
            f"Brier score (calibrated): {d['calibrated_brier']:.4f}",
            f"Improvement: {d['brier_improvement_pct']:.1f}%",
            "",
            "--- RELIABILITY DIAGRAM ---",
            f"{'Prediction Range':>20} | {'Actual UP%':>10} | "
            f"{'Cal. Prob':>10} | {'Count':>6}",
            "-" * 55,
        ]

        for b in d.get("bin_stats", []):
            lo, hi = b["raw_prediction_range"]
            report_lines.append(
                f"  [{lo:+.4f}, {hi:+.4f}] | "
                f"{b['actual_up_frequency']:>9.1%} | "
                f"{b['calibrated_probability']:>9.1%} | "
                f"{b['count']:>5}"
            )

        # Per-asset breakdown
        if d.get("per_asset"):
            report_lines.extend(["", "--- PER-ASSET ACCURACY ---"])
            for asset, stats in sorted(d["per_asset"].items()):
                report_lines.append(
                    f"  {asset}: accuracy={stats['accuracy']:.1%} "
                    f"UP_rate={stats['up_rate']:.1%} n={stats['n']}"
                )

        # Calibration mapping table
        report_lines.extend([
            "",
            "--- CALIBRATION MAPPING ---",
            "  (what our model says → what probability that maps to)",
        ])

        if self.calibration_curve is not None:
            test_preds = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
            for p in test_preds:
                prob = self.predict_probability(p)
                report_lines.append(
                    f"  prediction={p:+.3f} → P(UP)={prob:.1%}"
                )

        report_text = "\n".join(report_lines)

        report_path = os.path.join(save_dir, "calibration_report.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        logger.info(f"Calibration report saved to {report_path}")
        print(report_text)

        # Reliability diagram
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            bins = d.get("bin_stats", [])
            if bins:
                centers = [b["bin_center"] for b in bins]
                actual = [b["actual_up_frequency"] for b in bins]
                calibrated = [b["calibrated_probability"] for b in bins]
                counts = [b["count"] for b in bins]

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Reliability diagram
                ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
                prob_centers = [(c + 1) / 2 for c in centers]
                ax1.scatter(
                    prob_centers, actual,
                    s=[c * 2 for c in counts],
                    c="steelblue", alpha=0.7, label="Raw bins"
                )
                ax1.scatter(
                    calibrated, actual,
                    s=[c * 2 for c in counts],
                    c="coral", alpha=0.7, marker="^", label="Calibrated"
                )
                ax1.set_xlabel("Predicted P(UP)")
                ax1.set_ylabel("Actual P(UP)")
                ax1.set_title("Reliability Diagram")
                ax1.legend()
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)

                # Bin count histogram
                ax2.bar(range(len(counts)), counts, color="steelblue", alpha=0.7)
                ax2.set_xlabel("Prediction Bin")
                ax2.set_ylabel("Count")
                ax2.set_title("Sample Distribution Across Bins")

                plt.tight_layout()
                plot_path = os.path.join(save_dir, "reliability_diagram.png")
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.info(f"Reliability diagram saved to {plot_path}")
        except ImportError:
            logger.info("matplotlib not available — skipping plot")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    analyzer = CalibrationAnalyzer(db_path="data/renaissance_bot.db")

    predictions, outcomes, assets = analyzer.load_matched_data()
    analyzer._last_assets = assets  # Store for per-asset stats

    print(f"\nMatched pairs: {len(predictions)}")
    if len(predictions) > 0:
        print(f"  Prediction range: [{predictions.min():.4f}, "
              f"{predictions.max():.4f}]")
        print(f"  UP outcomes: {outcomes.sum()}/{len(outcomes)} "
              f"({outcomes.mean():.1%})")

        diag = analyzer.fit_calibration(predictions, outcomes)
        print(f"\nRaw accuracy: {diag['raw_accuracy']:.1%}")
        print(f"Calibration method: {diag['chosen_method']}")
        print(f"Brier: {diag['raw_brier_score']:.4f} → "
              f"{diag['calibrated_brier']:.4f}")

        for pred in [-0.3, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3]:
            prob = analyzer.predict_probability(pred)
            print(f"  prediction={pred:+.3f} → P(UP)={prob:.1%}")

        analyzer.save()
        analyzer.generate_report()
    else:
        print("NO MATCHED DATA — ensure polymarket_5m_history and "
              "ml_predictions tables overlap")
