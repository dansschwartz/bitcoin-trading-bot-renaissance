"""
polymarket_edge_simulation.py — Historical edge simulation.

Simulates Polymarket 5m trading over historical data using:
1. Matched ML predictions (if available)
2. Parametric accuracy sweep (to find minimum viable accuracy)

Output: P&L at various edge thresholds, optimal threshold, drawdown analysis.

Usage:
    python polymarket_edge_simulation.py
"""

import sqlite3
import json
import logging
import os
import random
import numpy as np
from datetime import datetime
from typing import Dict
from polymarket_probability_mapper import ProbabilityMapper

logger = logging.getLogger(__name__)


def run_simulation(
    db_path: str = "data/renaissance_bot.db",
    bankroll: float = 500.0,
    min_edge_thresholds: list = None,
    model_accuracies: list = None,
    save_dir: str = "data/calibration/",
) -> Dict:
    """
    Simulate P&L across the historical dataset.

    Two modes:
    1. Real predictions: Uses matched ML predictions from calibration data
    2. Parametric: Sweeps model accuracy from 50% to 60% to find break-even

    Returns dict of results by (accuracy, threshold).
    """
    if min_edge_thresholds is None:
        min_edge_thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    if model_accuracies is None:
        model_accuracies = [0.50, 0.52, 0.53, 0.54, 0.55, 0.57, 0.60]

    mapper = ProbabilityMapper()
    mapper.load()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT asset, window_start, resolved, crowd_yes_open,
               price_change_pct
        FROM polymarket_5m_history
        WHERE resolved IS NOT NULL
        ORDER BY window_start
    """).fetchall()

    # Also try to load matched ML predictions
    ml_predictions = {}
    try:
        pred_rows = conn.execute("""
            SELECT h.asset, h.window_start, h.resolved, h.crowd_yes_open,
                   p.prediction
            FROM polymarket_5m_history h
            JOIN ml_predictions p ON p.product_id = (h.asset || '-USD')
                AND p.model_name = 'meta_ensemble'
                AND p.timestamp >= datetime(h.window_start, 'unixepoch', '-2 minutes')
                AND p.timestamp <= datetime(h.window_end, 'unixepoch', '+2 minutes')
            WHERE h.resolved IS NOT NULL
            ORDER BY h.window_start
        """).fetchall()
        for r in pred_rows:
            key = (r["asset"], r["window_start"])
            if key not in ml_predictions:
                ml_predictions[key] = r["prediction"]
    except Exception as e:
        logger.debug(f"Could not load matched predictions: {e}")

    conn.close()

    if not rows:
        print("NO HISTORICAL DATA — run PolymarketHistoryCollector first")
        return {}

    print(f"Simulating on {len(rows)} resolved 5-minute markets")
    print(f"Bankroll: ${bankroll:.2f}")
    print(f"Matched ML predictions: {len(ml_predictions)}")

    all_results = {}

    # --- Mode 1: Real ML predictions (if available) ---
    if ml_predictions:
        print("\n" + "=" * 70)
        print("MODE 1: REAL ML PREDICTIONS")
        print("=" * 70)

        for threshold in min_edge_thresholds:
            br = bankroll
            bets = 0
            wins = 0
            losses = 0
            pnl_series = [0.0]

            for row in rows:
                key = (row["asset"], row["window_start"])
                if key not in ml_predictions:
                    continue

                pred = ml_predictions[key]
                crowd_yes = row["crowd_yes_open"] or 0.50
                outcome = row["resolved"]

                edge_info = mapper.get_edge(
                    asset=row["asset"],
                    raw_prediction=pred,
                    crowd_yes_price=crowd_yes,
                )

                if edge_info["edge_pct"] < threshold:
                    continue

                bet_size = min(
                    br * edge_info["kelly_fraction"],
                    br * 0.10,
                    25.0,
                )
                if bet_size < 1.0 or br < 10:
                    continue

                won = (
                    (edge_info["direction"] == "UP" and outcome == 1)
                    or (edge_info["direction"] == "DOWN" and outcome == 0)
                )

                if won:
                    profit = (
                        bet_size * edge_info["profit_if_win"]
                        / edge_info["entry_price"]
                    )
                    br += profit
                    wins += 1
                else:
                    br -= bet_size
                    losses += 1

                bets += 1
                pnl_series.append(br - bankroll)

            max_dd = _max_drawdown(pnl_series)

            all_results[("real", threshold)] = {
                "mode": "real_predictions",
                "threshold_pct": threshold,
                "bets": bets,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / bets if bets > 0 else 0,
                "final_bankroll": round(br, 2),
                "total_pnl": round(br - bankroll, 2),
                "roi_pct": round((br - bankroll) / bankroll * 100, 1),
                "max_drawdown": round(max_dd, 2),
            }

        _print_results_table("Real ML Predictions", {
            k[1]: v for k, v in all_results.items() if k[0] == "real"
        })

    # --- Mode 2: Parametric accuracy sweep ---
    print("\n" + "=" * 70)
    print("MODE 2: PARAMETRIC ACCURACY SWEEP")
    print("(What accuracy do we NEED to be profitable?)")
    print("=" * 70)

    best_threshold = 0.03  # Use a single threshold for the sweep
    for accuracy in model_accuracies:
        br = bankroll
        bets = 0
        wins = 0
        losses = 0
        pnl_series = [0.0]

        for row in rows:
            crowd_yes = row["crowd_yes_open"] or 0.50
            outcome = row["resolved"]

            # Simulate model prediction with given accuracy
            random.seed(int(row["window_start"]) + hash(row["asset"]))
            model_correct = random.random() < accuracy

            if model_correct:
                sim_pred = 0.15 if outcome == 1 else -0.15
            else:
                sim_pred = -0.15 if outcome == 1 else 0.15

            edge_info = mapper.get_edge(
                asset=row["asset"],
                raw_prediction=sim_pred,
                crowd_yes_price=crowd_yes,
            )

            if edge_info["edge_pct"] < best_threshold:
                continue

            bet_size = min(
                br * edge_info["kelly_fraction"],
                br * 0.10,
                25.0,
            )
            if bet_size < 1.0 or br < 10:
                continue

            won = (
                (edge_info["direction"] == "UP" and outcome == 1)
                or (edge_info["direction"] == "DOWN" and outcome == 0)
            )

            if won:
                profit = (
                    bet_size * edge_info["profit_if_win"]
                    / edge_info["entry_price"]
                )
                br += profit
                wins += 1
            else:
                br -= bet_size
                losses += 1

            bets += 1
            pnl_series.append(br - bankroll)

        max_dd = _max_drawdown(pnl_series)

        all_results[("parametric", accuracy)] = {
            "mode": "parametric",
            "accuracy": accuracy,
            "threshold_pct": best_threshold,
            "bets": bets,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / bets if bets > 0 else 0,
            "final_bankroll": round(br, 2),
            "total_pnl": round(br - bankroll, 2),
            "roi_pct": round((br - bankroll) / bankroll * 100, 1),
            "max_drawdown": round(max_dd, 2),
        }

    # Print parametric results
    print(f"\n{'Accuracy':>10} | {'Bets':>5} | {'W/L':>7} | "
          f"{'Win%':>6} | {'P&L':>9} | {'ROI':>7} | {'MaxDD':>7}")
    print("-" * 65)

    for k, r in sorted(all_results.items()):
        if k[0] != "parametric":
            continue
        print(
            f"  {r['accuracy']:>7.0%}   | {r['bets']:>5} | "
            f"{r['wins']:>3}/{r['losses']:<3} | {r['win_rate']:>5.1%} | "
            f"${r['total_pnl']:>+8.2f} | {r['roi_pct']:>+5.1f}% | "
            f"${r['max_drawdown']:>6.2f}"
        )

    # Save all results
    os.makedirs(save_dir, exist_ok=True)
    save_data = {}
    for k, v in all_results.items():
        save_key = f"{k[0]}_{k[1]}"
        save_data[save_key] = v

    with open(os.path.join(save_dir, "edge_simulation.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {save_dir}edge_simulation.json")

    return all_results


def _max_drawdown(pnl_series: list) -> float:
    """Compute max drawdown from a P&L series."""
    arr = np.array(pnl_series)
    max_dd = 0.0
    peak = arr[0]
    for v in arr:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _print_results_table(title: str, results: dict):
    """Print a formatted results table."""
    print(f"\n{title}:")
    print(f"{'Threshold':>10} | {'Bets':>5} | {'W/L':>7} | "
          f"{'Win%':>6} | {'P&L':>9} | {'ROI':>7} | {'MaxDD':>7}")
    print("-" * 65)
    for t, r in sorted(results.items()):
        print(
            f"  {r['threshold_pct']:>7.0%}   | {r['bets']:>5} | "
            f"{r['wins']:>3}/{r['losses']:<3} | {r['win_rate']:>5.1%} | "
            f"${r['total_pnl']:>+8.2f} | {r['roi_pct']:>+5.1f}% | "
            f"${r['max_drawdown']:>6.2f}"
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_simulation()
