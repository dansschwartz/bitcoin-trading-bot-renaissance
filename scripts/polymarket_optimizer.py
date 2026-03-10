#!/usr/bin/env python3
"""Polymarket Strategy A Optimizer.

Grid-searches tunable parameters against historical bet data to find
the optimal configuration for each asset and timeframe.

Usage:
    .venv/bin/python3 scripts/polymarket_optimizer.py
"""

import sqlite3
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "renaissance_bot.db"


@dataclass
class ParamSet:
    """One parameter combination to evaluate."""
    confidence_min: float       # Entry threshold (e.g., 52.0)
    confidence_max: float       # Cap high confidence (e.g., 53.0 or 100)
    token_cost_min: float       # Min crowd price (e.g., 0.15)
    token_cost_max: float       # Max crowd price (e.g., 0.50)
    max_bet_usd: float          # Bet cap (e.g., 20)
    kelly_fraction: float       # Kelly fraction (e.g., 0.5)


def load_bets(db_path: str = str(DB_PATH)) -> list[dict]:
    """Load all resolved bets from the database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, asset, entry_side, entry_token_cost, entry_confidence,
               total_invested, total_tokens, avg_cost, status, exit_price,
               exit_reason, pnl, return_pct, timeframe, opened_at, exit_at
        FROM polymarket_bets
        WHERE status IN ('WON', 'LOST', 'CLOSED')
          AND total_invested > 0
        ORDER BY exit_at
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def simulate_kelly_bet(prob: float, token_cost: float, kelly_frac: float,
                       bankroll: float, max_bet: float, min_bet: float = 5.0,
                       max_pct: float = 0.05, max_sizing_br: float = 1000.0) -> float:
    """Replicate the Kelly sizing logic."""
    b = (1.0 - token_cost) / (token_cost + 1e-10)
    p = max(0.5, min(0.99, prob))
    q = 1.0 - p
    kelly = (p * b - q) / (b + 1e-10)
    kelly = max(0.0, kelly)
    if kelly <= 0:
        return 0.0
    frac_kelly = kelly * kelly_frac
    sizing_br = min(bankroll, max_sizing_br)
    bet = sizing_br * frac_kelly
    ceiling = min(sizing_br * max_pct, max_bet)
    bet = max(min_bet, min(bet, ceiling))
    return round(bet, 2)


def evaluate_params(bets: list[dict], params: ParamSet,
                    asset_filter: str = "", tf_filter: int = 0,
                    initial_bankroll: float = 500.0) -> dict:
    """Simulate trading with given parameters. Returns performance stats."""
    bankroll = initial_bankroll
    trades = 0
    wins = 0
    total_pnl = 0.0
    max_drawdown = 0.0
    peak = bankroll
    daily_pnl: dict[str, float] = {}

    for bet in bets:
        # Asset/timeframe filter
        if asset_filter and bet["asset"] != asset_filter:
            continue
        if tf_filter and bet["timeframe"] != tf_filter:
            continue

        conf = bet["entry_confidence"] or 0
        token_cost = bet["entry_token_cost"] or 0

        # Apply parameter gates
        if conf < params.confidence_min or conf > params.confidence_max:
            continue
        if token_cost < params.token_cost_min or token_cost > params.token_cost_max:
            continue

        # Re-simulate Kelly sizing with our params
        prob = conf / 100.0
        sim_bet = simulate_kelly_bet(
            prob, token_cost, params.kelly_fraction,
            bankroll, params.max_bet_usd,
        )
        if sim_bet <= 0:
            continue

        # Use actual outcome direction but scale P&L to our simulated bet size
        actual_invested = bet["total_invested"] or 0
        actual_pnl = bet["pnl"] or 0
        if actual_invested > 0:
            # Scale P&L proportionally to our simulated bet size
            pnl = actual_pnl * (sim_bet / actual_invested)
        else:
            pnl = 0

        bankroll += pnl
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1

        # Track drawdown
        if bankroll > peak:
            peak = bankroll
        dd = (peak - bankroll) / peak * 100 if peak > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

        # Daily tracking
        day = (bet.get("exit_at") or "")[:10]
        if day:
            daily_pnl[day] = daily_pnl.get(day, 0) + pnl

        # Bankruptcy check
        if bankroll <= 0:
            break

    win_rate = wins / trades * 100 if trades > 0 else 0
    avg_pnl = total_pnl / trades if trades > 0 else 0
    worst_day = min(daily_pnl.values()) if daily_pnl else 0
    best_day = max(daily_pnl.values()) if daily_pnl else 0

    return {
        "trades": trades,
        "wins": wins,
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(avg_pnl, 4),
        "final_bankroll": round(bankroll, 2),
        "max_drawdown_pct": round(max_drawdown, 1),
        "worst_day": round(worst_day, 2),
        "best_day": round(best_day, 2),
        "return_pct": round(total_pnl / initial_bankroll * 100, 2),
    }


def main():
    print("=" * 80)
    print("POLYMARKET STRATEGY A — PARAMETER OPTIMIZER")
    print("=" * 80)

    bets = load_bets()
    if not bets:
        print("No bets found in database!")
        return

    print(f"\nLoaded {len(bets)} resolved bets")

    # Count by asset/timeframe
    from collections import Counter
    asset_counts = Counter((b["asset"], b["timeframe"]) for b in bets)
    for (a, tf), cnt in sorted(asset_counts.items()):
        print(f"  {a} {tf}m: {cnt} bets")

    # ── Current production parameters ──
    print("\n" + "─" * 80)
    print("CURRENT PRODUCTION PARAMETERS")
    print("─" * 80)

    current = ParamSet(
        confidence_min=52.0,
        confidence_max=100.0,
        token_cost_min=0.15,
        token_cost_max=0.85,
        max_bet_usd=50.0,
        kelly_fraction=0.5,
    )

    for label, asset, tf in [
        ("ALL", "", 0),
        ("BTC 15m", "BTC", 15), ("BTC 5m", "BTC", 5),
        ("SOL 15m", "SOL", 15), ("SOL 5m", "SOL", 5),
        ("XRP 15m", "XRP", 15), ("XRP 5m", "XRP", 5),
    ]:
        result = evaluate_params(bets, current, asset, tf)
        if result["trades"] > 0:
            print(f"  {label:10s}: {result['trades']:4d} trades | "
                  f"P&L: ${result['total_pnl']:+8.2f} | "
                  f"WR: {result['win_rate']:5.1f}% | "
                  f"DD: {result['max_drawdown_pct']:5.1f}% | "
                  f"Avg: ${result['avg_pnl']:+.4f}")

    # ── Grid search ──
    print("\n" + "─" * 80)
    print("GRID SEARCH — FINDING OPTIMAL PARAMETERS")
    print("─" * 80)

    # Parameter grid
    conf_mins = [51.0, 51.5, 52.0, 52.5, 53.0]
    conf_maxs = [52.5, 53.0, 54.0, 55.0, 100.0]
    cost_mins = [0.15, 0.20, 0.25]
    cost_maxs = [0.40, 0.50, 0.60, 0.85]
    max_bets = [10.0, 15.0, 20.0, 30.0, 50.0]
    kelly_fracs = [0.25, 0.35, 0.50, 0.65]

    segments = [
        ("ALL ASSETS", "", 0),
        ("SOL 15m", "SOL", 15),
        ("SOL 5m", "SOL", 5),
        ("BTC 15m", "BTC", 15),
        ("BTC 5m", "BTC", 5),
        ("XRP 15m", "XRP", 15),
    ]

    for seg_label, seg_asset, seg_tf in segments:
        print(f"\n{'=' * 70}")
        print(f"  OPTIMIZING: {seg_label}")
        print(f"{'=' * 70}")

        best_pnl = -999999
        best_params = None
        best_result = None
        tested = 0

        for cm, cx, tcm, tcx, mb, kf in itertools.product(
            conf_mins, conf_maxs, cost_mins, cost_maxs, max_bets, kelly_fracs
        ):
            if cm >= cx:
                continue  # Invalid range

            ps = ParamSet(
                confidence_min=cm, confidence_max=cx,
                token_cost_min=tcm, token_cost_max=tcx,
                max_bet_usd=mb, kelly_fraction=kf,
            )
            result = evaluate_params(bets, ps, seg_asset, seg_tf)
            tested += 1

            # Require minimum trade count for significance
            min_trades = 15 if seg_asset else 30
            if result["trades"] < min_trades:
                continue

            # Optimization target: total P&L with drawdown penalty
            score = result["total_pnl"] - result["max_drawdown_pct"] * 0.5

            if score > best_pnl:
                best_pnl = score
                best_params = ps
                best_result = result

        print(f"  Tested {tested:,} parameter combinations")

        if best_params and best_result:
            print(f"\n  BEST PARAMETERS:")
            print(f"    Confidence:  [{best_params.confidence_min:.1f}%, {best_params.confidence_max:.1f}%]")
            print(f"    Token cost:  [{best_params.token_cost_min:.2f}, {best_params.token_cost_max:.2f}]")
            print(f"    Max bet:     ${best_params.max_bet_usd:.0f}")
            print(f"    Kelly frac:  {best_params.kelly_fraction:.2f}")
            print(f"\n  RESULTS:")
            print(f"    Trades:      {best_result['trades']}")
            print(f"    Win rate:    {best_result['win_rate']:.1f}%")
            print(f"    Total P&L:   ${best_result['total_pnl']:+.2f}")
            print(f"    Avg P&L:     ${best_result['avg_pnl']:+.4f}")
            print(f"    Max DD:      {best_result['max_drawdown_pct']:.1f}%")
            print(f"    Best day:    ${best_result['best_day']:+.2f}")
            print(f"    Worst day:   ${best_result['worst_day']:+.2f}")
            print(f"    Final BR:    ${best_result['final_bankroll']:.2f}")
        else:
            print(f"  No parameter set met minimum trade requirements.")

    # ── Quick sensitivity analysis ──
    print("\n" + "─" * 80)
    print("SENSITIVITY ANALYSIS — KEY PARAMETER SWEEPS")
    print("─" * 80)

    base = ParamSet(52.0, 100.0, 0.15, 0.85, 50.0, 0.5)

    # Confidence max sweep
    print("\n  Confidence Cap (all else = production defaults):")
    for cap in [52.5, 53.0, 53.5, 54.0, 55.0, 100.0]:
        ps = ParamSet(52.0, cap, 0.15, 0.85, 50.0, 0.5)
        r = evaluate_params(bets, ps)
        if r["trades"] > 0:
            print(f"    Cap {cap:5.1f}%: {r['trades']:4d} trades | "
                  f"P&L: ${r['total_pnl']:+8.2f} | WR: {r['win_rate']:5.1f}%")

    # Token cost max sweep
    print("\n  Token Cost Cap (all else = production defaults):")
    for tcap in [0.30, 0.40, 0.50, 0.60, 0.70, 0.85]:
        ps = ParamSet(52.0, 100.0, 0.15, tcap, 50.0, 0.5)
        r = evaluate_params(bets, ps)
        if r["trades"] > 0:
            print(f"    Cap {tcap:.2f}: {r['trades']:4d} trades | "
                  f"P&L: ${r['total_pnl']:+8.2f} | WR: {r['win_rate']:5.1f}%")

    # Max bet sweep
    print("\n  Max Bet USD (all else = production defaults):")
    for mb in [5, 10, 15, 20, 30, 50]:
        ps = ParamSet(52.0, 100.0, 0.15, 0.85, float(mb), 0.5)
        r = evaluate_params(bets, ps)
        if r["trades"] > 0:
            print(f"    Max ${mb:3d}: {r['trades']:4d} trades | "
                  f"P&L: ${r['total_pnl']:+8.2f} | WR: {r['win_rate']:5.1f}%")

    # Kelly fraction sweep
    print("\n  Kelly Fraction (all else = production defaults):")
    for kf in [0.15, 0.25, 0.35, 0.50, 0.65, 0.80]:
        ps = ParamSet(52.0, 100.0, 0.15, 0.85, 50.0, kf)
        r = evaluate_params(bets, ps)
        if r["trades"] > 0:
            print(f"    Kelly {kf:.2f}: {r['trades']:4d} trades | "
                  f"P&L: ${r['total_pnl']:+8.2f} | WR: {r['win_rate']:5.1f}%")

    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
