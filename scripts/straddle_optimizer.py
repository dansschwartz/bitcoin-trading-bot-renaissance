#!/usr/bin/env python3
"""
Straddle Parameter Optimizer v2 — Full Capital Velocity Optimization

Simulates overlapping straddles over 24 hours of continuous BTC price data.
Tests entry frequency, hold time, leg size, stop/trail parameters, and
vol-scaling across ~300 parameter combinations in two phases.

Phase 1: 144 combos testing structure (entry freq, check speed, hold time, trail)
Phase 2: ~150 combos fine-tuning around Phase 1 winner (stop, activation, size, vol)

READ-ONLY analysis — changes nothing in production.

Usage:
    .venv/bin/python3 scripts/straddle_optimizer.py
"""

import argparse
import json
import sys
import time
import statistics
import requests
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

TOTAL_CAPITAL = 1000
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Per-asset vol scaling base (must match config.json vol_scaling_base_vol)
VOL_BASE: dict[str, float] = {
    "BTC": 15.0,
    "ETH": 15.0,
    "SOL": 25.0,
}
ALL_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _paths_for_symbol(symbol: str) -> tuple[Path, Path]:
    """Return (price_cache_path, report_path) for a given symbol."""
    base = symbol.replace("USDT", "").lower()
    return (
        DATA_DIR / f"{base}_24h_1s_cache.json",
        PROJECT_DIR / f"STRADDLE_OPTIMIZATION_REPORT_{base.upper()}.md",
    )

# Current production parameters
CURRENT = {
    "entry_interval": 60, "leg_size": 5.0, "stop_loss": 6,
    "trail_activation": 12, "trail_distance": 15,
    "max_hold": 900, "check_interval": 1, "vol_scaling": "none",
}

# Phase 1 grid: structure search (144 combos)
P1_CHECK = [1, 2, 5]
P1_TRAIL = [1, 2, 3, 4]
P1_ENTRY = [10, 20, 30, 60]
P1_HOLD = [30, 60, 120]

# Phase 2 grid: fine-tuning (variable count based on constraint)
P2_STOP = [3, 4, 5, 6, 7, 8, 10]
P2_ACT = [1, 2, 3, 4]
P2_SIZE = [50, 75, 100]
P2_VOL = ["none", "proportional"]


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    entry_interval: int       # seconds between new straddles
    leg_size: float           # USD per leg
    stop_loss: float          # bps
    trail_activation: float   # bps
    trail_distance: float     # bps
    max_hold: int             # seconds
    check_interval: int       # seconds
    vol_scaling: str          # 'none' or 'proportional'

    def label(self) -> str:
        return (f"entry={self.entry_interval}s/size=${self.leg_size:.0f}/"
                f"stop={self.stop_loss}/act={self.trail_activation}/"
                f"trail={self.trail_distance}/hold={self.max_hold}s/"
                f"chk={self.check_interval}s/vol={self.vol_scaling}")

    def short_label(self) -> str:
        return (f"{self.entry_interval}s/${self.leg_size:.0f}/"
                f"{self.stop_loss}/{self.trail_activation}/{self.trail_distance}/"
                f"{self.max_hold}s/{self.check_interval}s")


@dataclass
class SimResult:
    config: SimConfig
    total_straddles: int = 0
    completed: int = 0
    skipped_capital: int = 0
    net_pnl_usd: float = 0.0
    avg_net_bps: float = 0.0
    win_rate: float = 0.0
    avg_winner_bps: float = 0.0
    avg_loser_bps: float = 0.0
    avg_capture_pct: float = 0.0
    avg_hold_sec: float = 0.0
    max_concurrent: int = 0
    max_drawdown_usd: float = 0.0
    capital_utilization_pct: float = 0.0
    trades_per_hour: float = 0.0
    pnl_per_hour: float = 0.0
    sharpe: float = 0.0
    skip_rate_pct: float = 0.0
    phase: str = ""


# ═══════════════════════════════════════════════════════════════
# STEP 1: FETCH CONTINUOUS BTC PRICE SERIES
# ═══════════════════════════════════════════════════════════════

def fetch_prices(
    hours: int = 24,
    symbol: str = "BTCUSDT",
    cache_path: Optional[Path] = None,
) -> list[tuple[int, float]]:
    """
    Fetch continuous 1s klines from Binance for any symbol.
    Falls back to 1m + linear interpolation if 1s unavailable.
    Caches result locally.
    """
    if cache_path is None:
        cache_path = DATA_DIR / f"{symbol.lower()}_24h_1s_cache.json"

    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) > 1000:
            age_h = (time.time() * 1000 - cached[-1][0]) / 3.6e6
            gap_ms = cached[1][0] - cached[0][0] if len(cached) > 1 else 0
            print(f"Cache: {len(cached):,} ticks, {gap_ms}ms res, {age_h:.1f}h old")
            if age_h < 48:
                return [(int(p[0]), float(p[1])) for p in cached]
            print("Cache stale, re-fetching...")

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - hours * 3_600_000

    for interval, step, label in [("1s", 1000, "1-second"), ("1m", 60_000, "1-minute")]:
        prices: list[tuple[int, float]] = []
        cur = start_ms
        reqs = 0
        print(f"Fetching {hours}h of {label} {symbol} from Binance...")

        while cur < end_ms:
            try:
                r = requests.get(BINANCE_URL, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": min(cur + 999 * step, end_ms),
                    "limit": 1000,
                }, timeout=10)

                if r.status_code == 400 and interval == "1s":
                    print(f"  1s not supported for this range, falling back...")
                    prices = []
                    break
                if r.status_code == 429:
                    print("  Rate limited, waiting 10s...")
                    time.sleep(10)
                    continue
                r.raise_for_status()

                data = r.json()
                if not data:
                    break
                for c in data:
                    prices.append((int(c[0]), float(c[4])))
                cur = int(data[-1][0]) + step
                reqs += 1
                if reqs % 20 == 0:
                    print(f"  {len(prices):,} candles ({reqs} requests)...")
                time.sleep(0.05)

            except requests.exceptions.HTTPError:
                raise
            except Exception as e:
                print(f"  Error: {e}, retrying...")
                time.sleep(1)

        if not prices:
            continue

        print(f"  Done: {len(prices):,} {label} candles in {reqs} requests")

        # Interpolate 1m → 1s if needed
        is_interpolated = False
        if step >= 60_000 and len(prices) > 1:
            print("  Interpolating to 1s resolution...")
            is_interpolated = True
            interp: list[tuple[int, float]] = []
            for i in range(len(prices) - 1):
                t0, p0 = prices[i]
                t1, p1 = prices[i + 1]
                n = int((t1 - t0) / 1000)
                for s in range(n):
                    interp.append((t0 + s * 1000, p0 + (p1 - p0) * s / n))
            interp.append(prices[-1])
            prices = interp
            print(f"  Interpolated: {len(prices):,} 1s ticks")

        with open(cache_path, "w") as f:
            json.dump(prices, f)
        res_label = f"{label} (interpolated to 1s)" if is_interpolated else label
        print(f"  Cached to {cache_path} [{res_label}]")
        return prices

    print(f"ERROR: Could not fetch {symbol} price data from Binance")
    sys.exit(1)


def compute_vols(prices: list[tuple[int, float]], window: int = 60) -> dict[int, float]:
    """
    Compute rolling realized volatility (bps std dev) as a proxy for
    the production vol model. Window = number of 1s returns.
    """
    if len(prices) < window + 2:
        return {}

    closes = np.array([p[1] for p in prices])
    rets = np.diff(closes) / closes[:-1] * 10_000  # returns in bps

    vols: dict[int, float] = {}
    # Rolling std using a sliding window
    for i in range(window, len(rets)):
        vols[prices[i + 1][0]] = float(np.std(rets[i - window:i]))

    return vols


# ═══════════════════════════════════════════════════════════════
# STEP 2: FULL-SYSTEM SIMULATOR
# ═══════════════════════════════════════════════════════════════

def simulate(
    prices: list[tuple[int, float]],
    config: SimConfig,
    vols: Optional[dict[int, float]] = None,
    vol_base: float = 15.0,
) -> SimResult:
    """
    Simulate overlapping straddles over continuous price data.
    Manages capital, concurrent positions, and exit logic per-leg.
    Exactly mirrors straddle_engine.py exit priority:
      stop_loss > trail_stop > timeout
    """
    open_straddles: list[dict] = []
    closed_straddles: list[dict] = []
    straddle_count = 0
    skipped = 0
    last_entry_ms = 0
    last_check_ms = 0
    capital_deployed = 0.0
    max_concurrent = 0
    running_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0
    deploy_samples: list[float] = []

    entry_ms = config.entry_interval * 1000
    check_ms = config.check_interval * 1000
    hold_ms = config.max_hold * 1000
    straddle_cost = config.leg_size * 2

    if not prices:
        return SimResult(config=config)

    for ts, price in prices:
        # ── ENTRY ──
        if (ts - last_entry_ms) >= entry_ms:
            last_entry_ms = ts
            if capital_deployed + straddle_cost > TOTAL_CAPITAL:
                skipped += 1
            else:
                stop = config.stop_loss
                act = config.trail_activation
                dist = config.trail_distance

                if config.vol_scaling == "proportional" and vols:
                    v = vols.get(ts)
                    if v and v > 0:
                        ratio = max(0.3, min(v / vol_base, 3.0))
                        stop *= ratio
                        act *= ratio
                        dist *= ratio

                straddle_count += 1
                s = {
                    "id": straddle_count,
                    "entry_price": price,
                    "entry_ms": ts,
                    "long": {
                        "peak": 0.0, "trail_active": False, "trail_peak": 0.0,
                        "closed": False, "exit_bps": 0.0, "reason": "", "exit_ms": 0,
                        "stop": stop, "act": act, "dist": dist,
                    },
                    "short": {
                        "peak": 0.0, "trail_active": False, "trail_peak": 0.0,
                        "closed": False, "exit_bps": 0.0, "reason": "", "exit_ms": 0,
                        "stop": stop, "act": act, "dist": dist,
                    },
                }
                open_straddles.append(s)
                capital_deployed += straddle_cost

        # ── EXIT CHECK (only at check_interval) ──
        if (ts - last_check_ms) < check_ms:
            continue
        last_check_ms = ts
        deploy_samples.append(capital_deployed)
        max_concurrent = max(max_concurrent, len(open_straddles))

        newly_closed: list[dict] = []
        for s in open_straddles:
            ep = s["entry_price"]
            age = ts - s["entry_ms"]

            for key in ("long", "short"):
                leg = s[key]
                if leg["closed"]:
                    continue

                # P&L in basis points
                if key == "long":
                    pnl = (price - ep) / ep * 10_000
                else:
                    pnl = (ep - price) / ep * 10_000

                # Track peak favorable
                if pnl > leg["peak"]:
                    leg["peak"] = pnl

                # 1) Stop loss (highest priority, inclusive <=)
                if pnl <= -leg["stop"]:
                    leg["closed"] = True
                    leg["exit_bps"] = pnl
                    leg["reason"] = "stop"
                    leg["exit_ms"] = ts
                    continue

                # 2) Trail activation
                if not leg["trail_active"] and pnl >= leg["act"]:
                    leg["trail_active"] = True
                    leg["trail_peak"] = pnl

                # Trail peak update
                if leg["trail_active"] and pnl > leg["trail_peak"]:
                    leg["trail_peak"] = pnl

                # 3) Trail stop
                if leg["trail_active"] and (leg["trail_peak"] - pnl) >= leg["dist"]:
                    leg["closed"] = True
                    leg["exit_bps"] = pnl
                    leg["reason"] = "trail"
                    leg["exit_ms"] = ts
                    continue

                # 4) Timeout (lowest priority)
                if age >= hold_ms:
                    leg["closed"] = True
                    leg["exit_bps"] = pnl
                    leg["reason"] = "timeout"
                    leg["exit_ms"] = ts

            # Both legs closed?
            if s["long"]["closed"] and s["short"]["closed"]:
                s["net_bps"] = s["long"]["exit_bps"] + s["short"]["exit_bps"]
                s["net_usd"] = s["net_bps"] / 10_000 * config.leg_size
                newly_closed.append(s)
                running_pnl += s["net_usd"]
                capital_deployed -= straddle_cost
                if running_pnl > peak_pnl:
                    peak_pnl = running_pnl
                dd = peak_pnl - running_pnl
                if dd > max_dd:
                    max_dd = dd

        for s in newly_closed:
            open_straddles.remove(s)
            closed_straddles.append(s)

    # Force-close remaining at simulation end
    if prices and open_straddles:
        fp = prices[-1][1]
        fts = prices[-1][0]
        for s in open_straddles:
            for key in ("long", "short"):
                leg = s[key]
                if not leg["closed"]:
                    if key == "long":
                        leg["exit_bps"] = (fp - s["entry_price"]) / s["entry_price"] * 10_000
                    else:
                        leg["exit_bps"] = (s["entry_price"] - fp) / s["entry_price"] * 10_000
                    leg["closed"] = True
                    leg["reason"] = "sim_end"
                    leg["exit_ms"] = fts
            s["net_bps"] = s["long"]["exit_bps"] + s["short"]["exit_bps"]
            s["net_usd"] = s["net_bps"] / 10_000 * config.leg_size
            closed_straddles.append(s)

    # ── AGGREGATE ──
    if not closed_straddles:
        return SimResult(config=config)

    nets = [s["net_bps"] for s in closed_straddles]
    usds = [s["net_usd"] for s in closed_straddles]
    winners = [s for s in closed_straddles if s["net_bps"] > 0]
    losers = [s for s in closed_straddles if s["net_bps"] <= 0]

    # Winner capture: how much of peak the winner retained at exit
    captures: list[float] = []
    for s in winners:
        w_peak = max(s["long"]["peak"], s["short"]["peak"])
        w_exit = max(s["long"]["exit_bps"], s["short"]["exit_bps"])
        if w_peak > 0:
            captures.append(w_exit / w_peak * 100)

    # Hold times (seconds)
    holds: list[float] = []
    for s in closed_straddles:
        h = max(
            (s["long"]["exit_ms"] - s["entry_ms"]) / 1000 if s["long"]["exit_ms"] else 0,
            (s["short"]["exit_ms"] - s["entry_ms"]) / 1000 if s["short"]["exit_ms"] else 0,
        )
        holds.append(h)

    hrs = (prices[-1][0] - prices[0][0]) / 3.6e6
    if hrs <= 0:
        hrs = 1

    # Sharpe ratio (annualized from per-trade P&L)
    sharpe = 0.0
    if len(nets) > 1:
        mn = statistics.mean(nets)
        sd = statistics.stdev(nets)
        if sd > 0:
            trades_per_year = len(closed_straddles) / hrs * 8760
            sharpe = (mn / sd) * (trades_per_year ** 0.5)

    total_entry_attempts = straddle_count + skipped
    skip_pct = (skipped / total_entry_attempts * 100) if total_entry_attempts > 0 else 0

    return SimResult(
        config=config,
        total_straddles=straddle_count,
        completed=len(closed_straddles),
        skipped_capital=skipped,
        net_pnl_usd=round(sum(usds), 4),
        avg_net_bps=round(statistics.mean(nets), 4),
        win_rate=round(len(winners) / len(closed_straddles) * 100, 1),
        avg_winner_bps=round(statistics.mean([w["net_bps"] for w in winners]), 2) if winners else 0,
        avg_loser_bps=round(statistics.mean([l["net_bps"] for l in losers]), 2) if losers else 0,
        avg_capture_pct=round(statistics.mean(captures), 1) if captures else 0,
        avg_hold_sec=round(statistics.mean(holds), 1),
        max_concurrent=max_concurrent,
        max_drawdown_usd=round(max_dd, 4),
        capital_utilization_pct=round(
            statistics.mean(deploy_samples) / TOTAL_CAPITAL * 100, 1
        ) if deploy_samples else 0,
        trades_per_hour=round(len(closed_straddles) / hrs, 1),
        pnl_per_hour=round(sum(usds) / hrs, 4),
        sharpe=round(sharpe, 1),
        skip_rate_pct=round(skip_pct, 1),
    )


# ═══════════════════════════════════════════════════════════════
# STEP 3: TWO-PHASE GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def build_phase_1() -> list[SimConfig]:
    """Phase 1: structural search — 144 combos."""
    configs: list[SimConfig] = []
    for check in P1_CHECK:
        for trail in P1_TRAIL:
            for entry in P1_ENTRY:
                for hold in P1_HOLD:
                    configs.append(SimConfig(
                        entry_interval=entry, leg_size=100, stop_loss=6,
                        trail_activation=2, trail_distance=trail,
                        max_hold=hold, check_interval=check, vol_scaling="none",
                    ))
    return configs


def build_phase_2(winner: SimConfig) -> list[SimConfig]:
    """Phase 2: fine-tune stop/act/size/vol around Phase 1 winner."""
    configs: list[SimConfig] = []
    for stop in P2_STOP:
        for act in P2_ACT:
            if act >= stop:
                continue
            for size in P2_SIZE:
                for vol in P2_VOL:
                    configs.append(SimConfig(
                        entry_interval=winner.entry_interval,
                        leg_size=size,
                        stop_loss=stop,
                        trail_activation=act,
                        trail_distance=winner.trail_distance,
                        max_hold=winner.max_hold,
                        check_interval=winner.check_interval,
                        vol_scaling=vol,
                    ))
    return configs


def run_batch(
    label: str,
    configs: list[SimConfig],
    prices: list[tuple[int, float]],
    vols: dict[int, float],
    phase_tag: str,
    vol_base: float = 15.0,
) -> list[SimResult]:
    """Run a batch of simulations with progress reporting."""
    print(f"\n{'='*60}")
    print(f"  {label}: {len(configs)} configurations")
    print(f"{'='*60}")

    results: list[SimResult] = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        result = simulate(prices, cfg, vols, vol_base=vol_base)
        result.phase = phase_tag
        results.append(result)

        if (i + 1) % 25 == 0 or (i + 1) == len(configs):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(configs) - i - 1) / rate if rate > 0 else 0
            best_so_far = max(results, key=lambda r: r.pnl_per_hour)
            print(f"  [{i+1}/{len(configs)}] {rate:.1f}/s ETA {eta:.0f}s | "
                  f"best so far: ${best_so_far.pnl_per_hour:.4f}/hr "
                  f"({best_so_far.config.short_label()})")

    elapsed = time.time() - t0
    print(f"  Complete in {elapsed:.1f}s")

    results.sort(key=lambda r: r.pnl_per_hour, reverse=True)
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 4: REPORT GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_report(
    p1_results: list[SimResult],
    p2_results: list[SimResult],
    current_result: SimResult,
    all_results: list[SimResult],
    prices: list[tuple[int, float]],
    data_resolution: str,
    symbol: str = "BTCUSDT",
) -> str:
    """Generate comprehensive markdown report."""
    L: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    hrs = (prices[-1][0] - prices[0][0]) / 3.6e6
    price_range = (min(p[1] for p in prices), max(p[1] for p in prices))
    best = all_results[0]
    asset = symbol.replace("USDT", "")

    L.append(f"# {asset} Straddle Parameter Optimization Report v2")
    L.append("## Full Capital Velocity Analysis")
    L.append("")
    L.append(f"**Generated**: {now}")
    L.append(f"**Asset**: {symbol}")
    L.append(f"**Data**: {len(prices):,} price ticks, {hrs:.1f} hours, {data_resolution}")
    L.append(f"**Price range**: ${price_range[0]:,.2f} – ${price_range[1]:,.2f}")
    L.append(f"**Base capital**: ${TOTAL_CAPITAL:,}")
    L.append(f"**Phase 1 combos**: {len(p1_results)} (structure)")
    L.append(f"**Phase 2 combos**: {len(p2_results)} (fine-tuning)")
    L.append(f"**Total simulations**: {len(all_results)}")
    L.append("")

    # ── Section 1: Top 20 Overall ──
    L.append("## 1. Top 20 Parameter Combinations (by P&L/hour)")
    L.append("")
    L.append("| # | Entry | Size | Stop | Act | Trail | Hold | Chk | Vol | "
             "P&L/hr | Total $ | Trades/hr | Win% | Capture | Max DD | Util% | Skip% | Sharpe |")
    L.append("|---|-------|------|------|-----|-------|------|-----|-----|"
             "--------|---------|-----------|------|---------|--------|-------|-------|--------|")
    for i, r in enumerate(all_results[:20]):
        c = r.config
        L.append(
            f"| {i+1} | {c.entry_interval}s | ${c.leg_size:.0f} | {c.stop_loss} | "
            f"{c.trail_activation} | {c.trail_distance} | {c.max_hold}s | {c.check_interval}s | "
            f"{c.vol_scaling[0]} | ${r.pnl_per_hour:.4f} | ${r.net_pnl_usd:.2f} | "
            f"{r.trades_per_hour:.0f} | {r.win_rate:.0f}% | {r.avg_capture_pct:.0f}% | "
            f"${r.max_drawdown_usd:.2f} | {r.capital_utilization_pct:.0f}% | "
            f"{r.skip_rate_pct:.0f}% | {r.sharpe:.0f} |"
        )
    L.append("")

    # ── Section 2: Capital Velocity Analysis ──
    L.append("## 2. Capital Velocity Analysis")
    L.append("")
    L.append("Best result per entry interval:")
    L.append("")
    L.append("| Entry | Best Config | Trades/hr | P&L/hr | Total $ | "
             "Concurrent | Utilization | Skip Rate |")
    L.append("|-------|-------------|-----------|--------|---------|"
             "------------|-------------|-----------|")
    seen_entries = set()
    for r in all_results:
        ei = r.config.entry_interval
        if ei in seen_entries:
            continue
        seen_entries.add(ei)
        c = r.config
        L.append(
            f"| {ei}s | stop={c.stop_loss}/act={c.trail_activation}/"
            f"trail={c.trail_distance}/hold={c.max_hold}s | "
            f"{r.trades_per_hour:.0f} | ${r.pnl_per_hour:.4f} | ${r.net_pnl_usd:.2f} | "
            f"{r.max_concurrent} | {r.capital_utilization_pct:.0f}% | {r.skip_rate_pct:.0f}% |"
        )
    L.append("")

    # Throughput calculation for best
    if best.trades_per_hour > 0:
        notional_per_day = best.trades_per_hour * 24 * best.config.leg_size
        capital_turns = notional_per_day / TOTAL_CAPITAL
        L.append(f"**Best config throughput**: {best.trades_per_hour:.0f} trades/hr "
                 f"x ${best.config.leg_size:.0f}/leg x 24hr = "
                 f"**${notional_per_day:,.0f} daily notional** on ${TOTAL_CAPITAL:,} capital "
                 f"({capital_turns:.0f}x daily turnover)")
    L.append("")

    # ── Section 3: Hold Time Impact ──
    L.append("## 3. Hold Time Impact")
    L.append("")
    L.append("| Hold Time | Avg P&L/hr | Avg Win% | Avg Trades/hr | Avg Hold (actual) |")
    L.append("|-----------|------------|----------|---------------|-------------------|")
    for hold in sorted(set(r.config.max_hold for r in all_results)):
        subset = [r for r in all_results if r.config.max_hold == hold]
        if subset:
            avg_pnl = statistics.mean(r.pnl_per_hour for r in subset)
            avg_wr = statistics.mean(r.win_rate for r in subset)
            avg_tph = statistics.mean(r.trades_per_hour for r in subset)
            avg_actual = statistics.mean(r.avg_hold_sec for r in subset)
            L.append(f"| {hold}s | ${avg_pnl:.4f} | {avg_wr:.1f}% | "
                     f"{avg_tph:.0f} | {avg_actual:.1f}s |")
    L.append("")

    # ── Section 4: Check Interval Impact ──
    L.append("## 4. Check Interval Impact")
    L.append("")
    L.append("| Interval | Best P&L/hr | Best Config | Avg P&L/hr |")
    L.append("|----------|------------|-------------|------------|")
    for chk in sorted(set(r.config.check_interval for r in all_results)):
        subset = [r for r in all_results if r.config.check_interval == chk]
        if subset:
            subset.sort(key=lambda r: r.pnl_per_hour, reverse=True)
            best_chk = subset[0]
            avg_pnl = statistics.mean(r.pnl_per_hour for r in subset)
            c = best_chk.config
            L.append(
                f"| {chk}s | ${best_chk.pnl_per_hour:.4f} | "
                f"stop={c.stop_loss}/act={c.trail_activation}/trail={c.trail_distance}/"
                f"hold={c.max_hold}s | ${avg_pnl:.4f} |"
            )
    L.append("")

    # ── Section 5: Vol-Scaling Comparison ──
    L.append("## 5. Vol-Scaling Comparison")
    L.append("")
    vol_none = [r for r in p2_results if r.config.vol_scaling == "none"]
    vol_prop = [r for r in p2_results if r.config.vol_scaling == "proportional"]

    L.append("| Metric | Fixed | Proportional |")
    L.append("|--------|-------|--------------|")
    if vol_none and vol_prop:
        for metric_name, getter in [
            ("Best P&L/hr", lambda rs: f"${max(r.pnl_per_hour for r in rs):.4f}"),
            ("Avg P&L/hr", lambda rs: f"${statistics.mean(r.pnl_per_hour for r in rs):.4f}"),
            ("Best Win%", lambda rs: f"{max(r.win_rate for r in rs):.1f}%"),
            ("Avg Win%", lambda rs: f"{statistics.mean(r.win_rate for r in rs):.1f}%"),
            ("Best Capture%", lambda rs: f"{max(r.avg_capture_pct for r in rs):.0f}%"),
            ("Combos tested", lambda rs: str(len(rs))),
        ]:
            L.append(f"| {metric_name} | {getter(vol_none)} | {getter(vol_prop)} |")
    elif not vol_prop:
        L.append("| Note | All Phase 2 results | No proportional combos |")
    L.append("")

    # ── Section 6: Sensitivity Tables ──
    L.append("## 6. Parameter Sensitivity")
    L.append("")

    sensitivity_params = [
        ("Stop Loss", "stop_loss", P2_STOP, p2_results),
        ("Trail Activation", "trail_activation", P2_ACT, p2_results),
        ("Trail Distance", "trail_distance", P1_TRAIL, p1_results),
        ("Entry Interval", "entry_interval", P1_ENTRY, p1_results),
        ("Max Hold", "max_hold", P1_HOLD, p1_results),
        ("Check Interval", "check_interval", P1_CHECK, p1_results),
        ("Leg Size", "leg_size", P2_SIZE, p2_results),
    ]

    for param_label, attr, values, source in sensitivity_params:
        L.append(f"### {param_label}")
        L.append("")
        L.append(f"| Value | Avg P&L/hr | Avg Win% | Count |")
        L.append(f"|-------|------------|----------|-------|")
        for v in sorted(values):
            subset = [r for r in source if getattr(r.config, attr) == v]
            if subset:
                avg_pnl = statistics.mean(r.pnl_per_hour for r in subset)
                avg_wr = statistics.mean(r.win_rate for r in subset)
                val_label = f"${v:.0f}" if attr == "leg_size" else (f"{v}s" if attr in ("entry_interval", "max_hold", "check_interval") else str(v))
                L.append(f"| {val_label} | ${avg_pnl:.4f} | {avg_wr:.1f}% | {len(subset)} |")
        L.append("")

    # ── Section 7: Cluster Analysis ──
    L.append("## 7. Cluster Analysis — Top 10")
    L.append("")
    top10 = all_results[:10]
    if len(top10) >= 10:
        for attr, label in [
            ("entry_interval", "Entry interval"),
            ("leg_size", "Leg size"),
            ("stop_loss", "Stop loss"),
            ("trail_activation", "Trail activation"),
            ("trail_distance", "Trail distance"),
            ("max_hold", "Max hold"),
            ("check_interval", "Check interval"),
            ("vol_scaling", "Vol scaling"),
        ]:
            vals = [getattr(r.config, attr) for r in top10]
            if isinstance(vals[0], str):
                mode = max(set(vals), key=vals.count)
                L.append(f"- **{label}**: values={sorted(set(vals))}, mode={mode}")
            else:
                mode = max(set(vals), key=vals.count)
                spread = max(vals) - min(vals)
                L.append(f"- **{label}**: min={min(vals)}, max={max(vals)}, "
                         f"mode={mode}, spread={spread}")

        # Verdict
        numeric_attrs = ["stop_loss", "trail_activation", "trail_distance",
                         "entry_interval", "max_hold", "check_interval"]
        spreads = []
        for attr in numeric_attrs:
            vals = [getattr(r.config, attr) for r in top10]
            unique = len(set(vals))
            spreads.append(unique)

        avg_unique = statistics.mean(spreads)
        L.append("")
        if avg_unique <= 2:
            L.append("**Verdict**: Top 10 cluster tightly — HIGH confidence in optimal region.")
        elif avg_unique <= 3:
            L.append("**Verdict**: Top 10 show moderate clustering — GOOD confidence.")
        else:
            L.append("**Verdict**: Top 10 are dispersed — optimal region is broad/uncertain.")
    L.append("")

    # ── Section 8: Recommendation ──
    L.append("## 8. Recommendation")
    L.append("")

    # Robust: median of top 5
    top5 = all_results[:5]
    robust = SimConfig(
        entry_interval=sorted(r.config.entry_interval for r in top5)[2],
        leg_size=sorted(r.config.leg_size for r in top5)[2],
        stop_loss=sorted(r.config.stop_loss for r in top5)[2],
        trail_activation=sorted(r.config.trail_activation for r in top5)[2],
        trail_distance=sorted(r.config.trail_distance for r in top5)[2],
        max_hold=sorted(r.config.max_hold for r in top5)[2],
        check_interval=sorted(r.config.check_interval for r in top5)[2],
        vol_scaling=max(set(r.config.vol_scaling for r in top5),
                        key=[r.config.vol_scaling for r in top5].count),
    )

    L.append("### Robust (median of top 5)")
    L.append(f"- `entry_interval`: **{robust.entry_interval}s**")
    L.append(f"- `leg_size`: **${robust.leg_size:.0f}**")
    L.append(f"- `stop_loss_bps`: **{robust.stop_loss}**")
    L.append(f"- `trail_activation_bps`: **{robust.trail_activation}**")
    L.append(f"- `trail_distance_bps`: **{robust.trail_distance}**")
    L.append(f"- `max_hold_seconds`: **{robust.max_hold}s**")
    L.append(f"- `check_interval`: **{robust.check_interval}s**")
    L.append(f"- `vol_scaling`: **{robust.vol_scaling}**")
    L.append("")

    bc = best.config
    L.append("### Aggressive (absolute best)")
    L.append(f"- `entry_interval`: **{bc.entry_interval}s**")
    L.append(f"- `leg_size`: **${bc.leg_size:.0f}**")
    L.append(f"- `stop_loss_bps`: **{bc.stop_loss}**")
    L.append(f"- `trail_activation_bps`: **{bc.trail_activation}**")
    L.append(f"- `trail_distance_bps`: **{bc.trail_distance}**")
    L.append(f"- `max_hold_seconds`: **{bc.max_hold}s**")
    L.append(f"- `check_interval`: **{bc.check_interval}s**")
    L.append(f"- `vol_scaling`: **{bc.vol_scaling}**")
    L.append(f"- P&L/hour: **${best.pnl_per_hour:.4f}**")
    L.append(f"- Win rate: **{best.win_rate:.0f}%**")
    L.append(f"- Sharpe: **{best.sharpe:.0f}**")
    L.append("")

    # Current comparison
    L.append("### Current vs Recommended")
    L.append("")
    L.append("| Parameter | Current | Robust | Aggressive |")
    L.append("|-----------|---------|--------|------------|")
    param_map = [
        ("entry_interval", "entry_interval", "s"),
        ("leg_size", "leg_size", "$"),
        ("stop_loss", "stop_loss", " bps"),
        ("trail_activation", "trail_activation", " bps"),
        ("trail_distance", "trail_distance", " bps"),
        ("max_hold", "max_hold", "s"),
        ("check_interval", "check_interval", "s"),
        ("vol_scaling", "vol_scaling", ""),
    ]
    for display_key, attr, suffix in param_map:
        cur_val = CURRENT[display_key]
        rob_val = getattr(robust, attr)
        agg_val = getattr(bc, attr)
        if isinstance(cur_val, str):
            L.append(f"| {display_key} | {cur_val} | {rob_val} | {agg_val} |")
        elif suffix == "$":
            L.append(f"| {display_key} | ${cur_val:.0f} | ${rob_val:.0f} | ${agg_val:.0f} |")
        else:
            L.append(f"| {display_key} | {cur_val}{suffix} | {rob_val}{suffix} | {agg_val}{suffix} |")
    L.append("")

    # Performance comparison
    L.append("| Metric | Current | Robust (est.) | Aggressive |")
    L.append("|--------|---------|---------------|------------|")
    L.append(f"| P&L/hour | ${current_result.pnl_per_hour:.4f} | — | ${best.pnl_per_hour:.4f} |")
    L.append(f"| Total P&L (24h) | ${current_result.net_pnl_usd:.2f} | — | ${best.net_pnl_usd:.2f} |")
    L.append(f"| Win rate | {current_result.win_rate:.0f}% | — | {best.win_rate:.0f}% |")
    L.append(f"| Trades/hour | {current_result.trades_per_hour:.0f} | — | {best.trades_per_hour:.0f} |")
    L.append(f"| Max drawdown | ${current_result.max_drawdown_usd:.2f} | — | ${best.max_drawdown_usd:.2f} |")
    L.append(f"| Capital util | {current_result.capital_utilization_pct:.0f}% | — | {best.capital_utilization_pct:.0f}% |")
    L.append("")

    # ── Section 9: Projections ──
    L.append("## 9. Projected Revenue at $1,000 Capital")
    L.append("")
    L.append("Based on best configuration over the test period:")
    L.append("")
    daily = best.pnl_per_hour * 24
    monthly = daily * 30
    annual = daily * 365
    daily_pct = daily / TOTAL_CAPITAL * 100
    monthly_pct = monthly / TOTAL_CAPITAL * 100
    annual_pct = annual / TOTAL_CAPITAL * 100

    L.append(f"| Period | P&L | Return |")
    L.append(f"|--------|-----|--------|")
    L.append(f"| Daily | ${daily:.2f} | {daily_pct:.2f}% |")
    L.append(f"| Monthly | ${monthly:.2f} | {monthly_pct:.1f}% |")
    L.append(f"| Annual (simple) | ${annual:.2f} | {annual_pct:.0f}% |")
    L.append("")

    L.append("**Caveats**:")
    L.append("- Projections assume market conditions similar to test period")
    L.append("- No execution issues (slippage, API latency, downtime)")
    L.append("- $0 maker fees (MEXC LIMIT_MAKER orders)")
    L.append("- Single 24h period — needs multi-day validation")
    if "interpolated" in data_resolution.lower():
        L.append("- Data was interpolated from 1-minute candles (misses intra-minute spikes)")
    L.append("")

    # ── Bottom 5 ──
    L.append("## Appendix: Bottom 5 Configurations")
    L.append("")
    L.append("| # | Config | P&L/hr | Total $ | Win% |")
    L.append("|---|--------|--------|---------|------|")
    for i, r in enumerate(all_results[-5:]):
        rank = len(all_results) - 4 + i
        L.append(f"| {rank} | {r.config.short_label()} | ${r.pnl_per_hour:.4f} | "
                 f"${r.net_pnl_usd:.2f} | {r.win_rate:.0f}% |")
    L.append("")

    L.append("---")
    L.append("")
    L.append("*Generated by `scripts/straddle_optimizer.py` v2 — READ-ONLY analysis, "
             "no production changes.*")
    L.append("")
    L.append('*"Every dollar has a clock. The faster it recycles through winning '
             'trades, the harder your capital works."*')

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_optimizer(symbol: str, hours: int = 24) -> None:
    """Run full optimization for a single symbol."""
    cache_path, report_path = _paths_for_symbol(symbol)
    asset = symbol.replace("USDT", "")
    vol_base = VOL_BASE.get(asset, 15.0)

    print("=" * 60)
    print(f"  Straddle Parameter Optimizer v2 — {asset}")
    print("  Full Capital Velocity Analysis")
    print("=" * 60)
    print(f"  Symbol: {symbol}")
    print(f"  Base capital: ${TOTAL_CAPITAL:,}")
    print(f"  Vol scaling base: {vol_base} bps")
    print()

    # ── Step 1: Fetch prices ──
    prices = fetch_prices(hours=hours, symbol=symbol, cache_path=cache_path)
    hrs = (prices[-1][0] - prices[0][0]) / 3.6e6
    gap_ms = prices[1][0] - prices[0][0] if len(prices) > 1 else 0
    data_resolution = f"{'1-second native' if gap_ms <= 1000 else f'{gap_ms}ms (interpolated to 1s)'}"
    price_min = min(p[1] for p in prices)
    price_max = max(p[1] for p in prices)
    print(f"  {len(prices):,} ticks, {hrs:.1f}h, ${price_min:,.2f}–${price_max:,.2f}")

    # ── Step 2: Compute vol predictions ──
    print("\nComputing rolling volatility (60s window)...")
    vols = compute_vols(prices, window=60)
    if vols:
        vol_vals = list(vols.values())
        print(f"  {len(vols):,} vol predictions, "
              f"mean={statistics.mean(vol_vals):.2f} bps, "
              f"std={statistics.stdev(vol_vals):.2f} bps")

    # ── Step 3: Simulate current production params ──
    print("\nSimulating current production parameters...")
    current_cfg = SimConfig(
        entry_interval=CURRENT["entry_interval"],
        leg_size=CURRENT["leg_size"],
        stop_loss=CURRENT["stop_loss"],
        trail_activation=CURRENT["trail_activation"],
        trail_distance=CURRENT["trail_distance"],
        max_hold=CURRENT["max_hold"],
        check_interval=CURRENT["check_interval"],
        vol_scaling=CURRENT["vol_scaling"],
    )
    current_result = simulate(prices, current_cfg, vols, vol_base=vol_base)
    print(f"  Current: {current_result.completed} straddles, "
          f"${current_result.net_pnl_usd:.2f} total, "
          f"${current_result.pnl_per_hour:.4f}/hr, "
          f"{current_result.win_rate:.0f}% WR")

    # ── Step 4: Phase 1 — Structure ──
    p1_configs = build_phase_1()
    p1_results = run_batch("PHASE 1: Structure Search", p1_configs, prices, vols, "P1",
                           vol_base=vol_base)

    p1_best = p1_results[0]
    print(f"\n  Phase 1 winner: {p1_best.config.label()}")
    print(f"  P&L/hr: ${p1_best.pnl_per_hour:.4f}, "
          f"{p1_best.trades_per_hour:.0f} trades/hr, "
          f"{p1_best.win_rate:.0f}% WR")

    # ── Step 5: Phase 2 — Fine-tune ──
    p2_configs = build_phase_2(p1_best.config)
    p2_results = run_batch("PHASE 2: Fine-Tuning", p2_configs, prices, vols, "P2",
                           vol_base=vol_base)

    if p2_results:
        p2_best = p2_results[0]
        print(f"\n  Phase 2 winner: {p2_best.config.label()}")
        print(f"  P&L/hr: ${p2_best.pnl_per_hour:.4f}, "
              f"{p2_best.trades_per_hour:.0f} trades/hr, "
              f"{p2_best.win_rate:.0f}% WR")

    # ── Combine and rank ──
    all_results = sorted(
        p1_results + p2_results,
        key=lambda r: r.pnl_per_hour,
        reverse=True,
    )

    # ── Step 6: Generate report ──
    print("\nGenerating report...")
    report = generate_report(
        p1_results, p2_results, current_result, all_results, prices,
        data_resolution, symbol=symbol,
    )
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # ── Quick summary ──
    best = all_results[0]
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY — {asset}")
    print(f"{'='*60}")
    print(f"  Best config:     {best.config.label()}")
    print(f"  P&L/hour:        ${best.pnl_per_hour:.4f}")
    print(f"  Total P&L (24h): ${best.net_pnl_usd:.2f}")
    print(f"  Win rate:         {best.win_rate:.0f}%")
    print(f"  Trades/hour:      {best.trades_per_hour:.0f}")
    print(f"  Max concurrent:   {best.max_concurrent}")
    print(f"  Capital util:     {best.capital_utilization_pct:.0f}%")
    print(f"  Max drawdown:     ${best.max_drawdown_usd:.2f}")
    print(f"  Sharpe:           {best.sharpe:.0f}")
    print()
    print(f"  Current config:   ${current_result.pnl_per_hour:.4f}/hr "
          f"({current_result.win_rate:.0f}% WR)")
    improvement = best.pnl_per_hour - current_result.pnl_per_hour
    if current_result.pnl_per_hour != 0:
        improvement_pct = improvement / abs(current_result.pnl_per_hour) * 100
        print(f"  Improvement:      ${improvement:.4f}/hr ({improvement_pct:+.0f}%)")
    else:
        print(f"  Improvement:      ${improvement:.4f}/hr")
    print()
    daily = best.pnl_per_hour * 24
    print(f"  Projected daily:  ${daily:.2f} ({daily/TOTAL_CAPITAL*100:.2f}%)")
    print(f"  Projected monthly: ${daily*30:.2f} ({daily*30/TOTAL_CAPITAL*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Straddle Parameter Optimizer v2")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Binance symbol (e.g. BTCUSDT, ETHUSDT, SOLUSDT)")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to fetch")
    parser.add_argument("--all", action="store_true",
                        help="Run optimizer for all assets (BTC, ETH, SOL)")
    args = parser.parse_args()

    if args.all:
        for sym in ALL_SYMBOLS:
            run_optimizer(sym, hours=args.hours)
            print("\n\n")
    else:
        run_optimizer(args.symbol.upper(), hours=args.hours)


if __name__ == "__main__":
    main()
