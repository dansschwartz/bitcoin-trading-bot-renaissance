"""Systems Engineer's knowledge atoms — execution, costs, reliability."""
import numpy as np
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects


def kyle_lambda(volume: float, price_impact: float) -> dict:
    """Kyle's lambda: price impact per unit of volume.
    Lambda = delta_price / volume. Lower is better (more liquid).
    For our pairs: lambda ~ 0.001-0.01 per $1K traded."""
    lam = price_impact / max(volume, 1e-12)
    return {"lambda": round(lam, 8), "impact_per_1k": round(lam * 1000, 6),
            "liquid": lam < 0.001, "illiquid": lam > 0.01}


def implementation_shortfall(decision_price: float, exec_price: float, side: str) -> dict:
    """Implementation Shortfall = gap between decision price and execution price.
    IS = (exec_price - decision_price) / decision_price for buys.
    IS = (decision_price - exec_price) / decision_price for sells.
    Positive IS = cost. Typical for crypto: 1-5 bps."""
    if side.lower() in ("buy", "long"):
        is_pct = (exec_price - decision_price) / max(decision_price, 1e-12)
    else:
        is_pct = (decision_price - exec_price) / max(decision_price, 1e-12)
    return {"is_pct": round(is_pct, 6), "is_bps": round(is_pct * 10000, 2),
            "favorable": is_pct < 0, "acceptable": abs(is_pct) < 0.0005}


def devil_decomposition(entries) -> dict:
    """Decompose Devil Tracker costs into components.
    Components: slippage, fees, adverse selection, timing cost.
    Renaissance found Devil eats 50-75% of gross edge."""
    if not entries:
        return {"error": "no entries"}
    total_cost = sum(e.get("devil_cost", 0) for e in entries)
    total_slippage = sum(e.get("slippage_cost", 0) for e in entries)
    total_fees = sum(e.get("fee_cost", 0) for e in entries)
    total_adverse = sum(e.get("adverse_selection", 0) for e in entries)
    n = len(entries)
    return {"total_cost": round(total_cost, 4),
            "components": {
                "slippage": round(total_slippage, 4),
                "fees": round(total_fees, 4),
                "adverse_selection": round(total_adverse, 4),
                "other": round(total_cost - total_slippage - total_fees - total_adverse, 4),
            },
            "avg_cost_per_trade": round(total_cost / max(n, 1), 4),
            "n_trades": n}


def queue_utilization(request_rate: float, service_rate: float) -> dict:
    """M/M/1 queue model + Little's Law for system capacity.
    rho = lambda/mu. rho > 0.8 = degraded. rho > 0.95 = critical.
    Avg wait = rho / (mu * (1-rho))."""
    rho = request_rate / max(service_rate, 1e-12)
    if rho >= 1:
        return {"rho": round(rho, 4), "status": "overloaded",
                "avg_wait": float('inf'), "avg_queue": float('inf')}
    avg_wait = rho / (service_rate * (1 - rho))
    avg_queue = rho**2 / (1 - rho)
    return {"rho": round(rho, 4),
            "avg_wait_sec": round(avg_wait, 4),
            "avg_queue_length": round(avg_queue, 2),
            "status": "critical" if rho > 0.95 else "degraded" if rho > 0.8 else "healthy",
            "headroom_pct": round((1 - rho) * 100, 1)}


def slippage_simulator(mid_price: float, spread_bps: float, size_usd: float,
                       daily_volume_usd: float = 1_000_000) -> dict:
    """Realistic slippage model for paper trading.
    Components: half-spread + market impact (sqrt model).
    Impact = k * sqrt(size / daily_volume). k ~ 0.1 for crypto."""
    half_spread_cost = mid_price * spread_bps / 10000 / 2
    participation = size_usd / max(daily_volume_usd, 1)
    impact_bps = 0.1 * np.sqrt(participation) * 10000
    total_bps = spread_bps / 2 + impact_bps
    total_cost = size_usd * total_bps / 10000
    return {"half_spread_bps": round(spread_bps / 2, 2),
            "impact_bps": round(impact_bps, 2),
            "total_slippage_bps": round(total_bps, 2),
            "total_cost_usd": round(total_cost, 4),
            "participation_rate": round(participation * 100, 4)}


def mtbf_estimate(error_timestamps) -> dict:
    """Mean Time Between Failures from error log timestamps.
    MTBF < 1h is unacceptable. Target: MTBF > 24h."""
    ts = sorted(error_timestamps)
    if len(ts) < 2:
        return {"mtbf_hours": float('inf'), "n_failures": len(ts)}
    intervals = np.diff(ts)
    if hasattr(intervals[0], 'total_seconds'):
        intervals = np.array([i.total_seconds() / 3600 for i in intervals])
    else:
        intervals = np.array(intervals, dtype=float) / 3600
    mtbf = float(np.mean(intervals))
    return {"mtbf_hours": round(mtbf, 2), "n_failures": len(ts),
            "min_interval_hours": round(float(np.min(intervals)), 2),
            "max_interval_hours": round(float(np.max(intervals)), 2),
            "acceptable": mtbf > 1.0, "good": mtbf > 24.0}


def fmea_rpn(severity: int, probability: int, detection: int) -> dict:
    """Failure Mode and Effects Analysis — Risk Priority Number.
    RPN = S * P * D. Scale 1-10 each. RPN > 100: immediate action.
    RPN > 200: critical. Max = 1000."""
    rpn = severity * probability * detection
    return {"rpn": rpn, "severity": severity, "probability": probability,
            "detection": detection,
            "priority": "critical" if rpn > 200 else "high" if rpn > 100 else "medium" if rpn > 50 else "low"}


KB.register_many([
    Atom("sys.kyle_lambda", "Kyle's Lambda", "systems_engineer",
         kyle_lambda.__doc__, atype=AType.FORMULA, affects=Affects.EXECUTION|Affects.COST, formula=kyle_lambda),
    Atom("sys.implementation_shortfall", "Implementation Shortfall", "systems_engineer",
         implementation_shortfall.__doc__, atype=AType.FORMULA, affects=Affects.EXECUTION|Affects.COST, formula=implementation_shortfall),
    Atom("sys.devil_decomposition", "Devil Cost Decomposition", "systems_engineer",
         devil_decomposition.__doc__, atype=AType.FORMULA, affects=Affects.COST, formula=devil_decomposition),
    Atom("sys.queue_utilization", "Queue/Capacity Model", "systems_engineer",
         queue_utilization.__doc__, atype=AType.FORMULA, affects=Affects.EXECUTION, formula=queue_utilization),
    Atom("sys.slippage_simulator", "Slippage Simulator", "systems_engineer",
         slippage_simulator.__doc__, atype=AType.FORMULA, affects=Affects.COST|Affects.EXECUTION, formula=slippage_simulator,
         crypto_specific=True),
    Atom("sys.mtbf", "Mean Time Between Failures", "systems_engineer",
         mtbf_estimate.__doc__, atype=AType.FORMULA, affects=Affects.RISK, formula=mtbf_estimate),
    Atom("sys.fmea_rpn", "FMEA Risk Priority Number", "systems_engineer",
         fmea_rpn.__doc__, atype=AType.FORMULA, affects=Affects.RISK, formula=fmea_rpn),
    # Thresholds
    Atom("sys.fee_schedule", "Exchange Fee Schedule", "systems_engineer",
         "MEXC: 0 bps maker, 1 bps taker. Binance: 2 bps maker, 4 bps taker. "
         "Round trip taker: MEXC 2bps, Binance 8bps, cross-exchange 5bps.",
         atype=AType.THRESHOLD, affects=Affects.COST, crypto_specific=True,
         value={"mexc_maker": 0, "mexc_taker": 1, "binance_maker": 2, "binance_taker": 4}),
    Atom("sys.latency_budget", "Latency Budget", "systems_engineer",
         "5-min cycle budget: data fetch <15s, ML inference <5s, risk check <1s, order <3s. "
         "Total critical path < 25s. Remaining 275s is slack.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION,
         value={"data_fetch_s": 15, "ml_inference_s": 5, "risk_check_s": 1, "order_s": 3}),
    Atom("sys.rate_limits", "API Rate Limits", "systems_engineer",
         "Binance: 1200 req/min (weight-based). MEXC: ~600 req/min. "
         "At 40 pairs * 2 exchanges: 80 req/10s = 8 req/s. Well within limits.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION, crypto_specific=True,
         value={"binance_per_min": 1200, "mexc_per_min": 600, "our_rate": 8}),
    # Crypto exchange quirks
    Atom("sys.exchange_quirks", "Exchange-Specific Quirks", "systems_engineer",
         "MEXC: 403 rate limit is transient (self-heals). LIMIT_MAKER = 0% fee. "
         "Binance: TICK_SIZE precision (float). MEXC: DECIMAL_PLACES (int). "
         "VPS: asyncio event loop starvation during heavy ML inference.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION, crypto_specific=True),
    Atom("sys.order_type_guide", "Order Type Guide", "systems_engineer",
         "LIMIT_MAKER: 0% fee on MEXC, rejected if would cross spread. "
         "IOC: immediate-or-cancel, taker fee. "
         "Strategy: always try LIMIT_MAKER first, fall back to IOC if urgent.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION|Affects.COST, crypto_specific=True),
])
