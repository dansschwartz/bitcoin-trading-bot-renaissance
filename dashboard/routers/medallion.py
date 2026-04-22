"""Medallion module status endpoint (Audit 7)."""

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/medallion", tags=["medallion"])


def _safe_query(db_path: str, query: str, params: tuple = ()) -> Optional[Any]:
    """Execute a query and return the first row, or None on error."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def _safe_scalar(db_path: str, query: str, params: tuple = ()) -> Optional[Any]:
    """Execute a query and return a single scalar value."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _table_exists(db_path: str, table: str) -> bool:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        exists = cursor.fetchone()[0] > 0
        conn.close()
        return exists
    except Exception:
        return False


@router.get("/status")
async def medallion_status(request: Request):
    """
    Returns status of all Medallion-style modules:
    - Which modules are ACTIVE / INACTIVE / OBSERVATION
    - Key metrics from each
    - Any active alerts/warnings
    """
    cfg = request.app.state.dashboard_config
    db = cfg.db_path
    raw = cfg.raw

    modules: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # ── Phase 1 Modules ──

    # Signal Auto-Throttle (intra-day fast kill)
    st_cfg = raw.get("signal_throttle", {})
    modules.append({
        "name": "SignalAutoThrottle",
        "role": "Intra-day fast kill (accuracy < 45%)",
        "status": "ACTIVE" if st_cfg.get("enabled", False) else "INACTIVE",
        "config": {
            "kill_accuracy": st_cfg.get("kill_accuracy"),
            "short_window": st_cfg.get("short_window"),
        },
    })

    # Portfolio Health Monitor
    hm_cfg = raw.get("health_monitor", {})
    modules.append({
        "name": "PortfolioHealthMonitor",
        "role": "Rolling Sharpe-based size scaling",
        "status": "ACTIVE" if hm_cfg.get("enabled", False) else "INACTIVE",
        "config": {
            "full_size_sharpe": hm_cfg.get("full_size_sharpe"),
            "exits_only_sharpe": hm_cfg.get("exits_only_sharpe"),
        },
    })

    # Unified Portfolio Engine (correlation-aware sizing)
    pe_cfg = raw.get("portfolio_engine", {})
    modules.append({
        "name": "UnifiedPortfolioEngine",
        "role": "Correlation-aware position sizing",
        "status": "ACTIVE" if pe_cfg.get("enabled", False) else "INACTIVE",
        "config": {
            "correlation_threshold": pe_cfg.get("correlation_threshold"),
            "max_concentration": pe_cfg.get("max_concentration"),
        },
    })

    # Medallion Signal Analogs
    ma_cfg = raw.get("medallion_analogs", {})
    modules.append({
        "name": "MedallionSignalAnalogs",
        "role": "Sharp move reversion, seasonality, funding timing",
        "status": "ACTIVE" if ma_cfg.get("enabled", False) else "INACTIVE",
    })

    # Signal Validation Gate
    sv_cfg = raw.get("signal_validator", {})
    modules.append({
        "name": "SignalValidationGate",
        "role": "Statistical validation per signal per regime",
        "status": "ACTIVE" if sv_cfg else "INACTIVE",
    })

    # Data Validator
    modules.append({
        "name": "DataValidator",
        "role": "Input data integrity checks",
        "status": "ACTIVE",
    })

    # ── Phase 2 Modules ──

    # Devil Tracker
    devil_count = _safe_scalar(db, "SELECT COUNT(*) FROM devil_tracker") if _table_exists(db, "devil_tracker") else None
    devil_latest = _safe_query(db, "SELECT * FROM devil_tracker ORDER BY rowid DESC LIMIT 1") if _table_exists(db, "devil_tracker") else None
    modules.append({
        "name": "DevilTracker",
        "role": "Execution quality tracking (devil = slippage + cost)",
        "status": "ACTIVE" if _table_exists(db, "devil_tracker") else "INACTIVE",
        "metrics": {
            "total_entries": devil_count,
            "latest_entry": devil_latest,
        },
    })

    # Kelly Position Sizer
    ks_cfg = raw.get("kelly_sizer", {})
    modules.append({
        "name": "KellyPositionSizer",
        "role": "Fractional Kelly sizing from trade history (observation mode)",
        "status": "OBSERVATION",
        "config": {
            "kelly_fraction": ks_cfg.get("kelly_fraction"),
            "min_trades": ks_cfg.get("min_trades"),
        },
    })

    # Daily Signal Review (MedallionSignalThrottle)
    mst_cfg = raw.get("medallion_signal_throttle", {})
    modules.append({
        "name": "DailySignalReview",
        "role": "End-of-day P&L audit per signal type",
        "status": "OBSERVATION",
        "config": {
            "throttle_after_days": mst_cfg.get("throttle_after_consecutive_losing_days"),
            "disable_after_days": mst_cfg.get("disable_after_consecutive_losing_days"),
        },
    })

    # Medallion Regime Detector
    mrd_cfg = raw.get("medallion_regime_detector", {})
    modules.append({
        "name": "MedallionRegimeDetector",
        "role": "3-state HMM regime detection (observation mode vs RegimeOverlay)",
        "status": "OBSERVATION" if mrd_cfg.get("enabled", False) else "INACTIVE",
        "config": {
            "n_states": mrd_cfg.get("n_states"),
            "retrain_interval_hours": mrd_cfg.get("retrain_interval_hours"),
        },
    })

    # Insurance Premium Scanner
    is_cfg = raw.get("insurance_scanner", {})
    modules.append({
        "name": "InsurancePremiumScanner",
        "role": "Funding settlement / weekend / event premium detection",
        "status": "OBSERVATION" if is_cfg.get("enabled", False) else "INACTIVE",
    })

    # Medallion Portfolio Engine
    mpe_cfg = raw.get("medallion_portfolio_engine", {})
    modules.append({
        "name": "MedallionPortfolioEngine",
        "role": "Target/actual reconciliation (observation mode — drift logging)",
        "status": "OBSERVATION",
        "config": {
            "drift_threshold_pct": mpe_cfg.get("drift_threshold_pct"),
            "max_leverage": mpe_cfg.get("max_leverage"),
        },
    })

    # Leverage Manager
    lm_cfg = raw.get("leverage_manager", {})
    modules.append({
        "name": "LeverageManager",
        "role": "Consistency-based leverage scaling",
        "status": "ACTIVE" if lm_cfg.get("enabled", False) else "INACTIVE",
    })

    # Bar Aggregator
    modules.append({
        "name": "BarAggregator",
        "role": "5-min OHLCV bar aggregation from trade feed",
        "status": "ACTIVE" if _table_exists(db, "bars") else "INACTIVE",
        "metrics": {
            "bar_count": _safe_scalar(db, "SELECT COUNT(*) FROM bars") if _table_exists(db, "bars") else 0,
        },
    })

    # Beta Monitor
    bm_cfg = raw.get("beta_monitor", {})
    modules.append({
        "name": "BetaMonitor",
        "role": "Portfolio beta tracking",
        "status": "ACTIVE" if bm_cfg.get("enabled", False) else "INACTIVE",
    })

    # Sharpe Monitor
    sm_cfg = raw.get("sharpe_monitor", {})
    modules.append({
        "name": "SharpeMonitor",
        "role": "Rolling Sharpe health tracking",
        "status": "ACTIVE" if sm_cfg.get("enabled", False) else "INACTIVE",
    })

    # Capacity Monitor
    cm_cfg = raw.get("capacity_monitor", {})
    modules.append({
        "name": "CapacityMonitor",
        "role": "Capacity wall / slippage detection",
        "status": "ACTIVE" if cm_cfg.get("enabled", False) else "INACTIVE",
    })

    # ── Key Metrics (from DB) ──
    metrics: Dict[str, Any] = {}

    # Trade count
    trade_count = _safe_scalar(db, "SELECT COUNT(*) FROM trades") if _table_exists(db, "trades") else 0
    metrics["total_trades"] = trade_count

    # Latest decision
    latest_dec = _safe_query(
        db, "SELECT product_id, action, confidence, weighted_signal, timestamp "
            "FROM decisions ORDER BY rowid DESC LIMIT 1"
    ) if _table_exists(db, "decisions") else None
    metrics["latest_decision"] = latest_dec

    # Regime (from decisions table)
    latest_regime = _safe_scalar(
        db, "SELECT hmm_regime FROM decisions WHERE hmm_regime IS NOT NULL "
            "ORDER BY rowid DESC LIMIT 1"
    ) if _table_exists(db, "decisions") else None
    metrics["current_regime"] = latest_regime

    # ── Alerts ──
    alerts = list(request.app.state.active_alerts[-10:]) if hasattr(request.app.state, "active_alerts") else []

    # Summary
    active_count = sum(1 for m in modules if m["status"] == "ACTIVE")
    obs_count = sum(1 for m in modules if m["status"] == "OBSERVATION")
    inactive_count = sum(1 for m in modules if m["status"] == "INACTIVE")

    return {
        "summary": {
            "active": active_count,
            "observation": obs_count,
            "inactive": inactive_count,
            "total": len(modules),
        },
        "modules": modules,
        "metrics": metrics,
        "alerts": alerts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
