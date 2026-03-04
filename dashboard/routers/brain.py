"""Brain endpoints — ML ensemble, regime, VAE, confluence, cascade."""

import json
import logging
import sqlite3

from fastapi import APIRouter, Request

from dashboard import db_queries

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/brain", tags=["brain"])


@router.get("/ensemble")
async def ensemble_status(request: Request):
    """Current ML ensemble state (from bot emitter if live, else last DB snapshot)."""
    db = request.app.state.dashboard_config.db_path
    preds = db_queries.get_ml_predictions_history(db, hours=1)

    # Group by model name, take latest
    latest_by_model = {}
    for p in preds:
        model = p.get("model_name", "unknown")
        if model not in latest_by_model:
            latest_by_model[model] = p

    return {
        "models": latest_by_model,
        "model_count": len(latest_by_model),
    }


@router.get("/predictions/history")
async def prediction_history(request: Request, hours: int = 24):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_ml_predictions_history(db, hours=min(hours, 168))


@router.get("/regime")
async def regime_status(request: Request):
    db = request.app.state.dashboard_config.db_path
    history = db_queries.get_regime_history(db, limit=50)
    current = history[0] if history else None

    # Overlay live regime from emitter cache (has classifier + bar_count)
    emitter = getattr(request.app.state, "emitter", None)
    live = emitter.get_cached("regime") if emitter else None
    if live:
        current = {
            **(current or {}),
            "hmm_regime": live.get("hmm_regime", "unknown"),
            "confidence": live.get("confidence", 0.0),
            "classifier": live.get("classifier", "none"),
            "bar_count": live.get("bar_count", 0),
            "details": live.get("details", ""),
        }
    elif current:
        # DB fallback: derive classifier + bar_count when emitter is unavailable
        # (bot and dashboard run as separate processes)
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            bar_count = conn.execute(
                "SELECT COUNT(*) FROM five_minute_bars"
            ).fetchone()[0]
            conn.close()
            classifier = "hmm" if bar_count >= 200 else "bootstrap"
            current["classifier"] = classifier
            current["bar_count"] = bar_count
        except Exception as e:
            logger.debug(f"Bar count query failed: {e}")
            current.setdefault("classifier", "hmm")
            current.setdefault("bar_count", 0)

    return {
        "current": current,
        "history": history,
    }


@router.get("/confluence")
async def confluence_status(request: Request):
    """Confluence data — from emitter cache or reconstructed from DB."""
    # Primary: read from emitter's channel cache (works even with 0 WS clients)
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("confluence")
        if cached:
            return cached
    # Fallback: legacy WS relay cache
    legacy = getattr(request.app.state, "last_confluence", None)
    if legacy:
        return legacy

    # DB fallback: reconstruct confluence from latest decision's signals
    db = request.app.state.dashboard_config.db_path
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT reasoning, timestamp FROM decisions ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row and row[0]:
            reasoning = json.loads(row[0])
            signals = reasoning.get("signal_contributions", {})
            if signals:
                # Run confluence engine against latest signals
                from confluence_engine import ConfluenceEngine
                ce = ConfluenceEngine()
                result = ce.calculate_confluence_boost(signals)
                result["source"] = "db_fallback"
                result["decision_timestamp"] = row[1]
                return result
    except Exception as e:
        logger.debug(f"Confluence DB fallback failed: {e}")

    return {"status": "no_live_data", "message": "Waiting for bot cycle"}


@router.get("/regime/hierarchy")
async def regime_hierarchy(request: Request):
    """Hierarchical regime state — macro, crypto, micro, and model route."""
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("regime_hierarchy")
        if cached:
            return {"status": "live", **cached}
    return {
        "status": "waiting",
        "macro": {"regime": "UNKNOWN", "confidence": 0.0},
        "crypto": {"regime": "UNKNOWN", "confidence": 0.0},
        "router": None,
    }


@router.get("/vae")
async def vae_history(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_vae_history(db, limit=200)


@router.get("/crash")
async def crash_model_status(request: Request):
    """Multi-asset crash-regime LightGBM model status and latest predictions."""
    # Try emitter cache first (live data from bot)
    emitter = getattr(request.app.state, "emitter", None)
    if emitter:
        cached = emitter.get_cached("crash_models")
        if cached:
            return {"status": "live", **cached}

    # Fallback: load model metadata directly
    try:
        from crash_model_loader import CrashModelLoader
        loader = CrashModelLoader()
        return {
            "status": "static",
            **loader.get_state(),
        }
    except Exception as e:
        logger.debug(f"Crash model status fallback failed: {e}")
        return {"status": "unavailable", "model_count": 0, "models": {}}


@router.get("/cascade/collection")
async def cascade_collection_stats(request: Request):
    """Stats on Cascade data collection progress (Polymarket crowd pricing)."""
    db = request.app.state.dashboard_config.db_path
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row

        total_snapshots = conn.execute(
            "SELECT COUNT(*) as n FROM cascade_crowd_data"
        ).fetchone()['n']

        total_market_records = conn.execute(
            "SELECT COUNT(*) as n FROM cascade_market_snapshots"
        ).fetchone()['n']

        unique_markets = conn.execute(
            "SELECT COUNT(DISTINCT market_slug) as n FROM cascade_market_snapshots"
        ).fetchone()['n']

        first_ts = conn.execute(
            "SELECT MIN(timestamp) as ts FROM cascade_crowd_data"
        ).fetchone()['ts']

        last_ts = conn.execute(
            "SELECT MAX(timestamp) as ts FROM cascade_crowd_data"
        ).fetchone()['ts']

        conn.close()
        return {
            'total_snapshots': total_snapshots,
            'total_market_records': total_market_records,
            'unique_markets': unique_markets,
            'first_record': first_ts,
            'last_record': last_ts,
        }
    except Exception:
        return {'total_snapshots': 0, 'message': 'no data yet (tables not created)'}


@router.get("/correlation/eigenvalue")
async def eigenvalue_status(request: Request):
    """Council S3: Return eigenvalue concentration ratio + history from bot instance."""
    bot = getattr(request.app.state, "bot_instance", None)
    if bot and hasattr(bot, 'correlation_network'):
        cn = bot.correlation_network
        return {
            "enabled": cn.enabled,
            "eigenvalue_ratio": round(cn.get_eigenvalue_ratio(), 4),
            "concentrated": cn.should_reduce_positions(),
            "threshold": cn.concentration_threshold,
            "history": list(cn._eigenvalue_history)[-20:],
            "tracked_assets": len(cn._price_history),
        }
    return {
        "enabled": False,
        "eigenvalue_ratio": 0.0,
        "concentrated": False,
        "message": "correlation network not available",
    }


@router.get("/volatility")
async def volatility_status(request: Request):
    """Volatility prediction model status and recent predictions."""
    bot = getattr(request.app.state, "bot_instance", None)
    result = {"model_loaded": False, "recent_predictions": {}}

    if bot and hasattr(bot, 'real_time_pipeline'):
        proc = bot.real_time_pipeline.processor
        if hasattr(proc, '_vol_model') and proc._vol_model is not None:
            result["model_loaded"] = True
            meta = proc._vol_meta or {}
            result["model_info"] = {
                "n_features": meta.get("n_features", "?"),
                "test_mae": meta.get("test_mae", "?"),
                "trained_at": meta.get("trained_at", "?"),
                "label_percentiles": meta.get("label_percentiles", {}),
            }

    # Get recent volatility predictions from pending predictions
    if bot and hasattr(bot, '_pending_predictions'):
        for pid, pred in bot._pending_predictions.items():
            mag = pred.get('predicted_magnitude_bps')
            if mag is not None:
                result["recent_predictions"][pid] = round(mag, 1)

    return result


@router.get("/sub-bar/status")
async def sub_bar_status(request: Request):
    """Sub-bar scanner status and recent events."""
    bot = getattr(request.app.state, "bot_instance", None)
    if bot and hasattr(bot, 'sub_bar_scanner') and bot.sub_bar_scanner is not None:
        return bot.sub_bar_scanner.get_status()

    return {
        "enabled": False,
        "running": False,
        "observation_mode": True,
        "message": "sub-bar scanner not initialized",
    }


@router.get("/sub-bar/events")
async def sub_bar_events(request: Request):
    """Recent sub-bar trigger events from database."""
    cfg = getattr(request.app.state, "dashboard_config", None)
    db_path = getattr(cfg, "db_path", None) if cfg else None
    if not db_path:
        return {"events": [], "total": 0}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
        rows = conn.execute(
            """SELECT id, timestamp, product_id, trigger_type, side,
                      entry_price, current_price, pnl_bps, expected_magnitude_bps,
                      observation_mode, action_taken, details
               FROM sub_bar_events
               ORDER BY id DESC LIMIT 50"""
        ).fetchall()
        conn.close()

        events = []
        for r in rows:
            events.append({
                "id": r[0], "timestamp": r[1], "product_id": r[2],
                "trigger_type": r[3], "side": r[4],
                "entry_price": r[5], "current_price": r[6],
                "pnl_bps": round(r[7], 1) if r[7] else 0,
                "expected_magnitude_bps": round(r[8], 1) if r[8] else 0,
                "observation_mode": bool(r[9]), "action_taken": r[10],
                "details": r[11],
            })
        return {"events": events, "total": len(events)}
    except Exception as e:
        return {"events": [], "total": 0, "error": str(e)}
