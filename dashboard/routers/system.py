"""System status and configuration endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/status")
async def system_status(request: Request):
    cfg = request.app.state.dashboard_config
    db = cfg.db_path
    start_time = request.app.state.start_time

    uptime_seconds = (datetime.now(timezone.utc) - start_time).total_seconds()
    cycle_count = db_queries.get_cycle_count(db)
    trade_count = db_queries.get_trade_count(db)
    open_positions = db_queries.get_open_positions(db)

    # Dynamic universe: pairs the bot is actually scanning (from recent decisions)
    active_ids = db_queries.get_active_product_ids(db, hours=2)
    product_ids = active_ids if active_ids else cfg.product_ids

    # Batch-fetch latest prices for all active pairs
    batch_prices = db_queries.get_latest_prices_batch(db, product_ids)
    latest_prices = {}
    for pid, lp in batch_prices.items():
        latest_prices[pid] = {
            "price": lp.get("price", 0),
            "bid": lp.get("bid"),
            "ask": lp.get("ask"),
            "timestamp": lp.get("timestamp"),
        }

    ws_clients = request.app.state.ws_manager.active_count

    return {
        "status": "OPERATIONAL",
        "uptime_seconds": round(uptime_seconds),
        "cycle_count": cycle_count,
        "trade_count": trade_count,
        "open_position_count": len(open_positions),
        "paper_trading": cfg.paper_trading,
        "product_ids": product_ids,
        "latest_prices": latest_prices,
        "ws_clients": ws_clients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/config")
async def system_config(request: Request):
    cfg = request.app.state.dashboard_config
    db = cfg.db_path
    active_ids = db_queries.get_active_product_ids(db, hours=2)
    return {
        "flags": cfg.flags,
        "signal_weights": cfg.signal_weights,
        "dashboard": cfg.dashboard,
        "product_ids": active_ids if active_ids else cfg.product_ids,
        "paper_trading": cfg.paper_trading,
    }


@router.get("/prices/{product_id}")
async def price_history(request: Request, product_id: str, limit: int = 200):
    """Recent price data from market_data table."""
    db = request.app.state.dashboard_config.db_path
    rows = db_queries.get_recent_market_data(db, product_id=product_id, limit=min(limit, 1000))
    # Reverse to chronological order (DB returns newest first)
    rows.reverse()
    return rows
