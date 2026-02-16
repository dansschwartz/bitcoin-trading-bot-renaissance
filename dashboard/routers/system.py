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

    # Latest price per product with change %
    latest_prices = {}
    for pid in cfg.product_ids:
        lp = db_queries.get_latest_price(db, pid)
        if lp:
            price = lp.get("price", 0)
            # Compute 1h price change
            change_pct = 0.0
            try:
                old_prices = db_queries.get_recent_market_data(db, product_id=pid, limit=500)
                # old_prices is newest-first from DB
                if len(old_prices) > 1:
                    oldest = old_prices[-1]
                    oldest_price = oldest.get("price", price)
                    if oldest_price > 0:
                        change_pct = round((price - oldest_price) / oldest_price * 100, 2)
            except Exception:
                pass
            latest_prices[pid] = {
                "price": price,
                "bid": lp.get("bid"),
                "ask": lp.get("ask"),
                "timestamp": lp.get("timestamp"),
                "change_pct": change_pct,
            }

    ws_clients = request.app.state.ws_manager.active_count

    return {
        "status": "OPERATIONAL",
        "uptime_seconds": round(uptime_seconds),
        "cycle_count": cycle_count,
        "trade_count": trade_count,
        "open_position_count": len(open_positions),
        "paper_trading": cfg.paper_trading,
        "product_ids": cfg.product_ids,
        "latest_prices": latest_prices,
        "ws_clients": ws_clients,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/config")
async def system_config(request: Request):
    cfg = request.app.state.dashboard_config
    return {
        "flags": cfg.flags,
        "signal_weights": cfg.signal_weights,
        "dashboard": cfg.dashboard,
        "product_ids": cfg.product_ids,
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
