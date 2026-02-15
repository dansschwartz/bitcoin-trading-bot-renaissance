"""Trade and position endpoints."""

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api", tags=["trades"])


@router.get("/positions/open")
async def open_positions(request: Request):
    db = request.app.state.dashboard_config.db_path
    positions = db_queries.get_open_positions(db)
    # Enrich with latest prices if available
    for pos in positions:
        pid = pos.get("product_id", "BTC-USD")
        lp = db_queries.get_latest_price(db, pid)
        if lp:
            current_price = lp.get("price", 0.0)
            entry_price = pos.get("entry_price", 0.0)
            size = pos.get("size", 0.0)
            side = pos.get("side", "BUY")
            if side == "BUY":
                pos["unrealized_pnl"] = round((current_price - entry_price) * size, 2)
            else:
                pos["unrealized_pnl"] = round((entry_price - current_price) * size, 2)
            pos["current_price"] = current_price
        else:
            pos["unrealized_pnl"] = 0.0
            pos["current_price"] = pos.get("entry_price", 0.0)
    return positions


@router.get("/trades/closed")
async def closed_trades(request: Request, limit: int = 50, offset: int = 0):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_closed_trades(db, limit=min(limit, 500), offset=offset)


@router.get("/trades/{trade_id}")
async def trade_detail(request: Request, trade_id: int):
    db = request.app.state.dashboard_config.db_path
    result = db_queries.get_trade_by_id(db, trade_id)
    if not result:
        return {"error": "Trade not found"}
    return result


@router.get("/trades/{trade_id}/lifecycle")
async def trade_lifecycle(request: Request, trade_id: int):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_trade_lifecycle(db, trade_id)
