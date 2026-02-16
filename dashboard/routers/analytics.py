"""Analytics endpoints — PnL, equity curves, distributions, heatmaps."""

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

RANGE_MAP = {"1H": 1, "4H": 4, "1D": 24, "1W": 168, "1M": 720, "ALL": 8760}


def _parse_range(range_str: str) -> int:
    return RANGE_MAP.get(range_str.upper(), 24)


@router.get("/equity")
async def equity_curve(request: Request, range: str = "1D"):
    db = request.app.state.dashboard_config.db_path
    hours = _parse_range(range)
    return db_queries.get_equity_curve(db, hours=hours)


@router.get("/pnl")
async def pnl_summary(request: Request, range: str = "1D"):
    db = request.app.state.dashboard_config.db_path
    hours = _parse_range(range)
    return db_queries.get_pnl_summary(db, hours=hours)


@router.get("/by-regime")
async def by_regime(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_performance_by_regime(db)


@router.get("/by-execution")
async def by_execution(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_performance_by_execution(db)


@router.get("/distribution")
async def return_distribution(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_return_distribution(db)


@router.get("/calendar")
async def calendar_heatmap(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_calendar_pnl(db)


@router.get("/hourly")
async def hourly_heatmap(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_hourly_pnl(db)


@router.get("/benchmark")
async def benchmark(request: Request, range: str = "1D", product_id: str = "BTC-USD"):
    db = request.app.state.dashboard_config.db_path
    hours = _parse_range(range)
    return db_queries.get_benchmark_equity(db, hours=hours, product_id=product_id)
