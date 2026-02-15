"""Backtest endpoints — runs, results, live vs backtest comparison."""

from fastapi import APIRouter, Request

from dashboard import db_queries

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.get("/runs")
async def backtest_runs(request: Request):
    db = request.app.state.dashboard_config.db_path
    return db_queries.get_backtest_runs(db)


@router.get("/runs/{run_id}")
async def backtest_result(request: Request, run_id: int):
    db = request.app.state.dashboard_config.db_path
    result = db_queries.get_backtest_result(db, run_id)
    if not result:
        return {"error": "Backtest run not found"}
    return result


@router.get("/compare/{run_id}")
async def compare_live_vs_backtest(request: Request, run_id: int):
    db = request.app.state.dashboard_config.db_path
    backtest = db_queries.get_backtest_result(db, run_id)
    if not backtest:
        return {"error": "Backtest run not found"}

    live_metrics = db_queries.get_risk_metrics(db)
    live_pnl = db_queries.get_pnl_summary(db, hours=8760)

    return {
        "backtest": backtest,
        "live": {
            "risk_metrics": live_metrics,
            "pnl_summary": live_pnl,
        },
    }
