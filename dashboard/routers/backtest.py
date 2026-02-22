"""Backtest endpoints — runs, results, live vs backtest comparison, and runner control."""

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dashboard import db_queries

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


# ─── Existing read endpoints ──────────────────────────────────────────────

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


# ─── Runner endpoints (start / status / download) ─────────────────────────

class BacktestConfig(BaseModel):
    pairs: List[str] = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD", "AVAX-USD", "DOGE-USD"]
    cost_bps: float = 0.0065
    lookahead: int = 6
    new_denom: float = 0.02
    new_buy_thresh: float = 0.015
    new_sell_thresh: float = -0.015
    new_conf_floor: float = 0.48
    new_signal_scale: float = 10.0
    new_exit_bars: int = 6
    new_pos_min: float = 50.0
    new_pos_max: float = 300.0
    new_pos_base: float = 100.0
    old_denom: float = 0.05
    old_buy_thresh: float = 0.01
    old_sell_thresh: float = -0.01
    old_conf_floor: float = 0.505
    old_signal_scale: float = 1.0
    old_exit_bars: int = 1
    old_pos_usd: float = 75.0


@router.post("/start")
async def start_backtest(request: Request, config: BacktestConfig):
    """Start a new backtest with the given configuration."""
    manager = request.app.state.backtest_manager
    try:
        result = manager.start(config.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/status")
async def backtest_status(request: Request):
    """Return current backtest job state (for polling / page-reload recovery)."""
    manager = request.app.state.backtest_manager
    return manager.status()


@router.get("/download")
async def download_backtest_csv(request: Request):
    """Download the most recent backtest CSV result."""
    manager = request.app.state.backtest_manager
    status = manager.status()
    csv_path = status.get("csv_path")

    if not csv_path or not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail="No backtest CSV available")

    filename = os.path.basename(csv_path)
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
