"""FastAPI dashboard server — REST + WebSocket + static file serving."""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from dashboard.config import DashboardConfig
from dashboard.event_emitter import DashboardEventEmitter
from dashboard.ws_manager import ConnectionManager

from dashboard.routers import system, decisions, trades, analytics, brain, risk, backtest, medallion, devil

logger = logging.getLogger(__name__)

FRONTEND_BUILD = Path(__file__).resolve().parent / "frontend" / "dist"


def create_app(
    config_path: str | None = None,
    emitter: DashboardEventEmitter | None = None,
) -> FastAPI:
    app = FastAPI(title="Renaissance Dashboard", version="1.0.0")

    # CORS (allow Vite dev server during development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Shared state
    cfg = DashboardConfig(config_path)
    app.state.dashboard_config = cfg
    app.state.ws_manager = ConnectionManager()
    app.state.emitter = emitter or DashboardEventEmitter()
    app.state.start_time = datetime.now(timezone.utc)
    app.state.last_confluence = None
    app.state.active_alerts: list = []

    # Routers
    app.include_router(system.router)
    app.include_router(decisions.router)
    app.include_router(trades.router)
    app.include_router(analytics.router)
    app.include_router(brain.router)
    app.include_router(risk.router)
    app.include_router(backtest.router)
    app.include_router(medallion.router)
    app.include_router(devil.router)

    # WebSocket endpoint
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        mgr = app.state.ws_manager
        await mgr.connect(ws)
        q = app.state.emitter.subscribe()
        try:
            # Relay events from emitter to this client
            relay = asyncio.create_task(_relay(q, ws))
            # Keep alive — also receive pings from client
            while True:
                try:
                    await asyncio.wait_for(ws.receive_text(), timeout=30)
                except asyncio.TimeoutError:
                    # Send heartbeat
                    from dashboard import db_queries
                    db = cfg.db_path
                    await ws.send_json({
                        "channel": "heartbeat",
                        "data": {
                            "status": "OPERATIONAL",
                            "uptime": (datetime.now(timezone.utc) - app.state.start_time).total_seconds(),
                            "cycle_count": db_queries.get_cycle_count(db),
                            "ws_clients": mgr.active_count,
                        },
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.debug(f"WS error: {e}")
        finally:
            relay.cancel()
            app.state.emitter.unsubscribe(q)
            mgr.disconnect(ws)

    async def _relay(q, ws: WebSocket):
        import queue as _queue
        try:
            while True:
                try:
                    msg = await asyncio.to_thread(q.get, timeout=2)
                except _queue.Empty:
                    continue
                await ws.send_json(msg)
                # Cache certain channels for REST fallback
                ch = msg.get("channel", "")
                if ch == "confluence":
                    app.state.last_confluence = msg.get("data")
                elif ch == "risk.alert":
                    alerts = app.state.active_alerts
                    alerts.append(msg.get("data"))
                    app.state.active_alerts = alerts[-100:]  # Keep last 100
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # Health check
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

    # Serve frontend static build (must be last — catches all remaining routes)
    if FRONTEND_BUILD.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="frontend")

    return app


def main():
    parser = argparse.ArgumentParser(description="Renaissance Dashboard Server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    import uvicorn

    app = create_app(config_path=args.config)
    logger.info(f"Starting dashboard on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
