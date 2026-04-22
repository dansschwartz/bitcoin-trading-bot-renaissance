"""BacktestJobManager — spawns backtest_ml_pipeline.py as a subprocess and
streams JSON progress to the dashboard via WebSocket events.

Only one backtest can run at a time.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from dashboard.event_emitter import DashboardEventEmitter

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "backtest_ml_pipeline.py"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python3"


class BacktestJobManager:
    """Singleton-style manager for one backtest subprocess at a time."""

    def __init__(self, emitter: DashboardEventEmitter, db_path: str) -> None:
        self._emitter = emitter
        self._db_path = db_path
        self._state: str = "idle"  # idle | running | complete | error
        self._progress: Dict[str, Any] = {}
        self._summary: Optional[Dict[str, Any]] = None
        self._csv_path: Optional[str] = None
        self._error_msg: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._config: Optional[Dict[str, Any]] = None

    @property
    def is_running(self) -> bool:
        return self._state == "running"

    def status(self) -> Dict[str, Any]:
        """Return current job state for REST polling / recovery."""
        result: Dict[str, Any] = {"state": self._state}
        if self._state == "running":
            result["progress"] = self._progress
            result["config"] = self._config
        elif self._state == "complete":
            result["summary"] = self._summary
            result["csv_path"] = self._csv_path
            result["config"] = self._config
        elif self._state == "error":
            result["error"] = self._error_msg
            result["config"] = self._config
        return result

    def start(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Start a backtest subprocess with the given config.

        Returns {"status": "started"} on success.
        Raises ValueError if already running.
        """
        if self.is_running:
            raise ValueError("A backtest is already running")

        # Reset state
        self._state = "running"
        self._progress = {}
        self._summary = None
        self._csv_path = None
        self._error_msg = None
        self._config = config

        # Write config to temp file
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="backtest_cfg_", delete=False
        )
        json.dump(config, tmp)
        tmp.close()
        config_path = tmp.name

        # Determine python executable
        python = str(PYTHON) if PYTHON.exists() else sys.executable

        cmd = [
            python,
            str(BACKTEST_SCRIPT),
            "--json-config", config_path,
            "--json-progress",
        ]

        logger.info(f"Starting backtest subprocess: {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(PROJECT_ROOT),
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as e:
            self._state = "error"
            self._error_msg = str(e)
            raise

        # Launch async monitor
        loop = asyncio.get_event_loop()
        self._monitor_task = loop.create_task(self._monitor(config_path))

        return {"status": "started"}

    async def _monitor(self, config_path: str) -> None:
        """Read subprocess stdout line-by-line, parse JSON, emit WS events."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        try:
            while True:
                line = await asyncio.to_thread(proc.stdout.readline)
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")

                if msg_type == "progress":
                    self._progress = msg
                    self._emitter.emit_sync("backtest.progress", {
                        "state": "running",
                        **msg,
                    })

                elif msg_type == "complete":
                    self._summary = msg.get("summary")
                    self._csv_path = msg.get("csv_path")

            # Wait for process to finish
            await asyncio.to_thread(proc.wait)

            if proc.returncode == 0:
                self._state = "complete"
                self._emitter.emit_sync("backtest.progress", {
                    "state": "complete",
                    "summary": self._summary,
                    "csv_path": self._csv_path,
                })
                logger.info(f"Backtest completed. CSV: {self._csv_path}")
            else:
                stderr_output = ""
                if proc.stderr:
                    stderr_output = await asyncio.to_thread(proc.stderr.read)
                self._state = "error"
                self._error_msg = stderr_output[-2000:] if stderr_output else f"exit code {proc.returncode}"
                self._emitter.emit_sync("backtest.progress", {
                    "state": "error",
                    "error": self._error_msg,
                })
                logger.error(f"Backtest failed (rc={proc.returncode}): {self._error_msg[:200]}")

        except asyncio.CancelledError:
            if proc.poll() is None:
                proc.kill()
            self._state = "error"
            self._error_msg = "Cancelled"
        except Exception as e:
            self._state = "error"
            self._error_msg = str(e)
            logger.exception("Backtest monitor error")
        finally:
            # Clean up temp config file
            try:
                os.unlink(config_path)
            except OSError:
                pass
            self._process = None
