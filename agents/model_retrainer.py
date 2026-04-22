"""ModelRetrainer — async wrapper around retrain_weekly with model_ledger logging.

Runs model retraining in a background thread (via ThreadPoolExecutor) to avoid
blocking the async event loop. Logs results to the model_ledger table and emits
EventBus events for monitoring.

Usage from coordinator:
    retrainer = ModelRetrainer(db_path, models_dir, event_bus)
    await retrainer.run_retraining(proposal_id=42, epochs=50, rolling_days=180)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Timeout for a single retraining run (4 hours)
_RETRAIN_TIMEOUT_SECONDS = 4 * 3600


class ModelRetrainer:
    """Async wrapper around scripts.training.retrain_weekly."""

    def __init__(
        self,
        db_path: str,
        models_dir: str = "models/trained",
        event_bus: Optional[Any] = None,
    ) -> None:
        self.db_path = db_path
        self.models_dir = Path(models_dir).resolve()
        self.event_bus = event_bus
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retrain")
        self.is_running: bool = False
        self._last_results: Optional[Dict[str, Any]] = None

    async def run_retraining(
        self,
        proposal_id: Optional[int] = None,
        epochs: int = 50,
        rolling_days: int = 180,
        full_history: bool = False,
    ) -> Dict[str, Any]:
        """Run retrain_weekly() in a background thread.

        Returns the results dict from retrain_weekly (model name -> metrics).
        """
        if self.is_running:
            logger.warning("Retraining already in progress, skipping")
            return {"error": "already_running"}

        self.is_running = True
        start_time = datetime.now(timezone.utc)

        if self.event_bus:
            self.event_bus.emit("retrain.started", {
                "proposal_id": proposal_id,
                "epochs": epochs,
                "rolling_days": rolling_days,
                "started_at": start_time.isoformat(),
            })

        logger.info(
            "Starting model retraining (epochs=%d, rolling_days=%d, proposal=%s)",
            epochs, rolling_days, proposal_id,
        )

        try:
            loop = asyncio.get_running_loop()
            results, exit_code = await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    self._run_sync,
                    epochs,
                    rolling_days,
                    full_history,
                ),
                timeout=_RETRAIN_TIMEOUT_SECONDS,
            )

            self._last_results = results
            self._log_to_model_ledger(results, proposal_id)

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
            logger.info(
                "Retraining completed in %.1fs (exit_code=%d, models=%d)",
                elapsed, exit_code, len(results),
            )

            if self.event_bus:
                self.event_bus.emit("retrain.completed", {
                    "proposal_id": proposal_id,
                    "exit_code": exit_code,
                    "models_trained": len(results),
                    "elapsed_seconds": elapsed,
                })

            return {"results": results, "exit_code": exit_code, "elapsed": elapsed}

        except asyncio.TimeoutError:
            logger.error("Retraining timed out after %ds", _RETRAIN_TIMEOUT_SECONDS)
            if self.event_bus:
                self.event_bus.emit("retrain.timeout", {
                    "proposal_id": proposal_id,
                    "timeout_seconds": _RETRAIN_TIMEOUT_SECONDS,
                })
            return {"error": "timeout", "exit_code": 1}

        except Exception as exc:
            logger.error("Retraining failed: %s", exc)
            if self.event_bus:
                self.event_bus.emit("retrain.failed", {
                    "proposal_id": proposal_id,
                    "error": str(exc),
                })
            return {"error": str(exc), "exit_code": 1}

        finally:
            self.is_running = False

    def _run_sync(
        self,
        epochs: int,
        rolling_days: int,
        full_history: bool,
    ) -> tuple:
        """Synchronous wrapper — imports and calls retrain_weekly directly."""
        from scripts.training.retrain_weekly import retrain_weekly

        return retrain_weekly(
            epochs=epochs,
            rolling_days=rolling_days,
            full_history=full_history,
            auto_deploy=True,
        )

    def _log_to_model_ledger(
        self,
        results: Dict[str, Any],
        proposal_id: Optional[int],
    ) -> None:
        """Write one row per model to the model_ledger table."""
        from agents.db_schema import insert_model_ledger

        now = datetime.now(timezone.utc).isoformat()

        for model_name, metrics in results.items():
            if not isinstance(metrics, dict):
                continue

            file_path = metrics.get("model_path", "")
            file_hash = self._file_hash(file_path) if file_path and os.path.exists(file_path) else None
            accuracy = metrics.get("new_accuracy") or metrics.get("accuracy")
            status = "active" if metrics.get("deployed", False) else "candidate"
            version = now  # Use timestamp as version

            try:
                insert_model_ledger(
                    db_path=self.db_path,
                    model_name=model_name,
                    model_version=version,
                    accuracy=accuracy,
                    file_path=file_path,
                    file_hash=file_hash,
                    status=status,
                    proposal_id=proposal_id,
                )
            except Exception as exc:
                logger.debug("Failed to log model %s to ledger: %s", model_name, exc)

    @staticmethod
    def _file_hash(path: str) -> str:
        """SHA-256 hash of file (first 16 hex chars)."""
        h = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    h.update(chunk)
            return h.hexdigest()[:16]
        except OSError:
            return ""
