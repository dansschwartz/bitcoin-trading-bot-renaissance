"""
Watchdog — separate process that monitors the bot's heartbeat and restarts it.

The main bot process writes a heartbeat file (``data/.heartbeat``) periodically.
The watchdog checks its mtime; if it becomes stale (default > 30 seconds) the
watchdog terminates the existing process and spawns a new one.

A **circuit breaker** limits restarts to 5 per hour.  If that threshold is
exceeded the watchdog halts and emits a critical log entry.

Usage
-----
Run as a standalone process::

    python -m recovery.watchdog
    python -m recovery.watchdog --heartbeat data/.heartbeat --timeout 30 \\
        --command "python run_renaissance_bot.py --run"

The ``--command`` flag sets the shell command used to launch the bot.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

logger = logging.getLogger("recovery.watchdog")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_HEARTBEAT = _PROJECT_ROOT / "data" / ".heartbeat"
_DEFAULT_TIMEOUT = 30  # seconds
_DEFAULT_CHECK_INTERVAL = 5  # seconds
_MAX_RESTARTS_PER_HOUR = 5
_DEFAULT_COMMAND = f"{sys.executable} {_PROJECT_ROOT / 'run_renaissance_bot.py'} --run"


class Watchdog:
    """
    Monitors the heartbeat file and restarts the bot when it goes stale.

    Parameters
    ----------
    heartbeat_path : str | Path
        Path to the heartbeat sentinel file.
    timeout_seconds : float
        Maximum heartbeat age before the bot is considered dead.
    check_interval : float
        How often (in seconds) to check the heartbeat.
    command : str | list[str]
        Shell command (or argv list) to start the bot.
    max_restarts_per_hour : int
        Circuit-breaker threshold.
    """

    def __init__(
        self,
        heartbeat_path: str = str(_DEFAULT_HEARTBEAT),
        timeout_seconds: float = _DEFAULT_TIMEOUT,
        check_interval: float = _DEFAULT_CHECK_INTERVAL,
        command: str = _DEFAULT_COMMAND,
        max_restarts_per_hour: int = _MAX_RESTARTS_PER_HOUR,
    ) -> None:
        self._heartbeat_path = str(heartbeat_path)
        self._timeout = timeout_seconds
        self._interval = check_interval
        self._command = command
        self._max_restarts = max_restarts_per_hour

        self._process: Optional[subprocess.Popen] = None
        self._restart_times: Deque[float] = deque()
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop — blocks until SIGINT/SIGTERM or circuit-breaker trip."""
        self._running = True
        self._install_signal_handlers()

        logger.info(
            "Watchdog starting.  heartbeat=%s  timeout=%ds  command=%s",
            self._heartbeat_path, self._timeout, self._command,
        )

        # Initial bot launch
        self._start_bot()

        while self._running:
            time.sleep(self._interval)

            if not self._running:
                break

            # Check if the process is still alive
            if self._process is not None and self._process.poll() is not None:
                exit_code = self._process.returncode
                logger.warning(
                    "Bot process exited with code %d — will restart", exit_code,
                )
                self._process = None
                if not self._circuit_breaker_ok():
                    break
                self._start_bot()
                continue

            # Check heartbeat staleness
            age = self._heartbeat_age()
            if age > self._timeout:
                logger.warning(
                    "Heartbeat stale (%.1f s > %d s threshold) — restarting bot",
                    age, self._timeout,
                )
                if not self._circuit_breaker_ok():
                    break
                self._kill_bot()
                self._start_bot()

        # Cleanup on exit
        self._kill_bot()
        logger.info("Watchdog stopped.")

    def stop(self) -> None:
        """Signal the watchdog to stop."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heartbeat_age(self) -> float:
        """Return the age of the heartbeat file in seconds."""
        try:
            mtime = os.path.getmtime(self._heartbeat_path)
            return time.time() - mtime
        except OSError:
            return float("inf")

    def _start_bot(self) -> None:
        """Spawn the bot process."""
        logger.info("Starting bot: %s", self._command)
        try:
            if isinstance(self._command, str):
                self._process = subprocess.Popen(
                    self._command,
                    shell=True,
                    cwd=str(_PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                self._process = subprocess.Popen(
                    self._command,
                    cwd=str(_PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._restart_times.append(time.time())
            logger.info("Bot started with PID %d", self._process.pid)
        except Exception as exc:
            logger.critical("Failed to start bot: %s", exc)
            self._process = None

    def _kill_bot(self) -> None:
        """Terminate the bot process if it is running."""
        if self._process is None:
            return
        pid = self._process.pid
        try:
            # Try graceful SIGTERM first
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
                logger.info("Bot (PID %d) terminated gracefully.", pid)
            except subprocess.TimeoutExpired:
                # Force kill
                self._process.kill()
                self._process.wait(timeout=5)
                logger.warning("Bot (PID %d) killed forcefully.", pid)
        except Exception as exc:
            logger.warning("Error stopping bot (PID %d): %s", pid, exc)
        finally:
            self._process = None

    def _circuit_breaker_ok(self) -> bool:
        """Return True if we are allowed to restart (haven't exceeded the limit).

        Prunes timestamps older than 1 hour.
        """
        now = time.time()
        cutoff = now - 3600
        while self._restart_times and self._restart_times[0] < cutoff:
            self._restart_times.popleft()

        if len(self._restart_times) >= self._max_restarts:
            logger.critical(
                "CIRCUIT BREAKER TRIPPED: %d restarts in the last hour "
                "(limit=%d). Watchdog halting.  Manual intervention required.",
                len(self._restart_times),
                self._max_restarts,
            )
            self._running = False
            return False
        return True

    def _install_signal_handlers(self) -> None:
        """Register SIGTERM and SIGINT handlers to stop the watchdog cleanly."""
        def _handler(signum: int, frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Watchdog received %s — shutting down", sig_name)
            self._running = False

        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)


# ---------------------------------------------------------------------------
# __main__ support
# ---------------------------------------------------------------------------

def _main() -> None:
    """CLI entry point for ``python -m recovery.watchdog``."""
    parser = argparse.ArgumentParser(
        description="Renaissance Trading Bot — Watchdog Process",
    )
    parser.add_argument(
        "--heartbeat",
        default=str(_DEFAULT_HEARTBEAT),
        help=f"Path to heartbeat file (default: {_DEFAULT_HEARTBEAT})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help=f"Heartbeat staleness threshold in seconds (default: {_DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=_DEFAULT_CHECK_INTERVAL,
        help=f"Check interval in seconds (default: {_DEFAULT_CHECK_INTERVAL})",
    )
    parser.add_argument(
        "--command",
        default=_DEFAULT_COMMAND,
        help=f"Command to start the bot (default: {_DEFAULT_COMMAND})",
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=_MAX_RESTARTS_PER_HOUR,
        help=f"Max restarts per hour before circuit breaker trips (default: {_MAX_RESTARTS_PER_HOUR})",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    watchdog = Watchdog(
        heartbeat_path=args.heartbeat,
        timeout_seconds=args.timeout,
        check_interval=args.interval,
        command=args.command,
        max_restarts_per_hour=args.max_restarts,
    )
    watchdog.run()


if __name__ == "__main__":
    _main()
