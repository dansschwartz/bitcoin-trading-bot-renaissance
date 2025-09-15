"""
Renaissance Technologies Enhanced Logging System
Complete audit trail with alternative data monitoring and dashboard integration.
"""

import json
import logging
import logging.handlers
import re
import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import asyncio
import uuid

# Renaissance Technologies specific imports
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class RenaissanceLogLevel:
    """Renaissance-specific log levels"""
    SIGNAL_GENERATION = 25
    REGIME_CHANGE = 26
    ALTERNATIVE_DATA = 27
    CONFIDENCE_CHANGE = 28
    PERFORMANCE_ATTRIBUTION = 29


# Add custom log levels
logging.addLevelName(RenaissanceLogLevel.SIGNAL_GENERATION, 'SIGNAL')
logging.addLevelName(RenaissanceLogLevel.REGIME_CHANGE, 'REGIME')
logging.addLevelName(RenaissanceLogLevel.ALTERNATIVE_DATA, 'ALTDATA')
logging.addLevelName(RenaissanceLogLevel.CONFIDENCE_CHANGE, 'CONFIDENCE')
logging.addLevelName(RenaissanceLogLevel.PERFORMANCE_ATTRIBUTION, 'PERFORMANCE')


class RenaissanceJSONFormatter(logging.Formatter):
    """Enhanced JSON formatter for Renaissance data"""

    def __init__(self):
        super().__init__()
        self.hostname = self._get_hostname()

    def format(self, record: logging.LogRecord) -> str:
        """Format with Renaissance-specific data"""

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "hostname": self.hostname,
            "thread": record.thread,
            "session_id": getattr(record, 'session_id', 'unknown')
        }

        # Add Renaissance-specific data
        if hasattr(record, 'renaissance_data'):
            log_data["renaissance"] = record.renaissance_data

        # Add signal breakdown if present
        if hasattr(record, 'signal_breakdown'):
            log_data["signal_breakdown"] = record.signal_breakdown

        # Add market regime data
        if hasattr(record, 'regime_data'):
            log_data["regime"] = record.regime_data

        # Add alternative data status
        if hasattr(record, 'alt_data_status'):
            log_data["alternative_data"] = record.alt_data_status

        # Add performance metrics
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                log_data["system_metrics"] = {
                    "cpu_percent": process.cpu_percent(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "memory_percent": process.memory_percent()
                }
            except Exception:
                pass

        return json.dumps(log_data, default=str)


class RenaissanceAuditLogger:
    """Complete audit trail logger for Renaissance Technologies integration"""

    def __init__(self, name: str = "renaissance_bot",
                 log_level: str = "INFO",
                 audit_file: Optional[Path] = None,
                 dashboard_file: Optional[Path] = None):

        self.name = name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.audit_file = audit_file or Path("logs/renaissance_audit.jsonl")
        self.dashboard_file = dashboard_file or Path("logs/dashboard_data.json")

        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.run_counter = 0

        # Thread safety
        self._lock = threading.Lock()

        # Dashboard data structure
        self.dashboard_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "current_regime": "unknown",
            "alternative_data_status": {},
            "signal_components": {},
            "positions": [],
            "performance": {
                "total_pnl": 0.0,
                "daily_pnl": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            },
            "last_update": datetime.now().isoformat()
        }

        # Initialize logger
        self.logger = self._setup_logger()
        self._setup_audit_logging()

    def _setup_logger(self) -> logging.Logger:
        """Setup Renaissance-enhanced logger"""

        bot_logger = logging.getLogger(self.name)
        bot_logger.setLevel(self.log_level)
        bot_logger.handlers.clear()

        # Console handler with Renaissance formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        bot_logger.addHandler(console_handler)

        # JSON file handler for structured logging
        if self.audit_file:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            json_handler = logging.FileHandler(self.audit_file, mode='a')
            json_handler.setLevel(logging.DEBUG)
            json_handler.setFormatter(RenaissanceJSONFormatter())
            bot_logger.addHandler(json_handler)

        bot_logger.propagate = False
        return bot_logger

    def log_trading_cycle_start(self, cycle_id: str, market_data: Dict[str, Any]):
        """Log start of trading cycle with complete input data"""

        self.run_counter += 1

        audit_data = {
            "cycle_id": cycle_id,
            "run_number": self.run_counter,
            "phase": "CYCLE_START",
            "input_data": {
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }
        }

        self.logger.info(
            f"Trading Cycle {self.run_counter} Started - ID: {cycle_id}",
            extra={
                'session_id': self.session_id,
                'renaissance_data': audit_data
            }
        )

    def log_signal_generation(self, cycle_id: str, signal_result: Dict[str, Any],
                              components: Dict[str, float], regime_data: Dict[str, Any]):
        """Log complete signal generation with Renaissance breakdown"""

        # Extract signal components for audit
        signal_breakdown = {
            "order_flow": components.get('order_flow', 0.0),
            "order_book": components.get('order_book', 0.0),
            "volume": components.get('volume', 0.0),
            "macd": components.get('macd', 0.0),
            "rsi": components.get('rsi', 0.0),
            "bollinger": components.get('bollinger', 0.0),
            "renaissance": components.get('renaissance', 0.0)
        }

        # Update dashboard data
        self.dashboard_data["signal_components"] = signal_breakdown
        self.dashboard_data["current_regime"] = regime_data.get('regime', 'unknown')

        self.logger.log(
            RenaissanceLogLevel.SIGNAL_GENERATION,
            f"Signal Generated: {signal_result['action']} (confidence: {signal_result['confidence']:.3f})",
            extra={
                'session_id': self.session_id,
                'signal_breakdown': signal_breakdown,
                'regime_data': regime_data,
                'renaissance_data': {
                    "cycle_id": cycle_id,
                    "phase": "SIGNAL_GENERATION",
                    "signal_result": signal_result,
                    "components": components
                }
            }
        )

    def log_alternative_data_status(self, alt_data_status: Dict[str, Any]):
        """Log alternative data feed status"""

        self.dashboard_data["alternative_data_status"] = alt_data_status

        self.logger.log(
            RenaissanceLogLevel.ALTERNATIVE_DATA,
            f"Alternative Data Status Updated",
            extra={
                'session_id': self.session_id,
                'alt_data_status': alt_data_status
            }
        )

    def log_regime_change(self, old_regime: str, new_regime: str,
                          regime_metrics: Dict[str, Any]):
        """Log market regime change"""

        self.dashboard_data["current_regime"] = new_regime

        self.logger.log(
            RenaissanceLogLevel.REGIME_CHANGE,
            f"Regime Change: {old_regime} → {new_regime}",
            extra={
                'session_id': self.session_id,
                'regime_data': {
                    "old_regime": old_regime,
                    "new_regime": new_regime,
                    "metrics": regime_metrics,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    def log_trade_execution(self, cycle_id: str, execution_result: Dict[str, Any]):
        """Log trade execution with Renaissance context"""

        if execution_result.get('executed'):
            self.dashboard_data["performance"]["total_trades"] += 1

        self.logger.info(
            f"Trade Execution: {execution_result.get('action', 'NONE')}",
            extra={
                'session_id': self.session_id,
                'renaissance_data': {
                    "cycle_id": cycle_id,
                    "phase": "TRADE_EXECUTION",
                    "execution_result": execution_result
                }
            }
        )

    def log_cycle_completion(self, cycle_id: str, cycle_summary: Dict[str, Any]):
        """Log completion of trading cycle"""

        # Update dashboard data
        self.dashboard_data["last_update"] = datetime.now().isoformat()

        # Save dashboard data
        self._save_dashboard_data()

        self.logger.info(
            f"Trading Cycle {self.run_counter} Completed - Duration: {cycle_summary.get('duration', 0):.2f}s",
            extra={
                'session_id': self.session_id,
                'renaissance_data': {
                    "cycle_id": cycle_id,
                    "phase": "CYCLE_COMPLETION",
                    "summary": cycle_summary
                }
            }
        )

    def _save_dashboard_data(self):
        """Save current dashboard data to JSON file"""

        try:
            self.dashboard_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.dashboard_file, 'w') as f:
                json.dump(self.dashboard_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save dashboard data: {e}")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data.copy()