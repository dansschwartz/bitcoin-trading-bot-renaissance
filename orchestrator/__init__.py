"""
Multi-Bot Orchestrator
======================
Coordinates multiple bot instances via file-based heartbeats.
"""

from orchestrator.bot_manager import BotOrchestrator, BotInstance
from orchestrator.heartbeat import HeartbeatWriter

__all__ = ["BotOrchestrator", "BotInstance", "HeartbeatWriter"]
