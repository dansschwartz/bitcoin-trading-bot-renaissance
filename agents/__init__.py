"""Doc 15: Autonomous Agent Coordination & Self-Improvement System.

Provides BaseAgent, EventBus, 7 agent wrappers, AgentCoordinator,
ObservationCollector, SafetyGate, DeploymentMonitor, and QuantResearcher.
"""

from agents.base import BaseAgent
from agents.event_bus import EventBus
from agents.coordinator import AgentCoordinator

__all__ = [
    "BaseAgent",
    "EventBus",
    "AgentCoordinator",
]
