"""AgentCoordinator — instantiates agents, wires event subscriptions, orchestrates weekly research."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agents.base import BaseAgent
from agents.event_bus import EventBus
from agents.data_agent import DataCollectionAgent
from agents.signal_agent import SignalAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.portfolio_agent import PortfolioAgent
from agents.monitoring_agent import MonitoringAgent
from agents.meta_agent import MetaAgent
from agents.db_schema import ensure_agent_tables, log_agent_event

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Central coordinator for the Doc 15 agent system.

    Instantiates all 7 agents, wires event subscriptions, and provides:
    - ``on_cycle_complete(cycle_data)`` — called after each trading cycle
    - ``run_weekly_check_loop()`` — async background task for weekly research
    - ``get_all_statuses()`` — aggregated status for dashboard
    """

    def __init__(
        self,
        bot: Any,
        db_path: str,
        config: Dict[str, Any],
        bot_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.db_path = db_path
        self.config = config
        self.logger = bot_logger or logger
        self.event_bus = EventBus(max_history=1000)
        self._cycle_count: int = 0
        self._start_time: float = time.time()

        # Ensure agent DB tables exist
        ensure_agent_tables(db_path)

        # ── Instantiate agents ──
        common = dict(event_bus=self.event_bus, db_path=db_path, config=config)

        self.data_agent = DataCollectionAgent(
            market_data_provider=getattr(bot, "market_data_provider", None),
            bar_aggregator=getattr(bot, "bar_aggregator", None),
            data_validator=getattr(bot, "data_validator", None),
            **common,
        )
        self.signal_agent = SignalAgent(
            signal_scorecard=getattr(bot, "_signal_scorecard", None),
            signal_throttle=getattr(bot, "signal_throttle", None),
            ml_bridge=getattr(bot, "ml_bridge", None),
            **common,
        )
        self.risk_agent = RiskAgent(
            risk_gateway=getattr(bot, "risk_gateway", None),
            regime_overlay=getattr(bot, "regime_overlay", None),
            **common,
        )
        self.execution_agent = ExecutionAgent(
            devil_tracker=getattr(bot, "devil_tracker", None),
            position_manager=getattr(bot, "position_manager", None),
            **common,
        )
        self.portfolio_agent = PortfolioAgent(
            portfolio_engine=getattr(bot, "medallion_portfolio_engine", None),
            position_reevaluator=getattr(bot, "position_reevaluator", None),
            kelly_sizer=getattr(bot, "kelly_sizer", None),
            **common,
        )
        self.monitoring_agent = MonitoringAgent(
            sharpe_monitor=getattr(bot, "sharpe_monitor_medallion", None),
            beta_monitor=getattr(bot, "beta_monitor", None),
            capacity_monitor=getattr(bot, "capacity_monitor", None),
            **common,
        )
        self.meta_agent = MetaAgent(
            adaptive_weights=getattr(bot, "_adaptive_weight_blend", None),
            genetic_optimizer=getattr(bot, "genetic_optimizer", None),
            signal_weights=config.get("signal_weights", {}),
            **common,
        )

        self.agents: List[BaseAgent] = [
            self.data_agent,
            self.signal_agent,
            self.risk_agent,
            self.execution_agent,
            self.portfolio_agent,
            self.monitoring_agent,
            self.meta_agent,
        ]

        # ── Wire cross-agent event subscriptions ──
        self.event_bus.subscribe(
            "risk.circuit_breaker",
            self._on_circuit_breaker,
            subscriber="coordinator",
        )
        self.event_bus.subscribe(
            "signal.trade_signal",
            self._on_trade_signal,
            subscriber="coordinator",
        )

        self.logger.info(
            "AgentCoordinator: ACTIVE — %d agents initialized", len(self.agents),
        )

    # ── Cycle hook ──

    def on_cycle_complete(self, cycle_data: Dict[str, Any]) -> None:
        """Called at the end of every trading cycle.  Fan-out to all agents."""
        self._cycle_count += 1
        for agent in self.agents:
            try:
                agent.on_cycle_complete(cycle_data)
            except Exception as exc:
                self.logger.debug("Agent %s cycle hook error: %s", agent.name, exc)

        # Emit cycle event
        self.event_bus.emit("cycle.complete", {
            "cycle": self._cycle_count,
            "product_id": cycle_data.get("product_id", ""),
            "action": cycle_data.get("action", "HOLD"),
        })

    # ── Status for dashboard ──

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """Return status dicts for all agents (for /api/agents/status)."""
        statuses = []
        for agent in self.agents:
            try:
                statuses.append(agent.get_status())
            except Exception as exc:
                statuses.append({
                    "name": agent.name,
                    "state": "error",
                    "error": str(exc),
                })
        return statuses

    def get_all_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        """Compile observations from all agents for weekly report."""
        observations: Dict[str, Any] = {}
        for agent in self.agents:
            try:
                observations[agent.name] = agent.get_observations(window_hours)
            except Exception as exc:
                observations[agent.name] = {"error": str(exc)}
        return observations

    # ── Weekly research loop ──

    async def run_weekly_check_loop(self) -> None:
        """Background task: checks hourly if it's time for weekly research.

        The actual research is handled by QuantResearcherAgent (Phase D).
        This loop just triggers it at the configured schedule.
        """
        researcher_cfg = self.config.get("quant_researcher", {})
        if not researcher_cfg.get("enabled", False):
            self.logger.info("AgentCoordinator: quant_researcher disabled, weekly loop idle")
            # Still run loop but skip research
            while True:
                await asyncio.sleep(3600)
                continue

        schedule_day = researcher_cfg.get("schedule_day", "sunday").lower()
        schedule_hour = researcher_cfg.get("schedule_hour_utc", 2)
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }
        target_day = day_map.get(schedule_day, 6)
        last_run_week: Optional[int] = None

        self.logger.info(
            "AgentCoordinator: weekly research scheduled for %s %02d:00 UTC",
            schedule_day, schedule_hour,
        )

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                now = datetime.now(timezone.utc)
                iso_week = now.isocalendar()[1]

                if (
                    now.weekday() == target_day
                    and now.hour == schedule_hour
                    and iso_week != last_run_week
                ):
                    self.logger.info("AgentCoordinator: triggering weekly research")
                    last_run_week = iso_week
                    try:
                        from agents.quant_researcher import QuantResearcherAgent
                        researcher = QuantResearcherAgent(
                            event_bus=self.event_bus,
                            db_path=self.db_path,
                            config=self.config,
                            coordinator=self,
                        )
                        await researcher.run_weekly_research()
                    except Exception as exc:
                        self.logger.error("Weekly research failed: %s", exc)
                        log_agent_event(
                            self.db_path,
                            agent_name="coordinator",
                            event_type="research_failed",
                            channel="coordinator.error",
                            payload={"error": str(exc)},
                            severity="error",
                        )
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Weekly check loop error: %s", exc)
                await asyncio.sleep(60)

    # ── Internal event handlers ──

    def _on_circuit_breaker(self, channel: str, payload: Dict[str, Any]) -> None:
        """Log circuit breaker activation to DB."""
        try:
            log_agent_event(
                self.db_path,
                agent_name="risk",
                event_type="circuit_breaker",
                channel=channel,
                payload=payload,
                severity="warning",
            )
        except Exception:
            pass

    def _on_trade_signal(self, channel: str, payload: Dict[str, Any]) -> None:
        """Log significant trade signals to DB."""
        confidence = payload.get("confidence", 0.0)
        if confidence >= 0.7:
            try:
                log_agent_event(
                    self.db_path,
                    agent_name="signal",
                    event_type="high_confidence_signal",
                    channel=channel,
                    payload=payload,
                    severity="info",
                )
            except Exception:
                pass
