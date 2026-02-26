"""AgentCoordinator — instantiates agents, wires event subscriptions, orchestrates weekly research and deployment."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
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
from agents.deployment_monitor import DeploymentMonitor
from agents.config_deployer import ConfigDeployer
from agents.model_retrainer import ModelRetrainer
from agents.proposal import (
    DeploymentMode,
    ProposalStatus,
    REQUIRES_APPROVAL,
    SANDBOX_HOURS,
)

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

        # ── Deployment infrastructure ──
        config_path = str(Path(db_path).resolve().parent.parent / "config" / "config.json")
        models_dir = str(Path(db_path).resolve().parent.parent / "models" / "trained")
        self.config_deployer = ConfigDeployer(config_path)
        self.deployment_monitor = DeploymentMonitor(db_path, config)
        self.model_retrainer = ModelRetrainer(db_path, models_dir, self.event_bus)
        self._active_backups: Dict[int, str] = {}  # proposal_id -> backup_path

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

        # Triggered research — specific events trigger specific researchers
        self._trigger_cooldowns: Dict[str, float] = {}  # researcher -> last trigger timestamp

        self.event_bus.subscribe(
            "monitoring.model_degradation",
            lambda ch, payload: self._maybe_trigger_research("linguist", payload),
            subscriber="coordinator",
        )
        self.event_bus.subscribe(
            "monitoring.sharpe_alert",
            lambda ch, payload: self._maybe_trigger_research("mathematician", payload),
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
        """Background task: checks hourly if it's time for weekly research or retraining.

        The actual research is handled by QuantResearcherAgent (Phase D).
        Model retraining is handled by ModelRetrainer.
        """
        researcher_cfg = self.config.get("quant_researcher", {})
        retrain_cfg = self.config.get("model_retraining", {})

        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }

        # Research schedule
        research_enabled = researcher_cfg.get("enabled", False)
        research_day = day_map.get(researcher_cfg.get("schedule_day", "sunday").lower(), 6)
        research_hour = researcher_cfg.get("schedule_hour_utc", 2)
        last_research_week: Optional[int] = None

        # Retraining schedule
        retrain_enabled = retrain_cfg.get("enabled", False)
        retrain_day = day_map.get(retrain_cfg.get("schedule_day", "saturday").lower(), 5)
        retrain_hour = retrain_cfg.get("schedule_hour_utc", 6)
        last_retrain_week: Optional[int] = None

        # Daily scan schedule (Systems Engineer)
        daily_scan_enabled = researcher_cfg.get("daily_scan_enabled", False)
        daily_scan_hour = researcher_cfg.get("daily_scan_hour_utc", 0)
        last_daily_scan_date: Optional[str] = None

        if research_enabled:
            self.logger.info(
                "AgentCoordinator: weekly research scheduled for %s %02d:00 UTC",
                researcher_cfg.get("schedule_day", "sunday"), research_hour,
            )
        else:
            self.logger.info("AgentCoordinator: quant_researcher disabled")

        if retrain_enabled:
            self.logger.info(
                "AgentCoordinator: weekly retraining scheduled for %s %02d:00 UTC",
                retrain_cfg.get("schedule_day", "saturday"), retrain_hour,
            )
        else:
            self.logger.info("AgentCoordinator: model_retraining disabled")

        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                now = datetime.now(timezone.utc)
                iso_week = now.isocalendar()[1]
                today = now.strftime("%Y-%m-%d")

                # ── Daily Systems Engineer scan ──
                if (
                    daily_scan_enabled
                    and now.hour == daily_scan_hour
                    and today != last_daily_scan_date
                ):
                    last_daily_scan_date = today
                    self.logger.info("AgentCoordinator: triggering daily Systems Engineer scan")
                    try:
                        from agents.quant_researcher import QuantResearcherAgent
                        daily_cfg = {
                            **researcher_cfg,
                            "research_mode": "council",
                            "researchers": ["systems_engineer"],
                            "council_max_turns_per_researcher": 20,
                            "council_max_minutes_per_researcher": 15,
                            "enabled": True,
                        }
                        researcher = QuantResearcherAgent(
                            event_bus=self.event_bus,
                            db_path=self.db_path,
                            config={**self.config, "quant_researcher": daily_cfg},
                            coordinator=self,
                        )
                        await researcher.run_weekly_research()
                    except Exception as exc:
                        self.logger.error("Daily scan failed: %s", exc)

                # ── Weekly research ──
                if (
                    research_enabled
                    and now.weekday() == research_day
                    and now.hour == research_hour
                    and iso_week != last_research_week
                ):
                    self.logger.info("AgentCoordinator: triggering weekly research")
                    last_research_week = iso_week
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

                # ── Weekly model retraining ──
                if (
                    retrain_enabled
                    and now.weekday() == retrain_day
                    and now.hour == retrain_hour
                    and iso_week != last_retrain_week
                    and not self.model_retrainer.is_running
                ):
                    self.logger.info("AgentCoordinator: triggering scheduled model retraining")
                    last_retrain_week = iso_week
                    asyncio.create_task(
                        self._run_model_retrain(
                            proposal_id=None,  # scheduled, not from proposal
                            retrain_args={
                                "epochs": retrain_cfg.get("epochs", 50),
                                "rolling_days": retrain_cfg.get("rolling_days", 180),
                            },
                        )
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Weekly check loop error: %s", exc)
                await asyncio.sleep(60)

    # ── Deployment loop ──

    async def run_deployment_loop(self) -> None:
        """Background task: every 5 min, process pending deployments and check sandboxes."""
        self.logger.info("AgentCoordinator: deployment loop started (every 5 min)")
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes
                await self._process_pending_deployments()
                await self._check_completed_sandboxes()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.error("Deployment loop error: %s", exc)
                await asyncio.sleep(60)

    async def _process_pending_deployments(self) -> None:
        """Find safety_passed proposals and deploy/sandbox/retrain them."""
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            rows = conn.execute(
                """SELECT id, title, deployment_mode, config_changes
                   FROM proposals WHERE status = ?""",
                (ProposalStatus.SAFETY_PASSED.value,),
            ).fetchall()
            conn.close()
        except Exception as exc:
            self.logger.debug("_process_pending_deployments query failed: %s", exc)
            return

        for row in rows:
            proposal_id, title, mode_str, config_changes_json = row
            config_changes = json.loads(config_changes_json) if config_changes_json else {}

            try:
                mode = DeploymentMode(mode_str)
            except ValueError:
                self.logger.warning("Unknown deployment mode '%s' for proposal %d", mode_str, proposal_id)
                continue

            self.logger.info(
                "Processing proposal %d '%s' (mode=%s)", proposal_id, title, mode_str,
            )

            try:
                if mode == DeploymentMode.MODEL_RETRAIN:
                    # Launch retraining in background
                    retrain_args = config_changes.get("retrain_args", {})
                    asyncio.create_task(
                        self._run_model_retrain(proposal_id, retrain_args)
                    )
                    self._update_proposal_status(proposal_id, ProposalStatus.SANDBOXING)

                elif mode == DeploymentMode.PARAMETER_TUNE:
                    # Deploy immediately (no sandbox, no approval)
                    backup_path, _ = self.config_deployer.apply_changes(config_changes, proposal_id)
                    self._active_backups[proposal_id] = str(backup_path)
                    self.deployment_monitor.deploy_proposal(proposal_id)
                    self.logger.info("Proposal %d deployed immediately (parameter_tune)", proposal_id)

                elif REQUIRES_APPROVAL.get(mode, True):
                    # Needs human approval first
                    self._update_proposal_status(proposal_id, ProposalStatus.AWAITING_APPROVAL)
                    self.logger.info("Proposal %d awaiting human approval (mode=%s)", proposal_id, mode_str)

                else:
                    # Apply config + start sandbox (MODIFY_EXISTING)
                    backup_path, _ = self.config_deployer.apply_changes(config_changes, proposal_id)
                    self._active_backups[proposal_id] = str(backup_path)
                    # Create a minimal Proposal object for the monitor
                    from agents.proposal import Proposal
                    p = Proposal(
                        title=title,
                        description="",
                        category="",
                        deployment_mode=mode,
                        proposal_id=proposal_id,
                    )
                    self.deployment_monitor.start_sandbox(p)
                    self.logger.info("Proposal %d deployed to sandbox (%dh)", proposal_id, SANDBOX_HOURS.get(mode, 24))

            except Exception as exc:
                self.logger.error("Failed to process proposal %d: %s", proposal_id, exc)
                log_agent_event(
                    self.db_path,
                    agent_name="coordinator",
                    event_type="deployment_error",
                    channel="coordinator.deploy",
                    payload={"proposal_id": proposal_id, "error": str(exc)},
                    severity="error",
                )

    async def _check_completed_sandboxes(self) -> None:
        """Check for sandboxes that have ended and decide deploy vs rollback."""
        completed = self.deployment_monitor.check_sandbox_completion()
        if not completed:
            return

        for item in completed:
            proposal_id = item["id"]
            sandbox_start = item.get("sandbox_start", "")

            try:
                metrics_before = self._get_baseline_metrics(sandbox_start)
                metrics_after = self._get_current_metrics()

                result = self.deployment_monitor.evaluate_sandbox_result(
                    proposal_id, metrics_before, metrics_after,
                )

                if result["action"] == "deploy":
                    self.deployment_monitor.deploy_proposal(proposal_id)
                    self.logger.info(
                        "Proposal %d promoted from sandbox to deployed", proposal_id,
                    )
                else:
                    reason = "; ".join(result.get("reasons", ["metric degradation"]))
                    # Rollback config
                    if proposal_id in self._active_backups:
                        self.config_deployer.restore(Path(self._active_backups[proposal_id]))
                        del self._active_backups[proposal_id]
                    else:
                        self.config_deployer.rollback_proposal(proposal_id)
                    self.deployment_monitor.rollback_proposal(proposal_id, reason)
                    self.logger.warning(
                        "Proposal %d rolled back: %s", proposal_id, reason,
                    )

            except Exception as exc:
                self.logger.error("Sandbox evaluation failed for proposal %d: %s", proposal_id, exc)

    async def _run_model_retrain(
        self, proposal_id: int, retrain_args: Dict[str, Any],
    ) -> None:
        """Run model retraining and deploy/rollback based on result."""
        self.logger.info("Starting model retraining for proposal %d", proposal_id)
        try:
            result = await self.model_retrainer.run_retraining(
                proposal_id=proposal_id,
                epochs=retrain_args.get("epochs", 50),
                rolling_days=retrain_args.get("rolling_days", 180),
                full_history=retrain_args.get("full_history", False),
            )

            exit_code = result.get("exit_code", 1)
            if exit_code == 0:
                self.deployment_monitor.deploy_proposal(proposal_id)
                self.logger.info("Model retraining succeeded — proposal %d deployed", proposal_id)
            else:
                error_msg = result.get("error", f"exit_code={exit_code}")
                self.deployment_monitor.rollback_proposal(proposal_id, f"retrain failed: {error_msg}")
                self.logger.warning("Model retraining failed — proposal %d rolled back", proposal_id)

        except Exception as exc:
            self.logger.error("Model retrain task failed for proposal %d: %s", proposal_id, exc)
            self.deployment_monitor.rollback_proposal(proposal_id, f"exception: {exc}")

    def _update_proposal_status(self, proposal_id: int, status: ProposalStatus) -> None:
        """Update a proposal's status in the DB."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "UPDATE proposals SET status=?, updated_at=? WHERE id=?",
                (status.value, now, proposal_id),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            self.logger.debug("_update_proposal_status failed: %s", exc)

    def _get_baseline_metrics(self, sandbox_start: str) -> Dict[str, float]:
        """Get performance metrics from before the sandbox started."""
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute(
                """SELECT AVG(sharpe) as sharpe, MAX(max_drawdown) as drawdown,
                          AVG(win_rate) as win_rate
                   FROM daily_performance WHERE date < ?""",
                (sandbox_start[:10],),  # use date part
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                return {
                    "sharpe": row[0] or 0.0,
                    "drawdown": row[1] or 0.0,
                    "win_rate": row[2] or 0.0,
                }
        except Exception as exc:
            self.logger.debug("_get_baseline_metrics failed: %s", exc)
        return {"sharpe": 0.0, "drawdown": 0.05, "win_rate": 0.5}

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get recent performance metrics (last 7 days)."""
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute(
                """SELECT AVG(sharpe) as sharpe, MAX(max_drawdown) as drawdown,
                          AVG(win_rate) as win_rate
                   FROM daily_performance
                   ORDER BY date DESC LIMIT 7"""
            ).fetchone()
            conn.close()
            if row and row[0] is not None:
                return {
                    "sharpe": row[0] or 0.0,
                    "drawdown": row[1] or 0.0,
                    "win_rate": row[2] or 0.0,
                }
        except Exception as exc:
            self.logger.debug("_get_current_metrics failed: %s", exc)
        return {"sharpe": 0.0, "drawdown": 0.05, "win_rate": 0.5}

    # ── Triggered research ──

    def _maybe_trigger_research(self, researcher_name: str, payload: Dict[str, Any]) -> None:
        """Trigger a single-researcher session if cooldown allows."""
        now = time.time()
        researcher_cfg = self.config.get("quant_researcher", {})
        cooldown_hours = researcher_cfg.get("trigger_cooldown_hours", 6)
        last = self._trigger_cooldowns.get(researcher_name, 0)

        if now - last < cooldown_hours * 3600:
            self.logger.debug("Trigger for %s suppressed (cooldown)", researcher_name)
            return

        # Count today's triggers
        today_triggers = sum(
            1 for ts in self._trigger_cooldowns.values()
            if now - ts < 86400
        )
        max_daily = researcher_cfg.get("max_triggers_per_day", 3)
        if today_triggers >= max_daily:
            self.logger.debug("Daily trigger limit reached (%d/%d)", today_triggers, max_daily)
            return

        self._trigger_cooldowns[researcher_name] = now
        self.logger.info("Triggered research: %s (reason: %s)", researcher_name, payload)

        log_agent_event(
            self.db_path,
            agent_name="coordinator",
            event_type="research_triggered",
            channel="coordinator.trigger",
            payload={"researcher": researcher_name, "reason": str(payload)},
            severity="info",
        )

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
