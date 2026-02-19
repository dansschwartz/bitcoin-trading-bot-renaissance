"""QuantResearcherAgent — the Outer Loop.

Weekly:
1. ObservationCollector compiles a structured report (pure Python, no LLM).
2. Report + research prompt written to ``data/research_sessions/``.
3. Claude Code launched via ``subprocess.run(['claude', ...])`` with timeout.
4. Session output parsed for ``proposals.json``.
5. Each proposal evaluated through SafetyGate.
6. Approved proposals stored in ``proposals`` table.

Can be run manually:
    python -m agents.quant_researcher --manual
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agents.base import BaseAgent
from agents.observation_collector import ObservationCollector
from agents.safety_gate import SafetyGate, PAPER_TRADING_ONLY
from agents.proposal import Proposal, ProposalStatus, DeploymentMode
from agents.db_schema import insert_proposal, log_agent_event

if TYPE_CHECKING:
    from agents.coordinator import AgentCoordinator

logger = logging.getLogger(__name__)


class QuantResearcherAgent(BaseAgent):
    """Outer Loop agent — launches weekly Claude Code research sessions."""

    def __init__(
        self,
        event_bus: Any,
        db_path: str,
        config: Dict[str, Any],
        coordinator: Optional[AgentCoordinator] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("quant_researcher", event_bus, db_path, config, **kwargs)
        self.coordinator = coordinator
        self.researcher_cfg = config.get("quant_researcher", {})
        self.max_proposals = self.researcher_cfg.get("max_proposals_per_week", 5)
        self.max_session_minutes = self.researcher_cfg.get("max_session_minutes", 120)
        self.model = self.researcher_cfg.get("model", "sonnet")
        self._state = "idle"
        self._project_root = Path(self.db_path).resolve().parent.parent
        self._claude_available = self._check_claude_available()

    def _check_claude_available(self) -> bool:
        """Check if the claude CLI is available on PATH."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=10,
            )
            version = (result.stdout or "").strip()
            self.logger.info("Claude CLI available: %s", version or "unknown version")
            return True
        except FileNotFoundError:
            self.logger.info("Claude CLI not found on PATH — research sessions will skip Claude launch")
            return False
        except Exception as exc:
            self.logger.debug("Claude CLI check failed: %s", exc)
            return False

    def get_status(self) -> Dict[str, Any]:
        status = self._base_status()
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM weekly_reports"
            ).fetchone()
            status["total_reports"] = row[0] if row else 0
            row = conn.execute(
                "SELECT COUNT(*) FROM proposals"
            ).fetchone()
            status["total_proposals"] = row[0] if row else 0
            # Latest report date
            row = conn.execute(
                "SELECT generated_at FROM weekly_reports ORDER BY id DESC LIMIT 1"
            ).fetchone()
            status["last_report"] = row[0] if row else None
            conn.close()
        except Exception as exc:
            status["db_error"] = str(exc)
        return status

    def get_observations(self, window_hours: int = 168) -> Dict[str, Any]:
        self._record_run()
        obs: Dict[str, Any] = {"agent": self.name}
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            # Recent proposals
            rows = conn.execute(
                """SELECT id, title, category, status, deployment_mode,
                          backtest_sharpe, created_at
                   FROM proposals ORDER BY id DESC LIMIT 10"""
            ).fetchall()
            obs["recent_proposals"] = [
                {
                    "id": r[0], "title": r[1], "category": r[2],
                    "status": r[3], "mode": r[4], "sharpe": r[5],
                    "created": r[6],
                }
                for r in rows
            ]
            # Improvement log
            rows = conn.execute(
                """SELECT change_type, description, reverted
                   FROM improvement_log
                   ORDER BY id DESC LIMIT 10"""
            ).fetchall()
            obs["recent_improvements"] = [
                {"type": r[0], "desc": r[1], "reverted": bool(r[2])}
                for r in rows
            ]
            conn.close()
        except Exception as exc:
            self._record_error(exc)
            obs["error"] = str(exc)
        return obs

    async def run_weekly_research(self) -> None:
        """Execute one weekly research cycle."""
        self._state = "researching"
        self.logger.info("QuantResearcher: starting weekly research cycle")
        self._record_run()

        try:
            # Step 1: Compile observation report
            collector = ObservationCollector(
                db_path=self.db_path,
                config=self.config,
                coordinator=self.coordinator,
            )
            report = collector.compile_weekly_report()
            filepath = collector.save_report(report)
            self.logger.info("Weekly report compiled: %s", filepath)

            # Step 2: Prepare research session directory
            session_dir = self._prepare_session(report)

            # Step 3: Launch Claude Code (if enabled)
            if self.researcher_cfg.get("enabled", False):
                proposals = self._launch_claude_session(session_dir)
            else:
                self.logger.info("QuantResearcher: Claude Code disabled, skipping launch")
                proposals = []

            # Step 4: Evaluate proposals through safety gate
            if proposals:
                self._evaluate_proposals(proposals, report)

            self.event_bus.emit("researcher.cycle_complete", {
                "report_path": filepath,
                "proposals_found": len(proposals),
            })

        except Exception as exc:
            self._record_error(exc)
            self.logger.error("Weekly research failed: %s", exc)
            log_agent_event(
                self.db_path,
                agent_name=self.name,
                event_type="research_error",
                channel="researcher.error",
                payload={"error": str(exc)},
                severity="error",
            )
        finally:
            self._state = "idle"

    def _prepare_session(self, report: Dict[str, Any]) -> Path:
        """Write report + prompt to a session directory."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        session_dir = self._project_root / "data" / "research_sessions" / ts
        session_dir.mkdir(parents=True, exist_ok=True)

        # Write report
        (session_dir / "weekly_report.json").write_text(
            json.dumps(report, indent=2, default=str)
        )

        # Build and write research prompt
        prompt = self._build_research_prompt(
            report,
            session_dir=session_dir,
            project_root=self._project_root,
        )
        (session_dir / "research_prompt.md").write_text(prompt)

        self.logger.info("Research session prepared at %s", session_dir)
        return session_dir

    def _build_research_prompt(
        self,
        report: Dict[str, Any],
        session_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
    ) -> str:
        """Build the research prompt for Claude Code."""
        summary = report.get("summary", {})
        signals = report.get("signals", {})
        regimes = report.get("regimes", {})
        config_snap = report.get("config_snapshot", {})

        if project_root is None:
            project_root = self._project_root
        db_path_abs = Path(self.db_path).resolve()
        config_path_abs = (project_root / "config" / "config.json").resolve()

        # Identify focus areas
        focus_areas: List[str] = []
        worst_signals = signals.get("per_signal_pnl", [])
        if worst_signals:
            losers = [s for s in worst_signals if s.get("pnl", 0) < 0]
            if losers:
                focus_areas.append(
                    f"Worst signals: {', '.join(s['signal'] for s in losers[:3])}"
                )

        sharpe = summary.get("sharpe_7d")
        if sharpe is not None and sharpe < 0.5:
            focus_areas.append(f"Low Sharpe ratio: {sharpe:.3f}")

        win_rate = summary.get("win_rate", 0)
        if win_rate < 0.5:
            focus_areas.append(f"Win rate below 50%: {win_rate:.1%}")

        output_dir_str = str(session_dir) if session_dir else "data/research_sessions/<timestamp>"

        prompt = f"""# Weekly Quant Research Session

## System Paths
- **Project root:** `{project_root}`
- **Database:** `{db_path_abs}` (read-only for analysis)
- **Config:** `{config_path_abs}`
- **Output dir:** `{output_dir_str}`
- **Models dir:** `{project_root / 'models' / 'trained'}`

## Performance Summary (last 7 days)
- Total P&L: ${summary.get('total_pnl', 0):.2f}
- Total Trades: {summary.get('total_trades', 0)}
- Win Rate: {summary.get('win_rate', 0):.1%}
- Sharpe (7d): {summary.get('sharpe_7d', 'N/A')}
- Open Positions: {summary.get('open_positions', 0)}

## Focus Areas
{chr(10).join(f'- {f}' for f in focus_areas) if focus_areas else '- No critical issues identified'}

## Current Signal Weights
```json
{json.dumps(config_snap.get('signal_weights', {}), indent=2)}
```

## Regime Distribution
```json
{json.dumps(regimes.get('regime_distribution', []), indent=2, default=str)}
```

## Your Task
Analyze the weekly report at `{output_dir_str}/weekly_report.json`.
Query the database for deeper analysis:
```bash
sqlite3 "{db_path_abs}" "SELECT ..."
```
Read source code at `{project_root}` to understand the system.

Produce a file `proposals.json` in `{output_dir_str}` with up to {self.max_proposals} improvement proposals.

## Model Retraining
If model accuracy is below target (< 52%), you may propose a `model_retrain` deployment.
Set `"deployment_mode": "model_retrain"` and include retrain arguments in `config_changes`:
```json
{{"retrain_args": {{"epochs": 50, "rolling_days": 180, "full_history": false}}}}
```
The system will run `retrain_weekly()` automatically. Retraining has its own safety gate
(new model must beat old by >= -1% accuracy).

## Hard Safety Limits (CANNOT be changed)
- PAPER_TRADING_ONLY = True
- ABSOLUTE_MAX_POSITION_USD = 10,000
- ABSOLUTE_MAX_DRAWDOWN_PCT = 10%
- ABSOLUTE_MAX_LEVERAGE = 3.0
- ABSOLUTE_MAX_DAILY_TRADES = 200

## Allowed Proposal Categories
- `parameter_tune` — adjust signal weights, thresholds, intervals (no sandbox, no approval)
- `modify_existing` — change existing module behavior (24h sandbox, no approval)
- `new_feature` — add new capability (72h sandbox, human approval required)
- `new_strategy` — add new trading strategy (168h sandbox, human approval required)
- `model_retrain` — retrain ML models (no sandbox, has own safety gate)

## Output Format (proposals.json)
```json
[
  {{
    "title": "Short descriptive title",
    "description": "What to change and why",
    "category": "parameter_tune",
    "deployment_mode": "parameter_tune",
    "config_changes": {{"signal_weights": {{"rsi": 0.08}}}},
    "backtest_sharpe": 1.2,
    "backtest_drawdown": 0.03,
    "backtest_accuracy": 0.54,
    "backtest_sample_size": 150,
    "backtest_p_value": 0.02,
    "notes": "Additional context"
  }}
]
```

Focus on high-impact, low-risk improvements first. Parameter tunes are preferred
because they deploy immediately without sandbox overhead.
"""
        return prompt

    def _launch_claude_session(self, session_dir: Path) -> List[Dict[str, Any]]:
        """Launch Claude Code subprocess and parse proposals."""
        prompt_file = session_dir / "research_prompt.md"
        timeout_seconds = self.max_session_minutes * 60

        self.logger.info(
            "Launching Claude Code session (timeout=%dm)", self.max_session_minutes,
        )

        try:
            result = subprocess.run(
                [
                    "claude",
                    "--print",
                    "--model", self.model,
                    "--max-turns", "50",
                    "-p", prompt_file.read_text(),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(self._project_root),
            )

            # Save raw output
            (session_dir / "claude_output.txt").write_text(result.stdout or "")
            if result.stderr:
                (session_dir / "claude_stderr.txt").write_text(result.stderr)

            # Try to find proposals.json
            proposals_file = session_dir / "proposals.json"
            if proposals_file.exists():
                proposals = json.loads(proposals_file.read_text())
                self.logger.info("Claude session produced %d proposals", len(proposals))
                return proposals
            else:
                # Try to extract JSON from stdout
                return self._extract_proposals_from_output(result.stdout or "")

        except subprocess.TimeoutExpired:
            self.logger.warning("Claude Code session timed out after %dm", self.max_session_minutes)
            return []
        except FileNotFoundError:
            self.logger.warning("'claude' CLI not found — skipping Claude Code session")
            return []
        except Exception as exc:
            self.logger.error("Claude Code session failed: %s", exc)
            return []

    def _extract_proposals_from_output(self, output: str) -> List[Dict[str, Any]]:
        """Try to extract proposals JSON from Claude's raw output."""
        import re
        # Look for JSON array in output
        matches = re.findall(r'\[[\s\S]*?\]', output)
        for match in reversed(matches):  # Try last match first
            try:
                data = json.loads(match)
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    if "title" in data[0]:
                        return data
            except (json.JSONDecodeError, IndexError):
                continue
        return []

    def _evaluate_proposals(
        self, proposals: List[Dict[str, Any]], report: Dict[str, Any],
    ) -> None:
        """Run each proposal through SafetyGate and store results."""
        summary = report.get("summary", {})
        current_sharpe = summary.get("sharpe_7d", 0.0) or 0.0

        # Estimate current max drawdown from daily P&L
        daily_pnl = report.get("portfolio", {}).get("daily_pnl", [])
        current_dd = 0.05  # default
        if daily_pnl:
            pnls = [d.get("pnl", 0) for d in daily_pnl]
            peak = 0.0
            max_dd = 0.0
            running = 0.0
            for p in pnls:
                running += p
                if running > peak:
                    peak = running
                dd = (peak - running) / max(peak, 1.0)
                if dd > max_dd:
                    max_dd = dd
            if max_dd > 0:
                current_dd = max_dd

        gate = SafetyGate(current_sharpe=current_sharpe, current_max_dd=current_dd)

        accepted = 0
        for p_data in proposals[:self.max_proposals]:
            try:
                proposal = Proposal.from_dict(p_data)
                results = gate.evaluate(proposal)

                # Store in DB regardless of pass/fail
                proposal_dict = proposal.to_dict()
                proposal_dict["safety_gate_results"] = results
                pid = insert_proposal(self.db_path, proposal_dict)
                proposal.proposal_id = pid

                if results["overall"]:
                    accepted += 1
                    self.event_bus.emit("researcher.proposal_accepted", {
                        "id": pid,
                        "title": proposal.title,
                        "category": proposal.category,
                    })
                else:
                    self.event_bus.emit("researcher.proposal_rejected", {
                        "id": pid,
                        "title": proposal.title,
                        "reasons": [
                            g.get("reason") for g in results["gates"].values()
                            if not g["passed"]
                        ],
                    })

            except Exception as exc:
                self.logger.warning("Failed to evaluate proposal: %s", exc)

        self.logger.info(
            "QuantResearcher: %d/%d proposals accepted by SafetyGate",
            accepted, min(len(proposals), self.max_proposals),
        )


# ── Standalone / manual entry point ──

def _main() -> None:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Quant Researcher Agent")
    parser.add_argument("--manual", action="store_true", help="Run one research cycle manually")
    parser.add_argument("--no-claude", action="store_true", help="Skip Claude Code launch (report only)")
    parser.add_argument("--report-only", action="store_true", help="Only compile report, no Claude Code")
    parser.add_argument("--db", default="data/renaissance_bot.db", help="DB path")
    parser.add_argument("--config", default="config/config.json", help="Config path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    config: Dict[str, Any] = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)

    if args.report_only:
        from agents.db_schema import ensure_agent_tables
        ensure_agent_tables(args.db)
        collector = ObservationCollector(db_path=args.db, config=config)
        report = collector.compile_weekly_report()
        filepath = collector.save_report(report)
        print(json.dumps(report.get("summary", {}), indent=2))
        print(f"\nFull report: {filepath}")
        return

    if args.manual:
        from agents.event_bus import EventBus
        from agents.db_schema import ensure_agent_tables
        ensure_agent_tables(args.db)

        bus = EventBus()
        config_copy = dict(config)
        config_copy.setdefault("quant_researcher", {})
        # --no-claude disables Claude launch; --manual alone respects config
        if args.no_claude:
            config_copy["quant_researcher"]["enabled"] = False

        researcher = QuantResearcherAgent(
            event_bus=bus,
            db_path=args.db,
            config=config_copy,
        )
        asyncio.run(researcher.run_weekly_research())
        claude_status = "disabled (--no-claude)" if args.no_claude else (
            "enabled" if config_copy["quant_researcher"].get("enabled") else "disabled (config)"
        )
        print(f"\nManual research cycle complete (Claude Code: {claude_status}).")


if __name__ == "__main__":
    _main()
