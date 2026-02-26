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
import sys
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

    @staticmethod
    def _clean_env() -> dict:
        """Return env dict with CLAUDECODE unset so nested sessions work."""
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        return env

    def _check_claude_available(self) -> bool:
        """Check if the claude CLI is available on PATH."""
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True, text=True, timeout=10,
                env=self._clean_env(),
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

            # Step 3: Launch research session (single or council mode)
            research_mode = self.researcher_cfg.get("research_mode", "single")

            if research_mode == "council":
                proposals = self._run_council_cycle(session_dir, report)
            else:
                # Existing single-researcher mode (PRESERVED)
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
                env=self._clean_env(),
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

    # ── Council mode methods ──

    def _run_council_cycle(
        self, session_dir: Path, report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Run the full Executive Research Council cycle.

        Phase 1: Hypothesis generation — 5 specialist sessions
        Phase 2: Peer review — 5 review sessions
        Phase 3: Consensus scoring — pure Python

        Returns proposals that pass consensus threshold (for SafetyGate).
        """
        researchers = self.researcher_cfg.get("researchers", [
            "mathematician", "cryptographer", "physicist",
            "linguist", "systems_engineer",
        ])
        max_turns = self.researcher_cfg.get("council_max_turns", 30)
        max_minutes = self.researcher_cfg.get("council_max_minutes", 25)
        project_root = self._project_root

        # Prepare snapshot
        snapshot_script = project_root / "scripts" / "prepare_research_snapshot.sh"
        if snapshot_script.exists():
            try:
                subprocess.run(
                    ["bash", str(snapshot_script)], timeout=120,
                    capture_output=True, cwd=str(project_root),
                )
                self.logger.info("Research snapshot created")
            except Exception as exc:
                self.logger.warning("Snapshot creation failed (continuing): %s", exc)

        snapshot_dir = project_root / "data" / "research_snapshots" / "latest"

        # Phase 1: Hypothesis generation
        self.logger.info(
            "Council Phase 1: Hypothesis generation (%d researchers)", len(researchers),
        )
        for name in researchers:
            self.logger.info("  Launching researcher: %s", name)
            try:
                self._launch_researcher_session(
                    researcher_name=name,
                    phase="hypothesize",
                    session_dir=session_dir,
                    snapshot_dir=snapshot_dir,
                    project_root=project_root,
                    max_turns=max_turns,
                    timeout_minutes=max_minutes,
                )
            except Exception as exc:
                self.logger.warning("Researcher %s hypothesis session failed: %s", name, exc)

        # Phase 2: Peer review
        self.logger.info(
            "Council Phase 2: Peer review (%d researchers)", len(researchers),
        )
        for name in researchers:
            self.logger.info("  Launching reviewer: %s", name)
            try:
                self._launch_researcher_session(
                    researcher_name=name,
                    phase="review",
                    session_dir=session_dir,
                    snapshot_dir=snapshot_dir,
                    project_root=project_root,
                    max_turns=max(10, max_turns // 2),
                    timeout_minutes=max(10, max_minutes // 2),
                )
            except Exception as exc:
                self.logger.warning("Researcher %s review session failed: %s", name, exc)

        # Phase 3: Consensus scoring
        self.logger.info("Council Phase 3: Consensus scoring")
        try:
            scripts_dir = str(project_root / "scripts")
            if scripts_dir not in sys.path:
                sys.path.insert(0, scripts_dir)
            from score_proposals import score
            passing_proposals = score(session_dir)
        except Exception as exc:
            self.logger.error("Consensus scoring failed: %s", exc)
            passing_proposals = self._collect_all_proposals(session_dir, researchers)

        # Update outcome ledger
        try:
            updater = project_root / "scripts" / "update_council_memory.py"
            if updater.exists():
                subprocess.run(
                    [str(project_root / ".venv" / "bin" / "python3"), str(updater)],
                    timeout=30, capture_output=True, cwd=str(project_root),
                )
        except Exception:
            pass

        self.logger.info(
            "Council cycle complete: %d proposals pass consensus", len(passing_proposals),
        )
        return passing_proposals

    def _launch_researcher_session(
        self,
        researcher_name: str,
        phase: str,
        session_dir: Path,
        snapshot_dir: Path,
        project_root: Path,
        max_turns: int = 30,
        timeout_minutes: int = 25,
    ) -> None:
        """Launch a single Claude Code session for one researcher in one phase."""
        # Load researcher profile
        profile_path = project_root / "researchers" / f"{researcher_name}.md"
        if not profile_path.exists():
            self.logger.warning("Researcher profile not found: %s", profile_path)
            return
        researcher_profile = profile_path.read_text()

        # Load journal (institutional memory)
        journal_path = (
            project_root / "data" / "council_memory" / "journals"
            / f"{researcher_name}_journal.md"
        )
        journal_content = journal_path.read_text() if journal_path.exists() else "(No previous sessions)"

        # Load outcome ledger
        ledger_path = project_root / "data" / "council_memory" / "outcome_ledger.json"
        ledger_content = ledger_path.read_text() if ledger_path.exists() else "{}"

        # Create per-researcher output directory
        if phase == "hypothesize":
            output_dir = session_dir / "proposals" / researcher_name
        else:
            output_dir = session_dir / "reviews" / researcher_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build prompt
        if phase == "hypothesize":
            prompt = self._build_hypothesis_prompt(
                researcher_name, researcher_profile, journal_content,
                ledger_content, session_dir, snapshot_dir, output_dir,
            )
        elif phase == "review":
            prompt = self._build_review_prompt(
                researcher_name, researcher_profile, journal_content,
                session_dir, output_dir,
            )
        else:
            self.logger.error("Unknown phase: %s", phase)
            return

        # Save prompt for debugging
        (output_dir / f"{phase}_prompt.md").write_text(prompt)

        # Launch Claude Code
        try:
            result = subprocess.run(
                ["claude", "--print", "--max-turns", str(max_turns), "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60,
                cwd=str(project_root),
                env=self._clean_env(),
            )
            (output_dir / "claude_output.txt").write_text(result.stdout or "")
            if result.stderr:
                (output_dir / "claude_stderr.txt").write_text(result.stderr)

            # Check for proposals.json or reviews.json
            expected_file = "proposals.json" if phase == "hypothesize" else "reviews.json"
            if not (output_dir / expected_file).exists():
                extracted = self._extract_proposals_from_output(result.stdout or "")
                if extracted:
                    (output_dir / expected_file).write_text(
                        json.dumps(extracted, indent=2, default=str)
                    )

            # Harvest updated journal
            self._harvest_journal_update(researcher_name, result.stdout or "", project_root)

        except subprocess.TimeoutExpired:
            self.logger.warning(
                "Researcher %s %s timed out after %dm",
                researcher_name, phase, timeout_minutes,
            )
        except FileNotFoundError:
            self.logger.warning("'claude' CLI not found — skipping session")
        except Exception as exc:
            self.logger.error("Researcher %s %s failed: %s", researcher_name, phase, exc)

    def _build_hypothesis_prompt(
        self,
        name: str,
        profile: str,
        journal: str,
        ledger: str,
        session_dir: Path,
        snapshot_dir: Path,
        output_dir: Path,
    ) -> str:
        """Build the hypothesis-phase prompt for a researcher."""
        report_path = session_dir / "weekly_report.json"
        project_root = self._project_root

        # Load dynamic brief if available
        brief_path = project_root / "data" / "council_memory" / "briefs" / f"{name}_brief.md"
        brief = brief_path.read_text() if brief_path.exists() else "(No brief available)"

        return f"""You are the {name.replace('_', ' ').title()} on the Executive Research Council.

{profile}

KNOWLEDGE LIBRARY — import and use:
    import sys; sys.path.insert(0, 'researchers')
    from knowledge.registry import KB
    from knowledge.atoms import *
    print(KB.manifest("{name}"))
    result = KB.execute("math.kelly_optimal", p=0.54, b=1.2)
    from knowledge.shared.data_loader import load_pair_csv, get_aligned_returns
    from knowledge.shared.queries import weekly_performance, correlation_matrix
    from knowledge.shared.dead_ends import is_dead_end

DYNAMIC BRIEF:
{brief}

YOUR RESEARCH JOURNAL (institutional memory — what you've done before):
{journal}

SHARED OUTCOME LEDGER (what the whole council has deployed/rolled back):
{ledger}

WEEKLY REPORT: Read the file at {report_path}
DATABASE SNAPSHOT: {snapshot_dir}/trading_snapshot.db (read-only SQLite — query freely)
HISTORICAL DATA: {snapshot_dir}/training_data/ (5-year CSVs per pair)

YOUR TASK:
1. Analyze the weekly report through YOUR specific scientific lens
2. Import and run your diagnostic: KB.diagnostic("{name}")
3. Check dead ends before proposing: from knowledge.shared.dead_ends import is_dead_end
4. Check your journal — build on standing hypotheses, do NOT repropose failed ideas
5. Generate 1-3 improvement proposals based on your domain expertise
6. For each proposal, write implementation code and run a backtest if possible:
   .venv/bin/python3 -m backtesting.engine --walk-forward --pairs BTC-USD ETH-USD SOL-USD --total-months 3 --train-months 2 --test-months 1
7. Save results to {output_dir}/proposals.json using this exact format:
[
  {{
    "title": "Short descriptive title",
    "description": "What to change and why",
    "category": "parameter_tune",
    "deployment_mode": "parameter_tune",
    "config_changes": {{}},
    "backtest_sharpe": 0.0,
    "backtest_drawdown": 0.0,
    "backtest_accuracy": 0.0,
    "backtest_sample_size": 0,
    "backtest_p_value": 0.0,
    "expected_improvement_bps": 0.0,
    "notes": ""
  }}
]

FINAL TASK: Update your research journal at data/council_memory/journals/{name}_journal.md
with what you proposed, what you learned, and standing hypotheses for next week.

CONSTRAINTS:
- NEVER modify risk_gateway.py, safety limits, or circuit breakers
- NEVER propose increasing leverage
- Save ALL work to {output_dir}/ — never modify production code directly
"""

    def _build_review_prompt(
        self,
        name: str,
        profile: str,
        journal: str,
        session_dir: Path,
        output_dir: Path,
    ) -> str:
        """Build the peer-review prompt for a researcher."""
        researchers = [
            'mathematician', 'cryptographer', 'physicist',
            'linguist', 'systems_engineer',
        ]
        other_proposals: Dict[str, Any] = {}
        for other in researchers:
            if other == name:
                continue
            prop_file = session_dir / "proposals" / other / "proposals.json"
            if prop_file.exists():
                try:
                    other_proposals[other] = json.loads(prop_file.read_text())
                except Exception:
                    pass

        if not other_proposals:
            (output_dir / "reviews.json").write_text("[]")
            return "No proposals from other researchers to review. Session complete."

        return f"""You are the {name.replace('_', ' ').title()} on the Executive Research Council.
You are now in PEER REVIEW mode.

{profile}

YOUR JOURNAL (for context on your own domain expertise):
{journal}

YOUR TASK: Review the following proposals from your fellow researchers.
For each proposal, provide your verdict:
- ENDORSE — the proposal is sound from your domain perspective
- CHALLENGE — the proposal has issues but could be improved (explain how)
- REJECT — the proposal is fundamentally flawed (explain why)

PROPOSALS TO REVIEW:
{json.dumps(other_proposals, indent=2, default=str)}

Save your reviews to {output_dir}/reviews.json in this format:
[
  {{
    "researcher": "name_of_proposer",
    "proposal_title": "title from their proposal",
    "verdict": "endorse|challenge|reject",
    "reasoning": "Your domain-specific analysis (2-4 sentences)",
    "suggested_improvements": "If challenging, what would fix it",
    "confidence": 0.8
  }}
]

Be rigorous. Challenge weak reasoning. Endorse only what you'd stake your reputation on.
"""

    def _collect_all_proposals(
        self, session_dir: Path, researchers: List[str],
    ) -> List[Dict[str, Any]]:
        """Fallback: collect all proposals without consensus filtering."""
        all_props: List[Dict[str, Any]] = []
        for name in researchers:
            prop_file = session_dir / "proposals" / name / "proposals.json"
            if prop_file.exists():
                try:
                    props = json.loads(prop_file.read_text())
                    for p in props:
                        p["_source_researcher"] = name
                    all_props.extend(props)
                except Exception:
                    pass
        return all_props

    def _harvest_journal_update(
        self, researcher_name: str, stdout: str, project_root: Path,
    ) -> None:
        """Check if the researcher updated their journal during the session."""
        journal_path = (
            project_root / "data" / "council_memory" / "journals"
            / f"{researcher_name}_journal.md"
        )
        if journal_path.exists():
            lines = len(journal_path.read_text().splitlines())
            self.logger.info("Journal for %s: %d lines", researcher_name, lines)

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
    parser.add_argument("--council", action="store_true", help="Run full council cycle instead of single researcher")
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

    if args.manual or args.council:
        from agents.event_bus import EventBus
        from agents.db_schema import ensure_agent_tables
        ensure_agent_tables(args.db)

        bus = EventBus()
        config_copy = dict(config)
        config_copy.setdefault("quant_researcher", {})
        # --no-claude disables Claude launch; --manual alone respects config
        if args.no_claude:
            config_copy["quant_researcher"]["enabled"] = False
        # --council overrides mode
        if args.council:
            config_copy["quant_researcher"]["research_mode"] = "council"
            config_copy["quant_researcher"]["enabled"] = True

        researcher = QuantResearcherAgent(
            event_bus=bus,
            db_path=args.db,
            config=config_copy,
        )
        asyncio.run(researcher.run_weekly_research())
        mode = config_copy["quant_researcher"].get("research_mode", "single")
        claude_status = "disabled (--no-claude)" if args.no_claude else (
            f"enabled ({mode} mode)" if config_copy["quant_researcher"].get("enabled") else "disabled (config)"
        )
        print(f"\nManual research cycle complete (Claude Code: {claude_status}).")


if __name__ == "__main__":
    _main()
