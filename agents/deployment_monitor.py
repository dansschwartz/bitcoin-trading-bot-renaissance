"""DeploymentMonitor — sandbox tracking and auto-rollback.

Monitors proposals in sandbox mode.  When the sandbox period ends, compares
live metrics against pre-deployment baselines.  If degradation is detected
beyond tolerance, automatically rolls back config changes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from agents.proposal import (
    DeploymentMode,
    Proposal,
    ProposalStatus,
    SANDBOX_HOURS,
    REQUIRES_APPROVAL,
)
from agents.db_schema import log_improvement

logger = logging.getLogger(__name__)


class DeploymentMonitor:
    """Track sandboxed proposals and auto-rollback on degradation."""

    def __init__(self, db_path: str, config: Dict[str, Any]) -> None:
        self.db_path = db_path
        self.config = config
        # Rollback thresholds (relative degradation)
        self._sharpe_degradation_threshold = 0.3  # 30% Sharpe decline triggers rollback
        self._drawdown_increase_threshold = 0.5   # 50% drawdown increase triggers rollback

    def should_sandbox(self, proposal: Proposal) -> bool:
        """Return True if this proposal needs sandboxing."""
        hours = SANDBOX_HOURS.get(proposal.deployment_mode, 0)
        return hours > 0

    def needs_human_approval(self, proposal: Proposal) -> bool:
        """Return True if this deployment mode requires human sign-off."""
        return REQUIRES_APPROVAL.get(proposal.deployment_mode, True)

    def start_sandbox(self, proposal: Proposal) -> None:
        """Mark a proposal as entering sandbox mode."""
        now = datetime.now(timezone.utc)
        hours = SANDBOX_HOURS.get(proposal.deployment_mode, 24)
        end = now + timedelta(hours=hours)

        proposal.sandbox_start = now.isoformat()
        proposal.sandbox_end = end.isoformat()
        proposal.status = ProposalStatus.SANDBOXING

        # Update DB
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                """UPDATE proposals SET status=?, sandbox_start=?, sandbox_end=?,
                   updated_at=? WHERE id=?""",
                (
                    ProposalStatus.SANDBOXING.value,
                    proposal.sandbox_start,
                    proposal.sandbox_end,
                    now.isoformat(),
                    proposal.proposal_id,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.warning("Failed to update proposal sandbox in DB: %s", exc)

        logger.info(
            "Sandbox started for proposal '%s' (%dh, ends %s)",
            proposal.title, hours, proposal.sandbox_end,
        )

    def check_sandbox_completion(self) -> List[Dict[str, Any]]:
        """Check for proposals whose sandbox period has ended.

        Returns list of proposals ready for deployment or rollback evaluation.
        """
        now = datetime.now(timezone.utc).isoformat()
        completed: List[Dict[str, Any]] = []

        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            rows = conn.execute(
                """SELECT id, title, category, deployment_mode, sandbox_start,
                          sandbox_end, config_changes, safety_gate_results
                   FROM proposals
                   WHERE status = 'sandboxing' AND sandbox_end <= ?""",
                (now,),
            ).fetchall()
            conn.close()

            for row in rows:
                completed.append({
                    "id": row[0],
                    "title": row[1],
                    "category": row[2],
                    "deployment_mode": row[3],
                    "sandbox_start": row[4],
                    "sandbox_end": row[5],
                    "config_changes": json.loads(row[6]) if row[6] else None,
                    "safety_gate_results": json.loads(row[7]) if row[7] else None,
                })
        except Exception as exc:
            logger.warning("check_sandbox_completion error: %s", exc)

        return completed

    def evaluate_sandbox_result(
        self,
        proposal_id: int,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare pre/post sandbox metrics and decide deploy vs rollback.

        Parameters
        ----------
        metrics_before : dict with keys 'sharpe', 'drawdown', 'win_rate'
        metrics_after : same structure, measured during sandbox
        """
        result: Dict[str, Any] = {
            "proposal_id": proposal_id,
            "action": "deploy",  # default
            "reasons": [],
        }

        sharpe_before = metrics_before.get("sharpe", 0.0)
        sharpe_after = metrics_after.get("sharpe", 0.0)
        dd_before = metrics_before.get("drawdown", 0.0)
        dd_after = metrics_after.get("drawdown", 0.0)

        # Check Sharpe degradation
        if sharpe_before > 0 and sharpe_after < sharpe_before * (1 - self._sharpe_degradation_threshold):
            result["action"] = "rollback"
            result["reasons"].append(
                f"Sharpe degraded {sharpe_before:.3f} → {sharpe_after:.3f} "
                f"(>{self._sharpe_degradation_threshold*100:.0f}% decline)"
            )

        # Check drawdown increase
        if dd_before > 0 and dd_after > dd_before * (1 + self._drawdown_increase_threshold):
            result["action"] = "rollback"
            result["reasons"].append(
                f"Drawdown increased {dd_before:.3f} → {dd_after:.3f} "
                f"(>{self._drawdown_increase_threshold*100:.0f}% increase)"
            )

        result["metrics_before"] = metrics_before
        result["metrics_after"] = metrics_after

        return result

    def deploy_proposal(self, proposal_id: int) -> None:
        """Mark a proposal as deployed."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "UPDATE proposals SET status=?, deployed_at=?, updated_at=? WHERE id=?",
                (ProposalStatus.DEPLOYED.value, now, now, proposal_id),
            )
            conn.commit()
            conn.close()
            log_improvement(
                self.db_path,
                proposal_id=proposal_id,
                change_type="deploy",
                description=f"Proposal {proposal_id} deployed after sandbox",
            )
        except Exception as exc:
            logger.error("deploy_proposal failed: %s", exc)

    def rollback_proposal(self, proposal_id: int, reason: str = "") -> None:
        """Mark a proposal as rolled back."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute(
                "UPDATE proposals SET status=?, rollback_at=?, updated_at=?, notes=? WHERE id=?",
                (ProposalStatus.ROLLED_BACK.value, now, now, reason, proposal_id),
            )
            conn.commit()
            conn.close()
            log_improvement(
                self.db_path,
                proposal_id=proposal_id,
                change_type="rollback",
                description=f"Proposal {proposal_id} rolled back: {reason}",
            )
        except Exception as exc:
            logger.error("rollback_proposal failed: %s", exc)

    def get_active_sandboxes(self) -> List[Dict[str, Any]]:
        """Return all currently sandboxing proposals."""
        try:
            conn = sqlite3.connect(
                f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0,
            )
            rows = conn.execute(
                """SELECT id, title, category, deployment_mode,
                          sandbox_start, sandbox_end
                   FROM proposals WHERE status = 'sandboxing'
                   ORDER BY sandbox_end"""
            ).fetchall()
            conn.close()
            return [
                {
                    "id": r[0], "title": r[1], "category": r[2],
                    "deployment_mode": r[3], "sandbox_start": r[4],
                    "sandbox_end": r[5],
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("get_active_sandboxes error: %s", exc)
            return []
