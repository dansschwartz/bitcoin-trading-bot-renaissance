"""SafetyGate — 5 independent gates, ALL must pass for a proposal to be approved.

Hard-coded thresholds — intentionally NOT configurable. No agent can weaken
these limits.

Gates:
  1. Sharpe: backtest_sharpe >= current_sharpe - 0.1
  2. Drawdown: backtest_max_dd <= current_max_dd * 1.2
  3. Accuracy: backtest_accuracy >= 0.50
  4. Significance: p_value < 0.05 (parameter_tune exempt)
  5. Sample size: backtest_trades >= 100
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from agents.proposal import DeploymentMode, Proposal, ProposalStatus

logger = logging.getLogger(__name__)

# ── Hard-coded absolute safety limits — NEVER make these configurable ──
ABSOLUTE_MAX_POSITION_USD = 10_000
ABSOLUTE_MAX_DRAWDOWN_PCT = 0.10
ABSOLUTE_MAX_LEVERAGE = 3.0
ABSOLUTE_MAX_DAILY_TRADES = 200
PAPER_TRADING_ONLY = True


class SafetyGate:
    """Evaluate a proposal against 5 independent safety gates."""

    def __init__(
        self,
        current_sharpe: float = 0.0,
        current_max_dd: float = 0.05,
    ) -> None:
        self.current_sharpe = current_sharpe
        self.current_max_dd = current_max_dd

    def evaluate(self, proposal: Proposal) -> Dict[str, Any]:
        """Run all 5 gates.  Returns a dict with per-gate results + overall pass/fail."""
        results: Dict[str, Any] = {
            "overall": True,
            "gates": {},
        }

        gates = [
            self._gate_sharpe(proposal),
            self._gate_drawdown(proposal),
            self._gate_accuracy(proposal),
            self._gate_significance(proposal),
            self._gate_sample_size(proposal),
        ]

        for gate_result in gates:
            gate_name = gate_result["gate"]
            results["gates"][gate_name] = gate_result
            if not gate_result["passed"]:
                results["overall"] = False

        # Check absolute limits in config_changes
        abs_check = self._check_absolute_limits(proposal)
        results["gates"]["absolute_limits"] = abs_check
        if not abs_check["passed"]:
            results["overall"] = False

        # Update proposal
        proposal.safety_gate_results = results
        if results["overall"]:
            proposal.status = ProposalStatus.SAFETY_PASSED
        else:
            proposal.status = ProposalStatus.SAFETY_FAILED

        logger.info(
            "SafetyGate: proposal '%s' — %s (%d/%d gates passed)",
            proposal.title,
            "PASSED" if results["overall"] else "FAILED",
            sum(1 for g in results["gates"].values() if g["passed"]),
            len(results["gates"]),
        )

        return results

    # ── Individual gates ──

    def _gate_sharpe(self, p: Proposal) -> Dict[str, Any]:
        """Gate 1: backtest_sharpe >= current_sharpe - 0.1"""
        threshold = self.current_sharpe - 0.1
        value = p.backtest_sharpe
        if value is None:
            return {"gate": "sharpe", "passed": False, "reason": "no backtest Sharpe provided",
                    "threshold": threshold, "value": None}
        passed = value >= threshold
        return {
            "gate": "sharpe",
            "passed": passed,
            "threshold": round(threshold, 4),
            "value": round(value, 4),
            "reason": None if passed else f"backtest Sharpe {value:.4f} < threshold {threshold:.4f}",
        }

    def _gate_drawdown(self, p: Proposal) -> Dict[str, Any]:
        """Gate 2: backtest_max_dd <= current_max_dd * 1.2"""
        threshold = self.current_max_dd * 1.2
        value = p.backtest_drawdown
        if value is None:
            return {"gate": "drawdown", "passed": False, "reason": "no backtest drawdown provided",
                    "threshold": threshold, "value": None}
        passed = value <= threshold
        return {
            "gate": "drawdown",
            "passed": passed,
            "threshold": round(threshold, 4),
            "value": round(value, 4),
            "reason": None if passed else f"backtest drawdown {value:.4f} > threshold {threshold:.4f}",
        }

    def _gate_accuracy(self, p: Proposal) -> Dict[str, Any]:
        """Gate 3: backtest_accuracy >= 0.50"""
        threshold = 0.50
        value = p.backtest_accuracy
        if value is None:
            return {"gate": "accuracy", "passed": False, "reason": "no backtest accuracy provided",
                    "threshold": threshold, "value": None}
        passed = value >= threshold
        return {
            "gate": "accuracy",
            "passed": passed,
            "threshold": threshold,
            "value": round(value, 4),
            "reason": None if passed else f"backtest accuracy {value:.4f} < {threshold}",
        }

    def _gate_significance(self, p: Proposal) -> Dict[str, Any]:
        """Gate 4: p_value < 0.05 (parameter_tune exempt)."""
        if p.deployment_mode == DeploymentMode.PARAMETER_TUNE:
            return {"gate": "significance", "passed": True,
                    "reason": "parameter_tune exempt", "threshold": 0.05, "value": None}
        threshold = 0.05
        value = p.backtest_p_value
        if value is None:
            return {"gate": "significance", "passed": False, "reason": "no p-value provided",
                    "threshold": threshold, "value": None}
        passed = value < threshold
        return {
            "gate": "significance",
            "passed": passed,
            "threshold": threshold,
            "value": round(value, 6),
            "reason": None if passed else f"p-value {value:.6f} >= {threshold}",
        }

    def _gate_sample_size(self, p: Proposal) -> Dict[str, Any]:
        """Gate 5: backtest_trades >= 100."""
        threshold = 100
        value = p.backtest_sample_size
        if value is None:
            return {"gate": "sample_size", "passed": False, "reason": "no sample size provided",
                    "threshold": threshold, "value": None}
        passed = value >= threshold
        return {
            "gate": "sample_size",
            "passed": passed,
            "threshold": threshold,
            "value": value,
            "reason": None if passed else f"sample size {value} < {threshold}",
        }

    def _check_absolute_limits(self, p: Proposal) -> Dict[str, Any]:
        """Verify proposal doesn't exceed hard-coded absolute limits."""
        violations: list[str] = []
        if p.config_changes:
            changes = p.config_changes
            # Check position size
            max_pos = changes.get("risk_management", {}).get("position_limit")
            if max_pos is not None and max_pos > ABSOLUTE_MAX_POSITION_USD:
                violations.append(
                    f"position_limit {max_pos} > absolute max {ABSOLUTE_MAX_POSITION_USD}"
                )
            # Check drawdown
            max_dd = changes.get("risk_management", {}).get("max_drawdown_pct")
            if max_dd is not None and max_dd > ABSOLUTE_MAX_DRAWDOWN_PCT:
                violations.append(
                    f"max_drawdown_pct {max_dd} > absolute max {ABSOLUTE_MAX_DRAWDOWN_PCT}"
                )
            # Check leverage
            max_lev = changes.get("risk_management", {}).get("max_leverage")
            if max_lev is not None and max_lev > ABSOLUTE_MAX_LEVERAGE:
                violations.append(
                    f"max_leverage {max_lev} > absolute max {ABSOLUTE_MAX_LEVERAGE}"
                )
            # Check daily trades
            max_trades = changes.get("risk_management", {}).get("max_daily_trades")
            if max_trades is not None and max_trades > ABSOLUTE_MAX_DAILY_TRADES:
                violations.append(
                    f"max_daily_trades {max_trades} > absolute max {ABSOLUTE_MAX_DAILY_TRADES}"
                )
            # Check paper trading flag
            paper = changes.get("trading", {}).get("paper_trading")
            if paper is not None and paper is False and PAPER_TRADING_ONLY:
                violations.append("cannot disable paper trading (PAPER_TRADING_ONLY=True)")

        passed = len(violations) == 0
        return {
            "gate": "absolute_limits",
            "passed": passed,
            "violations": violations,
            "reason": "; ".join(violations) if violations else None,
        }
