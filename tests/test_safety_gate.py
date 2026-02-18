"""Tests for agents.safety_gate.SafetyGate — all 5 gates pass/fail."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.safety_gate import SafetyGate, ABSOLUTE_MAX_POSITION_USD, PAPER_TRADING_ONLY
from agents.proposal import Proposal, ProposalStatus, DeploymentMode


class TestSafetyGate:
    """All 5 gates pass/fail, deployment mode logic."""

    def _make_proposal(self, **kwargs) -> Proposal:
        defaults = {
            "title": "Test Proposal",
            "description": "Test",
            "category": "parameter_tune",
            "deployment_mode": DeploymentMode.PARAMETER_TUNE,
            "backtest_sharpe": 1.0,
            "backtest_drawdown": 0.03,
            "backtest_accuracy": 0.55,
            "backtest_sample_size": 200,
            "backtest_p_value": 0.01,
        }
        defaults.update(kwargs)
        return Proposal(**defaults)

    def test_all_gates_pass(self):
        gate = SafetyGate(current_sharpe=0.5, current_max_dd=0.05)
        proposal = self._make_proposal()
        result = gate.evaluate(proposal)
        assert result["overall"] is True
        assert proposal.status == ProposalStatus.SAFETY_PASSED

    def test_sharpe_gate_fail(self):
        gate = SafetyGate(current_sharpe=2.0, current_max_dd=0.05)
        proposal = self._make_proposal(backtest_sharpe=0.5)
        result = gate.evaluate(proposal)
        assert result["gates"]["sharpe"]["passed"] is False
        assert result["overall"] is False

    def test_sharpe_gate_marginal_pass(self):
        gate = SafetyGate(current_sharpe=1.0, current_max_dd=0.05)
        # threshold = 1.0 - 0.1 = 0.9
        proposal = self._make_proposal(backtest_sharpe=0.9)
        result = gate.evaluate(proposal)
        assert result["gates"]["sharpe"]["passed"] is True

    def test_drawdown_gate_fail(self):
        gate = SafetyGate(current_sharpe=0.5, current_max_dd=0.03)
        # threshold = 0.03 * 1.2 = 0.036
        proposal = self._make_proposal(backtest_drawdown=0.05)
        result = gate.evaluate(proposal)
        assert result["gates"]["drawdown"]["passed"] is False

    def test_accuracy_gate_fail(self):
        gate = SafetyGate()
        proposal = self._make_proposal(backtest_accuracy=0.49)
        result = gate.evaluate(proposal)
        assert result["gates"]["accuracy"]["passed"] is False

    def test_accuracy_gate_at_boundary(self):
        gate = SafetyGate()
        proposal = self._make_proposal(backtest_accuracy=0.50)
        result = gate.evaluate(proposal)
        assert result["gates"]["accuracy"]["passed"] is True

    def test_significance_gate_fail(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            deployment_mode=DeploymentMode.MODIFY_EXISTING,
            backtest_p_value=0.06,
        )
        result = gate.evaluate(proposal)
        assert result["gates"]["significance"]["passed"] is False

    def test_significance_gate_parameter_tune_exempt(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            deployment_mode=DeploymentMode.PARAMETER_TUNE,
            backtest_p_value=None,  # No p-value, but exempt
        )
        result = gate.evaluate(proposal)
        assert result["gates"]["significance"]["passed"] is True

    def test_sample_size_gate_fail(self):
        gate = SafetyGate()
        proposal = self._make_proposal(backtest_sample_size=50)
        result = gate.evaluate(proposal)
        assert result["gates"]["sample_size"]["passed"] is False

    def test_sample_size_gate_pass(self):
        gate = SafetyGate()
        proposal = self._make_proposal(backtest_sample_size=100)
        result = gate.evaluate(proposal)
        assert result["gates"]["sample_size"]["passed"] is True

    def test_missing_backtest_data_fails(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            backtest_sharpe=None,
            backtest_drawdown=None,
            backtest_accuracy=None,
            backtest_sample_size=None,
        )
        result = gate.evaluate(proposal)
        assert result["overall"] is False

    def test_absolute_limits_position(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            config_changes={"risk_management": {"position_limit": 50000}},
        )
        result = gate.evaluate(proposal)
        assert result["gates"]["absolute_limits"]["passed"] is False
        assert "position_limit" in result["gates"]["absolute_limits"]["violations"][0]

    def test_absolute_limits_paper_trading(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            config_changes={"trading": {"paper_trading": False}},
        )
        result = gate.evaluate(proposal)
        assert result["gates"]["absolute_limits"]["passed"] is False

    def test_absolute_limits_clean_pass(self):
        gate = SafetyGate()
        proposal = self._make_proposal(
            config_changes={"signal_weights": {"rsi": 0.12}},
        )
        result = gate.evaluate(proposal)
        assert result["gates"]["absolute_limits"]["passed"] is True

    def test_multiple_gates_fail(self):
        gate = SafetyGate(current_sharpe=2.0, current_max_dd=0.02)
        proposal = self._make_proposal(
            backtest_sharpe=0.1,
            backtest_drawdown=0.5,
            backtest_accuracy=0.3,
        )
        result = gate.evaluate(proposal)
        assert result["overall"] is False
        failed = [g for g in result["gates"].values() if not g["passed"]]
        assert len(failed) >= 3
