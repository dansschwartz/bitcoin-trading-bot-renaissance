"""Proposal, ProposalStatus, DeploymentMode dataclasses for the agent system."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class ProposalStatus(str, Enum):
    PENDING = "pending"
    SAFETY_PASSED = "safety_passed"
    SAFETY_FAILED = "safety_failed"
    SANDBOXING = "sandboxing"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"
    REJECTED = "rejected"
    AWAITING_APPROVAL = "awaiting_approval"


class DeploymentMode(str, Enum):
    """Determines sandbox duration and human-approval requirements."""
    PARAMETER_TUNE = "parameter_tune"       # No sandbox, no approval
    MODIFY_EXISTING = "modify_existing"     # 24h sandbox, no approval
    NEW_FEATURE = "new_feature"             # 72h sandbox, human approval
    NEW_STRATEGY = "new_strategy"           # 168h sandbox, human approval
    MODEL_RETRAIN = "model_retrain"         # No sandbox (has own safety gate), no approval
    INFRASTRUCTURE = "infrastructure"       # No sandbox, human approval required


# Sandbox durations in hours
SANDBOX_HOURS: Dict[DeploymentMode, int] = {
    DeploymentMode.PARAMETER_TUNE: 0,
    DeploymentMode.MODIFY_EXISTING: 24,
    DeploymentMode.NEW_FEATURE: 72,
    DeploymentMode.NEW_STRATEGY: 168,
    DeploymentMode.MODEL_RETRAIN: 0,
    DeploymentMode.INFRASTRUCTURE: 0,
}

# Which modes require human approval
REQUIRES_APPROVAL: Dict[DeploymentMode, bool] = {
    DeploymentMode.PARAMETER_TUNE: False,
    DeploymentMode.MODIFY_EXISTING: False,
    DeploymentMode.NEW_FEATURE: True,
    DeploymentMode.NEW_STRATEGY: True,
    DeploymentMode.MODEL_RETRAIN: False,
    DeploymentMode.INFRASTRUCTURE: True,
}


@dataclass
class Proposal:
    """A suggested improvement from the quant researcher."""
    title: str
    description: str
    category: str                                   # e.g. "signal_weight", "threshold", "model"
    deployment_mode: DeploymentMode = DeploymentMode.PARAMETER_TUNE
    status: ProposalStatus = ProposalStatus.PENDING

    # What to change
    config_changes: Optional[Dict[str, Any]] = None
    code_diff: Optional[str] = None

    # Backtest evidence
    backtest_sharpe: Optional[float] = None
    backtest_drawdown: Optional[float] = None
    backtest_accuracy: Optional[float] = None
    backtest_sample_size: Optional[int] = None
    backtest_p_value: Optional[float] = None

    # Safety gate results
    safety_gate_results: Optional[Dict[str, Any]] = None

    # Lifecycle timestamps (ISO format)
    created_at: Optional[str] = None
    sandbox_start: Optional[str] = None
    sandbox_end: Optional[str] = None
    deployed_at: Optional[str] = None
    rollback_at: Optional[str] = None

    # Metadata
    source: str = "quant_researcher"
    notes: Optional[str] = None
    proposal_id: Optional[int] = None               # DB id after insertion

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["deployment_mode"] = self.deployment_mode.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Proposal:
        data = dict(data)  # shallow copy
        if "deployment_mode" in data and isinstance(data["deployment_mode"], str):
            data["deployment_mode"] = DeploymentMode(data["deployment_mode"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ProposalStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
