"""ConfigDeployer — atomic config backup, apply, and rollback.

Handles the lifecycle of config changes from proposals:
1. Backup current config before changes
2. Apply changes via deep merge
3. Rollback to backup if needed
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfigDeployer:
    """Manages config.json changes with atomic writes and backup/rollback."""

    def __init__(
        self,
        config_path: str,
        backup_dir: str = "data/config_backups",
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.backup_dir = Path(backup_dir).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> Dict[str, Any]:
        """Load current config from disk."""
        with open(self.config_path) as f:
            return json.load(f)

    def save_config(self, config: Dict[str, Any]) -> None:
        """Atomic write: write to tmp file then rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self.config_path.parent),
            suffix=".json.tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                json.dump(config, f, indent=4)
                f.write("\n")
            os.replace(tmp_path, str(self.config_path))
            logger.info("Config saved atomically to %s", self.config_path)
        except Exception:
            # Clean up tmp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def backup(self, proposal_id: int) -> Path:
        """Snapshot current config before applying changes.

        Returns the backup file path.
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_name = f"config_proposal_{proposal_id}_{ts}.json"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(str(self.config_path), str(backup_path))
        logger.info("Config backed up: %s", backup_path)
        return backup_path

    def restore(self, backup_path: Path) -> bool:
        """Restore config from a backup file."""
        backup_path = Path(backup_path)
        if not backup_path.exists():
            logger.error("Backup file not found: %s", backup_path)
            return False
        try:
            # Validate the backup is valid JSON
            with open(backup_path) as f:
                json.load(f)
            shutil.copy2(str(backup_path), str(self.config_path))
            logger.info("Config restored from %s", backup_path)
            return True
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to restore config: %s", exc)
            return False

    def rollback_proposal(self, proposal_id: int) -> bool:
        """Find and restore the backup for a given proposal.

        Searches backup_dir for files matching the proposal_id.
        """
        pattern = f"config_proposal_{proposal_id}_*.json"
        matches = sorted(self.backup_dir.glob(pattern))
        if not matches:
            logger.error("No backup found for proposal %d", proposal_id)
            return False
        # Use the most recent backup for this proposal
        return self.restore(matches[-1])

    def apply_changes(
        self,
        config_changes: Dict[str, Any],
        proposal_id: int,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Backup current config, apply changes, save.

        Returns (backup_path, new_config).
        """
        backup_path = self.backup(proposal_id)
        config = self.load_config()
        new_config = self._deep_merge(config, config_changes)
        self.save_config(new_config)
        logger.info(
            "Config changes applied for proposal %d (backup: %s)",
            proposal_id, backup_path,
        )
        return backup_path, new_config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge overlay into base (overlay wins on conflict)."""
        result = dict(base)
        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigDeployer._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
