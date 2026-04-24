"""Abstract base class that all simulation models implement."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import numpy as np


class SimulationModel(ABC):
    """Base class for price-path simulation models.

    Every concrete model must implement ``calibrate`` and ``simulate``.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._calibrated: bool = False
        self._parameters: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def calibrate(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Calibrate model parameters from empirical data.

        Args:
            returns: array of log returns (length N-1 for N prices).
            prices: array of close prices (length N).

        Returns:
            Dict of calibrated parameters.
        """

    @abstractmethod
    def simulate(self, S0: float, n_steps: int, n_simulations: int,
                 dt: float = 1.0 / 252, seed: Optional[int] = None) -> np.ndarray:
        """Generate simulated price paths.

        Returns:
            ndarray of shape ``(n_simulations, n_steps + 1)`` starting at *S0*.
        """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self.__class__.__name__

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated
