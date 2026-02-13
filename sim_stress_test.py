"""Stress testing: black swan injection, correlation stress, liquidity crisis."""

import logging
import numpy as np
from typing import Any, Dict, Optional

from sim_config import DEFAULT_CONFIG


class SimStressTest:
    """Inject extreme scenarios into simulated price paths.

    Supported scenarios:
    - Flash crash (single-day shock)
    - COVID-style sustained decline
    - Luna-style death spiral (accelerating feedback)
    - Correlation stress (force corr → 1 during crisis)
    - Liquidity crisis (multiply transaction costs)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or DEFAULT_CONFIG.get("stress_test", {})
        self.logger = logger or logging.getLogger(__name__)

        self.flash_crash_pct = cfg.get("flash_crash_pct", -0.30)
        self.covid_decline_days = cfg.get("covid_decline_days", 30)
        self.covid_total_decline = cfg.get("covid_total_decline", -0.50)
        self.death_spiral_feedback = cfg.get("death_spiral_feedback", 0.02)
        self.death_spiral_duration = cfg.get("death_spiral_duration", 20)
        self.liquidity_crisis_multiplier = cfg.get("liquidity_crisis_multiplier", 5.0)

    # ------------------------------------------------------------------
    # Flash crash
    # ------------------------------------------------------------------

    def inject_flash_crash(self, paths: np.ndarray, crash_day: int,
                           crash_pct: Optional[float] = None) -> np.ndarray:
        """Inject a single-day crash at *crash_day*.

        All prices from ``crash_day`` onward are multiplied by ``(1 + crash_pct)``.
        """
        pct = crash_pct if crash_pct is not None else self.flash_crash_pct
        if crash_day < 0 or crash_day >= paths.shape[1]:
            return paths.copy()

        stressed = paths.copy()
        factor = 1.0 + pct
        stressed[:, crash_day:] *= factor
        return stressed

    # ------------------------------------------------------------------
    # COVID-style sustained decline
    # ------------------------------------------------------------------

    def inject_covid_decline(self, paths: np.ndarray, start_day: int,
                             duration: Optional[int] = None,
                             total_decline: Optional[float] = None) -> np.ndarray:
        """Apply a sustained daily decline over *duration* days starting at *start_day*.

        The cumulative decline equals ``total_decline`` (e.g. -0.50 for -50%).
        """
        dur = duration if duration is not None else self.covid_decline_days
        decline = total_decline if total_decline is not None else self.covid_total_decline
        daily_factor = (1.0 + decline) ** (1.0 / dur)

        stressed = paths.copy()
        n_steps = paths.shape[1]
        cumulative = 1.0
        for t in range(start_day, min(start_day + dur, n_steps)):
            cumulative *= daily_factor
            stressed[:, t:] *= daily_factor

        return stressed

    # ------------------------------------------------------------------
    # Death spiral (Luna-style)
    # ------------------------------------------------------------------

    def inject_death_spiral(self, paths: np.ndarray, start_day: int,
                            feedback: Optional[float] = None,
                            duration: Optional[int] = None) -> np.ndarray:
        """Luna-style accelerating decline.

        Each day's decline feeds back into the next:
            decline[t] = base_decline * (1 + feedback)^(t - start_day)
        Starting from -5% on day 1.
        """
        fb = feedback if feedback is not None else self.death_spiral_feedback
        dur = duration if duration is not None else self.death_spiral_duration
        base_decline = -0.05

        stressed = paths.copy()
        n_steps = paths.shape[1]
        for t_offset in range(dur):
            t = start_day + t_offset
            if t >= n_steps:
                break
            daily_decline = base_decline * (1.0 + fb) ** t_offset
            daily_decline = max(daily_decline, -0.50)  # cap at -50% per day
            factor = 1.0 + daily_decline
            stressed[:, t:] *= factor

        return stressed

    # ------------------------------------------------------------------
    # Correlation stress
    # ------------------------------------------------------------------

    def stress_correlation(
        self,
        multi_asset_paths: Dict[str, np.ndarray],
        crisis_start: int,
        crisis_end: int,
    ) -> Dict[str, np.ndarray]:
        """Force all asset returns toward perfect correlation during crisis period.

        For each simulation path, during [crisis_start, crisis_end], the return
        of every asset is replaced by the cross-asset average return + small noise.
        """
        assets = list(multi_asset_paths.keys())
        if len(assets) < 2:
            return {k: v.copy() for k, v in multi_asset_paths.items()}

        stressed = {k: v.copy() for k, v in multi_asset_paths.items()}
        n_sims = stressed[assets[0]].shape[0]
        n_steps = stressed[assets[0]].shape[1]
        end = min(crisis_end, n_steps - 1)

        for t in range(max(crisis_start, 1), end + 1):
            # Compute cross-asset average return at this step
            avg_return = np.zeros(n_sims)
            for asset in assets:
                log_ret = np.log(stressed[asset][:, t] / stressed[asset][:, t - 1])
                avg_return += log_ret
            avg_return /= len(assets)

            # Replace each asset's return with the average + tiny noise
            rng = np.random.default_rng(t)
            for asset in assets:
                noise = rng.normal(0, 0.001, n_sims)
                new_price = stressed[asset][:, t - 1] * np.exp(avg_return + noise)
                # Propagate the change forward
                ratio = new_price / stressed[asset][:, t]
                stressed[asset][:, t:] *= ratio[:, np.newaxis]

        return stressed

    # ------------------------------------------------------------------
    # Liquidity crisis
    # ------------------------------------------------------------------

    def inject_liquidity_crisis(
        self,
        cost_params: Dict[str, float],
        multiplier: Optional[float] = None,
    ) -> Dict[str, float]:
        """Multiply all transaction cost parameters by *multiplier*.

        Returns a new cost parameter dict suitable for SimTransactionCostModel.
        """
        mult = multiplier if multiplier is not None else self.liquidity_crisis_multiplier
        scaled = {}
        for key, val in cost_params.items():
            if isinstance(val, (int, float)):
                scaled[key] = val * mult
            else:
                scaled[key] = val
        return scaled
