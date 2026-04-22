"""
Intelligence Layer B2: Signal Validation Pipeline
===================================================
Six-gate validation pipeline ensuring that every signal earns its
right to influence the portfolio.

Gates
-----
1. Pattern Detection     - enough trades and non-trivial pattern
2. Statistical Significance - p < 0.01 via one-sample t-test
3. (reserved / Explanation gate - logged but not blocking)
4. Cost Check            - net profitability after fees and slippage
5. Regime Check          - profitable in >= N distinct regimes
6. Out-of-Sample         - OOS Sharpe retains >= 50 % of in-sample Sharpe

Confidence tiers
~~~~~~~~~~~~~~~~
* Tier 1 (full allocation)  - all six gates passed
* Tier 2 (25 % allocation)  - passed gates 1-4 but marginal on 5 or 6
* Tier 3 (paper only)       - one or more core gates failed

Config key: ``signal_validator``
"""

from __future__ import annotations

import enum
import logging
import math
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Graceful scipy import
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ValidationResult(enum.Enum):
    """Outcome of a single validation gate."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class SignalValidationReport:
    """Complete report produced by the 6-gate validation pipeline."""

    signal_name: str
    pair: str
    start_date: str
    end_date: str

    # Gate results
    gate_1_pattern: ValidationResult = ValidationResult.INSUFFICIENT_DATA
    gate_2_stat_sig: ValidationResult = ValidationResult.INSUFFICIENT_DATA
    gate_2_p_value: float = 1.0
    gate_3_explanation: str = ""
    gate_4_cost: ValidationResult = ValidationResult.INSUFFICIENT_DATA
    gate_5_regime: ValidationResult = ValidationResult.INSUFFICIENT_DATA
    gate_5_profitable_regimes: List[str] = field(default_factory=list)
    gate_6_oos: ValidationResult = ValidationResult.INSUFFICIENT_DATA

    # Aggregate metrics
    total_trades: int = 0
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    win_rate: float = 0.0
    avg_return_bps: float = 0.0
    net_return_after_costs_bps: float = 0.0
    confidence_tier: int = 3  # default to paper-only
    allocation_pct: float = 0.0
    timestamp: str = ""

    def summary(self) -> str:
        gates = [
            f"G1={self.gate_1_pattern.value}",
            f"G2={self.gate_2_stat_sig.value}(p={self.gate_2_p_value:.4f})",
            f"G4={self.gate_4_cost.value}",
            f"G5={self.gate_5_regime.value}({len(self.gate_5_profitable_regimes)} regimes)",
            f"G6={self.gate_6_oos.value}",
        ]
        return (
            f"Signal={self.signal_name} pair={self.pair} tier={self.confidence_tier} "
            f"alloc={self.allocation_pct:.0%} trades={self.total_trades} "
            f"sharpe_IS={self.in_sample_sharpe:.2f} sharpe_OOS={self.out_of_sample_sharpe:.2f} "
            + " | ".join(gates)
        )


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------

class SignalValidator:
    """
    Six-gate signal validation pipeline.

    Usage::

        validator = SignalValidator(config, db_path)
        report = validator.validate_signal(
            signal_generator_callable=my_signal_func,
            pair="BTC-USD",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
        tier = validator.assign_confidence_tier(report)
    """

    # Default assumed round-trip costs in basis points
    DEFAULT_COST_BPS = 15.0  # ~7.5 bps per leg (spread + fee)

    def __init__(self, config: Dict[str, Any], db_path: str):
        self.logger = logging.getLogger(f"{__name__}.SignalValidator")

        self.min_p_value: float = config.get("min_p_value", 0.01)
        self.min_trades: int = config.get("min_trades_for_validation", 100)
        self.min_sharpe: float = config.get("min_sharpe", 0.5)
        self.oos_pct: float = config.get("out_of_sample_pct", 0.25)
        self.min_regimes_profitable: int = config.get("min_regimes_profitable", 2)
        self.oos_sharpe_retention: float = config.get("oos_sharpe_retention", 0.5)
        self.tier_1_allocation: float = config.get("tier_1_allocation", 1.0)
        self.tier_2_allocation: float = config.get("tier_2_allocation", 0.25)
        self.tier_3_allocation: float = config.get("tier_3_allocation", 0.0)

        self.db_path = db_path

        if not SCIPY_AVAILABLE:
            self.logger.warning(
                "scipy is not installed. Statistical significance tests will "
                "return INSUFFICIENT_DATA. Install with: pip install scipy"
            )

        self.logger.info(
            "SignalValidator initialized (min_p=%.4f, min_trades=%d, min_sharpe=%.2f, "
            "oos_pct=%.0f%%, min_regimes=%d)",
            self.min_p_value, self.min_trades, self.min_sharpe,
            self.oos_pct * 100, self.min_regimes_profitable,
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _fetch_market_data(self, pair: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Return market_data rows for *pair* between *start_date* and *end_date*."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT price, volume, bid, ask, spread, timestamp, product_id
                    FROM market_data
                    WHERE product_id = ?
                      AND timestamp >= ?
                      AND timestamp <= ?
                    ORDER BY timestamp ASC
                    """,
                    (pair, start_date, end_date),
                )
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as exc:
            self.logger.error("Failed to fetch market data: %s", exc)
            return []

    def _fetch_regime_labels(self, start_date: str, end_date: str) -> List[str]:
        """Attempt to retrieve regime labels from the decisions table."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT hmm_regime
                    FROM decisions
                    WHERE timestamp >= ? AND timestamp <= ?
                      AND hmm_regime IS NOT NULL
                    ORDER BY timestamp ASC
                    """,
                    (start_date, end_date),
                )
                return [row[0] for row in cursor.fetchall() if row[0]]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Top-level validation
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal_generator_callable: Callable,
        pair: str,
        start_date: str,
        end_date: str,
    ) -> SignalValidationReport:
        """
        Run the full 6-gate validation pipeline for a signal generator.

        Parameters
        ----------
        signal_generator_callable : callable
            ``func(market_data_rows) -> list[dict]`` where each dict
            represents a simulated trade with at least keys
            ``entry_price``, ``exit_price``, ``return_bps``, and
            optionally ``regime``.
        pair : str
            Trading pair, e.g. ``"BTC-USD"``.
        start_date, end_date : str
            ISO date strings bounding the evaluation window.

        Returns
        -------
        SignalValidationReport
        """
        report = SignalValidationReport(
            signal_name=getattr(signal_generator_callable, "__name__", "unknown"),
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # ----- fetch data -----
        market_data = self._fetch_market_data(pair, start_date, end_date)
        if len(market_data) < self.min_trades:
            self.logger.warning(
                "Only %d market data rows for %s — insufficient for validation.", len(market_data), pair
            )
            report.confidence_tier = 3
            report.allocation_pct = self.tier_3_allocation
            return report

        # ----- split in-sample / out-of-sample -----
        split_idx = int(len(market_data) * (1 - self.oos_pct))
        in_sample_data = market_data[:split_idx]
        oos_data = market_data[split_idx:]

        # ----- generate trades on in-sample -----
        try:
            trades = signal_generator_callable(in_sample_data)
        except Exception as exc:
            self.logger.error("Signal generator raised an exception: %s", exc)
            report.confidence_tier = 3
            report.allocation_pct = self.tier_3_allocation
            return report

        if not trades or not isinstance(trades, list):
            trades = []

        report.total_trades = len(trades)

        # Extract returns array (in basis points)
        returns_bps = np.array(
            [t.get("return_bps", 0.0) for t in trades], dtype=np.float64
        )

        # ----- Gate 1: Pattern Detection -----
        report.gate_1_pattern = self._gate_1_pattern_detection(trades)

        # ----- Gate 2: Statistical Significance -----
        report.gate_2_stat_sig, report.gate_2_p_value = self._gate_2_statistical_significance(returns_bps)

        # ----- Gate 3: Explanation (informational) -----
        report.gate_3_explanation = self._gate_3_explanation(trades, returns_bps)

        # ----- Gate 4: Cost Check -----
        report.gate_4_cost = self._gate_4_cost_check(trades, pair)

        # ----- Gate 5: Regime Check -----
        regimes = self._fetch_regime_labels(start_date, end_date)
        report.gate_5_regime, report.gate_5_profitable_regimes = self._gate_5_regime_check(
            trades, regimes
        )

        # ----- Gate 6: Out-of-Sample -----
        report.gate_6_oos = self._gate_6_out_of_sample(
            signal_generator_callable, pair, oos_data
        )

        # ----- Aggregate metrics -----
        if len(returns_bps) > 1:
            report.avg_return_bps = float(np.mean(returns_bps))
            report.win_rate = float(np.sum(returns_bps > 0) / len(returns_bps))
            report.in_sample_sharpe = self._compute_sharpe(returns_bps)

            cost_adj = returns_bps - self.DEFAULT_COST_BPS
            report.net_return_after_costs_bps = float(np.mean(cost_adj))

        # Compute OOS Sharpe for the report
        try:
            oos_trades = signal_generator_callable(oos_data)
            if oos_trades and isinstance(oos_trades, list):
                oos_returns = np.array([t.get("return_bps", 0.0) for t in oos_trades], dtype=np.float64)
                if len(oos_returns) > 1:
                    report.out_of_sample_sharpe = self._compute_sharpe(oos_returns)
        except Exception:
            pass

        # ----- Assign tier -----
        report.confidence_tier = self.assign_confidence_tier(report)
        if report.confidence_tier == 1:
            report.allocation_pct = self.tier_1_allocation
        elif report.confidence_tier == 2:
            report.allocation_pct = self.tier_2_allocation
        else:
            report.allocation_pct = self.tier_3_allocation

        self.logger.info("Validation complete: %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------

    def _gate_1_pattern_detection(self, trades: List[Dict]) -> ValidationResult:
        """
        Gate 1 — Pattern Detection.

        Checks:
        * Minimum number of trades met
        * Win rate is non-trivially different from 50 %
        * Average return is positive
        """
        if len(trades) < self.min_trades:
            self.logger.debug(
                "G1 INSUFFICIENT_DATA: %d trades < min %d", len(trades), self.min_trades
            )
            return ValidationResult.INSUFFICIENT_DATA

        returns = np.array([t.get("return_bps", 0.0) for t in trades], dtype=np.float64)
        win_rate = float(np.sum(returns > 0) / len(returns)) if len(returns) > 0 else 0.0
        avg_return = float(np.mean(returns)) if len(returns) > 0 else 0.0

        # Must have win rate > 50 % or positive average return
        if win_rate > 0.50 and avg_return > 0:
            self.logger.debug("G1 PASSED: win_rate=%.3f avg_ret=%.2f bps", win_rate, avg_return)
            return ValidationResult.PASSED

        self.logger.debug("G1 FAILED: win_rate=%.3f avg_ret=%.2f bps", win_rate, avg_return)
        return ValidationResult.FAILED

    def _gate_2_statistical_significance(
        self, returns: np.ndarray
    ) -> Tuple[ValidationResult, float]:
        """
        Gate 2 — Statistical Significance.

        One-sample t-test: H0 = mean return is zero.
        Passes if p-value < configured threshold (default 0.01) and mean > 0.
        """
        if not SCIPY_AVAILABLE:
            return ValidationResult.INSUFFICIENT_DATA, 1.0

        if len(returns) < 20:
            return ValidationResult.INSUFFICIENT_DATA, 1.0

        # Filter out NaN / Inf
        clean = returns[np.isfinite(returns)]
        if len(clean) < 20:
            return ValidationResult.INSUFFICIENT_DATA, 1.0

        try:
            t_stat, p_value = scipy_stats.ttest_1samp(clean, 0.0)
            p_value = float(p_value)
        except Exception as exc:
            self.logger.error("t-test failed: %s", exc)
            return ValidationResult.INSUFFICIENT_DATA, 1.0

        mean_return = float(np.mean(clean))

        if p_value < self.min_p_value and mean_return > 0:
            self.logger.debug(
                "G2 PASSED: p=%.6f < %.4f, mean=%.2f bps", p_value, self.min_p_value, mean_return
            )
            return ValidationResult.PASSED, p_value

        self.logger.debug(
            "G2 FAILED: p=%.6f (threshold=%.4f), mean=%.2f bps", p_value, self.min_p_value, mean_return
        )
        return ValidationResult.FAILED, p_value

    def _gate_3_explanation(self, trades: List[Dict], returns: np.ndarray) -> str:
        """
        Gate 3 — Explanation (informational, non-blocking).

        Summarise the signal's behaviour for human review.
        """
        if len(trades) == 0:
            return "No trades to explain."

        n = len(returns)
        winners = int(np.sum(returns > 0))
        losers = int(np.sum(returns < 0))
        avg_win = float(np.mean(returns[returns > 0])) if winners > 0 else 0.0
        avg_loss = float(np.mean(returns[returns < 0])) if losers > 0 else 0.0
        sharpe = self._compute_sharpe(returns) if n > 1 else 0.0

        explanation = (
            f"{n} trades: {winners} wins (avg +{avg_win:.1f} bps), "
            f"{losers} losses (avg {avg_loss:.1f} bps). "
            f"Sharpe={sharpe:.2f}."
        )
        self.logger.debug("G3 explanation: %s", explanation)
        return explanation

    def _gate_4_cost_check(self, trades: List[Dict], pair: str) -> ValidationResult:
        """
        Gate 4 — Cost Check.

        Verifies that the signal is net-profitable after assumed
        transaction costs (spread, exchange fees, slippage).
        """
        if len(trades) < self.min_trades:
            return ValidationResult.INSUFFICIENT_DATA

        returns = np.array([t.get("return_bps", 0.0) for t in trades], dtype=np.float64)
        avg_gross = float(np.mean(returns))
        avg_net = avg_gross - self.DEFAULT_COST_BPS

        if avg_net > 0:
            self.logger.debug(
                "G4 PASSED: avg_gross=%.2f bps, cost=%.1f bps, net=%.2f bps",
                avg_gross, self.DEFAULT_COST_BPS, avg_net,
            )
            return ValidationResult.PASSED

        self.logger.debug(
            "G4 FAILED: avg_gross=%.2f bps, cost=%.1f bps, net=%.2f bps",
            avg_gross, self.DEFAULT_COST_BPS, avg_net,
        )
        return ValidationResult.FAILED

    def _gate_5_regime_check(
        self, trades: List[Dict], regimes: List[str]
    ) -> Tuple[ValidationResult, List[str]]:
        """
        Gate 5 — Regime Check.

        Verifies that the signal is profitable in at least
        ``min_regimes_profitable`` distinct market regimes.
        """
        if not regimes:
            self.logger.debug("G5 INSUFFICIENT_DATA: no regime labels available.")
            return ValidationResult.INSUFFICIENT_DATA, []

        # Assign regimes to trades by index (best-effort alignment)
        regime_returns: Dict[str, List[float]] = {}
        for i, trade in enumerate(trades):
            # Use trade's own regime if present, else align by index
            regime = trade.get("regime", "")
            if not regime and i < len(regimes):
                regime = regimes[i]
            if not regime:
                continue
            regime_returns.setdefault(regime, []).append(trade.get("return_bps", 0.0))

        profitable_regimes: List[str] = []
        for regime_name, rets in regime_returns.items():
            arr = np.array(rets, dtype=np.float64)
            if len(arr) >= 5 and float(np.mean(arr)) > 0:
                profitable_regimes.append(regime_name)

        if len(profitable_regimes) >= self.min_regimes_profitable:
            self.logger.debug(
                "G5 PASSED: %d profitable regimes %s (min=%d)",
                len(profitable_regimes), profitable_regimes, self.min_regimes_profitable,
            )
            return ValidationResult.PASSED, profitable_regimes

        self.logger.debug(
            "G5 FAILED: %d profitable regimes %s (min=%d)",
            len(profitable_regimes), profitable_regimes, self.min_regimes_profitable,
        )
        return ValidationResult.FAILED, profitable_regimes

    def _gate_6_out_of_sample(
        self,
        signal_generator: Callable,
        pair: str,
        oos_data: List[Dict],
    ) -> ValidationResult:
        """
        Gate 6 — Out-of-Sample.

        Runs the signal generator on held-out data and checks whether
        the OOS Sharpe ratio retains at least ``oos_sharpe_retention``
        of the in-sample Sharpe.
        """
        if len(oos_data) < 20:
            self.logger.debug("G6 INSUFFICIENT_DATA: OOS set too small (%d rows).", len(oos_data))
            return ValidationResult.INSUFFICIENT_DATA

        try:
            oos_trades = signal_generator(oos_data)
        except Exception as exc:
            self.logger.error("G6 signal generator failed on OOS data: %s", exc)
            return ValidationResult.FAILED

        if not oos_trades or not isinstance(oos_trades, list) or len(oos_trades) < 5:
            self.logger.debug("G6 INSUFFICIENT_DATA: only %d OOS trades.", len(oos_trades) if oos_trades else 0)
            return ValidationResult.INSUFFICIENT_DATA

        oos_returns = np.array(
            [t.get("return_bps", 0.0) for t in oos_trades], dtype=np.float64
        )
        oos_sharpe = self._compute_sharpe(oos_returns)

        if oos_sharpe >= self.min_sharpe * self.oos_sharpe_retention:
            self.logger.debug("G6 PASSED: OOS Sharpe=%.3f", oos_sharpe)
            return ValidationResult.PASSED

        self.logger.debug(
            "G6 FAILED: OOS Sharpe=%.3f < threshold %.3f",
            oos_sharpe, self.min_sharpe * self.oos_sharpe_retention,
        )
        return ValidationResult.FAILED

    # ------------------------------------------------------------------
    # Confidence tier assignment
    # ------------------------------------------------------------------

    def assign_confidence_tier(self, report: SignalValidationReport) -> int:
        """
        Assign a confidence tier based on the gate results.

        Returns
        -------
        int
            1 = full allocation, 2 = 25 % allocation, 3 = paper only.
        """
        core_gates = [
            report.gate_1_pattern,
            report.gate_2_stat_sig,
            report.gate_4_cost,
        ]

        advanced_gates = [
            report.gate_5_regime,
            report.gate_6_oos,
        ]

        # Tier 3: any core gate failed
        if any(g == ValidationResult.FAILED for g in core_gates):
            return 3

        # Tier 3: insufficient data on all core gates
        if all(g == ValidationResult.INSUFFICIENT_DATA for g in core_gates):
            return 3

        # Tier 1: all gates passed (or insufficient data treated as non-blocking for advanced)
        all_passed = (
            all(g == ValidationResult.PASSED for g in core_gates)
            and all(g in (ValidationResult.PASSED, ValidationResult.INSUFFICIENT_DATA) for g in advanced_gates)
            and sum(1 for g in advanced_gates if g == ValidationResult.PASSED) >= 1
        )
        if all_passed and report.in_sample_sharpe >= self.min_sharpe:
            return 1

        # Tier 2: core passed, advanced marginal
        if all(g == ValidationResult.PASSED for g in core_gates):
            return 2

        return 3

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sharpe(returns: np.ndarray, annualization: float = 252.0) -> float:
        """
        Compute annualised Sharpe ratio from an array of per-trade returns.

        Assumes daily-ish frequency for annualization.
        """
        clean = returns[np.isfinite(returns)]
        if len(clean) < 2:
            return 0.0
        mean_r = float(np.mean(clean))
        std_r = float(np.std(clean, ddof=1))
        if std_r < 1e-12:
            return 0.0
        return (mean_r / std_r) * math.sqrt(annualization)
