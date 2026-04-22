"""Reporting module: CSV, JSON, Parquet export + matplotlib/seaborn plots."""

import json
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional

from sim_config import (
    BacktestResult, SimulationResult, ValidationScore, DEFAULT_CONFIG,
)

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy import stats as sp_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class SimReporter:
    """Generate all output artefacts for the simulation system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None):
        cfg = config or DEFAULT_CONFIG.get("output", {})
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = cfg.get("output_dir", "sim_output")
        self.plot_format = cfg.get("plot_format", "png")
        self.plot_dpi = cfg.get("plot_dpi", 150)
        self.save_csv = cfg.get("save_csv", True)
        self.save_json = cfg.get("save_json", True)
        self.save_plots = cfg.get("save_plots", True)
        self.save_parquet = cfg.get("save_parquet", True)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.output_dir, filename)

    # ------------------------------------------------------------------
    # Data export
    # ------------------------------------------------------------------

    def save_simulation_paths(self, result: SimulationResult,
                              fmt: str = "parquet") -> str:
        """Save simulation paths as Parquet or CSV."""
        filename = f"sim_{result.model_name}_{result.asset}_paths"
        df = pd.DataFrame(result.paths.T)
        df.columns = [f"sim_{i}" for i in range(result.n_simulations)]

        if fmt == "parquet" and self.save_parquet:
            path = self._path(filename + ".parquet")
            try:
                df.to_parquet(path, index=False)
            except Exception:
                path = self._path(filename + ".csv")
                df.to_csv(path, index=False)
        else:
            path = self._path(filename + ".csv")
            df.to_csv(path, index=False)

        self.logger.info(f"Saved paths: {path}")
        return path

    def save_metrics_json(self, metrics: Dict[str, Any], filename: str) -> str:
        path = self._path(filename)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, cls=_NumpyEncoder)
        self.logger.info(f"Saved metrics: {path}")
        return path

    def save_summary_csv(self, summaries: List[Dict[str, Any]],
                         filename: str) -> str:
        path = self._path(filename)
        pd.DataFrame(summaries).to_csv(path, index=False)
        self.logger.info(f"Saved summary: {path}")
        return path

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_simulation_fan(self, result: SimulationResult,
                            empirical_prices: Optional[np.ndarray] = None,
                            filename: Optional[str] = None) -> Optional[str]:
        """Fan chart: median, 5/25/75/95 percentiles + optional empirical overlay."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"fan_{result.model_name}_{result.asset}.{self.plot_format}"
        paths = result.paths
        x = np.arange(paths.shape[1])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(x, np.percentile(paths, 5, axis=0),
                         np.percentile(paths, 95, axis=0),
                         alpha=0.15, color="steelblue", label="5-95%")
        ax.fill_between(x, np.percentile(paths, 25, axis=0),
                         np.percentile(paths, 75, axis=0),
                         alpha=0.3, color="steelblue", label="25-75%")
        ax.plot(x, np.median(paths, axis=0), color="navy", linewidth=1.5, label="Median")

        if empirical_prices is not None:
            n = min(len(empirical_prices), len(x))
            ax.plot(x[:n], empirical_prices[:n], color="red",
                    linewidth=1.5, linestyle="--", label="Empirical")

        ax.set_title(f"Simulation Fan — {result.model_name} ({result.asset})")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved fan plot: {path}")
        return path

    def plot_distribution_comparison(
        self,
        empirical_returns: np.ndarray,
        simulated_returns: np.ndarray,
        model_name: str,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """Side-by-side: KDE PDF overlay + QQ plot."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"dist_{model_name}.{self.plot_format}"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # PDF overlay
        if SEABORN_AVAILABLE:
            sns.kdeplot(empirical_returns, ax=ax1, label="Empirical", color="red")
            sns.kdeplot(simulated_returns[:len(empirical_returns) * 5], ax=ax1,
                        label="Simulated", color="steelblue")
        else:
            ax1.hist(empirical_returns, bins=80, density=True, alpha=0.5,
                     label="Empirical", color="red")
            ax1.hist(simulated_returns[:50000], bins=80, density=True, alpha=0.5,
                     label="Simulated", color="steelblue")
        ax1.set_title(f"Return Distribution — {model_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        # QQ plot
        if SCIPY_AVAILABLE:
            emp_sorted = np.sort(empirical_returns)
            n_emp = len(emp_sorted)
            sim_quantiles = np.quantile(simulated_returns,
                                        np.linspace(0, 1, n_emp))
            ax2.scatter(emp_sorted, sim_quantiles, s=3, alpha=0.5, color="steelblue")
            mn = min(emp_sorted.min(), sim_quantiles.min())
            mx = max(emp_sorted.max(), sim_quantiles.max())
            ax2.plot([mn, mx], [mn, mx], "r--", linewidth=1)
        ax2.set_title("QQ Plot")
        ax2.set_xlabel("Empirical Quantiles")
        ax2.set_ylabel("Simulated Quantiles")
        ax2.grid(alpha=0.3)

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved distribution plot: {path}")
        return path

    def plot_validation_scorecard(self, scores: List[ValidationScore],
                                  filename: Optional[str] = None) -> Optional[str]:
        """Bar chart of composite scores per model, color-coded."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"scorecard.{self.plot_format}"
        names = [s.model_name for s in scores]
        values = [s.composite_score for s in scores]
        colors = ["#2ecc71" if v >= 7 else "#f39c12" if v >= 4 else "#e74c3c"
                  for v in values]

        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.5), 5))
        bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylim(0, 10)
        ax.set_ylabel("Composite Score (0-10)")
        ax.set_title("Model Validation Scorecard")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved scorecard: {path}")
        return path

    def plot_equity_curves(self, results: List[BacktestResult],
                           filename: Optional[str] = None) -> Optional[str]:
        """Multi-strategy equity curves with max-drawdown shading."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"equity_curves.{self.plot_format}"
        fig, ax = plt.subplots(figsize=(12, 6))

        for r in results:
            eq = r.equity_curve
            x = np.arange(len(eq))
            ax.plot(x, eq, linewidth=1.3,
                    label=f"{r.strategy_name} ({r.asset})")

            # Drawdown shading
            running_max = np.maximum.accumulate(eq)
            dd = (eq - running_max) / np.maximum(running_max, 1e-9)
            ax.fill_between(x, eq, running_max, alpha=0.1, color="red")

        ax.set_title("Strategy Equity Curves")
        ax.set_xlabel("Day")
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved equity curves: {path}")
        return path

    def plot_regime_timeline(self, prices: np.ndarray,
                             regimes: np.ndarray,
                             regime_labels: Optional[Dict[int, str]] = None,
                             filename: Optional[str] = None) -> Optional[str]:
        """Price chart with background colour bands for each regime."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"regime_timeline.{self.plot_format}"
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(prices))
        ax.plot(x, prices, color="black", linewidth=1)

        palette = ["#3498db", "#2ecc71", "#f1c40f", "#e74c3c", "#9b59b6",
                    "#1abc9c", "#e67e22", "#95a5a6"]
        unique_regimes = np.unique(regimes[:len(prices)])
        for r in unique_regimes:
            mask = regimes[:len(prices)] == r
            label = regime_labels.get(r, str(r)) if regime_labels else str(r)
            color = palette[int(r) % len(palette)]
            for i in range(len(mask)):
                if mask[i]:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color=color,
                               label=label if i == np.argmax(mask) else None)

        ax.set_title("Regime Timeline")
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")
        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left")
        ax.grid(alpha=0.3)

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved regime timeline: {path}")
        return path

    def plot_correlation_heatmap(self, corr_matrix: np.ndarray,
                                 labels: List[str],
                                 filename: Optional[str] = None) -> Optional[str]:
        """Correlation heatmap with annotations."""
        if not MATPLOTLIB_AVAILABLE or not self.save_plots:
            return None

        fname = filename or f"correlation_heatmap.{self.plot_format}"
        fig, ax = plt.subplots(figsize=(8, 6))

        if SEABORN_AVAILABLE:
            df = pd.DataFrame(corr_matrix, index=labels, columns=labels)
            sns.heatmap(df, annot=True, fmt=".2f", cmap="RdYlGn",
                        center=0, ax=ax, vmin=-1, vmax=1)
        else:
            im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticklabels(labels)
            plt.colorbar(im, ax=ax)

        ax.set_title("Cross-Asset Correlation Matrix")

        path = self._path(fname)
        fig.savefig(path, dpi=self.plot_dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Saved correlation heatmap: {path}")
        return path

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_full_report(self, all_results: Dict[str, Any]) -> str:
        """Orchestrate all exports and generate a summary markdown file."""
        lines = ["# Renaissance Simulation Report\n"]

        # Simulation results
        sim_results = all_results.get("simulation_results", {})
        for key, result in sim_results.items():
            if isinstance(result, SimulationResult):
                self.save_simulation_paths(result)
                lines.append(f"## {result.model_name} — {result.asset}")
                lines.append(f"- Simulations: {result.n_simulations}")
                lines.append(f"- Steps: {result.n_steps}")
                lines.append(f"- Parameters: {result.parameters}\n")

        # Validation scores
        scores = all_results.get("validation_scores", [])
        if scores:
            self.plot_validation_scorecard(scores)
            lines.append("## Validation Scorecard")
            for s in scores:
                lines.append(
                    f"- **{s.model_name}** ({s.asset}): "
                    f"{s.composite_score:.1f}/10 "
                    f"(KS={s.ks_stat:.3f}, ACF={s.acf_rmse:.3f})"
                )
            lines.append("")

        # Backtest results
        backtest_results = all_results.get("backtest_results", [])
        if backtest_results:
            self.plot_equity_curves(backtest_results)
            lines.append("## Backtest Results")
            for r in backtest_results:
                m = r.metrics
                lines.append(
                    f"- **{r.strategy_name}** ({r.asset}): "
                    f"Sharpe={m.get('sharpe_ratio', 0):.2f}, "
                    f"Return={m.get('total_return', 0):.1%}, "
                    f"MaxDD={m.get('max_drawdown', 0):.1%}"
                )
            lines.append("")

        # Metrics JSON
        metrics = all_results.get("metrics", {})
        if metrics:
            self.save_metrics_json(metrics, "simulation_metrics.json")

        # Write markdown summary
        report_path = self._path("report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        self.logger.info(f"Full report generated: {report_path}")
        return report_path
