"""
Ghost Runner — Shadow Trading Validation System
=================================================
Runs parallel "ghost" trades alongside the main bot to validate strategy
stability. Ghost trades use the same signal pipeline but different parameters
to test whether alternative configurations would perform better.

"Every strategy runs in shadow for 6 months before getting real capital."
"""

import asyncio
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np


class GhostRunner:
    """
    Shadow trading system that tracks hypothetical PnL from alternative
    parameter sets. Runs alongside the main bot using the same signals.
    """

    def __init__(self, bot, logger: Optional[logging.Logger] = None):
        self.bot = bot
        self.logger = logger or logging.getLogger(__name__)
        self.is_running = False

        # Ghost parameter sets to test
        self._param_sets = {
            "aggressive": {
                "buy_threshold": 0.03,
                "sell_threshold": -0.03,
                "min_confidence": 0.55,
            },
            "conservative": {
                "buy_threshold": 0.10,
                "sell_threshold": -0.10,
                "min_confidence": 0.70,
            },
            "momentum": {
                "buy_threshold": 0.05,
                "sell_threshold": -0.05,
                "min_confidence": 0.60,
            },
        }

        # Ghost positions: {param_set_name: {product_id: {"action": str, "price": float, "cycle": int}}}
        self._ghost_positions: Dict[str, Dict[str, Dict]] = defaultdict(dict)

        # Performance tracking per param set
        self._performance: Dict[str, Dict[str, Any]] = {}
        for name in self._param_sets:
            self._performance[name] = {
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "returns": deque(maxlen=200),
            }

        # Trade history for DB persistence
        self._trade_history: List[Dict[str, Any]] = []

        self.logger.info(f"Ghost Runner initialized with {len(self._param_sets)} parameter sets")

    async def run_validation_cycle(self):
        """Run ghost trading for all products using all parameter sets."""
        self.logger.debug("Ghost Runner: Starting validation cycle...")

        try:
            for product_id in self.bot.product_ids:
                market_data = await self.bot.collect_all_data(product_id)
                if not market_data:
                    continue

                # Get current price
                ticker = market_data.get('ticker', {})
                current_price = float(ticker.get('price', 0.0) or 0.0)
                if current_price <= 0:
                    continue

                # Generate signals (reuse main bot's pipeline)
                signals = await self.bot.generate_signals(market_data)
                weighted_signal, contributions = self.bot.calculate_weighted_signal(signals)

                # Evaluate each parameter set
                for param_name, params in self._param_sets.items():
                    self._evaluate_ghost(
                        param_name, params, product_id,
                        weighted_signal, current_price
                    )

        except Exception as e:
            self.logger.error(f"Ghost Runner cycle failed: {e}")

    def _evaluate_ghost(self, param_name: str, params: Dict, product_id: str,
                        weighted_signal: float, current_price: float):
        """Evaluate a single ghost parameter set for a product."""
        perf = self._performance[param_name]
        existing = self._ghost_positions[param_name].get(product_id)

        # Check for exit of existing ghost position
        if existing:
            entry_price = existing["price"]
            entry_action = existing["action"]
            cycles_held = getattr(self.bot, 'scan_cycle_count', 0) - existing.get("cycle", 0)

            # Simple exit: after 6 cycles or signal reversal
            should_exit = False
            if cycles_held >= 6:
                should_exit = True
                exit_reason = "max_hold"
            elif entry_action == "BUY" and weighted_signal < params["sell_threshold"]:
                should_exit = True
                exit_reason = "signal_reversal"
            elif entry_action == "SELL" and weighted_signal > params["buy_threshold"]:
                should_exit = True
                exit_reason = "signal_reversal"

            if should_exit:
                # Calculate ghost PnL
                if entry_action == "BUY":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price

                perf["trades"] += 1
                perf["pnl"] += pnl_pct
                perf["returns"].append(pnl_pct)
                if pnl_pct > 0:
                    perf["wins"] += 1
                else:
                    perf["losses"] += 1

                # Record trade
                self._trade_history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "param_set": param_name,
                    "product_id": product_id,
                    "action": entry_action,
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "pnl_pct": round(pnl_pct, 6),
                    "exit_reason": exit_reason,
                    "cycles_held": cycles_held,
                })

                # Clear position
                del self._ghost_positions[param_name][product_id]

                if perf["trades"] <= 5 or perf["trades"] % 20 == 0:
                    win_rate = perf["wins"] / max(perf["trades"], 1)
                    self.logger.info(
                        f"GHOST [{param_name}] {product_id}: {entry_action} exit "
                        f"PnL={pnl_pct:+.4f} | Cumulative: {perf['pnl']:+.4f} "
                        f"({perf['trades']} trades, {win_rate:.0%} win)"
                    )
                return

        # Check for new ghost entry (only if no existing position)
        if product_id not in self._ghost_positions[param_name]:
            confidence = abs(weighted_signal)
            if confidence >= params["min_confidence"]:
                if weighted_signal > params["buy_threshold"]:
                    action = "BUY"
                elif weighted_signal < params["sell_threshold"]:
                    action = "SELL"
                else:
                    return

                self._ghost_positions[param_name][product_id] = {
                    "action": action,
                    "price": current_price,
                    "cycle": getattr(self.bot, 'scan_cycle_count', 0),
                }

    async def start_ghost_loop(self, interval: int = 600):
        """Start the continuous ghost runner loop."""
        self.is_running = True
        self.logger.info(f"Ghost Runner Loop started (Interval: {interval}s)")

        while self.is_running:
            await self.run_validation_cycle()
            await asyncio.sleep(interval)

    def stop(self):
        """Stop the ghost runner loop."""
        self.is_running = False
        self.logger.info("Ghost Runner Loop stopped.")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Return comprehensive performance comparison."""
        summary = {}
        for name, perf in self._performance.items():
            trades = perf["trades"]
            wins = perf["wins"]
            returns = list(perf["returns"])

            entry = {
                "cumulative_pnl": round(perf["pnl"], 6),
                "total_trades": trades,
                "win_rate": round(wins / max(trades, 1), 3),
                "avg_return": round(float(np.mean(returns)), 6) if returns else 0.0,
                "params": self._param_sets[name],
            }

            if len(returns) >= 10:
                arr = np.array(returns)
                mean_r = float(np.mean(arr))
                std_r = float(np.std(arr, ddof=1))
                entry["sharpe"] = round(mean_r / std_r if std_r > 1e-8 else 0.0, 3)
            else:
                entry["sharpe"] = 0.0

            summary[name] = entry

        return summary

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent ghost trades."""
        return self._trade_history[-limit:]
