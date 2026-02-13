"""
Step 18: Ghost Runner Loop
Continuous out-of-sample validation loop that runs 'ghost' trades
in parallel with the main bot to validate strategy stability.
"""

import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

class GhostRunner:
    def __init__(self, bot, logger: Optional[logging.Logger] = None):
        self.bot = bot
        self.logger = logger or logging.getLogger(__name__)
        self.ghost_performance = {"pnl": 0.0, "trades": 0, "win_rate": 0.0}
        self.is_running = False

    async def run_validation_cycle(self):
        """Runs a validation cycle using current bot signals but different risk parameters."""
        self.logger.info("👻 Ghost Runner: Starting validation cycle...")
        
        try:
            # 1. Use the main bot to generate signals for all products
            for product_id in self.bot.product_ids:
                market_data = await self.bot.collect_all_data(product_id)
                if not market_data:
                    continue
                
                signals = await self.bot.generate_signals(market_data)
                weighted_signal, contributions = self.bot.calculate_weighted_signal(signals)
                
                # 2. Ghost Trading Logic (e.g., more aggressive thresholds)
                ghost_buy_threshold = 0.05 
                ghost_sell_threshold = -0.05
                
                action = "HOLD"
                if weighted_signal > ghost_buy_threshold:
                    action = "BUY"
                elif weighted_signal < ghost_sell_threshold:
                    action = "SELL"
                
                # 3. Simulate outcome (simplified)
                # In real ghost runner, we'd store this and check price in N minutes
                current_price = market_data.get('ticker', {}).get('price', 0.0)
                
                self.logger.info(f"👻 Ghost Decision for {product_id}: {action} at {current_price:.2f}")
                
                # Store ghost decision for future attribution
                if self.bot.db_enabled:
                    ghost_data = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "product_id": product_id,
                        "action": action,
                        "signal": weighted_signal,
                        "price": current_price,
                        "type": "GHOST_RUNNER"
                    }
                    # We could extend db_manager to store ghost trades
                    pass

        except Exception as e:
            self.logger.error(f"Ghost Runner cycle failed: {e}")

    async def start_ghost_loop(self, interval: int = 600):
        """Starts the continuous ghost runner loop."""
        self.is_running = True
        self.logger.info(f"👻 Ghost Runner Loop started (Interval: {interval}s)")
        
        while self.is_running:
            await self.run_validation_cycle()
            await asyncio.sleep(interval)

    def stop(self):
        self.is_running = False
        self.logger.info("👻 Ghost Runner Loop stopped.")
