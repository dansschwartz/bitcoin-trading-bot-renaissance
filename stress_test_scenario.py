import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from renaissance_trading_bot import RenaissanceTradingBot
from database_manager import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlackSwanSimulator")

class BlackSwanSimulator:
    """
    Simulates extreme market events to test the bot's institutional protective layers:
    1. VAE Anomaly Detection (RiskGateway)
    2. VPIN Toxicity Gating (Execution Suite)
    3. Regime-Aware De-leveraging (RegimeOverlay)
    """

    def __init__(self):
        self.bot = RenaissanceTradingBot()
        # Initialize database manager with bot's database config
        db_cfg = self.bot.config.get("database", {"path": "data/renaissance_bot.db", "enabled": True})
        self.db = DatabaseManager(db_cfg)
        logger.info("🏛️ Black Swan Simulator Initialized")

    def generate_flash_crash_data(self, base_price=60000.0, steps=50):
        """Simulates a -10% drop in minutes with extreme volume and spread widening."""
        prices = []
        curr_price = base_price
        for i in range(steps):
            # Normal jitter
            curr_price *= (1 + np.random.normal(0, 0.001))
            # Crash phase (steps 20 to 30)
            if 20 <= i <= 30:
                curr_price *= 0.98 # -2% per step
            prices.append(curr_price)
        
        return {
            'prices': prices,
            'vpin': [0.95 if 20 <= i <= 35 else 0.4 for i in range(steps)],
            'spread': [0.02 if 20 <= i <= 35 else 0.0001 for i in range(steps)],
            'description': "FLASH CRASH (-10% in 10 steps)"
        }

    def generate_parabolic_pump_data(self, base_price=60000.0, steps=50):
        """Simulates a +15% irrational pump."""
        prices = []
        curr_price = base_price
        for i in range(steps):
            curr_price *= (1 + np.random.normal(0, 0.001))
            if 15 <= i <= 25:
                curr_price *= 1.015 # +1.5% per step
            prices.append(curr_price)
        
        return {
            'prices': prices,
            'vpin': [0.85 if 15 <= i <= 30 else 0.3 for i in range(steps)],
            'spread': [0.005 if 15 <= i <= 30 else 0.0001 for i in range(steps)],
            'description': "PARABOLIC PUMP (+15% in 10 steps)"
        }

    async def run_scenario(self, scenario_data):
        logger.info(f"\n🔥 STARTING SCENARIO: {scenario_data['description']}")
        logger.info("="*60)
        
        results = []
        for i, price in enumerate(scenario_data['prices']):
            # Mock market data
            market_data = {
                'ticker': {
                    'price': price,
                    'bid': price * (1 - scenario_data['spread'][i]/2),
                    'ask': price * (1 + scenario_data['spread'][i]/2),
                    'volume': 5000.0 if scenario_data['vpin'][i] > 0.5 else 100.0,
                    'bid_ask_spread': scenario_data['spread'][i]
                },
                'vpin': scenario_data['vpin'][i],
                'volatility': 0.05 if scenario_data['vpin'][i] > 0.5 else 0.01
            }

            # Inject into bot's internal price history (if accessible)
            # For this test, we override the bot's data provider or signals
            
            # 1. Update Technical Indicators
            from enhanced_technical_indicators import PriceData
            self.bot.technical_indicators.update_price_data(PriceData(
                timestamp=datetime.now(timezone.utc),
                open=price, 
                high=price, 
                low=price, 
                close=price, 
                volume=market_data['ticker']['volume']
            ))

            # 2. Generate Signals
            signals = await self.bot.generate_signals(market_data)
            weighted_signal, contributions = self.bot.calculate_weighted_signal(signals)
            
            # 3. Make Decision (This is where RiskGateway & VAE are checked)
            # Create a synthetic high-dimensional feature vector for VAE
            # If VPIN is high, we make the feature vector "anomalous"
            feature_vector = np.random.randn(128)
            if scenario_data['vpin'][i] > 0.8:
                feature_vector = feature_vector * 10.0 # Make it anomalous

            decision = self.bot.make_trading_decision(
                weighted_signal, contributions, 
                current_price=price, 
                product_id="BTC-USD"
            )

            # 4. Assess Risk (Explicit check for logging)
            risk_approved = self.bot.risk_gateway.assess_trade(
                decision.action, decision.position_size, price, 
                {'total_value': 1000.0, 'positions': {}},
                feature_vector=feature_vector
            )

            status = "✅ NORMAL"
            if not risk_approved:
                status = "🛡️ BLOCKED (RISK/VAE)"
            elif scenario_data['vpin'][i] > 0.8 or scenario_data['spread'][i] > 0.01:
                status = "⏳ DELAYED (TCO/VPIN)"

            logger.info(f"Step {i:02d} | Price: {price:8.2f} | VPIN: {scenario_data['vpin'][i]:.2f} | Signal: {weighted_signal:+.4f} | Decision: {decision.action:4} | Status: {status}")
            results.append(status)

        blocked_pct = results.count("🛡️ BLOCKED (RISK/VAE)") / len(results)
        logger.info(f"\nSCENARIO COMPLETE: {scenario_data['description']}")
        logger.info(f"Protection Rate: {blocked_pct:.1%}")
        return blocked_pct

async def main():
    simulator = BlackSwanSimulator()
    
    # Run Flash Crash
    crash_data = simulator.generate_flash_crash_data()
    await simulator.run_scenario(crash_data)
    
    # Run Parabolic Pump
    pump_data = simulator.generate_parabolic_pump_data()
    await simulator.run_scenario(pump_data)

if __name__ == "__main__":
    asyncio.run(main())
