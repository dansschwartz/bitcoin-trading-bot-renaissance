#!/usr/bin/env python3
"""
Enhanced Data Pipeline Startup Script
Clean version without messy print statements
"""

import sys
import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

# Add the data_pipeline directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_data_pipeline import EnhancedDataPipeline


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/enhanced_pipeline_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_config():
    """Load configuration file"""
    config_path = Path(__file__).parent / "config" / "data_pipeline_config.json"

    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        print("Please ensure data_pipeline_config.json exists in the config directory")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)


def validate_config(config):
    """Validate configuration has required fields"""
    required_sections = ['coinbase', 'database', 'data_sources']
    missing_sections = [section for section in required_sections if section not in config]

    if missing_sections:
        print(f"❌ Missing configuration sections: {missing_sections}")
        return False

    # Check API credentials (warn if missing, don't fail)
    api_warnings = []

    if not config.get('twitter', {}).get('bearer_token'):
        api_warnings.append("Twitter API")

    if not config.get('reddit', {}).get('client_id'):
        api_warnings.append("Reddit API")

    if not config.get('glassnode', {}).get('api_key'):
        api_warnings.append("Glassnode API")

    if api_warnings:
        print(f"⚠️  Missing API credentials: {', '.join(api_warnings)}")
        print("   Some data sources will be unavailable")

    return True


async def test_pipeline_components(pipeline):
    """Test individual pipeline components"""
    print("\n🧪 Testing Pipeline Components...")

    # Test database connection
    try:
        await pipeline.db_manager.init_database()
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

    # Test Coinbase connection (most critical)
    try:
        test_data = await pipeline._collect_coinbase_data()
        if test_data and 'price' in test_data:
            print(f"✅ Coinbase API - BTC Price: ${test_data['price']:,.2f}")
        else:
            print("⚠️  Coinbase API connected but no price data")
    except Exception as e:
        print(f"❌ Coinbase API failed: {e}")

    # Test other APIs (non-blocking)
    api_tests = [
        # ("Twitter", lambda: pipeline.twitter_client.get_bitcoin_sentiment()),
        ("Reddit", lambda: pipeline.reddit_client.get_crypto_sentiment()),
        ("Glassnode", lambda: pipeline.glassnode_client.get_onchain_metrics()),
        ("Fear & Greed", lambda: pipeline.fear_greed_client.get_fear_greed_index())
    ]

    for api_name, test_func in api_tests:
        try:
            result = await test_func() if asyncio.iscoroutinefunction(test_func) else test_func()
            status = "✅" if result and not result.get('error') else "⚠️"
            print(f"{status} {api_name} API")
        except Exception:
            print(f"⚠️  {api_name} API (offline)")

    return True


async def run_single_collection(pipeline):
    """Run a single data collection cycle"""
    print("\n📊 Running Data Collection Test...")

    try:
        data = await pipeline.collect_all_data()

        if data:
            print("✅ Data collection successful!")

            # Show key metrics
            if data.get('market_data', {}).get('price'):
                price = data['market_data']['price']
                print(f"   BTC Price: ${price:,.2f}")

            if data.get('social_sentiment', {}).get('overall_sentiment'):
                sentiment = data['social_sentiment']['overall_sentiment']
                print(f"   Social Sentiment: {sentiment:.3f}")

            if data.get('onchain_metrics', {}).get('onchain_health_score'):
                health = data['onchain_metrics']['onchain_health_score']
                print(f"   On-chain Health: {health:.3f}")

            return True
        else:
            print("❌ No data collected")
            return False

    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False


def create_integration_helper():
    """Create integration helper files instead of printing messy code"""
    try:
        # Create integration helper in the src directory
        src_dir = Path(__file__).parent.parent / "src"
        src_dir.mkdir(exist_ok=True)

        integration_file = src_dir / "renaissance_integration.py"

        integration_code = '''"""
Renaissance Technologies Integration Helper
Use this to connect the enhanced data pipeline to your existing bot
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to path to access data_pipeline
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_data_pipeline import EnhancedDataPipeline
import json

class RenaissanceIntegration:
    """Helper class to integrate Renaissance-style signals with your existing bot"""

    def __init__(self):
        # Load pipeline configuration
        config_path = Path(__file__).parent.parent / "data_pipeline" / "config" / "data_pipeline_config.json"

        with open(config_path, 'r') as f:
            self.pipeline_config = json.load(f)

        self.pipeline = EnhancedDataPipeline(self.pipeline_config)
        self.alternative_weight = 0.20  # Start with 20%, increase gradually to 40%

    async def get_enhanced_signals(self):
        """Get Renaissance-style alternative data signals"""
        try:
            data = await self.pipeline.collect_all_data()

            signals = {
                'social_sentiment': 0,
                'onchain_health': 0.5,
                'network_momentum': 0.5,
                'fear_greed': 0.5,
                'timestamp': data.get('timestamp')
            }

            # Extract signals safely
            if data.get('social_sentiment'):
                signals['social_sentiment'] = data['social_sentiment'].get('overall_sentiment', 0)

            if data.get('onchain_metrics'):
                signals['onchain_health'] = data['onchain_metrics'].get('onchain_health_score', 0.5)
                signals['network_momentum'] = data['onchain_metrics'].get('network_momentum', 0.5)

            if data.get('alternative_data'):
                signals['fear_greed'] = data['alternative_data'].get('fear_greed_normalized', 0.5)

            return signals

        except Exception as e:
            print(f"Error getting enhanced signals: {e}")
            return {
                'social_sentiment': 0,
                'onchain_health': 0.5,
                'network_momentum': 0.5,
                'fear_greed': 0.5,
                'timestamp': None
            }

    def calculate_final_signal(self, existing_signals, alternative_signals=None):
        """
        Calculate Renaissance-style final signal

        Args:
            existing_signals: dict with keys: rsi, macd, bollinger, order_flow, volume
            alternative_signals: dict from get_enhanced_signals() (optional - will fetch if None)

        Returns:
            float: Final signal score combining technical + alternative data
        """

        if alternative_signals is None:
            # If not provided, use default neutral values
            alternative_signals = {
                'social_sentiment': 0,
                'onchain_health': 0.5,
                'network_momentum': 0.5,
                'fear_greed': 0.5
            }

        # Technical analysis weight (reduced from your current setup)
        technical_weight = 1.0 - self.alternative_weight

        # Calculate technical score (your existing logic)
        technical_score = (
            existing_signals.get('rsi', 0) * 0.25 +
            existing_signals.get('macd', 0) * 0.30 +
            existing_signals.get('bollinger', 0) * 0.20 +
            existing_signals.get('order_flow', 0) * 0.15 +
            existing_signals.get('volume', 0) * 0.10
        ) * technical_weight

        # Calculate alternative data score (Renaissance component)
        alternative_score = (
            alternative_signals.get('social_sentiment', 0) * 0.30 +
            alternative_signals.get('onchain_health', 0.5) * 0.25 +
            alternative_signals.get('network_momentum', 0.5) * 0.25 +
            alternative_signals.get('fear_greed', 0.5) * 0.20
        ) * self.alternative_weight

        return technical_score + alternative_score

    def set_alternative_weight(self, weight):
        """Set the weight for alternative data (0.0 to 0.4 recommended)"""
        self.alternative_weight = max(0.0, min(0.4, weight))

# USAGE EXAMPLE:
#
# async def your_existing_trading_function():
#     renaissance = RenaissanceIntegration()
#     
#     # Your existing signals (replace with your actual values)
#     existing_signals = {
#         'rsi': your_rsi_calculation(),
#         'macd': your_macd_calculation(),
#         'bollinger': your_bollinger_calculation(),
#         'order_flow': your_order_flow_calculation(),
#         'volume': your_volume_calculation()
#     }
#     
#     # Get Renaissance alternative data
#     alternative_signals = await renaissance.get_enhanced_signals()
#     
#     # Calculate final Renaissance-style signal
#     final_signal = renaissance.calculate_final_signal(existing_signals, alternative_signals)
#     
#     # Use final_signal in your existing trading logic
#     if final_signal > your_buy_threshold:
#         # Your buy logic
#         pass
#     elif final_signal < your_sell_threshold:
#         # Your sell logic
#         pass
#     else:
#         # Your hold logic
#         pass

if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        renaissance = RenaissanceIntegration()

        # Test with sample signals
        test_signals = {
            'rsi': 0.6,
            'macd': 0.7,
            'bollinger': 0.5,
            'order_flow': 0.8,
            'volume': 0.4
        }

        print("Testing Renaissance Integration...")
        alternative_signals = await renaissance.get_enhanced_signals()
        final_signal = renaissance.calculate_final_signal(test_signals, alternative_signals)

        print(f"Alternative signals: {alternative_signals}")
        print(f"Final signal: {final_signal:.3f}")

    asyncio.run(test_integration())
'''

        with open(integration_file, 'w') as f:
            f.write(integration_code)

        print(f"✅ Created integration helper: {integration_file}")
        return True

    except Exception as e:
        print(f"⚠️  Could not create integration file: {e}")
        return False


def show_integration_instructions():
    """Show clean integration instructions"""
    print("\n" + "=" * 50)
    print("🔗 INTEGRATION WITH YOUR EXISTING BOT")
    print("=" * 50)

    success = create_integration_helper()

    if success:
        print("""
✅ Integration helper created in your src/ folder!

NEXT STEPS:
1. Open: coinbot1/src/renaissance_integration.py
2. Copy the RenaissanceIntegration class to your main bot file
3. Replace the example signals with your actual signal calculations
4. Start with alternative_weight = 0.10 (10%)
5. Test in paper trading mode
6. Gradually increase to alternative_weight = 0.40 (40%)

The file contains:
- Complete integration code
- Usage examples
- Testing functions
- Clear documentation

This replaces your current signal fusion with Renaissance-style
alternative data integration.
""")
    else:
        print("""
⚠️  Could not create integration file automatically.

MANUAL INTEGRATION:
1. Import the enhanced_data_pipeline in your bot
2. Get alternative signals using pipeline.collect_all_data()
3. Combine with your existing signals using Renaissance weights:
   - Technical Analysis: 60% (down from ~75%)
   - Alternative Data: 40% (new component)

Check the documentation for detailed code examples.
""")


async def main():
    """Main startup function - clean and organized"""
    print("🚀 Enhanced Data Pipeline - Renaissance Technologies Style")
    print("=" * 60)

    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)

    # Load and validate configuration
    config = load_config()
    if not validate_config(config):
        sys.exit(1)

    # Initialize pipeline
    try:
        config_path = Path(__file__).parent / "config" / "data_pipeline_config.json"
        pipeline = EnhancedDataPipeline(str(config_path))  # ← New line
        print("✅ Enhanced Data Pipeline initialized")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        sys.exit(1)

    # Test components
    if not await test_pipeline_components(pipeline):
        print("⚠️  Some components failed - check API credentials")

    # Run collection test
    if not await run_single_collection(pipeline):
        print("❌ Data collection failed - check configuration")
        sys.exit(1)

    # Simple menu
    print("\n" + "=" * 40)
    print("🎯 OPTIONS:")
    print("=" * 40)
    print("1. Start continuous data collection")
    print("2. Test single collection cycle")
    print("3. Create integration helper for src/")
    print("4. Exit")

    while True:
        try:
            choice = input("\nChoice (1-4): ").strip()

            if choice == '1':
                print("\n🔄 Starting continuous collection (Ctrl+C to stop)...")
                try:
                    await pipeline.start_continuous_collection()
                except KeyboardInterrupt:
                    print("\n⏹️  Stopping...")
                    await pipeline.stop()
                    print("✅ Stopped gracefully")
                break

            elif choice == '2':
                await run_single_collection(pipeline)

            elif choice == '3':
                show_integration_instructions()

            elif choice == '4':
                print("👋 Goodbye!")
                break

            else:
                print("Please enter 1, 2, 3, or 4")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
