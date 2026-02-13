#!/usr/bin/env python3
"""
Renaissance Bitcoin Trading Bot - Main Runner
Complete startup and orchestration system
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
import logging
import shutil

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from renaissance_trading_bot import RenaissanceTradingBot

def setup_environment():
    """Setup required directories and configuration"""

    # Create required directories
    directories = [
        PROJECT_ROOT / 'logs',
        PROJECT_ROOT / 'data',
        PROJECT_ROOT / 'config',
        PROJECT_ROOT / 'output'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Create default config if it doesn't exist
    config_path = PROJECT_ROOT / 'config' / 'config.json'
    example_path = PROJECT_ROOT / 'config' / 'config.example.json'

    if not config_path.exists():
        if example_path.exists():
            shutil.copy(example_path, config_path)
            print(f"✅ Created config from template: {config_path}")
        else:
            default_config = {
                "trading": {
                    "product_id": "BTC-USD",
                    "cycle_interval_seconds": 300,
                    "paper_trading": True,
                    "sandbox": True
                },
                "risk_management": {
                    "daily_loss_limit": 500,
                    "position_limit": 1000,
                    "min_confidence": 0.65
                },
                "signal_weights": {
                    "order_flow": 0.32,
                    "order_book": 0.21,
                    "volume": 0.14,
                    "macd": 0.105,
                    "rsi": 0.115,
                    "bollinger": 0.095,
                    "alternative": 0.045
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/renaissance_bot.log"
                }
            }

            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)

            print(f"✅ Created default configuration: {config_path}")

    return str(config_path)

async def run_single_test():
    """Run a single trading cycle test"""
    print("🧪 Running Single Cycle Test...")
    print("-" * 50)

    try:
        config_path = setup_environment()
        bot = RenaissanceTradingBot(config_path)

        decision = await bot.execute_trading_cycle()

        print(f"\n📊 Test Results:")
        print(f"   Action: {decision.action}")
        print(f"   Confidence: {decision.confidence:.3f}")
        print(f"   Position Size: {decision.position_size:.3f}")
        print(f"   Timestamp: {decision.timestamp}")

        # Show signal breakdown
        if 'signal_contributions' in decision.reasoning:
            print(f"\n🔍 Signal Breakdown:")
            for signal, contribution in decision.reasoning['signal_contributions'].items():
                print(f"   {signal}: {contribution:.4f}")

        print("\n✅ Single test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Single test failed: {e}")
        return False

async def run_multiple_tests(num_cycles: int = 5):
    """Run multiple trading cycle tests"""
    print(f"🧪 Running {num_cycles} Trading Cycles Test...")
    print("-" * 50)

    try:
        config_path = setup_environment()
        bot = RenaissanceTradingBot(config_path)

        decisions = []
        for i in range(num_cycles):
            print(f"\n--- Cycle {i+1}/{num_cycles} ---")
            decision = await bot.execute_trading_cycle()
            decisions.append(decision)

            print(f"Decision: {decision.action} (Confidence: {decision.confidence:.3f})")
            await asyncio.sleep(2)  # Brief pause between cycles

        # Performance summary
        summary = bot.get_performance_summary()
        print(f"\n📊 Performance Summary:")
        print(f"   Total Decisions: {summary['total_decisions']}")
        print(f"   Action Distribution: {summary['action_distribution']}")
        print(f"   Average Confidence: {summary['average_confidence']:.3f}")
        print(f"   Average Position Size: {summary['average_position_size']:.3f}")

        print("\n✅ Multiple cycles test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Multiple cycles test failed: {e}")
        return False

async def run_continuous_trading(cycle_interval: int = 300):
    """Run continuous Renaissance trading"""
    print(f"🚀 Starting Continuous Renaissance Trading")
    print(f"   Cycle Interval: {cycle_interval} seconds ({cycle_interval/60:.1f} minutes)")
    print(f"   Paper Trading: ENABLED")
    print(f"   Started: {datetime.now()}")
    print("-" * 60)

    try:
        config_path = setup_environment()
        bot = RenaissanceTradingBot(config_path)

        print("🎯 Renaissance signal weights:")
        for signal, weight in bot.signal_weights.items():
            print(f"   {signal}: {weight:.3f} ({weight*100:.1f}%)")

        print(f"\n⚡ Starting trading loop...")
        await bot.run_continuous_trading(cycle_interval)

    except KeyboardInterrupt:
        print("\n🛑 Trading stopped by user")
    except Exception as e:
        print(f"❌ Continuous trading failed: {e}")

def validate_system():
    """Validate that all components are available"""
    print("🔍 Validating Renaissance System Components...")
    print("-" * 50)

    required_files = [
        'enhanced_config_manager.py',
        'microstructure_engine.py', 
        'enhanced_technical_indicators.py',
        'market_data_provider.py',
        'coinbase_client.py',
        'alternative_data_engine.py',
        'renaissance_signal_fusion.py',
        'renaissance_trading_bot.py'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
        else:
            print(f"✅ Found: {file_path}")

    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files!")
        print("Please ensure all Renaissance components are in the src/ directory")
        return False
    else:
        print(f"\n✅ All {len(required_files)} components validated!")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Renaissance Technologies Bitcoin Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_renaissance_bot.py --validate              # Validate system
  python run_renaissance_bot.py --test                  # Single test cycle
  python run_renaissance_bot.py --test-multiple 10      # Test 10 cycles
  python run_renaissance_bot.py --run                   # Continuous trading (5min cycles)
  python run_renaissance_bot.py --run --interval 60     # Continuous trading (1min cycles)
        """
    )

    parser.add_argument('--validate', action='store_true',
                       help='Validate system components')
    parser.add_argument('--test', action='store_true',
                       help='Run single trading cycle test')
    parser.add_argument('--test-multiple', type=int, metavar='N',
                       help='Run N trading cycles test')
    parser.add_argument('--run', action='store_true',
                       help='Run continuous trading')
    parser.add_argument('--interval', type=int, default=300,
                       help='Trading cycle interval in seconds (default: 300)')

    args = parser.parse_args()

    # Show header
    print("🏛️  RENAISSANCE TECHNOLOGIES BITCOIN TRADING BOT")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print()

    # Handle commands
    if args.validate:
        success = validate_system()
        sys.exit(0 if success else 1)

    elif args.test:
        success = asyncio.run(run_single_test())
        sys.exit(0 if success else 1)

    elif args.test_multiple:
        success = asyncio.run(run_multiple_tests(args.test_multiple))
        sys.exit(0 if success else 1)

    elif args.run:
        asyncio.run(run_continuous_trading(args.interval))

    else:
        print("🤔 No command specified. Use --help for options.")
        print("\nQuick start:")
        print("  1. python run_renaissance_bot.py --validate")
        print("  2. python run_renaissance_bot.py --test")
        print("  3. python run_renaissance_bot.py --run")

if __name__ == "__main__":
    main()
