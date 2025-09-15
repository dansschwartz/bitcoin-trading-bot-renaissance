"""
Main Trading Engine with Complete Safety Monitoring (Pandas-Free Version)
Production-ready Bitcoin trading bot with comprehensive safety features and system orchestration.
Enhanced with Renaissance Technologies integration - testing version without pandas dependency.
"""

import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
import os
import json
import time

# Skip pandas and Excel logging for now
# from trade_logger import log_comprehensive_data

from enhanced_data_pipeline import EnhancedDataPipeline

# Import Renaissance enhanced signal generator
from signal_generator import EnhancedSignalGenerator

# Handle optional imports
try:
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.interval import IntervalTrigger

    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    BlockingScheduler = None
    IntervalTrigger = None
    warnings.warn(
        "APScheduler not available. Install with: pip install APScheduler",
        UserWarning
    )

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    prometheus_client = None
    Counter = Histogram = Gauge = start_http_server = None
    warnings.warn(
        "Prometheus client not available. Install with: pip install prometheus_client",
        UserWarning
    )


# Simple JSON logging instead of Excel
def log_simple_data(data: Dict[str, Any], filename: str = None):
    """Simple JSON logging replacement for Excel logging"""
    if filename is None:
        filename = f"trading_log_{datetime.now().strftime('%Y%m%d')}.json"

    # Read existing data
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
        if not isinstance(existing_data, list):
            existing_data = []
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    # Add timestamp
    data['timestamp'] = datetime.now().isoformat()
    existing_data.append(data)

    # Write back
    with open(filename, 'w') as f:
        json.dump(existing_data, f, indent=2)

    print(f"📊 Logged data to {filename}")


print("🚀 Starting Renaissance Technologies Bitcoin Trading Bot (Testing Version)")
print("⚡ Pandas-free version for immediate testing")

try:
    # Initialize components
    print("🔧 Initializing Renaissance Technologies components...")

    data_pipeline = EnhancedDataPipeline()
    signal_generator = EnhancedSignalGenerator()

    print("✅ Components initialized successfully!")

    # Main trading loop
    iteration = 0
    while True:
        iteration += 1
        current_time = datetime.now()
        print(f"\n⏰ Iteration {iteration} - {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            # Generate Renaissance Technologies signals
            print("📡 Generating Renaissance Technologies signals...")
            signals = signal_generator.generate_signals('BTCUSDT')

            # Display results
            print(f"🎯 Renaissance Technologies Analysis:")
            print(f"   Final Signal: {signals.get('final_signal', 'HOLD')}")
            print(f"   Confidence: {signals.get('confidence', 0):.3f}")
            print(f"   Alternative Data Score: {signals.get('alternative_data_score', 0):.3f}")

            # Log to JSON instead of Excel
            log_data = {
                'iteration': iteration,
                'signal': signals.get('final_signal', 'HOLD'),
                'confidence': signals.get('confidence', 0),
                'alternative_data_score': signals.get('alternative_data_score', 0),
                'components': signals.get('components', {})
            }

            log_simple_data(log_data)

            print("✅ Renaissance Technologies cycle completed successfully!")

        except Exception as e:
            print(f"❌ Error in trading cycle: {e}")
            print(f"🔍 Error details: {type(e).__name__}")

        # Wait 5 minutes (300 seconds) - you can adjust this
        print("⏳ Waiting 5 minutes for next cycle...")
        time.sleep(300)

except KeyboardInterrupt:
    print("\n🛑 Renaissance Technologies Trading Bot stopped by user")
    print("📊 Check trading_log_*.json files for results")
except Exception as e:
    print(f"❌ Fatal error: {e}")
    print(f"🔍 Error type: {type(e).__name__}")
    sys.exit(1)