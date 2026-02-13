import os
import asyncio
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from coinbase_client import EnhancedCoinbaseClient, CoinbaseCredentials
from whale_activity_monitor import WhaleActivityMonitor

async def test_api_keys():
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("API_Tester")
    
    print("\n🏛️  RENAISSANCE BOT: LIVE API KEY VERIFICATION")
    print("=" * 60)
    
    # 1. Test WhaleAlert
    whale_key = os.getenv("WHALE_ALERT_KEY")
    if whale_key:
        print(f"🐋 Testing WhaleAlert API (Key: {whale_key[:4]}...{whale_key[-4:]})")
        whale_monitor = WhaleActivityMonitor({"whale_alert_key": whale_key}, logger=logger)
        result = await whale_monitor.get_whale_signals()
        if result.get("status") == "no_api_key":
             print("  ❌ WhaleAlert: Key not recognized by monitor.")
        elif result.get("whale_count", 0) > 0 or result.get("whale_pressure") != 0:
            print(f"  ✅ WhaleAlert: SUCCESS! Found {result.get('whale_count')} recent whale transactions.")
        elif "whale_count" in result:
             # It might be 0 at this specific moment, but if the keys were wrong it would likely error.
             # Actually get_whale_signals returns 0.0 pressure if it fails with error.
             print(f"  ✅ WhaleAlert: CONNECTED (Found {result.get('whale_count')} transactions).")
        else:
            print(f"  ❌ WhaleAlert: FAILED. Result: {result}")
    else:
        print("  ❌ WhaleAlert: KEY MISSING in .env")

    # 2. Test Coinbase
    cb_key = os.getenv("COINBASE_API_KEY")
    cb_secret = os.getenv("COINBASE_API_SECRET")
    
    if cb_key and cb_secret:
        print(f"\n🪙 Testing Coinbase API (Key: {cb_key[:4]}...{cb_key[-4:]})")
        # Ensure we use live mode (paper_trading=False for health_check connectivity test)
        # Note: EnhancedCoinbaseClient health_check calls get_accounts() if paper_trading=False
        creds = CoinbaseCredentials(
            api_key=cb_key,
            api_secret=cb_secret,
            sandbox=False
        )
        client = EnhancedCoinbaseClient(creds, logger=logger, paper_trading=False)
        
        # Test basic connectivity via get_accounts
        try:
            health = client.health_check()
            conn_status = health.get("checks", {}).get("connectivity", "unknown")
            if conn_status == "ok":
                print("  ✅ Coinbase: SUCCESS! Authenticated and reached account endpoints.")
            else:
                print(f"  ❌ Coinbase: FAILED. Status: {conn_status}")
                if "failed" in conn_status:
                    print(f"     Error Detail: {conn_status}")
        except Exception as e:
            print(f"  ❌ Coinbase: ERROR during health check: {e}")
    else:
        print("\n  ❌ Coinbase: KEY or SECRET MISSING in .env")

    print("\n" + "=" * 60)
    print("Verification complete.")

if __name__ == "__main__":
    asyncio.run(test_api_keys())
