import os
import sys
import json
import time
import sqlite3
import requests
import asyncio
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def check_api_keys(config):
    print("🔑 Checking API Keys...")
    
    keys_to_check = {
        "Coinbase": os.getenv("COINBASE_API_KEY"),
        "NewsAPI": os.getenv("NEWSAPI_KEY"),
        "Twitter": os.getenv("TWITTER_BEARER_TOKEN"),
        "Reddit Client ID": os.getenv("REDDIT_CLIENT_ID"),
        "WhaleAlert": os.getenv("WHALE_ALERT_KEY")
    }
    
    all_present = True
    for name, key in keys_to_check.items():
        if key:
            print(f"  ✅ {name:16}: PRESENT (ends in ...{key[-4:]})")
        else:
            print(f"  ❌ {name:16}: MISSING")
            all_present = False
    return all_present

async def check_network_latency():
    print("\n🌐 Checking Network Latency...")
    url = "https://api.coinbase.com/v2/time"
    latencies = []
    
    for i in range(3):
        try:
            start = time.time()
            response = requests.get(url, timeout=5)
            latency = (time.time() - start) * 1000
            if response.status_code == 200:
                print(f"  Ping {i+1}: {latency:.2f}ms")
                latencies.append(latency)
            else:
                print(f"  Ping {i+1}: FAILED (Status: {response.status_code})")
        except Exception as e:
            print(f"  Ping {i+1}: ERROR ({e})")
            
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"  ✅ Average Latency: {avg_latency:.2f}ms")
        return avg_latency < 500
    return False

def check_database_integrity(db_path):
    print("\n🗄️  Checking Database Integrity...")
    if not Path(db_path).exists():
        print(f"  ❌ Database not found at {db_path}")
        return False
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        conn.close()
        
        if result == "ok":
            print(f"  ✅ Database Integrity: {result}")
            return True
        else:
            print(f"  ❌ Database Integrity: {result}")
            return False
    except Exception as e:
        print(f"  ❌ Database Check Failed: {e}")
        return False

async def main():
    print("🏛️  RENAISSANCE BOT: PRE-FLIGHT COMBAT READINESS CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    config_path = Path("config/config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}
        print("⚠️  Warning: config/config.json not found, using environment variables only.\n")

    db_path = config.get("database", {}).get("path", "data/renaissance_bot.db")
    
    results = {
        "API Keys": check_api_keys(config),
        "Network": await check_network_latency(),
        "Database": check_database_integrity(db_path)
    }
    
    print("\n" + "=" * 60)
    if all(results.values()):
        print("🚀 ALL SYSTEMS GO! Bot is ready for combat.")
        sys.exit(0)
    else:
        print("⚠️  CAUTION: Some systems failed readiness checks.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
