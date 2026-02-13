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
        "WhaleAlert": os.getenv("WHALE_ALERT_KEY"),
        "Guavy": os.getenv("GUAVY_API_KEY")
    }
    
    all_present = True
    for name, key in keys_to_check.items():
        if key:
            print(f"  ✅ {name:16}: PRESENT (ends in ...{key[-4:]})")
        else:
            print(f"  ❌ {name:16}: MISSING")
            all_present = False
            
    # Optional legacy keys
    legacy_keys = {
        "NewsAPI": os.getenv("NEWSAPI_KEY"),
        "Twitter": os.getenv("TWITTER_BEARER_TOKEN"),
        "Reddit": os.getenv("REDDIT_CLIENT_ID")
    }
    for name, key in legacy_keys.items():
        if not key:
            print(f"  ℹ️  {name:16}: OPTIONAL (Legacy fallback)")
            
    return all_present

async def check_network_latency():
    print("\n🌐 Checking Network Latency...")
    urls = [
        ("Coinbase", "https://api.coinbase.com/v2/time"),
        ("Ollama (Local LLM)", "http://localhost:11434/api/tags"),
        ("Guavy API", "https://data.guavy.com/api/v1/ping")
    ]
    
    all_good = True
    for name, url in urls:
        try:
            start = time.time()
            # Use a short timeout for Ollama and Guavy check
            timeout = 2 if name in ["Ollama (Local LLM)", "Guavy API"] else 5
            
            headers = {}
            if name == "Guavy API":
                key = os.getenv("GUAVY_API_KEY")
                if key:
                    headers['Authorization'] = f'Bearer {key}'
                else:
                    print(f"  ❌ {name:20}: SKIPPED (No API Key)")
                    all_good = False
                    continue

            response = requests.get(url, headers=headers, timeout=timeout)
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                print(f"  ✅ {name:20}: {latency:.2f}ms")
            else:
                if name == "Ollama (Local LLM)":
                    print(f"  ⚠️  {name:20}: NOT RUNNING (Status: {response.status_code}) - Deep NLP will use mocks.")
                elif name == "Guavy API":
                    print(f"  ❌ {name:20}: AUTH FAILED (Status: {response.status_code})")
                    all_good = False
                else:
                    print(f"  ❌ {name:20}: FAILED (Status: {response.status_code})")
                    all_good = False
        except Exception as e:
            if name == "Ollama (Local LLM)":
                print(f"  ⚠️  {name:20}: NOT REACHABLE - Deep NLP will use mocks.")
            else:
                print(f"  ❌ {name:20}: ERROR ({e})")
                all_good = False
            
    return all_good

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

def check_ml_environment():
    print("\n🤖 Checking ML Environment...")
    
    components = {
        "PyTorch": "torch",
        "NumPy": "numpy",
        "Pandas": "pandas",
        "Scipy": "scipy",
        "Sklearn": "sklearn"
    }
    
    all_good = True
    for name, module in components.items():
        try:
            import importlib
            importlib.import_module(module)
            print(f"  ✅ {name:16}: INSTALLED")
        except ImportError:
            print(f"  ❌ {name:16}: MISSING")
            all_good = False
            
    # Check for model weights (simulated check for common paths)
    model_paths = [
        "models/cnn_lstm_weights.pt",
        "models/vae_anomaly_weights.pt"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            print(f"  ✅ Model Weights   : {path} FOUND")
        else:
            print(f"  ℹ️  Model Weights   : {path} NOT FOUND (Will use default/cold start)")
            
    return all_good

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
        "Database": check_database_integrity(db_path),
        "ML Env": check_ml_environment()
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
