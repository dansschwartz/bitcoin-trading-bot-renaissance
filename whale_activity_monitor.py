"""
Whale Activity & On-Chain Monitor
Tracks large transactions and exchange inflows/outflows as leading indicators.
"""

import logging
import asyncio
import os
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

class WhaleActivityMonitor:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = config.get("whale_alert_key") or os.getenv("WHALE_ALERT_KEY")
        self.min_value = config.get("whale_min_value", 500000) # $500k
        self.active = False
        
    async def get_whale_signals(self) -> Dict[str, Any]:
        """Fetch recent whale alerts and calculate pressure score."""
        if not self.api_key:
            return {"whale_pressure": 0.0, "whale_count": 0, "status": "no_api_key"}
            
        try:
            # Whale Alert API: https://docs.whale-alert.io/
            # Fetches transactions from the last hour
            start_time = int(datetime.now(timezone.utc).timestamp()) - 3600
            url = f"https://api.whale-alert.io/v1/transactions?api_key={self.api_key}&min_value={self.min_value}&start={start_time}"
            
            response = await asyncio.to_thread(requests.get, url, timeout=10)
            if response.status_code != 200:
                self.logger.warning(f"Whale Alert API error: {response.status_code}")
                return {"whale_pressure": 0.0, "whale_count": 0}
                
            data = response.json()
            transactions = data.get("transactions", [])
            
            whale_pressure = 0.0
            whale_count = len(transactions)
            
            for tx in transactions:
                # Logic: Inflows to exchanges are often bearish (selling pressure)
                # Outflows from exchanges are often bullish (holding pressure)
                # Transfer from wallet to wallet is neutral
                
                from_owner = tx.get("from", {}).get("owner_type", "unknown")
                to_owner = tx.get("to", {}).get("owner_type", "unknown")
                
                amount_usd = tx.get("amount_usd", 0)
                
                if from_owner == "unknown" and to_owner == "exchange":
                    whale_pressure -= (amount_usd / self.min_value) * 0.1 # Bearish
                elif from_owner == "exchange" and to_owner == "unknown":
                    whale_pressure += (amount_usd / self.min_value) * 0.1 # Bullish
                    
            return {
                "whale_pressure": max(min(whale_pressure, 1.0), -1.0),
                "whale_count": whale_count,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            self.logger.warning(f"Whale monitor failed: {e}")
            return {"whale_pressure": 0.0, "whale_count": 0}

    def get_mock_signal(self) -> Dict[str, Any]:
        """Returns a neutral mock signal for testing."""
        return {
            "whale_pressure": 0.05, # Slight bullish mock
            "whale_count": 3,
            "timestamp": datetime.now(timezone.utc)
        }
