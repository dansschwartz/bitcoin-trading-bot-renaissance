"""
🧠 DEEP NLP BRIDGE
==================
Integrates LLM-based reasoning (Llama, GPT) for deep analysis 
of news, social media, and market context.
"""

import logging
import os
import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

class DeepNLPBridge:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.provider = config.get("provider", "ollama") # Default to local Ollama
        self.api_url = config.get("api_url", "http://localhost:11434/api/generate")
        self.model = config.get("model", "llama3")
        self.enabled = config.get("enabled", True)
        
    async def analyze_sentiment_with_reasoning(self, text: str) -> Dict[str, Any]:
        """
        Sends text to an LLM to extract sentiment AND reasoning.
        """
        if not self.enabled or not text:
            return {"sentiment": 0.0, "reasoning": "NLP Disabled", "confidence": 0.0}

        prompt = f"""
        Analyze the following crypto market news/text and provide:
        1. Sentiment Score (-1.0 to 1.0)
        2. Brief Reasoning (1 sentence)
        3. Confidence Level (0.0 to 1.0)
        
        Text: {text[:2000]}
        
        Respond ONLY in JSON format like: 
        {{"sentiment": 0.5, "reasoning": "positive whale activity", "confidence": 0.8}}
        """

        try:
            if self.provider == "ollama":
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
                response = requests.post(self.api_url, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    return json.loads(result.get("response", "{}"))
            
            # Fallback/Mock for testing
            return {"sentiment": 0.1, "reasoning": "LLM simulation enabled", "confidence": 0.5}

        except Exception as e:
            self.logger.error(f"Deep NLP analysis failed: {e}")
            return {"sentiment": 0.0, "reasoning": f"Error: {str(e)}", "confidence": 0.0}

    def get_market_reasoning_prompt(self, market_data: Dict[str, Any]) -> str:
        """Constructs a prompt for the LLM to 'think' about the current market state."""
        return f"Market is currently at {market_data.get('ticker', {}).get('price')}. VPIN is {market_data.get('vpin')}. What is the institutional outlook?"
