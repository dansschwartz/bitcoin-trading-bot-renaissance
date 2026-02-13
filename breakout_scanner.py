import asyncio
import ccxt
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class BreakoutScanner:
    """
    Renaissance Global Scanner: Identifies volatility expansion and momentum breakouts 
    across hundreds of assets in real-time.
    """
    def __init__(self, exchanges: List[str] = ["coinbase"], top_n: int = 30, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.exchanges = exchanges
        self.top_n = top_n
        self.clients: Dict[str, Any] = {}
        self.primary_exchange = exchanges[0] if exchanges else "coinbase"
        self._initialize_clients()
        
    def _initialize_clients(self):
        for ex_id in self.exchanges:
            try:
                # Use ccxt.async_support for better performance if needed, 
                # but staying with standard for compatibility with current pipeline
                client_class = getattr(ccxt, ex_id.lower())
                self.clients[ex_id] = client_class({'enableRateLimit': True})
                self.logger.info(f"BreakoutScanner initialized for {ex_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize scanner for {ex_id}: {e}")

    async def scan_all_exchanges(self) -> List[Dict[str, Any]]:
        """Scans all configured exchanges for high-probability breakout candidates."""
        all_breakouts = []
        for ex_id, client in self.clients.items():
            try:
                breakouts = await self.scan_exchange(ex_id, client)
                all_breakouts.extend(breakouts)
            except Exception as e:
                self.logger.error(f"Scanner error on {ex_id}: {e}")
        
        # Deduplicate and sort
        unique_breakouts = {}
        for b in all_breakouts:
            symbol = b['symbol']
            if symbol not in unique_breakouts or b['breakout_score'] > unique_breakouts[symbol]['breakout_score']:
                unique_breakouts[symbol] = b
                
        return sorted(unique_breakouts.values(), key=lambda x: x['breakout_score'], reverse=True)

    async def scan_exchange(self, ex_id: str, client: Any) -> List[Dict[str, Any]]:
        """Fetches tickers, filters for volume, and analyzes for breakouts."""
        try:
            # 1. Fetch all tickers
            tickers = await asyncio.to_thread(client.fetch_tickers)
            
            # 2. Filter for USD/USDT and high volume
            candidates = []
            
            # For secondary exchanges, check if coin is also on primary exchange
            # to ensure the bot can actually trade it.
            primary_symbols = []
            if ex_id != self.primary_exchange:
                 primary_client = self.clients.get(self.primary_exchange)
                 if primary_client:
                     markets = await asyncio.to_thread(primary_client.load_markets)
                     primary_symbols = list(markets.keys())

            for symbol, t in tickers.items():
                if not any(quote in symbol for quote in ['/USD', '/USDT']):
                    continue
                
                # Filter for tradeable on primary exchange if not on primary
                if primary_symbols and symbol not in primary_symbols:
                    # Try to match base currency
                    base = symbol.split('/')[0]
                    if not any(base in ps for ps in primary_symbols):
                        continue

                vol = t.get('quoteVolume') or 0
                if vol > 1000000: # Min $1M volume
                    candidates.append({
                        'symbol': symbol,
                        'volume': vol,
                        'change': t.get('percentage', 0),
                        'last': t.get('last')
                    })
            
            # Sort by volume and take top_n
            candidates = sorted(candidates, key=lambda x: x['volume'], reverse=True)[:self.top_n]
            self.logger.info(f"Analyzing {len(candidates)} candidates on {ex_id}...")
            
            # 3. Analyze each candidate in parallel
            tasks = [self._analyze_candidate(client, c) for c in candidates]
            results = await asyncio.gather(*tasks)
            
            return [r for r in results if r and r['is_breakout']]
            
        except Exception as e:
            self.logger.error(f"Failed to scan {ex_id}: {e}")
            return []

    async def _analyze_candidate(self, client: Any, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        symbol = candidate['symbol']
        try:
            # Fetch OHLCV
            # 1h timeframe gives a good balance between noise and trend
            limit = 100
            ohlcv = await asyncio.to_thread(client.fetch_ohlcv, symbol, timeframe='1h', limit=limit)
            if not ohlcv or len(ohlcv) < 50:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            close = df['close']
            volume = df['volume']
            
            # --- Indicators ---
            
            # 1. Bollinger Bands
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_bb = sma20 + (std20 * 2)
            lower_bb = sma20 - (std20 * 2)
            bw = (upper_bb - lower_bb) / sma20
            
            # 2. Keltner Channels (for squeeze detection)
            atr20 = self._calculate_atr(df, 20)
            upper_kc = sma20 + (atr20 * 1.5)
            lower_kc = sma20 - (atr20 * 1.5)
            
            # Squeeze: BB inside KC
            is_squeeze = (upper_bb.iloc[-1] < upper_kc.iloc[-1]) and (lower_bb.iloc[-1] > lower_kc.iloc[-1])
            
            # 3. Volume Surge
            vol_ma20 = volume.rolling(window=20).mean()
            vol_surge = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 0
            
            # 4. ADX (Trend Strength)
            adx = self._calculate_adx(df, 14)
            
            # 5. Price Position
            current_price = close.iloc[-1]
            is_breaking_bb = current_price > upper_bb.iloc[-2]
            
            # --- Scoring ---
            score = 0
            if is_breaking_bb: score += 40
            if is_squeeze: score += 20
            if vol_surge > 1.5: score += 15
            if vol_surge > 3.0: score += 10
            if adx > 25: score += 15
            
            # Multi-timeframe confirmation (check 4h trend if 1h is breakout)
            if score >= 60:
                ohlcv_4h = await asyncio.to_thread(client.fetch_ohlcv, symbol, timeframe='4h', limit=20)
                if ohlcv_4h and len(ohlcv_4h) > 10:
                    df4 = pd.DataFrame(ohlcv_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    # If 4h is also uptrending
                    if df4['close'].iloc[-1] > df4['close'].rolling(window=10).mean().iloc[-1]:
                        score += 10
            
            return {
                'symbol': symbol,
                'exchange': client.id,
                'price': current_price,
                'change_24h': candidate['change'],
                'volume_surge': vol_surge,
                'adx': adx,
                'is_squeeze': is_squeeze,
                'is_breakout': score >= 65,
                'breakout_score': score,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            # Quietly fail for individual symbol errors to keep scanner moving
            return None

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Simple ADX calculation."""
        try:
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = minus_dm.abs()
            
            tr = self._calculate_atr(df, 1) # True Range
            tr_smooth = tr.rolling(window=period).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr_smooth)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr_smooth)
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return float(adx.iloc[-1])
        except:
            return 0.0

if __name__ == "__main__":
    # Test script
    logging.basicConfig(level=logging.INFO)
    scanner = BreakoutScanner(exchanges=["coinbase", "kraken"])
    
    async def run_test():
        results = await scanner.scan_all_exchanges()
        print(f"\n🚀 FOUND {len(results)} BREAKOUT CANDIDATES:")
        for r in results:
            print(f"[{r['symbol']}] Score: {r['breakout_score']} | Price: {r['price']} | Vol Surge: {r['volume_surge']:.2f} | Squeeze: {r['is_squeeze']}")

    asyncio.run(run_test())
