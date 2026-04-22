import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class VolumeProfile:
    poc: float
    vah: float
    val: float
    high: float
    low: float
    total_volume: float
    bin_size: float
    profile: Dict[float, float]

class VolumeProfileEngine:
    """
    Institutional Volume Profile Engine.
    Calculates Point of Control (POC), Value Area High (VAH), and Value Area Low (VAL).
    Used to identify price areas of high and low liquidity.
    """
    def __init__(self, value_area_pct: float = 0.7):
        self.value_area_pct = value_area_pct
        self.logger = logging.getLogger(__name__)

    def calculate_profile(self, df: pd.DataFrame, bins: int = 50) -> Optional[VolumeProfile]:
        """Calculates volume profile from OHLCV data."""
        try:
            if df.empty or len(df) < 10:
                return None

            high = df['high'].max()
            low = df['low'].min()
            
            if high == low:
                return None

            bin_size = (high - low) / bins
            
            # Create price bins
            price_bins = np.linspace(low, high, bins + 1)
            volume_dist = np.zeros(bins)

            # Distribute volume across bins
            # Simplified: Use Close price for binning. 
            # More advanced: distribute between High and Low.
            for i in range(len(df)):
                c = df['close'].iloc[i]
                v = df['volume'].iloc[i]
                # Find which bin the price falls into
                bin_idx = min(int((c - low) / bin_size), bins - 1)
                volume_dist[bin_idx] += v

            # 1. POC: Bin with max volume
            poc_idx = np.argmax(volume_dist)
            poc = price_bins[poc_idx] + (bin_size / 2)

            # 2. Value Area
            total_volume = volume_dist.sum()
            target_va_volume = total_volume * self.value_area_pct
            
            current_va_volume = volume_dist[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx

            while current_va_volume < target_va_volume:
                # Check neighbors
                next_low_vol = volume_dist[low_idx - 1] if low_idx > 0 else 0
                next_high_vol = volume_dist[high_idx + 1] if high_idx < bins - 1 else 0

                if next_low_vol == 0 and next_high_vol == 0:
                    break

                if next_low_vol >= next_high_vol:
                    current_va_volume += next_low_vol
                    low_idx -= 1
                else:
                    current_va_volume += next_high_vol
                    high_idx += 1

            vah = price_bins[high_idx + 1]
            val = price_bins[low_idx]

            profile_dict = {float(price_bins[i]): float(volume_dist[i]) for i in range(bins)}

            return VolumeProfile(
                poc=float(poc),
                vah=float(vah),
                val=float(val),
                high=float(high),
                low=float(low),
                total_volume=float(total_volume),
                bin_size=float(bin_size),
                profile=profile_dict
            )

        except Exception as e:
            self.logger.error(f"Volume profile calculation failed: {e}")
            return None

    def get_profile_signal(self, current_price: float, profile: VolumeProfile) -> Dict[str, Any]:
        """Generates signals based on price position relative to Value Area."""
        if not profile:
            return {'signal': 0.0, 'status': 'no_profile'}

        # Position relative to Value Area
        if current_price > profile.vah:
            # Price above value area - potentially bullish breakout or overextended
            # Signal depends on distance from VAH
            strength = (current_price - profile.vah) / (profile.high - profile.vah) if profile.high > profile.vah else 0.1
            signal = min(strength, 1.0)
            status = "Above Value Area"
        elif current_price < profile.val:
            # Price below value area - potentially bearish breakout or oversold
            strength = (profile.val - current_price) / (profile.val - profile.low) if profile.val > profile.low else 0.1
            signal = -min(strength, 1.0)
            status = "Below Value Area"
        else:
            # Inside Value Area - Mean reversion to POC
            dist_to_poc = current_price - profile.poc
            max_dist = max(profile.vah - profile.poc, profile.poc - profile.val)
            # Reversion signal: negative if above POC, positive if below
            signal = - (dist_to_poc / max_dist) if max_dist > 0 else 0.0
            status = "Inside Value Area"

        return {
            'signal': float(signal),
            'status': status,
            'poc_dist': float(current_price - profile.poc),
            'vah_dist': float(current_price - profile.vah),
            'val_dist': float(current_price - profile.val)
        }
