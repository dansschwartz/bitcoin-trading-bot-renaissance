"""Deep crypto market structure knowledge shared across all researchers."""

FUNDING_MECHANICS = {
    "formula": "Funding Payment = Position Size * Funding Rate",
    "settlement": "Every 8h: 00:00, 08:00, 16:00 UTC",
    "rate_formula": "Premium Index + clamp(Interest Rate - Premium Index, -0.05%, 0.05%)",
    "interest_rate": "0.01% per 8h (0.03% daily, ~11% annualized)",
    "thresholds": {
        "strong_bearish": 0.0001,   # funding > 0.01%
        "strong_bullish": -0.0001,  # funding < -0.01%
        "extreme_bearish": 0.0003,  # funding > 0.03%
        "extreme_bullish": -0.0003, # funding < -0.03%
        "noise": 0.00005,           # |funding| < 0.005%
    },
    "strategy": "Pre-settlement (30-60min before): crowded side reduces positions -> predictable move",
}

LIQUIDATION_CASCADE = {
    "mechanism": "Price drop -> forced sells -> more drops -> positive feedback",
    "cross_collateral": "ETH as collateral for LINK -> ETH drop liquidates LINK positions",
    "detection_signals": [
        "OI declining + price declining -> liquidations happening NOW",
        "Funding > 0.03% + OI > 95th percentile -> cascade risk HIGH",
        "Cross-asset correlation spike (5-min rolling > 0.95)",
    ],
    "pair_risk": {
        "BTC": "Primary source. When BTC cascades, everything follows.",
        "ETH": "Secondary source. DeFi collateral creates extra pressure.",
        "SOL": "High beta to BTC. Cascades fastest after BTC.",
        "LINK": "Oracle token. DeFi liquidations force LINK selling.",
        "AVAX": "DeFi ecosystem. Thinner liquidity -> bigger moves.",
        "DOGE": "Retail-dominated. Leveraged retail on MEXC/Binance.",
    },
}

EXCHANGE_FEES = {
    "mexc": {"maker": 0.0, "taker": 0.0001, "notes": "0% maker is huge advantage for limit orders"},
    "binance": {"maker": 0.0002, "taker": 0.0004, "notes": "With BNB discount: 0.015%/0.03%"},
    "round_trip_taker": {"mexc": 0.0002, "binance": 0.0008, "cross_exchange": 0.0005},
}

MARKET_TIMING = {
    "peak_volume_utc": "16:00-20:00 (US + Europe overlap)",
    "lowest_volume": "Saturday-Sunday (30-50% of weekday)",
    "funding_settlements": ["00:00", "08:00", "16:00"],
    "options_expiry": "Last Friday of month (quarterly: Mar, Jun, Sep, Dec)",
}

SYSTEM_MAP = {
    "market_data_provider": {"produces": ["OHLCV bars", "ticker", "orderbook"], "file": "market_data_provider.py"},
    "ml_model_loader": {"produces": ["98-dim features", "per-model predictions"], "key_fn": "build_feature_sequence()"},
    "real_time_pipeline": {"produces": ["ensemble prediction", "confidence"], "file": "real_time_pipeline.py"},
    "regime_detector": {"produces": ["regime", "probabilities"], "states": ["low_vol", "trending", "high_vol"]},
    "portfolio_engine": {"produces": ["signal fusion", "trade decisions"], "file": "portfolio_engine.py"},
    "kelly_position_sizer": {"produces": ["position size USD"], "file": "kelly_position_sizer.py"},
    "risk_gateway": {"produces": ["approve/reject"], "WARNING": "NEVER MODIFY"},
    "devil_tracker": {"produces": ["cost measurements"], "file": "monitors/devil_tracker.py"},
}
