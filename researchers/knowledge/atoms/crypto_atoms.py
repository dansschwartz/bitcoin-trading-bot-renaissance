"""Crypto knowledge atoms — wrap crypto_knowledge.py constants as queryable atoms."""
from knowledge.registry import KB, Atom, Pair, Regime, Scale, AType, Affects
from knowledge.shared.crypto_knowledge import (
    FUNDING_MECHANICS, LIQUIDATION_CASCADE, EXCHANGE_FEES, MARKET_TIMING, SYSTEM_MAP
)


KB.register_many([
    # Funding mechanics
    Atom("ck.funding_formula", "Funding Rate Formula", "crypto",
         f"Funding Payment = Position Size * Rate. Settlement: {FUNDING_MECHANICS['settlement']}. "
         f"Rate: {FUNDING_MECHANICS['rate_formula']}.",
         atype=AType.REFERENCE, affects=Affects.SIGNALS|Affects.COST, crypto_specific=True,
         value=FUNDING_MECHANICS),
    Atom("ck.funding_thresholds", "Funding Rate Thresholds", "crypto",
         "Strong bearish: >0.01%. Strong bullish: <-0.01%. Extreme: >0.03%. Noise: <0.005%.",
         atype=AType.THRESHOLD, affects=Affects.SIGNALS, crypto_specific=True,
         value=FUNDING_MECHANICS["thresholds"]),
    Atom("ck.funding_strategy", "Pre-Settlement Strategy", "crypto",
         f"{FUNDING_MECHANICS['strategy']}",
         atype=AType.SIGNAL, affects=Affects.SIGNALS|Affects.EXECUTION, crypto_specific=True,
         scales=Scale.HOUR1|Scale.HOUR4),

    # Liquidation cascade
    Atom("ck.liquidation_mechanism", "Liquidation Cascade Mechanism", "crypto",
         f"{LIQUIDATION_CASCADE['mechanism']}. Cross-collateral: {LIQUIDATION_CASCADE['cross_collateral']}.",
         atype=AType.REFERENCE, affects=Affects.RISK, crypto_specific=True,
         regimes=Regime.HIGH_VOL|Regime.CRISIS),
    Atom("ck.liquidation_signals", "Liquidation Detection Signals", "crypto",
         "Signals: " + "; ".join(LIQUIDATION_CASCADE["detection_signals"]),
         atype=AType.SIGNAL, affects=Affects.RISK|Affects.SIGNALS, crypto_specific=True,
         regimes=Regime.HIGH_VOL|Regime.CRISIS),
    Atom("ck.pair_risk", "Per-Pair Cascade Risk", "crypto",
         "BTC: primary source. ETH: secondary/DeFi. SOL: high beta. "
         "LINK: oracle/DeFi. AVAX: thin liquidity. DOGE: retail leverage.",
         atype=AType.REFERENCE, affects=Affects.RISK, crypto_specific=True,
         value=LIQUIDATION_CASCADE["pair_risk"]),

    # Exchange fees
    Atom("ck.mexc_fees", "MEXC Fee Structure", "crypto",
         f"Maker: {EXCHANGE_FEES['mexc']['maker']*100}%. Taker: {EXCHANGE_FEES['mexc']['taker']*100}%. "
         f"{EXCHANGE_FEES['mexc']['notes']}.",
         atype=AType.THRESHOLD, affects=Affects.COST|Affects.EXECUTION, crypto_specific=True,
         value=EXCHANGE_FEES["mexc"]),
    Atom("ck.binance_fees", "Binance Fee Structure", "crypto",
         f"Maker: {EXCHANGE_FEES['binance']['maker']*100}%. Taker: {EXCHANGE_FEES['binance']['taker']*100}%. "
         f"{EXCHANGE_FEES['binance']['notes']}.",
         atype=AType.THRESHOLD, affects=Affects.COST, crypto_specific=True,
         value=EXCHANGE_FEES["binance"]),
    Atom("ck.round_trip_costs", "Round-Trip Trading Costs", "crypto",
         "Taker round-trip: MEXC 2bps, Binance 8bps, cross-exchange 5bps.",
         atype=AType.THRESHOLD, affects=Affects.COST|Affects.SIZING, crypto_specific=True,
         value=EXCHANGE_FEES["round_trip_taker"]),

    # Market timing
    Atom("ck.market_timing", "Crypto Market Timing", "crypto",
         f"Peak volume: {MARKET_TIMING['peak_volume_utc']}. "
         f"Lowest: {MARKET_TIMING['lowest_volume']}. "
         f"Funding: {', '.join(MARKET_TIMING['funding_settlements'])} UTC.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION|Affects.SIGNALS, crypto_specific=True,
         value=MARKET_TIMING),
    Atom("ck.options_expiry", "Options Expiry Calendar", "crypto",
         f"{MARKET_TIMING['options_expiry']}. Expect vol spike and price pinning near strikes.",
         atype=AType.REFERENCE, affects=Affects.RISK|Affects.SIGNALS, crypto_specific=True,
         scales=Scale.DAY1),

    # System map
    Atom("ck.system_architecture", "System Component Map", "crypto",
         "Pipeline: market_data -> ml_model_loader -> real_time_pipeline -> "
         "portfolio_engine -> kelly_sizer -> risk_gateway -> execution.",
         atype=AType.REFERENCE, affects=Affects.EXECUTION,
         value=SYSTEM_MAP),
    Atom("ck.risk_gateway_warning", "Risk Gateway — NEVER MODIFY", "crypto",
         "risk_gateway.py is the final safety check. NEVER modify thresholds, "
         "circuit breakers, or position limits. This is an immutable constraint.",
         atype=AType.REFERENCE, affects=Affects.RISK,
         value={"file": "risk_gateway.py", "action": "NEVER_MODIFY"}),
])
