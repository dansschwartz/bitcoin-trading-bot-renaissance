"""
bot/helpers.py — Setup, config, heartbeat, dashboard logging, and performance summary.

Extracted from RenaissanceTradingBot to reduce god-class size.
All functions accept `bot` as first argument (the bot instance).
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from logger import SecretMaskingFilter

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot, TradingDecision

logger = logging.getLogger(__name__)


def setup_logging(bot: "RenaissanceTradingBot", config: Dict[str, Any]) -> logging.Logger:
    """Setup comprehensive logging."""
    import renaissance_trading_bot as _rtb_mod

    log_cfg = config.get("logging", {})
    log_file = log_cfg.get("file", "logs/renaissance_bot.log")
    log_level = log_cfg.get("level", "INFO")

    log_path = (Path(_rtb_mod.__file__).resolve().parent / log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Use force=True to override any handlers set by imports
    logging.basicConfig(
        level=getattr(logging, str(log_level).upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_path, maxBytes=50 * 1024 * 1024, backupCount=5
            ),
            logging.StreamHandler()
        ],
        force=True,
    )

    # Apply secret masking to all handlers
    masking_filter = SecretMaskingFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(masking_filter)

    return logging.getLogger(__name__)


def load_config(bot: "RenaissanceTradingBot", config_path: Path) -> Dict[str, Any]:
    """Load bot configuration from JSON file."""
    default_config = {
        "trading": {
            "product_id": "BTC-USD",
            "cycle_interval_seconds": 300,
            "paper_trading": True,
            "sandbox": True
        },
        "risk_management": {
            "daily_loss_limit": 500,
            "position_limit": 1000,
            "min_confidence": 0.50
        },
        "signal_weights": {
            "order_flow": 0.32,
            "order_book": 0.21,
            "volume": 0.14,
            "macd": 0.105,
            "rsi": 0.115,
            "bollinger": 0.095,
            "alternative": 0.045
        },
        "data": {
            "candle_granularity": "ONE_MINUTE",
            "candle_lookback_minutes": 120,
            "order_book_depth": 10
        },
        "logging": {
            "file": "logs/renaissance_bot.log",
            "level": "INFO"
        }
    }

    if not config_path.exists():
        return default_config

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            return {**default_config, **loaded}
    except Exception as exc:
        print(f"Warning: failed to load config at {config_path}: {exc}")
        return default_config


def validate_config(bot: "RenaissanceTradingBot", config: Dict[str, Any]) -> None:
    """Validate critical config values at startup."""
    errors = []

    risk = config.get("risk_management", {})
    dl = risk.get("daily_loss_limit", 500)
    if not (0 < dl <= 100000):
        errors.append(f"daily_loss_limit={dl} out of range (0, 100000]")
    pl = risk.get("position_limit", 1000)
    if not (0 < pl <= 1000000):
        errors.append(f"position_limit={pl} out of range (0, 1000000]")
    mc = risk.get("min_confidence", 0.65)
    if not (0.0 < mc <= 1.0):
        errors.append(f"min_confidence={mc} out of range (0, 1.0]")

    trading = config.get("trading", {})
    interval = trading.get("cycle_interval_seconds", 300)
    if not (10 <= interval <= 3600):
        errors.append(f"cycle_interval_seconds={interval} out of range [10, 3600]")

    # Auto-normalize signal weights
    weights = config.get("signal_weights", {})
    if weights:
        total = sum(float(v) for v in weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            bot.logger.warning(f"Signal weights sum to {total:.3f}, normalizing to 1.0")
            for k in weights:
                weights[k] = float(weights[k]) / total

    if errors:
        for e in errors:
            bot.logger.error(f"CONFIG ERROR: {e}")
        raise ValueError(f"Invalid configuration: {'; '.join(errors)}")


def write_heartbeat(bot: "RenaissanceTradingBot") -> None:
    """Write heartbeat file for external monitoring."""
    try:
        bot.HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        heartbeat = {
            "alive": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cycle_count": len(bot.decision_history),
            "killed": bot._killed,
            "paper_mode": bot.coinbase_client.paper_trading,
        }
        with open(bot.HEARTBEAT_FILE, 'w') as f:
            json.dump(heartbeat, f)
    except Exception as e:
        bot.logger.warning(f"Heartbeat file write failed: {e}")


def log_consciousness_dashboard(
    bot: "RenaissanceTradingBot",
    product_id: str,
    decision: "TradingDecision",
    rt_result: Optional[Dict[str, Any]],
) -> None:
    """Displays the bot's 'Inner Thoughts' and ML consensus in a rich format."""
    bot.logger.info(f"\n" + "="*60 + f"\n🧠 CONSCIOUSNESS DASHBOARD: {product_id}\n" + "="*60)

    # 1. Decision Summary
    action_emoji = "🚀" if decision.action == "BUY" else "🔻" if decision.action == "SELL" else "⚖️"
    mode = decision.reasoning.get('execution_mode', 'TAKER')
    bot.logger.info(f"ACTION: {action_emoji} {decision.action} | MODE: {mode} | CONFIDENCE: {decision.confidence:.2%} | SIZE: {decision.position_size:.2%}")

    # 2. Market Regime & Global Intelligence
    regime = bot.regime_overlay.get_current_regime() if hasattr(bot.regime_overlay, 'get_current_regime') else "NORMAL"
    boost = bot.regime_overlay.get_confidence_boost()
    bot.logger.info(f"MARKET REGIME: {regime} | REGIME BOOST: {boost:+.4f}")

    # Whale & Lead-Lag Signals
    whale = decision.reasoning.get('whale_signals', {})
    w_pressure = whale.get('whale_pressure', 0.0)
    w_count = whale.get('whale_count', 0)
    w_emoji = "🐋" if abs(w_pressure) > 0.1 else "🌊"

    lead_lag = decision.reasoning.get('lead_lag_alpha', {})
    corr = lead_lag.get('correlation', 0.0)
    lag = lead_lag.get('lag_periods', 0)
    ll_emoji = "🔗" if abs(corr) > 0.7 else "⛓️"

    bot.logger.info(f"WHALE PRESSURE: {w_emoji} {w_pressure:+.4f} ({w_count} alerts) | LEAD-LAG: {ll_emoji} Corr:{corr:.2f} Lag:{lag}")

    # 2.5 Market Microstructure & Fractal Intelligence
    ms_metrics = bot.microstructure_engine.get_latest_metrics()
    vpin = ms_metrics.vpin if ms_metrics else 0.5
    v_emoji = "⚠️" if vpin > 0.7 else "⚖️"

    tech_signals = bot._get_tech(product_id).get_latest_signals()
    hurst = tech_signals.hurst_exponent if tech_signals else 0.5
    h_emoji = "📈" if hurst > 0.6 else "📉" if hurst < 0.4 else "↔️"
    h_status = "Trending" if hurst > 0.6 else "Mean-Rev" if hurst < 0.4 else "Random"

    bot.logger.info(f"VPIN TOXICITY: {v_emoji} {vpin:.4f} | HURST EXP: {h_emoji} {hurst:.4f} ({h_status})")

    # 2.6 Statistical Arbitrage Signal
    stat_arb = decision.reasoning.get('stat_arb', {})
    sa_signal = stat_arb.get('signal', 0.0)
    sa_z = stat_arb.get('z_score', 0.0)
    sa_emoji = "🎯" if abs(sa_signal) > 0.3 else "⚖️"
    bot.logger.info(f"STAT ARB SIGNAL: {sa_emoji} {sa_signal:+.4f} (Z-Score: {sa_z:+.2f})")

    # 2.7 Volume Profile Signal
    vp_signal = decision.reasoning.get('volume_profile_signal', 0.0)
    vp_status = decision.reasoning.get('volume_profile_status', 'Unknown')
    vp_emoji = "📊" if abs(vp_signal) > 0.3 else "⚖️"
    bot.logger.info(f"VOLUME PROFILE: {vp_emoji} {vp_signal:+.4f} ({vp_status})")

    # 2.8 High-Dimensional Intelligence (Fractal, Entropy, Quantum)
    fractal = decision.reasoning.get('fractal_intelligence', {})
    f_pattern = fractal.get('best_pattern', 'None')
    f_sim = fractal.get('similarity', 0.0)
    f_emoji = "🧬" if f_sim > 0.7 else "🧩"

    entropy = decision.reasoning.get('market_entropy', {})
    e_pred = entropy.get('predictability', 0.5)
    e_emoji = "🔮" if e_pred > 0.7 else "🌀"

    quantum = decision.reasoning.get('quantum_oscillator', {})
    q_state = quantum.get('current_energy_state', 0)
    q_prob = quantum.get('tunneling_probability', 0.0)
    q_emoji = "⚛️" if q_prob > 0.8 else "🔋"

    bot.logger.info(f"FRACTAL PATTERN: {f_emoji} {f_pattern} ({f_sim:.2%}) | ENTROPY PRED: {e_emoji} {e_pred:.4f}")
    bot.logger.info(f"QUANTUM STATE: {q_emoji} Level {q_state} | TUNNELING PROB: {q_prob:.2%}")

    # 3. ML Consensus (Step 12 Feature Fan-Out)
    if rt_result and 'predictions' in rt_result:
        bot.logger.info("-"*60 + "\n🤖 ML FEATURE FAN-OUT CONSENSUS\n" + "-"*60)
        preds = rt_result['predictions']
        for model, val in preds.items():
            m_emoji = "📈" if val > 0.1 else "📉" if val < -0.1 else "↔️"
            bot.logger.info(f"   {model:20} : {m_emoji} {val:+.4f}")

        # Aggregate Consensus
        model_values = list(preds.values())
        consensus = sum(model_values) / len(model_values) if model_values else 0
        c_emoji = "🔥" if abs(consensus) > 0.5 else "✅"
        bot.logger.info(f"AGGREGATE CONSENSUS: {c_emoji} {consensus:+.4f}")

    # 4. Step 9 Risk Check
    risk_check = decision.reasoning.get('risk_check', {})
    bot.logger.info("-"*60 + f"\n🛡️ RISK GATEWAY STATUS: {'ALLOWED' if decision.action != 'HOLD' or decision.reasoning.get('weighted_signal', 0) < 0.1 else 'BLOCKED'}\n" + "-"*60)
    bot.logger.info(f"Daily PnL: ${risk_check.get('daily_pnl', 0):.2f} / Limit: ${risk_check.get('daily_limit', 0):.2f}")
    bot.logger.info(
        f"Drawdown: {bot._current_drawdown_pct:.1%} from HWM ${bot._high_watermark_usd:,.2f} | "
        f"Weekly PnL: ${bot._weekly_pnl:,.2f}"
    )

    # 5. Persistence & Attribution (Step 13)
    if bot.db_enabled:
        bot.logger.info("-" * 60 + "\n💾 PERSISTENCE & ANALYTICS\n" + "-" * 60)
        bot.logger.info(f"Database: {bot.db_manager.db_path} | STATUS: ACTIVE")
        bot.logger.info(f"Historical Decisions: {len(bot.decision_history)}")

    # 6. Global Breakout Intelligence
    if bot.breakout_candidates:
        bot.logger.info("-" * 60 + "\n🚀 GLOBAL BREAKOUT INTELLIGENCE\n" + "-" * 60)
        for r in bot.breakout_candidates[:5]:
            b_emoji = "🔥" if r['breakout_score'] >= 80 else "✨"
            bot.logger.info(f"   {r['symbol']:15} : {b_emoji} Score {r['breakout_score']} | Vol Surge: {r['volume_surge']:.2f}x | {r['exchange']}")

    bot.logger.info("="*60 + "\n")


def get_performance_summary(bot: "RenaissanceTradingBot") -> Dict[str, Any]:
    """Get performance summary of the Renaissance bot."""
    if not bot.decision_history:
        return {"message": "No trading decisions yet"}

    recent_decisions = bot.decision_history[-20:]  # Last 20 decisions

    summary = {
        'total_decisions': len(bot.decision_history),
        'recent_decisions': len(recent_decisions),
        'action_distribution': {},
        'average_confidence': 0.0,
        'average_position_size': 0.0,
        'signal_weight_distribution': bot.signal_weights
    }

    # Calculate distributions
    actions = [d.action for d in recent_decisions]
    confidences = [d.confidence for d in recent_decisions]
    position_sizes = [d.position_size for d in recent_decisions]

    for action in ['BUY', 'SELL', 'HOLD']:
        summary['action_distribution'][action] = actions.count(action)

    if confidences:
        summary['average_confidence'] = np.mean(confidences)
    if position_sizes:
        summary['average_position_size'] = np.mean(position_sizes)

    return summary


# ──────────────────────────────────────────────
#  Signal Conversion Helpers (module-level)
# ──────────────────────────────────────────────

def signed_strength(signal: Any) -> float:
    """Convert an IndicatorOutput into a signed strength value."""
    if not signal:
        return 0.0
    direction = str(signal.signal).upper()
    strength = abs(float(signal.strength))
    if direction == "SELL":
        return -strength
    if direction == "BUY":
        return strength
    return 0.0


def continuous_rsi_signal(signal: Any) -> float:
    """Convert RSI to continuous signal: oversold(+1) <-> neutral(0) <-> overbought(-1)."""
    if not signal:
        return 0.0
    rsi_value = float(signal.value) if signal.value is not None else 50.0
    if np.isnan(rsi_value) or np.isinf(rsi_value):
        return 0.0
    return float(np.clip(-(rsi_value - 50.0) / 50.0, -1.0, 1.0))


def continuous_macd_signal(signal: Any) -> float:
    """Convert MACD histogram to continuous signal using metadata."""
    if not signal or not signal.metadata:
        return 0.0
    hist = signal.metadata.get('histogram', 0.0)
    if hist is None or (hasattr(hist, '__float__') and (np.isnan(float(hist)) or np.isinf(float(hist)))):
        return 0.0
    hist = float(hist)
    signal_line = abs(float(signal.metadata.get('signal_line', 1.0) or 1.0))
    if signal_line > 0:
        normalized = hist / signal_line
    else:
        normalized = hist
    return float(np.clip(normalized, -1.0, 1.0))


def continuous_bollinger_signal(signal: Any) -> float:
    """Convert Bollinger position to continuous signal: lower_band(+1) <-> mid(0) <-> upper_band(-1)."""
    if not signal:
        return 0.0
    position = float(signal.value) if signal.value is not None else 0.5
    if np.isnan(position) or np.isinf(position):
        return 0.0
    return float(np.clip(-(position - 0.5) * 2.0, -1.0, 1.0))


def continuous_obv_signal(signal: Any) -> float:
    """Convert OBV momentum to continuous signal using metadata instead of binary BUY/SELL."""
    if not signal or not signal.metadata:
        return signed_strength(signal)
    obv_momentum = signal.metadata.get('obv_momentum', 0.0)
    obv_change = signal.metadata.get('obv_change', 0.0)
    divergence = signal.metadata.get('divergence', 0)
    if obv_momentum is None:
        obv_momentum = 0.0
    if obv_change is None:
        obv_change = 0.0
    try:
        obv_momentum = float(obv_momentum)
        obv_change = float(obv_change)
        divergence = float(divergence)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(obv_momentum) or np.isinf(obv_momentum):
        return 0.0
    raw = obv_momentum * 3.0 + obv_change * 2.0 + divergence * 0.3
    return float(np.clip(raw, -1.0, 1.0))


def convert_ws_orderbook_to_snapshot(ws_ob: dict, last_price: float = 0.0) -> Any:
    """Convert WebSocket order_book dict to OrderBookSnapshot."""
    from analysis.microstructure_engine import OrderBookSnapshot, OrderBookLevel
    bids_dict = ws_ob.get('bids', {})
    asks_dict = ws_ob.get('asks', {})
    bid_levels = [OrderBookLevel(price=p, size=s) for p, s in sorted(bids_dict.items(), reverse=True)[:20]]
    ask_levels = [OrderBookLevel(price=p, size=s) for p, s in sorted(asks_dict.items())[:20]]
    return OrderBookSnapshot(
        timestamp=datetime.now(),
        bids=bid_levels,
        asks=ask_levels,
        last_price=last_price,
        last_size=0.0,
    )
