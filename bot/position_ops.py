"""
bot/position_ops.py — Position management operations extracted from RenaissanceTradingBot.

Covers: opening, closing, P&L computation, execution, exposure, deduplication,
spray/straddle price fetching, and state recovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from renaissance_trading_bot import RenaissanceTradingBot
    from renaissance_types import TradingDecision

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  P&L Computation
# ──────────────────────────────────────────────

def compute_realized_pnl(entry_price: float, close_price: float,
                          size: float, side: str) -> float:
    """Compute realized PnL from entry/close prices and position side."""
    if entry_price <= 0 or close_price <= 0 or size <= 0:
        return 0.0
    side_upper = side.upper() if isinstance(side, str) else str(side).upper()
    if side_upper in ("LONG", "BUY"):
        return (close_price - entry_price) * size
    elif side_upper in ("SHORT", "SELL"):
        return (entry_price - close_price) * size
    return 0.0


# ──────────────────────────────────────────────
#  Close Price Resolution
# ──────────────────────────────────────────────

async def resolve_close_price(bot: "RenaissanceTradingBot", pos) -> float:
    """Resolve the best available market price for closing a position.

    Fallback chain:
      1. pos.current_price (if recently updated and > 0)
      2. bot._last_prices cache
      3. Live Binance ticker fetch
      4. pos.entry_price (last resort — better than 0)
    """
    from data_providers.binance_spot_provider import to_binance_symbol

    _cpx = getattr(pos, 'current_price', 0.0) or 0.0

    if _cpx <= 0:
        _pair_id = getattr(pos, 'product_id', '') or (
            pos.position_id.rsplit('_', 2)[0] if '_' in pos.position_id else pos.position_id
        )
        if hasattr(bot, '_last_prices'):
            _cpx = bot._last_prices.get(_pair_id, 0.0)

    if _cpx <= 0:
        try:
            _pair_id = getattr(pos, 'product_id', '') or (
                pos.position_id.rsplit('_', 2)[0] if '_' in pos.position_id else pos.position_id
            )
            bsym = to_binance_symbol(_pair_id)
            ticker = await bot.binance_spot.fetch_ticker(bsym)
            if ticker and ticker.get('price', 0) > 0:
                _cpx = float(ticker['price'])
                bot.logger.info(
                    f"CLOSE PRICE RESOLVED via Binance: {_pair_id} = ${_cpx:,.4f}"
                )
        except Exception as e:
            bot.logger.debug(f"Binance ticker fetch for close price failed: {e}")

    if _cpx <= 0:
        _cpx = getattr(pos, 'entry_price', 0.0) or 0.0
        if _cpx > 0:
            bot.logger.warning(
                f"CLOSE PRICE FALLBACK: {pos.position_id} using entry price ${_cpx:,.4f}"
            )

    return float(_cpx)


# ──────────────────────────────────────────────
#  Dynamic Position Sizing
# ──────────────────────────────────────────────

def calculate_dynamic_position_size(bot: "RenaissanceTradingBot", product_id: str,
                                     confidence: float, weighted_signal: float,
                                     current_price: float) -> float:
    """Calculate dynamic position size using Step 10 Portfolio Optimizer"""
    try:
        universe_data = {
            'returns': np.array([weighted_signal * 0.01]),
            'market_cap': np.array([1.0]),
            'assets': [product_id]
        }

        market_data = {
            'bid_ask_spread': np.array([0.0005]),
            'market_impact': np.array([0.0002])
        }

        opt_result = bot.portfolio_optimizer.optimize_portfolio(universe_data, market_data)

        if 'weights' in opt_result:
            optimized_weight = float(opt_result['weights'][0])
            final_size = optimized_weight * confidence
            return float(np.clip(final_size, 0.0, 0.3))

        return min(confidence * 0.5, 0.3)

    except Exception as e:
        bot.logger.error(f"Portfolio optimization sizing failed: {e}")
        return min(confidence * 0.5, 0.3)


# ──────────────────────────────────────────────
#  Account Balance
# ──────────────────────────────────────────────

def fetch_account_balance(bot: "RenaissanceTradingBot") -> float:
    """Fetch current USD account balance, hard-capped to prevent phantom inflation.

    Paper trading short-sell accounting inflates the cash balance because
    borrowed-share liabilities aren't tracked.  We cap at INITIAL_CAPITAL
    so position sizing stays anchored to real capital.
    """
    INITIAL_CAPITAL = 10_000.0
    MAX_BALANCE = INITIAL_CAPITAL * 1.5

    try:
        portfolio = bot.coinbase_client.get_portfolio_breakdown()
        if "error" not in portfolio:
            balance = portfolio.get("total_balance_usd", 0.0)
            if balance > 0:
                balance = min(balance, MAX_BALANCE)
                bot._cached_balance_usd = balance
                return balance
    except Exception as e:
        bot.logger.debug(f"Balance fetch failed: {e}")
    return bot._cached_balance_usd if bot._cached_balance_usd > 0 else INITIAL_CAPITAL


# ──────────────────────────────────────────────
#  Smart Order Execution
# ──────────────────────────────────────────────

async def execute_smart_order(bot: "RenaissanceTradingBot", decision: "TradingDecision",
                               market_data: Dict[str, Any]):
    """Execute order through position manager (real or paper) with slippage analysis.

    Routes through MEXC (0% maker) for Binance-sourced pairs, Coinbase for legacy.
    """
    try:
        product_id = market_data.get('product_id', 'BTC-USD')
        current_price = decision.reasoning.get('current_price', 0.0)

        # Determine execution venue
        is_mexc_execution = (
            bot._universe_built
            and product_id in bot._pair_binance_symbols
        )

        # For MEXC execution: use limit order at best bid/ask for 0% maker fee
        if is_mexc_execution:
            ticker = market_data.get('ticker', {})
            if decision.action == 'BUY':
                limit_price = float(ticker.get('bid', current_price))
                limit_price *= 1.0001
            else:
                limit_price = float(ticker.get('ask', current_price))
                limit_price *= 0.9999
            current_price = limit_price if limit_price > 0 else current_price
            order_type = 'LIMIT_MAKER'
            execution_exchange = 'mexc'
        else:
            order_type = 'MARKET'
            execution_exchange = 'coinbase'

        order_details = {
            'product_id': product_id,
            'side': decision.action,
            'size': decision.position_size,
            'price': current_price,
            'type': order_type,
            'exchange': execution_exchange,
        }

        # 1. Analyze Slippage Risk
        slippage_risk = bot.slippage_protection.analyze_slippage_risk(order_details, market_data)
        bot.logger.info(f"Slippage risk for {product_id}: {slippage_risk.get('risk_level', 'UNKNOWN')}")

        # Council #4: Apply spread-based slippage in paper mode
        fill_price = current_price
        _slippage_bps = 0.0
        if bot.paper_trading and current_price > 0:
            ticker = market_data.get('ticker', {})
            _bid = bot._force_float(ticker.get('bid', 0))
            _ask = bot._force_float(ticker.get('ask', 0))
            if _bid > 0 and _ask > 0:
                half_spread_bps = ((_ask - _bid) / ((_ask + _bid) / 2)) * 10000 / 2.0
            else:
                half_spread_bps = 0.5
            _adv_cfg = bot.config.get('adverse_selection_bps', {})
            _pair_key = product_id.replace('-', '').replace('/', '').upper()
            if _pair_key in _adv_cfg:
                adverse_bps = float(_adv_cfg[_pair_key])
            else:
                try:
                    _vol_24h = bot._force_float(ticker.get('volume_24h') or ticker.get('volume', 0))
                    _daily_vol_usd = _vol_24h * current_price if _vol_24h > 0 else 0
                    if _daily_vol_usd > 50_000_000:
                        adverse_bps = float(_adv_cfg.get('__default_large_cap__', 0.20))
                    else:
                        adverse_bps = float(_adv_cfg.get('__default_small_cap__', 0.80))
                except Exception:
                    adverse_bps = float(_adv_cfg.get('__default_small_cap__', 0.80))
            floor_bps = float(bot.config.get('paper_trading', {}).get('slippage_floor_bps', 0.5))
            _slippage_bps = max(half_spread_bps + adverse_bps, floor_bps)
            slippage_frac = _slippage_bps / 10000.0
            if decision.action == 'BUY':
                fill_price = current_price * (1 + slippage_frac)
            else:
                fill_price = current_price * (1 - slippage_frac)

        # 2. Map action to position side
        side = "LONG" if decision.action == "BUY" else "SHORT"

        # 3. Execute through position manager
        success, message, position = bot.position_manager.open_position(
            product_id=product_id,
            side=side,
            size=decision.position_size,
            entry_price=fill_price,
        )

        exec_result = {
            'status': 'EXECUTED' if success else 'REJECTED',
            'message': message,
            'position_id': position.position_id if position else None,
            'execution_price': fill_price,
            'signal_price': current_price,
            'slippage_bps': _slippage_bps,
            'slippage': slippage_risk.get('predicted_slippage', 0.0),
            'exchange': execution_exchange,
            'order_type': order_type,
        }

        # Record trade cycle for anti-churn cooldown
        if success:
            bot._last_trade_cycle[product_id] = getattr(bot, 'scan_cycle_count', 0)
            if _slippage_bps > 0:
                bot.logger.info(
                    f"PAPER FILL: {decision.action} {product_id} signal=${current_price:.2f} "
                    f"fill=${fill_price:.2f} slippage={_slippage_bps:.1f}bps"
                )
            if is_mexc_execution:
                bot.logger.info(
                    f"MEXC LIMIT ORDER: {decision.action} {decision.position_size:.8f} "
                    f"{product_id} @ ${fill_price:.2f} (maker, 0% fee)"
                )

        # Devil Tracker — record fill with spread-calibrated slippage
        if success and bot.devil_tracker:
            try:
                _dtid = getattr(bot, '_last_devil_trade_id', {}).get(product_id)
                if _dtid:
                    bot.devil_tracker.record_order_submission(_dtid, current_price)
                    _ticker = market_data.get('ticker', {})
                    _bid = bot._force_float(_ticker.get('bid', 0))
                    _ask = bot._force_float(_ticker.get('ask', 0))
                    if _bid > 0 and _ask > 0:
                        _calibrated_fill = _ask if decision.action == 'BUY' else _bid
                    else:
                        _calibrated_fill = current_price
                    _fee_bps = 0.0 if is_mexc_execution else 5.0
                    _fill_fee = _fee_bps / 10000.0 * decision.position_size * current_price
                    bot.devil_tracker.record_fill(
                        _dtid,
                        fill_price=_calibrated_fill,
                        fill_quantity=decision.position_size,
                        fill_fee=_fill_fee,
                    )
            except Exception as _dt_err:
                bot.logger.debug(f"Devil tracker fill record failed: {_dt_err}")

        # 4. Persist Trade
        if success and bot.db_enabled:
            trade_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'product_id': product_id,
                'side': decision.action,
                'size': decision.position_size,
                'price': current_price,
                'status': 'EXECUTED',
                'algo_used': f'POSITION_MANAGER_{execution_exchange.upper()}',
                'slippage': slippage_risk.get('predicted_slippage', 0.0) if not is_mexc_execution else 0.0,
                'execution_time': 0.0,
            }
            bot._track_task(bot.db_manager.store_trade(trade_data))
            if position:
                bot._track_task(bot.db_manager.save_position({
                    'position_id': position.position_id,
                    'product_id': product_id,
                    'side': side,
                    'size': decision.position_size,
                    'entry_price': current_price,
                    'stop_loss_price': position.stop_loss_price,
                    'take_profit_price': position.take_profit_price,
                    'opened_at': position.entry_time.isoformat(),
                    'status': 'OPEN',
                }))

        # 4.3 Feed trade to BarAggregator
        if success and bot.bar_aggregator:
            try:
                bot.bar_aggregator.on_trade(
                    pair=product_id,
                    exchange=execution_exchange,
                    price=current_price,
                    quantity=decision.position_size,
                    side=decision.action.lower(),
                    timestamp=time.time(),
                )
            except Exception as e:
                bot.logger.warning(f"BarAggregator trade feed failed for {product_id}: {e}")

        # 4.5 Send monitoring alert for executed trade
        if success and bot.monitoring_alert_manager:
            try:
                bot._track_task(bot.monitoring_alert_manager.send_trade_alert({
                    'product_id': product_id,
                    'side': decision.action,
                    'size': decision.position_size,
                    'price': current_price,
                    'confidence': decision.confidence,
                    'slippage': slippage_risk.get('predicted_slippage', 0.0),
                }))
            except Exception as e:
                bot.logger.warning(f"Monitoring trade alert failed for {product_id}: {e}")

        # 5. Check daily loss after trade
        if bot.position_manager.daily_pnl < -bot.daily_loss_limit:
            bot.trigger_kill_switch(
                f"Daily loss limit breached: ${abs(bot.position_manager.daily_pnl):.2f}"
            )

        if exec_result['status'] == 'REJECTED':
            n_pos = len(bot.position_manager.positions)
            exp = bot.position_manager._calculate_total_exposure()
            lim = bot.position_manager.risk_limits.max_total_exposure_usd
            bot.logger.info(
                f"Smart execution complete: REJECTED ({product_id}) | {message} "
                f"| positions={n_pos}, exposure=${exp:.0f}, limit=${lim:.0f}, "
                f"trade_usd=${decision.position_size * current_price:.0f}"
            )
        else:
            bot.logger.info(f"Smart execution complete: {exec_result['status']} | {message}")
        return exec_result

    except Exception as e:
        bot.logger.error(f"Smart execution failed: {e}")
        return {'status': 'FAILED', 'error': str(e)}


# ──────────────────────────────────────────────
#  Spray / Straddle Price Fetching
# ──────────────────────────────────────────────

async def get_spray_prices(bot: "RenaissanceTradingBot", pairs: List[str]) -> Dict[str, float]:
    """Fetch current prices for token spray exit checks.

    Called every 5s by the exit loop.  Fetches live Binance tickers
    in parallel, falling back to ``_last_prices`` for any that fail.
    """
    prices: Dict[str, float] = {}
    provider = getattr(bot, 'binance_spot', None)
    pair_map = getattr(bot, '_pair_binance_symbols', {})

    if provider:
        fetchable = [(p, pair_map[p]) for p in pairs if pair_map.get(p)]
        if fetchable:
            try:
                results = await asyncio.gather(
                    *(provider.fetch_ticker(bsym) for _, bsym in fetchable),
                    return_exceptions=True,
                )
                for (pair, _), ticker in zip(fetchable, results):
                    if isinstance(ticker, dict) and ticker.get('price', 0) > 0:
                        prices[pair] = float(ticker['price'])
            except Exception as e:
                bot.logger.warning(f"Batch price fetch from provider failed: {e}")

    last = getattr(bot, '_last_prices', {})
    for pair in pairs:
        if pair not in prices and last.get(pair, 0) > 0:
            prices[pair] = float(last[pair])

    return prices


async def get_straddle_price(bot: "RenaissanceTradingBot", pair: str = '') -> Dict[str, float]:
    """Fetch current price for straddle exit checks.

    When called with a specific pair, fetches ONLY that pair (fast path
    for per-engine exit loops). Without args, fetches all engine pairs.
    """
    pairs_needed = set()
    if pair:
        pairs_needed.add(pair)
    else:
        for eng in bot.straddle_engines.values():
            pairs_needed.add(eng.pair)

    result: Dict[str, float] = {}
    provider = getattr(bot, 'binance_spot', None)
    pair_map = getattr(bot, '_pair_binance_symbols', {})

    for p in pairs_needed:
        binance_sym = pair_map.get(p, p.replace('-USD', 'USDT'))
        if provider:
            try:
                ticker = await provider.fetch_ticker(binance_sym)
                if isinstance(ticker, dict) and ticker.get('price', 0) > 0:
                    result[p] = float(ticker['price'])
                    continue
            except Exception as e:
                bot.logger.warning(f"Straddle price fetch failed for {p}: {e}")
        last = getattr(bot, '_last_prices', {})
        if last.get(p, 0) > 0:
            result[p] = float(last[p])

    return result


# ──────────────────────────────────────────────
#  Position Deduplication
# ──────────────────────────────────────────────

async def deduplicate_positions_on_startup(bot: "RenaissanceTradingBot") -> None:
    """Close duplicate and opposing positions found after DB restore.

    Rules:
    1. If multiple same-side positions exist for a product, keep the newest, close the rest.
    2. If opposing positions exist for a product (LONG + SHORT), close both and go flat.
    """
    from risk_management.position_manager import PositionSide, PositionStatus

    # Group open positions by product_id
    by_product: Dict[str, List] = defaultdict(list)
    with bot.position_manager._lock:
        for pos in list(bot.position_manager.positions.values()):
            if pos.status == PositionStatus.OPEN:
                by_product[pos.product_id].append(pos)

    closed_count = 0
    for product_id, positions in by_product.items():
        longs = [p for p in positions if p.side == PositionSide.LONG]
        shorts = [p for p in positions if p.side == PositionSide.SHORT]

        # Rule 2: Opposing positions — close ALL (go flat)
        if longs and shorts:
            bot.logger.warning(
                f"STARTUP DEDUP: {product_id} has {len(longs)} LONG + {len(shorts)} SHORT — closing all (go flat)"
            )
            for pos in longs + shorts:
                ok, msg = bot.position_manager.close_position(
                    pos.position_id, reason="startup_dedup_opposing"
                )
                if ok:
                    _cpx = await resolve_close_price(bot, pos)
                    _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                    _rpnl = compute_realized_pnl(
                        pos.entry_price, _cpx, pos.size, _side
                    )
                    bot._track_task(
                        bot.db_manager.close_position_record(
                            pos.position_id,
                            close_price=float(_cpx),
                            realized_pnl=float(_rpnl),
                            exit_reason="startup_dedup_opposing",
                        )
                    )
                    closed_count += 1
            continue

        # Rule 1: Duplicate same-side — keep newest, close rest
        for group in [longs, shorts]:
            if len(group) > 1:
                group.sort(key=lambda p: p.entry_time, reverse=True)
                keep = group[0]
                dupes = group[1:]
                bot.logger.warning(
                    f"STARTUP DEDUP: {product_id} has {len(group)} {keep.side.value} positions — "
                    f"keeping {keep.position_id}, closing {len(dupes)} duplicates"
                )
                for pos in dupes:
                    ok, msg = bot.position_manager.close_position(
                        pos.position_id, reason="startup_dedup_duplicate"
                    )
                    if ok:
                        _cpx = await resolve_close_price(bot, pos)
                        _side = pos.side.value if hasattr(pos.side, 'value') else str(pos.side)
                        _rpnl = compute_realized_pnl(
                            pos.entry_price, _cpx, pos.size, _side
                        )
                        bot._track_task(
                            bot.db_manager.close_position_record(
                                pos.position_id,
                                close_price=float(_cpx),
                                realized_pnl=float(_rpnl),
                                exit_reason="startup_dedup_duplicate",
                            )
                        )
                        closed_count += 1

    if closed_count > 0:
        bot.logger.info(f"STARTUP DEDUP: closed {closed_count} duplicate/opposing positions")


# ──────────────────────────────────────────────
#  State Recovery
# ──────────────────────────────────────────────

async def restore_state(bot: "RenaissanceTradingBot") -> None:
    """Restore positions and daily PnL from the database after restart."""
    try:
        open_positions = await bot.db_manager.get_open_positions()
        restored = 0
        net_position = 0.0
        for row in open_positions:
            from risk_management.position_manager import Position, PositionSide, PositionStatus
            pos = Position(
                position_id=row['position_id'],
                product_id=row['product_id'],
                side=PositionSide(row['side']),
                size=row['size'],
                entry_price=row['entry_price'],
                current_price=row['entry_price'],
                stop_loss_price=row.get('stop_loss_price'),
                take_profit_price=row.get('take_profit_price'),
                status=PositionStatus.OPEN,
                entry_time=datetime.fromisoformat(row['opened_at']),
            )
            bot.position_manager.positions[pos.position_id] = pos
            sign = 1.0 if pos.side == PositionSide.LONG else -1.0
            net_position += sign * pos.size
            restored += 1

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        daily_pnl = await bot.db_manager.get_daily_pnl(today)
        bot.position_manager.daily_pnl = daily_pnl
        bot.daily_pnl = daily_pnl
        bot.current_position = net_position

        if restored > 0 or daily_pnl != 0:
            bot.logger.info(
                f"State restored: {restored} open positions, "
                f"net_position={net_position:.6f}, daily_pnl=${daily_pnl:.2f}"
            )

        if not bot.paper_trading:
            recon = bot.position_manager.reconcile_with_exchange()
            if recon.get("status") == "MISMATCH":
                asyncio.ensure_future(
                    bot.alert_manager.send_alert("CRITICAL", "Position Mismatch",
                        f"{len(recon['discrepancies'])} discrepancies found on startup")
                )
    except Exception as e:
        bot.logger.warning(f"State recovery skipped: {e}")
