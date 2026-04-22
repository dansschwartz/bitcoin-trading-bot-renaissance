"""
Intelligence Layer D3: Insurance Premium Scanner
==================================================
Identifies systematic moments when other market participants de-risk,
creating predictable dislocations that can be harvested as
"insurance premiums".

Scans:
    1. Funding Settlement Premium   - 30 min window around 00:00/08:00/16:00 UTC
    2. Weekend Premium              - Friday-Sunday liquidity premium
    3. Scheduled Event Premium      - pre/post event dislocations (CPI, FOMC, etc.)

All scans use historical bar data from the SQLite database to compute
statistical significance of recurring patterns.

Config key: ``insurance_scanner``
"""

from __future__ import annotations

import logging
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Graceful scipy import
try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_stats = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class InsurancePremiumScanner:
    """
    Scans for systematic insurance-premium dislocations in crypto markets.

    These premiums arise because many participants de-risk around
    predictable events (funding settlement, weekends, macro releases).
    A disciplined counterparty can collect the spread.

    Config key: ``insurance_scanner``
    """

    # Standard crypto funding settlement hours (UTC)
    FUNDING_SETTLEMENT_HOURS: List[int] = [0, 8, 16]

    # Window (in minutes) before/after settlement to analyse
    SETTLEMENT_WINDOW_MINUTES: int = 30

    def __init__(self, config: Dict[str, Any], db_path: str):
        self.logger = logging.getLogger(f"{__name__}.InsurancePremiumScanner")

        self.enabled: bool = config.get("enabled", True)
        self.lookback_days: int = config.get("lookback_days", 90)
        self.min_observations: int = config.get("min_observations", 20)
        self.significance_level: float = config.get("significance_level", 0.05)
        self.min_dislocation_bps: float = config.get("min_dislocation_bps", 3.0)
        self.weekend_start_hour: int = config.get("weekend_start_hour", 20)  # Friday 20:00 UTC
        self.weekend_end_hour: int = config.get("weekend_end_hour", 22)  # Sunday 22:00 UTC

        self.db_path: str = db_path

        self.logger.info(
            "InsurancePremiumScanner initialized (lookback=%d days, "
            "min_obs=%d, sig_level=%.3f, min_bps=%.1f)",
            self.lookback_days, self.min_observations,
            self.significance_level, self.min_dislocation_bps,
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _fetch_market_data(
        self, pair: str, lookback_days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Fetch price/volume/spread rows for *pair* over the lookback period."""
        days = lookback_days or self.lookback_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT price, volume, bid, ask, spread, timestamp, product_id
                    FROM market_data
                    WHERE product_id = ?
                      AND timestamp >= ?
                    ORDER BY timestamp ASC
                    """,
                    (pair, cutoff),
                )
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
        except Exception as exc:
            self.logger.error("Failed to fetch market data for %s: %s", pair, exc)
            return []

    @staticmethod
    def _parse_timestamp(ts_str: str) -> Optional[datetime]:
        """Parse an ISO-format timestamp string into a timezone-aware datetime."""
        try:
            dt = datetime.fromisoformat(ts_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Scan 1: Funding Settlement Premium
    # ------------------------------------------------------------------

    def scan_funding_settlement_premium(self, pair: str) -> Dict[str, Any]:
        """
        Analyse price behaviour in the 30-minute windows before and after
        the three daily funding settlement times (00:00, 08:00, 16:00 UTC).

        Traders who need to close positions before settlement often create
        a predictable dislocation that mean-reverts shortly after.

        Returns
        -------
        dict with keys:
            premium_detected  : bool
            direction         : str ("long" / "short" / "none")
            avg_dislocation_bps : float
            p_value           : float
            recommended_action: str
            settlement_stats  : dict per-settlement-hour detail
        """
        result: Dict[str, Any] = {
            "premium_detected": False,
            "direction": "none",
            "avg_dislocation_bps": 0.0,
            "p_value": 1.0,
            "recommended_action": "no_action",
            "settlement_stats": {},
        }

        rows = self._fetch_market_data(pair)
        if len(rows) < self.min_observations * 3:
            self.logger.debug(
                "Funding scan: insufficient data (%d rows) for %s.", len(rows), pair
            )
            return result

        # Build price array with timestamps
        prices: List[float] = []
        timestamps: List[datetime] = []
        for r in rows:
            dt = self._parse_timestamp(r.get("timestamp", ""))
            if dt is not None:
                prices.append(float(r["price"]))
                timestamps.append(dt)

        if len(prices) < self.min_observations * 3:
            return result

        prices_arr = np.array(prices, dtype=np.float64)
        # Compute per-bar returns in basis points
        returns_bps = np.diff(np.log(np.maximum(prices_arr, 1e-9))) * 10_000

        # For each settlement hour, collect returns in the pre-window and
        # post-window, then compute the average "reversal" (post - pre).
        all_dislocations: List[float] = []
        settlement_detail: Dict[str, Any] = {}

        for settlement_hour in self.FUNDING_SETTLEMENT_HOURS:
            pre_returns: List[float] = []
            post_returns: List[float] = []

            for i in range(1, len(timestamps)):
                dt = timestamps[i]
                minutes_since_midnight = dt.hour * 60 + dt.minute

                # Settlement minute
                settlement_minute = settlement_hour * 60

                # Pre-window: [settlement - 30, settlement)
                pre_start = settlement_minute - self.SETTLEMENT_WINDOW_MINUTES
                pre_end = settlement_minute

                # Post-window: (settlement, settlement + 30]
                post_start = settlement_minute
                post_end = settlement_minute + self.SETTLEMENT_WINDOW_MINUTES

                if pre_start <= minutes_since_midnight < pre_end:
                    pre_returns.append(float(returns_bps[i - 1]))
                elif post_start < minutes_since_midnight <= post_end:
                    post_returns.append(float(returns_bps[i - 1]))

            n_pre = len(pre_returns)
            n_post = len(post_returns)

            if n_pre >= self.min_observations and n_post >= self.min_observations:
                pre_arr = np.array(pre_returns, dtype=np.float64)
                post_arr = np.array(post_returns, dtype=np.float64)

                avg_pre = float(np.mean(pre_arr))
                avg_post = float(np.mean(post_arr))
                dislocation = avg_post - avg_pre  # reversal size

                all_dislocations.append(dislocation)

                settlement_detail[f"{settlement_hour:02d}:00"] = {
                    "avg_pre_return_bps": round(avg_pre, 3),
                    "avg_post_return_bps": round(avg_post, 3),
                    "dislocation_bps": round(dislocation, 3),
                    "n_pre": n_pre,
                    "n_post": n_post,
                }
            else:
                settlement_detail[f"{settlement_hour:02d}:00"] = {
                    "avg_pre_return_bps": 0.0,
                    "avg_post_return_bps": 0.0,
                    "dislocation_bps": 0.0,
                    "n_pre": n_pre,
                    "n_post": n_post,
                }

        result["settlement_stats"] = settlement_detail

        if len(all_dislocations) == 0:
            return result

        avg_dislocation = float(np.mean(all_dislocations))
        result["avg_dislocation_bps"] = round(avg_dislocation, 3)

        # Statistical test: is the collection of dislocations significantly != 0?
        p_value = 1.0
        if SCIPY_AVAILABLE and len(all_dislocations) >= 3:
            try:
                _, p_value = scipy_stats.ttest_1samp(all_dislocations, 0.0)
                p_value = float(p_value)
            except Exception:
                p_value = 1.0
        result["p_value"] = round(p_value, 6)

        # Determine if premium is significant
        if abs(avg_dislocation) >= self.min_dislocation_bps and p_value < self.significance_level:
            result["premium_detected"] = True
            if avg_dislocation > 0:
                result["direction"] = "long"
                result["recommended_action"] = (
                    "buy_before_settlement_sell_after"
                )
            else:
                result["direction"] = "short"
                result["recommended_action"] = (
                    "sell_before_settlement_buy_after"
                )

        self.logger.info(
            "Funding settlement scan for %s: premium=%s dislocation=%.2f bps p=%.4f",
            pair, result["premium_detected"], avg_dislocation, p_value,
        )
        return result

    # ------------------------------------------------------------------
    # Scan 2: Weekend Premium
    # ------------------------------------------------------------------

    def scan_weekend_premium(self, pair: str) -> Dict[str, Any]:
        """
        Analyse returns during the weekend (Friday 20:00 UTC to Sunday 22:00 UTC)
        vs. weekday returns.

        Reduced weekend liquidity often causes a measurable spread-widening
        premium that reverts on Monday.

        Returns
        -------
        dict with keys:
            premium_detected : bool
            avg_weekend_return_bps : float
            avg_weekday_return_bps : float
            weekend_volatility     : float
            weekday_volatility     : float
            spread_ratio           : float (weekend / weekday volatility)
            p_value                : float
            recommended_action     : str
        """
        result: Dict[str, Any] = {
            "premium_detected": False,
            "avg_weekend_return_bps": 0.0,
            "avg_weekday_return_bps": 0.0,
            "weekend_volatility": 0.0,
            "weekday_volatility": 0.0,
            "spread_ratio": 1.0,
            "p_value": 1.0,
            "recommended_action": "no_action",
        }

        rows = self._fetch_market_data(pair)
        if len(rows) < self.min_observations * 5:
            self.logger.debug(
                "Weekend scan: insufficient data (%d rows) for %s.", len(rows), pair
            )
            return result

        # Segregate returns into weekend vs weekday
        weekend_returns: List[float] = []
        weekday_returns: List[float] = []
        weekend_spreads: List[float] = []
        weekday_spreads: List[float] = []

        for i in range(1, len(rows)):
            dt = self._parse_timestamp(rows[i].get("timestamp", ""))
            if dt is None:
                continue

            prev_price = float(rows[i - 1]["price"])
            cur_price = float(rows[i]["price"])
            if prev_price <= 0 or cur_price <= 0:
                continue

            ret_bps = math.log(cur_price / prev_price) * 10_000
            spread = float(rows[i].get("spread", 0.0))

            is_weekend = self._is_weekend(dt)
            if is_weekend:
                weekend_returns.append(ret_bps)
                weekend_spreads.append(spread)
            else:
                weekday_returns.append(ret_bps)
                weekday_spreads.append(spread)

        if len(weekend_returns) < self.min_observations or len(weekday_returns) < self.min_observations:
            self.logger.debug("Weekend scan: not enough weekend/weekday observations.")
            return result

        we_arr = np.array(weekend_returns, dtype=np.float64)
        wd_arr = np.array(weekday_returns, dtype=np.float64)

        result["avg_weekend_return_bps"] = round(float(np.mean(we_arr)), 3)
        result["avg_weekday_return_bps"] = round(float(np.mean(wd_arr)), 3)

        we_vol = float(np.std(we_arr))
        wd_vol = float(np.std(wd_arr))
        result["weekend_volatility"] = round(we_vol, 4)
        result["weekday_volatility"] = round(wd_vol, 4)
        result["spread_ratio"] = round(we_vol / max(wd_vol, 1e-9), 4)

        # Welch's t-test: are weekend and weekday returns different?
        p_value = 1.0
        if SCIPY_AVAILABLE:
            try:
                _, p_value = scipy_stats.ttest_ind(we_arr, wd_arr, equal_var=False)
                p_value = float(p_value)
            except Exception:
                p_value = 1.0
        result["p_value"] = round(p_value, 6)

        # Weekend spread analysis
        if weekend_spreads and weekday_spreads:
            avg_we_spread = float(np.mean(weekend_spreads))
            avg_wd_spread = float(np.mean(weekday_spreads))
            spread_premium_pct = (
                (avg_we_spread - avg_wd_spread) / max(avg_wd_spread, 1e-9) * 100
            )
        else:
            spread_premium_pct = 0.0

        # Determine if premium is significant
        return_diff = abs(result["avg_weekend_return_bps"] - result["avg_weekday_return_bps"])
        if return_diff >= self.min_dislocation_bps and p_value < self.significance_level:
            result["premium_detected"] = True
            if result["avg_weekend_return_bps"] < result["avg_weekday_return_bps"]:
                result["recommended_action"] = (
                    "buy_friday_close_sell_monday_open"
                )
            else:
                result["recommended_action"] = (
                    "sell_friday_close_buy_monday_open"
                )
        elif result["spread_ratio"] > 1.5:
            # Even if returns are not different, wider spreads mean
            # providing liquidity on weekends may be profitable
            result["premium_detected"] = True
            result["recommended_action"] = (
                "provide_weekend_liquidity"
            )

        self.logger.info(
            "Weekend premium scan for %s: premium=%s we_ret=%.2f wd_ret=%.2f spread_ratio=%.2f p=%.4f",
            pair, result["premium_detected"],
            result["avg_weekend_return_bps"], result["avg_weekday_return_bps"],
            result["spread_ratio"], p_value,
        )
        return result

    # ------------------------------------------------------------------
    # Scan 3: Scheduled Event Premium
    # ------------------------------------------------------------------

    def scan_scheduled_event_premium(
        self, pair: str, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyse price behaviour around scheduled macro events.

        Parameters
        ----------
        pair : str
            Trading pair.
        events : list of dict
            Each dict must contain at minimum:
            ``{"name": "CPI", "datetime": "2024-07-11T12:30:00+00:00"}``.
            Optional keys: ``pre_window_hours`` (default 4),
            ``post_window_hours`` (default 4).

        Returns
        -------
        dict with keys:
            premium_detected      : bool
            events_analysed       : int
            avg_pre_volatility_bps : float
            avg_post_volatility_bps: float
            avg_dislocation_bps   : float
            p_value               : float
            recommended_action    : str
            event_details         : list[dict]
        """
        result: Dict[str, Any] = {
            "premium_detected": False,
            "events_analysed": 0,
            "avg_pre_volatility_bps": 0.0,
            "avg_post_volatility_bps": 0.0,
            "avg_dislocation_bps": 0.0,
            "p_value": 1.0,
            "recommended_action": "no_action",
            "event_details": [],
        }

        if not events:
            self.logger.debug("Scheduled event scan: no events provided.")
            return result

        rows = self._fetch_market_data(pair)
        if len(rows) < self.min_observations:
            self.logger.debug(
                "Scheduled event scan: insufficient data (%d rows) for %s.", len(rows), pair
            )
            return result

        # Build price time series
        ts_prices: List[Tuple[datetime, float]] = []
        for r in rows:
            dt = self._parse_timestamp(r.get("timestamp", ""))
            if dt is not None:
                ts_prices.append((dt, float(r["price"])))

        if len(ts_prices) < self.min_observations:
            return result

        # Analyse each event
        all_dislocations: List[float] = []
        event_details: List[Dict[str, Any]] = []

        for evt in events:
            event_dt = self._parse_timestamp(evt.get("datetime", ""))
            if event_dt is None:
                continue

            event_name = evt.get("name", "unknown_event")
            pre_hours = evt.get("pre_window_hours", 4)
            post_hours = evt.get("post_window_hours", 4)

            pre_start = event_dt - timedelta(hours=pre_hours)
            post_end = event_dt + timedelta(hours=post_hours)

            pre_prices: List[float] = []
            post_prices: List[float] = []

            for ts, price in ts_prices:
                if pre_start <= ts < event_dt:
                    pre_prices.append(price)
                elif event_dt <= ts <= post_end:
                    post_prices.append(price)

            if len(pre_prices) < 3 or len(post_prices) < 3:
                continue

            pre_arr = np.array(pre_prices, dtype=np.float64)
            post_arr = np.array(post_prices, dtype=np.float64)

            # Compute log returns for volatility
            pre_rets = np.diff(np.log(np.maximum(pre_arr, 1e-9))) * 10_000
            post_rets = np.diff(np.log(np.maximum(post_arr, 1e-9))) * 10_000

            pre_vol = float(np.std(pre_rets)) if len(pre_rets) > 1 else 0.0
            post_vol = float(np.std(post_rets)) if len(post_rets) > 1 else 0.0

            # Dislocation: price move from end-of-pre to start-of-post
            pre_close = pre_arr[-1]
            post_open = post_arr[0]
            dislocation_bps = math.log(post_open / max(pre_close, 1e-9)) * 10_000

            # Mean reversion of dislocation in post window
            post_last = post_arr[-1]
            reversion_bps = math.log(post_last / max(post_open, 1e-9)) * 10_000

            all_dislocations.append(abs(dislocation_bps))

            event_details.append({
                "name": event_name,
                "datetime": event_dt.isoformat(),
                "pre_volatility_bps": round(pre_vol, 3),
                "post_volatility_bps": round(post_vol, 3),
                "dislocation_bps": round(dislocation_bps, 3),
                "reversion_bps": round(reversion_bps, 3),
                "reversion_pct": round(
                    abs(reversion_bps / max(abs(dislocation_bps), 1e-9)) * 100, 1
                ),
            })

        result["events_analysed"] = len(event_details)
        result["event_details"] = event_details

        if len(event_details) == 0:
            return result

        # Aggregate statistics
        pre_vols = [e["pre_volatility_bps"] for e in event_details]
        post_vols = [e["post_volatility_bps"] for e in event_details]
        result["avg_pre_volatility_bps"] = round(float(np.mean(pre_vols)), 3)
        result["avg_post_volatility_bps"] = round(float(np.mean(post_vols)), 3)
        result["avg_dislocation_bps"] = round(float(np.mean(all_dislocations)), 3)

        # Test: are dislocations significantly positive?
        p_value = 1.0
        if SCIPY_AVAILABLE and len(all_dislocations) >= 3:
            try:
                _, p_value = scipy_stats.ttest_1samp(all_dislocations, 0.0)
                p_value = float(p_value)
            except Exception:
                p_value = 1.0
        result["p_value"] = round(p_value, 6)

        # Check for consistent reversion
        reversion_pcts = [e.get("reversion_pct", 0.0) for e in event_details]
        avg_reversion = float(np.mean(reversion_pcts)) if reversion_pcts else 0.0

        if result["avg_dislocation_bps"] >= self.min_dislocation_bps and p_value < self.significance_level:
            result["premium_detected"] = True
            if avg_reversion > 50.0:
                result["recommended_action"] = (
                    "fade_event_dislocation"
                )
            else:
                result["recommended_action"] = (
                    "reduce_pre_event_then_enter_post_event"
                )

        self.logger.info(
            "Scheduled event scan for %s: premium=%s events=%d "
            "avg_dislocation=%.2f bps avg_reversion=%.1f%% p=%.4f",
            pair, result["premium_detected"], len(event_details),
            result["avg_dislocation_bps"], avg_reversion, p_value,
        )
        return result

    # ------------------------------------------------------------------
    # Combined scan
    # ------------------------------------------------------------------

    def get_all_premiums(
        self, pair: str, events: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Run all premium scans for *pair* and return a combined report.

        Parameters
        ----------
        pair : str
            Trading pair, e.g. ``"BTC-USD"``.
        events : list of dict, optional
            Scheduled events for scan 3. If omitted, the event scan
            returns a neutral result.

        Returns
        -------
        dict with keys:
            pair                     : str
            timestamp                : str (ISO)
            funding_settlement       : dict (scan 1 result)
            weekend                  : dict (scan 2 result)
            scheduled_events         : dict (scan 3 result)
            any_premium_detected     : bool
            total_premiums_found     : int
            combined_recommendation  : str
        """
        funding = self.scan_funding_settlement_premium(pair)
        weekend = self.scan_weekend_premium(pair)
        scheduled = self.scan_scheduled_event_premium(pair, events or [])

        premiums_found = sum([
            int(funding.get("premium_detected", False)),
            int(weekend.get("premium_detected", False)),
            int(scheduled.get("premium_detected", False)),
        ])

        # Build combined recommendation
        if premiums_found == 0:
            combined = "no_actionable_premiums"
        elif premiums_found == 1:
            # Use the single detected premium's recommendation
            for scan_result in [funding, weekend, scheduled]:
                if scan_result.get("premium_detected"):
                    combined = scan_result.get("recommended_action", "monitor")
                    break
            else:
                combined = "monitor"
        else:
            combined = "multiple_premiums_detected_review_allocation"

        result = {
            "pair": pair,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "funding_settlement": funding,
            "weekend": weekend,
            "scheduled_events": scheduled,
            "any_premium_detected": premiums_found > 0,
            "total_premiums_found": premiums_found,
            "combined_recommendation": combined,
        }

        self.logger.info(
            "All premiums scan for %s: %d premiums detected -> %s",
            pair, premiums_found, combined,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_weekend(self, dt: datetime) -> bool:
        """
        Return True if *dt* falls within the weekend liquidity window.

        Friday 20:00 UTC through Sunday 22:00 UTC.
        """
        weekday = dt.weekday()  # 0=Mon, 4=Fri, 6=Sun
        hour = dt.hour

        if weekday == 4 and hour >= self.weekend_start_hour:
            return True  # Friday after 20:00
        if weekday == 5:
            return True  # Saturday
        if weekday == 6 and hour < self.weekend_end_hour:
            return True  # Sunday before 22:00
        return False

    def __repr__(self) -> str:
        return (
            f"<InsurancePremiumScanner lookback={self.lookback_days}d "
            f"min_obs={self.min_observations} enabled={self.enabled}>"
        )
