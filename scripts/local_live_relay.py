#!/usr/bin/env python3
"""
local_live_relay.py — Relay Polymarket live bets from VPS to local CLOB execution.

The VPS is geo-blocked by Polymarket's CLOB. This script runs locally on Mac
where Polymarket access is unrestricted. It:
  1. Polls the VPS dashboard API for pending_relay bets
  2. Places them on the Polymarket CLOB using local wallet credentials
  3. Reports results back to VPS via the confirm endpoint

Usage:
    python scripts/local_live_relay.py                    # poll once
    python scripts/local_live_relay.py --loop              # poll every 15s
    python scripts/local_live_relay.py --loop --interval 5 # poll every 5s
    python scripts/local_live_relay.py --dry-run           # show pending bets without executing

Requires:
    - py-clob-client installed locally: pip install py-clob-client
    - config/polymarket_secrets.json with private_key
    - VPS dashboard running at VPS_URL:9090
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─── Configuration ───

VPS_URL = "http://178.128.216.112:9090"
PENDING_ENDPOINT = f"{VPS_URL}/api/polymarket/live/pending"
CONFIRM_ENDPOINT = f"{VPS_URL}/api/polymarket/live/confirm"

POLL_INTERVAL = 15  # seconds between polls
MAX_BET_USD = 2.00  # Safety cap — matches VPS setting

# Secrets file — same as VPS uses
SECRETS_FILE = Path(__file__).resolve().parent.parent / "config" / "polymarket_secrets.json"

# ─── Logging ───

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [RELAY] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("relay")


def load_secrets() -> dict:
    """Load wallet private key from local secrets file."""
    if not SECRETS_FILE.exists():
        logger.error(f"Secrets file not found: {SECRETS_FILE}")
        logger.error("Create it with: {\"private_key\": \"0x...\", \"wallet_address\": \"0x...\"}")
        sys.exit(1)
    with open(SECRETS_FILE) as f:
        data = json.load(f)
    if not data.get("private_key"):
        logger.error("private_key is empty in secrets file")
        sys.exit(1)
    return data


def init_clob_client(secrets: dict):
    """Initialize Polymarket CLOB client with local credentials."""
    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        logger.error("py-clob-client not installed. Run: pip install py-clob-client")
        sys.exit(1)

    client = ClobClient(
        host="https://clob.polymarket.com",
        chain_id=137,  # Polygon
        key=secrets["private_key"],
    )

    # Derive API credentials
    api_creds = client.create_or_derive_api_creds()
    client.set_api_creds(api_creds)

    # Verify connection
    server_time = client.get_server_time()
    logger.info(f"CLOB client connected (server_time={server_time})")

    return client


def fetch_pending_bets() -> list:
    """Poll VPS dashboard for pending relay bets."""
    try:
        resp = requests.get(PENDING_ENDPOINT, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"VPS returned {resp.status_code}")
            return []
        data = resp.json()
        return data.get("pending", [])
    except requests.ConnectionError:
        logger.warning("VPS unreachable — is the dashboard running?")
        return []
    except Exception as e:
        logger.warning(f"Fetch error: {e}")
        return []


def confirm_bet(bet_id: int, order_id: str, fill_status: str, error: str = "") -> bool:
    """Report execution result back to VPS."""
    try:
        resp = requests.post(
            CONFIRM_ENDPOINT,
            json={
                "bet_id": bet_id,
                "order_id": order_id,
                "fill_status": fill_status,
                "error": error,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            return result.get("ok", False)
        logger.warning(f"Confirm returned {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"Confirm error: {e}")
        return False


def execute_bet(client, bet: dict) -> dict:
    """Place a single bet on the CLOB. Returns execution result."""
    from py_clob_client.clob_types import OrderArgs
    from py_clob_client.order_builder.constants import BUY

    token_id = bet.get("token_id")
    entry_price = bet.get("order_price", 0)
    bet_amount = min(bet.get("order_size_usd", 0), MAX_BET_USD)
    shares = bet_amount / entry_price if entry_price > 0 else 0
    asset = bet.get("asset", "?")
    direction = bet.get("direction", "?")
    slug = bet.get("slug", "")

    if not token_id:
        return {"fill_status": "error", "order_id": "", "error": "missing token_id"}

    if bet_amount <= 0 or shares <= 0:
        return {"fill_status": "error", "order_id": "", "error": "invalid amount/shares"}

    logger.info(
        f"PLACING: {asset} {direction} @ ${entry_price:.3f} "
        f"| ${bet_amount:.2f} ({shares:.2f} shares) | {slug[:40]}"
    )

    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=round(entry_price, 2),
            size=round(shares, 2),
            side=BUY,
        )

        result = client.create_and_post_order(order_args)

        if result:
            order_id = (
                result.get("orderID")
                or result.get("orderIds", [""])[0]
                or result.get("id", "")
                or "unknown"
            )
            success = result.get("success", True)
            if success and order_id and order_id != "unknown":
                logger.info(f"FILLED: {order_id}")
                return {"fill_status": "placed", "order_id": order_id, "error": ""}
            else:
                err = json.dumps(result)[:300]
                logger.warning(f"REJECTED: {err}")
                return {"fill_status": "rejected", "order_id": order_id, "error": err}
        else:
            return {"fill_status": "unfilled", "order_id": "", "error": "empty result"}

    except Exception as e:
        err = str(e)[:500]
        logger.error(f"FAILED: {err}")
        return {"fill_status": "error", "order_id": "", "error": err}


def run_once(client, dry_run: bool = False) -> int:
    """Poll for pending bets and execute them. Returns count of bets processed."""
    pending = fetch_pending_bets()
    if not pending:
        return 0

    logger.info(f"Found {len(pending)} pending relay bet(s)")

    processed = 0
    for bet in pending:
        bet_id = bet.get("id")
        asset = bet.get("asset", "?")
        direction = bet.get("direction", "?")
        amount = bet.get("order_size_usd", 0)
        slug = bet.get("slug", "")[:40]

        if dry_run:
            logger.info(
                f"[DRY RUN] Would place: #{bet_id} {asset} {direction} "
                f"${amount:.2f} | {slug}"
            )
            processed += 1
            continue

        # Check if market is still open using window_start + timeframe
        window_start = bet.get("window_start", 0)
        timeframe = bet.get("timeframe", "")
        if window_start and window_start > 0:
            # Parse timeframe: "5m" → 300s, "15m" → 900s
            tf_sec = 300  # default 5m
            if timeframe:
                try:
                    tf_sec = int(timeframe.replace("m", "")) * 60
                except (ValueError, AttributeError):
                    pass
            market_end = window_start + tf_sec
            now_epoch = time.time()
            # Skip if market already ended (with 30s grace for execution)
            if now_epoch > market_end + 30:
                age = now_epoch - market_end
                logger.warning(
                    f"SKIP #{bet_id}: market expired {age:.0f}s ago "
                    f"(window_start={window_start}, tf={timeframe})"
                )
                confirm_bet(bet_id, "", "expired",
                            f"market ended {age:.0f}s before relay")
                processed += 1
                continue

        result = execute_bet(client, bet)
        ok = confirm_bet(
            bet_id,
            result.get("order_id", ""),
            result.get("fill_status", "error"),
            result.get("error", ""),
        )
        if ok:
            logger.info(f"Confirmed #{bet_id} → {result['fill_status']}")
        else:
            logger.warning(f"Confirm FAILED for #{bet_id}")

        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser(description="Local Polymarket live bet relay")
    parser.add_argument("--loop", action="store_true", help="Poll continuously")
    parser.add_argument("--interval", type=int, default=POLL_INTERVAL,
                        help=f"Seconds between polls (default {POLL_INTERVAL})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show pending bets without executing")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Polymarket Local Live Relay")
    logger.info(f"VPS: {VPS_URL}")
    logger.info(f"Mode: {'loop' if args.loop else 'once'}")
    if args.dry_run:
        logger.info("*** DRY RUN — no real orders will be placed ***")
    logger.info("=" * 60)

    # Initialize CLOB client (skip in dry-run mode)
    client = None
    if not args.dry_run:
        secrets = load_secrets()
        client = init_clob_client(secrets)

    if args.loop:
        logger.info(f"Polling every {args.interval}s — Ctrl+C to stop")
        consecutive_empty = 0
        while True:
            try:
                count = run_once(client, dry_run=args.dry_run)
                if count > 0:
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
                    # Log every 20th empty poll (5 min at 15s interval)
                    if consecutive_empty % 20 == 0:
                        logger.info(f"No pending bets ({consecutive_empty} empty polls)")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                logger.info("Stopped by user")
                break
    else:
        count = run_once(client, dry_run=args.dry_run)
        if count == 0:
            logger.info("No pending bets")


if __name__ == "__main__":
    main()
