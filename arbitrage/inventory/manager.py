"""
Inventory & Rebalancing Manager — tracks balances across exchanges
and signals when rebalancing is needed.

Cross-exchange arb CONSUMES inventory. After many trades:
  - Buy exchange accumulates base asset, depletes quote
  - Sell exchange accumulates quote, depletes base

Rebalancing is expensive (withdrawal fees + transfer time).
We rebalance INFREQUENTLY and in LARGE batches.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

logger = logging.getLogger("arb.inventory")


@dataclass
class InventorySnapshot:
    timestamp: datetime
    mexc_balances: Dict[str, Decimal]    # currency -> amount
    binance_balances: Dict[str, Decimal]
    imbalances: Dict[str, dict]          # currency -> {mexc_pct, binance_pct, needs_rebalance}


@dataclass
class RebalanceRequest:
    currency: str
    from_exchange: str
    to_exchange: str
    amount: Decimal
    network: str
    estimated_fee: Decimal
    estimated_time_minutes: int
    reason: str


class InventoryManager:

    TARGET_ALLOCATION = Decimal('0.50')     # 50/50 target
    REBALANCE_THRESHOLD = Decimal('0.70')   # Trigger at 70% on one side
    MIN_REBALANCE_USD = Decimal('500')
    CHECK_INTERVAL_MINUTES = 15

    PREFERRED_NETWORKS = {
        "USDT": ("TRC20", Decimal('1'), 3),      # network, fee_usd, time_minutes
        "USDC": ("SOL", Decimal('0.01'), 1),
        "BTC": ("BTC", Decimal('5'), 30),
        "ETH": ("ARB", Decimal('0.10'), 3),
        "SOL": ("SOL", Decimal('0.01'), 1),
        "BNB": ("BSC", Decimal('0.05'), 2),
        "XRP": ("XRP", Decimal('0.10'), 1),
        "DOGE": ("DOGE", Decimal('2'), 5),
        "ADA": ("ADA", Decimal('0.50'), 3),
        "AVAX": ("AVAX", Decimal('0.01'), 1),
        "LINK": ("ETH", Decimal('1'), 5),
        "DOT": ("DOT", Decimal('0.10'), 3),
    }

    MONITORED_CURRENCIES = [
        "USDT", "BTC", "ETH", "SOL", "BNB", "XRP",
        "DOGE", "ADA", "AVAX", "LINK", "DOT",
    ]

    def __init__(self, mexc_client, binance_client):
        self.mexc = mexc_client
        self.binance = binance_client
        self._snapshots: List[InventorySnapshot] = []
        self._pending_rebalances: List[RebalanceRequest] = []

    async def check_inventory(self) -> InventorySnapshot:
        """Take inventory snapshot and identify imbalances."""
        mexc_raw = await self.mexc.get_balances()
        binance_raw = await self.binance.get_balances()

        mexc_bals = {c: mexc_raw[c].total if c in mexc_raw else Decimal('0')
                     for c in self.MONITORED_CURRENCIES}
        binance_bals = {c: binance_raw[c].total if c in binance_raw else Decimal('0')
                        for c in self.MONITORED_CURRENCIES}

        imbalances = {}
        for currency in self.MONITORED_CURRENCIES:
            m = mexc_bals.get(currency, Decimal('0'))
            b = binance_bals.get(currency, Decimal('0'))
            total = m + b

            if total == 0:
                continue

            mexc_pct = m / total
            binance_pct = b / total

            needs_rebalance = (
                mexc_pct > self.REBALANCE_THRESHOLD
                or binance_pct > self.REBALANCE_THRESHOLD
            )

            imbalances[currency] = {
                'mexc_amount': float(m),
                'binance_amount': float(b),
                'total': float(total),
                'mexc_pct': float(mexc_pct),
                'binance_pct': float(binance_pct),
                'needs_rebalance': needs_rebalance,
            }

            if needs_rebalance:
                logger.warning(
                    f"IMBALANCE: {currency} — "
                    f"MEXC {float(mexc_pct)*100:.0f}% / Binance {float(binance_pct)*100:.0f}%"
                )

        snapshot = InventorySnapshot(
            timestamp=datetime.utcnow(),
            mexc_balances=mexc_bals,
            binance_balances=binance_bals,
            imbalances=imbalances,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def generate_rebalance_plan(self, snapshot: InventorySnapshot) -> List[RebalanceRequest]:
        """Generate rebalance requests for imbalanced currencies."""
        requests = []

        for currency, info in snapshot.imbalances.items():
            if not info['needs_rebalance']:
                continue

            total = Decimal(str(info['total']))
            mexc_pct = Decimal(str(info['mexc_pct']))
            binance_pct = Decimal(str(info['binance_pct']))

            if mexc_pct > self.REBALANCE_THRESHOLD:
                transfer_amount = (mexc_pct - self.TARGET_ALLOCATION) * total
                from_exchange = "mexc"
                to_exchange = "binance"
            else:
                transfer_amount = (binance_pct - self.TARGET_ALLOCATION) * total
                from_exchange = "binance"
                to_exchange = "mexc"

            network_info = self.PREFERRED_NETWORKS.get(currency)
            if not network_info:
                logger.warning(f"No preferred network for {currency}, skipping")
                continue

            network, est_fee, est_time = network_info

            request = RebalanceRequest(
                currency=currency,
                from_exchange=from_exchange,
                to_exchange=to_exchange,
                amount=transfer_amount,
                network=network,
                estimated_fee=est_fee,
                estimated_time_minutes=est_time,
                reason=f"Imbalance: {from_exchange} has {float(max(mexc_pct, binance_pct))*100:.0f}%",
            )
            requests.append(request)

            logger.info(
                f"REBALANCE PLAN: Transfer {float(transfer_amount):.4f} {currency} "
                f"from {from_exchange} to {to_exchange} via {network} "
                f"(est fee: ${float(est_fee)}, time: ~{est_time}min)"
            )

        return requests

    def get_summary(self) -> dict:
        if not self._snapshots:
            return {"status": "no_snapshots"}

        latest = self._snapshots[-1]
        imbalanced = [c for c, i in latest.imbalances.items() if i['needs_rebalance']]

        return {
            "snapshot_count": len(self._snapshots),
            "last_check": latest.timestamp.isoformat(),
            "currencies_monitored": len(latest.imbalances),
            "imbalanced_currencies": imbalanced,
            "balances": latest.imbalances,
        }
