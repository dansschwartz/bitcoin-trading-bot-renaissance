"""
Contract Address Verifier for Cross-Exchange Arbitrage Safety.

Prevents false positive arb signals caused by:
- Token migrations (old contract vs new contract)
- Token redenominations (1:100 swaps where ticker stays the same)
- Chain-specific tokens (native vs wrapped versions)
- Exchange-specific premium from suspended deposits/withdrawals

Three layers of verification (fastest to most thorough):
1. Price sanity check — >5% price difference = block (instant, no API needed)
2. Contract verification — compare on-chain addresses (cached, needs API keys)
3. Transfer status — deposits/withdrawals enabled (cached, needs API keys)

All three are independent. If API keys aren't available, layers 2+3 degrade
gracefully (permissive) and layer 1 still catches dangerous mismatches.
"""
import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Set

logger = logging.getLogger("arb.safety")

# Native chain tokens that have no contract address
NATIVE_TOKENS: Set[str] = {"BTC", "ETH", "SOL", "DOT", "ATOM", "XRP", "ADA", "ALGO", "NEAR"}


class ContractVerifier:
    """Verifies cross-exchange token identity before arb execution."""

    def __init__(
        self,
        mexc_client,
        binance_client,
        config: Optional[dict] = None,
        cache_ttl_hours: int = 24,
    ):
        self.mexc = mexc_client
        self.binance = binance_client
        self.cache_ttl = cache_ttl_hours * 3600
        self._config = config or {}

        # Token blocklist from config (manually maintained)
        safety_cfg = self._config.get("safety", {})
        self._blocklist: Set[str] = set(safety_cfg.get("blocked_tokens", []))
        self._price_divergence_pct: float = safety_cfg.get("max_price_divergence_pct", 5.0)

        # Cache: {exchange_name: {symbol: {network: {contract, deposit, withdraw}}}}
        self._currency_cache: Dict[str, Dict] = {}
        self._cache_timestamp: float = 0.0

        # Verification results cache: symbol -> bool
        self._verified: Dict[str, bool] = {}

        # Detected mismatches for dashboard
        self._mismatches: List[dict] = []
        self._price_blocks: List[dict] = []
        self._transfer_blocks: List[dict] = []

        # Stats
        self._stats = {
            "contract_checks": 0,
            "price_sanity_checks": 0,
            "price_sanity_blocks": 0,
            "contract_blocks": 0,
            "transfer_blocks": 0,
            "cache_refreshes": 0,
            "cache_refresh_failures": 0,
        }

    async def refresh_cache(self) -> None:
        """Fetch token network/contract info from both exchanges."""
        logger.info("CONTRACT VERIFIER: Refreshing token contract cache...")

        mexc_data = await self._fetch_exchange_currencies("mexc")
        binance_data = await self._fetch_exchange_currencies("binance")

        self._currency_cache = {"mexc": mexc_data, "binance": binance_data}
        self._cache_timestamp = time.time()
        self._stats["cache_refreshes"] += 1

        # Re-verify all previously checked symbols
        for symbol in list(self._verified.keys()):
            self._verified[symbol] = self._check_contract_match(symbol)

        logger.info(
            f"CONTRACT VERIFIER: Cached {len(mexc_data)} MEXC tokens, "
            f"{len(binance_data)} Binance tokens"
        )

    async def _fetch_exchange_currencies(self, exchange: str) -> Dict:
        """Fetch currency info from an exchange via ccxt.

        Returns: {symbol: {network: {"contract": "0x...", "deposit": bool, "withdraw": bool}}}
        """
        client = self.mexc if exchange == "mexc" else self.binance
        try:
            # Use the ccxt exchange object stored in the client
            ccxt_exchange = getattr(client, "_exchange", None)
            if ccxt_exchange is None:
                logger.warning(f"CONTRACT VERIFIER: No ccxt exchange on {exchange} client")
                return {}

            # fetch_currencies requires API keys on most exchanges
            # ccxt sync exchange — run in executor to avoid blocking
            if asyncio.iscoroutinefunction(getattr(ccxt_exchange, 'fetch_currencies', None)):
                currencies = await ccxt_exchange.fetch_currencies()
            else:
                loop = asyncio.get_event_loop()
                currencies = await loop.run_in_executor(None, ccxt_exchange.fetch_currencies)
            if not currencies:
                logger.info(f"CONTRACT VERIFIER: {exchange} returned 0 currencies (no API keys?)")
                return {}

            result = {}
            for symbol, info in currencies.items():
                networks = {}
                for net_id, net_info in (info.get("networks") or {}).items():
                    raw_info = net_info.get("info", {})
                    contract = (
                        raw_info.get("contract")
                        or raw_info.get("contractAddress")
                        or ""
                    )
                    networks[net_id] = {
                        "contract": contract.lower().strip() if contract else "",
                        "deposit": net_info.get("deposit", None),
                        "withdraw": net_info.get("withdraw", None),
                    }
                if networks:
                    result[symbol] = networks
            return result

        except Exception as e:
            logger.warning(f"CONTRACT VERIFIER: {exchange} fetch_currencies failed: {e}")
            self._stats["cache_refresh_failures"] += 1
            return {}

    def _check_contract_match(self, symbol: str) -> bool:
        """Check if a token has matching contract addresses on both exchanges.

        Returns True (allow) if:
        - At least one shared network has identical contract addresses
        - Token is a native chain token (BTC, ETH, etc.)
        - Token not found on one/both exchanges (permissive when we can't verify)

        Returns False (block) if:
        - All shared networks have different contract addresses
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol

        mexc_nets = self._currency_cache.get("mexc", {}).get(base, {})
        binance_nets = self._currency_cache.get("binance", {}).get(base, {})

        # If cache is empty (no API keys), be permissive
        if not self._currency_cache.get("mexc") and not self._currency_cache.get("binance"):
            return True

        if not mexc_nets or not binance_nets:
            logger.debug(
                f"CONTRACT VERIFIER: {base} not in cache for one/both exchanges, "
                f"allowing (permissive)"
            )
            return True

        shared_networks = set(mexc_nets.keys()) & set(binance_nets.keys())
        if not shared_networks:
            # No shared networks = can't transfer between exchanges
            logger.warning(
                f"CONTRACT VERIFIER: {base} has no shared networks. "
                f"MEXC: {list(mexc_nets.keys())}, Binance: {list(binance_nets.keys())}"
            )
            return False

        for network in shared_networks:
            mc = mexc_nets[network].get("contract", "")
            bc = binance_nets[network].get("contract", "")

            # Native tokens may have empty contracts
            if not mc and not bc:
                return True
            if mc and bc and mc == bc:
                return True

        # All shared networks have different contracts — mismatch
        self._mismatches.append({
            "symbol": base,
            "timestamp": time.time(),
            "shared_networks": list(shared_networks),
            "mexc_contracts": {n: mexc_nets[n].get("contract", "") for n in shared_networks},
            "binance_contracts": {n: binance_nets[n].get("contract", "") for n in shared_networks},
        })
        logger.warning(
            f"CONTRACT MISMATCH: {base} has different contracts on all shared networks!"
        )
        return False

    def check_transfers_enabled(self, symbol: str) -> dict:
        """Check deposit/withdrawal status for a token on both exchanges.

        Returns dict with status. 'can_arb' is True only if
        deposits AND withdrawals are enabled on at least one shared network.
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol
        mexc_nets = self._currency_cache.get("mexc", {}).get(base, {})
        binance_nets = self._currency_cache.get("binance", {}).get(base, {})

        result = {
            "symbol": base,
            "mexc_deposit": None,
            "mexc_withdraw": None,
            "binance_deposit": None,
            "binance_withdraw": None,
            "can_arb": True,  # Permissive default
        }

        if not mexc_nets or not binance_nets:
            return result  # Can't check, allow

        shared = set(mexc_nets.keys()) & set(binance_nets.keys())
        if not shared:
            result["can_arb"] = False
            return result

        # Check if ANY shared network has all four enabled
        for net in shared:
            m = mexc_nets[net]
            b = binance_nets[net]
            m_dep = m.get("deposit")
            m_wd = m.get("withdraw")
            b_dep = b.get("deposit")
            b_wd = b.get("withdraw")

            # If we have explicit data and all are enabled, it's good
            if m_dep and m_wd and b_dep and b_wd:
                result.update({
                    "mexc_deposit": True, "mexc_withdraw": True,
                    "binance_deposit": True, "binance_withdraw": True,
                    "can_arb": True,
                })
                return result

        # If we have data and none passed, check if any transfers are blocked
        any_blocked = False
        for net in shared:
            m = mexc_nets[net]
            b = binance_nets[net]
            for field in ["deposit", "withdraw"]:
                if m.get(field) is False or b.get(field) is False:
                    any_blocked = True

        if any_blocked:
            result["can_arb"] = False
            self._transfer_blocks.append({
                "symbol": base,
                "timestamp": time.time(),
                "reason": "transfers suspended on one or more shared networks",
            })
            logger.warning(f"TRANSFER CHECK: {base} — transfers suspended, blocking arb")

        return result

    def price_sanity_check(
        self, symbol: str, price_a: float, price_b: float
    ) -> bool:
        """Check if prices are within acceptable range.

        Returns True if prices are sane, False if suspicious.
        A >5% difference indicates different token versions, stale data,
        or exchange-specific premium from suspended transfers.
        """
        self._stats["price_sanity_checks"] += 1

        if price_a <= 0 or price_b <= 0:
            return False

        ratio = max(price_a, price_b) / min(price_a, price_b)
        threshold = 1.0 + (self._price_divergence_pct / 100.0)

        if ratio > threshold:
            self._stats["price_sanity_blocks"] += 1
            self._price_blocks.append({
                "symbol": symbol,
                "timestamp": time.time(),
                "price_a": price_a,
                "price_b": price_b,
                "ratio": ratio,
                "threshold": threshold,
            })
            # Keep only last 50 blocks
            if len(self._price_blocks) > 50:
                self._price_blocks = self._price_blocks[-50:]

            logger.warning(
                f"PRICE SANITY FAIL: {symbol} | "
                f"prices={price_a:.6f}/{price_b:.6f} ratio={ratio:.3f} "
                f"(threshold={threshold:.2f}) — BLOCKING"
            )
            return False

        return True

    def is_verified(self, symbol: str) -> bool:
        """Full verification: blocklist + contract match.

        Call this BEFORE evaluating any cross-exchange opportunity.
        Caches result so contract comparison only runs once per token.
        """
        base = symbol.split("/")[0] if "/" in symbol else symbol

        # Layer 0: Manual blocklist
        if base in self._blocklist:
            logger.debug(f"CONTRACT VERIFIER: {base} in blocklist")
            return False

        # Layer 2: Contract verification (cached)
        if base not in self._verified:
            self._stats["contract_checks"] += 1
            self._verified[base] = self._check_contract_match(base)
            status = "VERIFIED" if self._verified[base] else "BLOCKED"
            logger.info(f"CONTRACT VERIFIER: {base} -> {status}")

        if not self._verified[base]:
            self._stats["contract_blocks"] += 1
            return False

        # Layer 3: Transfer status (cached)
        transfer = self.check_transfers_enabled(symbol)
        if not transfer["can_arb"]:
            self._stats["transfer_blocks"] += 1
            return False

        return True

    def needs_refresh(self) -> bool:
        """Check if cache needs refresh."""
        return time.time() - self._cache_timestamp > self.cache_ttl

    def get_mismatches(self) -> List[dict]:
        """Return all detected contract mismatches."""
        return self._mismatches

    def get_stats(self) -> dict:
        """Return verification stats for dashboard."""
        verified = sum(1 for v in self._verified.values() if v)
        blocked = sum(1 for v in self._verified.values() if not v)
        return {
            "total_checked": len(self._verified),
            "verified": verified,
            "blocked": blocked,
            "blocklist": sorted(self._blocklist),
            "mismatches": [m["symbol"] for m in self._mismatches],
            "price_blocks_recent": len(self._price_blocks),
            "transfer_blocks_recent": len(self._transfer_blocks),
            "cache_age_minutes": round((time.time() - self._cache_timestamp) / 60, 1)
            if self._cache_timestamp > 0
            else None,
            "mexc_tokens_cached": len(self._currency_cache.get("mexc", {})),
            "binance_tokens_cached": len(self._currency_cache.get("binance", {})),
            **self._stats,
        }
