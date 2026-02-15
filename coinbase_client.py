"""
Coinbase Advanced Trade API Client
Reliable API client with retry logic, rate limiting, and paper trading simulation.
"""

import base64
import hashlib
import hmac
import logging
import random
import threading
import time
import uuid
import warnings
import json
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable

# Handle optional requests import
try:
    import requests
    from requests.adapters import HTTPAdapter

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None
    HTTPAdapter = None
    warnings.warn(
        "requests package not available. Install with: pip install requests",
        UserWarning
    )

# Handle optional pandas import
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None


class _CDPAuth:
    """Coinbase Developer Platform JWT auth using Ed25519 keys."""

    BASE_URL = "https://api.coinbase.com"

    def __init__(self, api_key: str, api_secret_b64: str, logger=None):
        self.api_key = api_key
        self.logger = logger or logging.getLogger(__name__)
        raw = base64.b64decode(api_secret_b64)
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        self._private_key = Ed25519PrivateKey.from_private_bytes(raw[:32])
        from cryptography.hazmat.primitives import serialization as _ser
        self._pem = self._private_key.private_bytes(
            encoding=_ser.Encoding.PEM,
            format=_ser.PrivateFormat.PKCS8,
            encryption_algorithm=_ser.NoEncryption(),
        )

    def _jwt(self, method: str, path: str) -> str:
        import jwt as pyjwt
        uri = f"{method} api.coinbase.com{path}"
        payload = {
            "sub": self.api_key,
            "iss": "cdp",
            "aud": ["cdp_service"],
            "nbf": int(time.time()),
            "exp": int(time.time()) + 120,
            "uris": [uri],
        }
        return pyjwt.encode(payload, self._pem, algorithm="EdDSA",
                            headers={"kid": self.api_key, "nonce": secrets.token_hex(16)})

    def request(self, method: str, path: str, params: Optional[Dict] = None,
                body: Optional[Dict] = None) -> Dict[str, Any]:
        token = self._jwt(method, path)
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"{self.BASE_URL}{path}"
        if method == "GET":
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        else:
            resp = requests.post(url, headers=headers, json=body, timeout=15)
        if not resp.ok:
            self.logger.debug("CDP %s %s -> %s: %s", method, path, resp.status_code, resp.text[:200])
        resp.raise_for_status()
        return resp.json()


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market_market_ioc"
    LIMIT = "limit_limit_gtc"
    STOP = "stop_limit_stop_limit_gtc"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_status: List[int] = None

    def __post_init__(self):
        if self.retry_on_status is None:
            self.retry_on_status = [429, 500, 502, 503, 504]


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_second: float = 10.0
    burst_limit: int = 20
    window_size: float = 1.0


@dataclass
class CoinbaseCredentials:
    """Coinbase API credentials"""
    api_key: str
    api_secret: str
    api_passphrase: str = ""
    sandbox: bool = True


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class RateLimitExceededError(Exception):
    """Exception raised when the rate limit is exceeded"""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for API reliability"""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise CircuitBreakerError("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as circuit_error:
                self._on_failure()
                raise circuit_error

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.tokens = config.burst_limit
        self.last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from the bucket"""
        with self._lock:
            now = time.time()

            # Add tokens based on time passed
            time_passed = now - self.last_update
            tokens_to_add = time_passed * self.config.requests_per_second
            self.tokens = min(self.config.burst_limit, self.tokens + tokens_to_add)
            self.last_update = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens"""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0

            needed_tokens = tokens - self.tokens
            return needed_tokens / self.config.requests_per_second


class PaperTradingSimulator:
    """Simulate trading operations for paper trading mode with real P&L tracking"""

    def __init__(self):
        self.orders = {}
        self.balances = {
            "USD": {"available": "10000.00", "hold": "0.00"},
            "BTC": {"available": "0.00", "hold": "0.00"}
        }
        self.order_counter = 1000
        self._lock = threading.Lock()
        self.logger = logging.getLogger("PaperTrader")
        self._last_prices: Dict[str, float] = {}  # product_id -> last known price

    def update_price(self, product_id: str, price: float):
        """Update the last known price for a product (called by bot each cycle)."""
        if price > 0:
            self._last_prices[product_id] = price

    def _ensure_currency(self, product_id: str):
        """Ensure balance entry exists for the base currency of a product."""
        base = product_id.split("-")[0] if "-" in product_id else product_id
        if base not in self.balances:
            self.balances[base] = {"available": "0.00", "hold": "0.00"}

    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate order creation with actual balance updates."""
        with self._lock:
            order_id = f"paper_order_{self.order_counter}"
            self.order_counter += 1

            product_id = order_data.get("product_id", "BTC-USD")
            side = order_data.get("side", "BUY")
            config = order_data.get("order_configuration", {})

            # Extract base_size/quote_size from nested order config
            # Coinbase API nests under market_market_ioc, limit_limit_gtc, etc.
            base_size = None
            quote_size = None
            for key, val in config.items():
                if isinstance(val, dict):
                    if "base_size" in val and base_size is None:
                        base_size = float(val["base_size"])
                    if "quote_size" in val and quote_size is None:
                        quote_size = float(val["quote_size"])
            # Fallback to top-level keys
            if base_size is None:
                base_size = float(config.get("base_size", 0.001))
            if quote_size is None:
                quote_size = float(config.get("quote_size", 0))
            fee_rate = 0.005  # 50bps simulated fee

            # Compute fill value using actual price with slippage simulation (Gap 6 fix)
            last_price = self._last_prices.get(product_id, 0)
            if quote_size > 0:
                fill_value = quote_size
            elif last_price > 0:
                # Simulate realistic slippage: 1-5 bps adverse fill
                slippage_bps = random.uniform(1.0, 5.0)
                slippage_mult = 1.0 + (slippage_bps / 10000.0) if side == "BUY" else 1.0 - (slippage_bps / 10000.0)
                fill_price = last_price * slippage_mult
                fill_value = base_size * fill_price
            else:
                fill_value = base_size * 50000  # ultimate fallback
            fee = fill_value * fee_rate

            # Update balances
            self._ensure_currency(product_id)
            base_currency = product_id.split("-")[0]
            usd_available = float(self.balances["USD"]["available"])
            base_available = float(self.balances[base_currency]["available"])

            if side == "BUY":
                cost = fill_value + fee
                if cost <= usd_available:
                    usd_available -= cost
                    base_available += base_size
                    self.logger.info(f"PAPER FILL: BUY {base_size:.6f} {base_currency} for ${fill_value:.2f} (fee ${fee:.2f})")
                else:
                    self.logger.warning(f"PAPER: Insufficient USD (${usd_available:.2f}) for ${cost:.2f} order")
            else:  # SELL
                if base_available >= base_size:
                    base_available -= base_size
                    usd_available += fill_value - fee
                    self.logger.info(f"PAPER FILL: SELL {base_size:.6f} {base_currency} for ${fill_value:.2f} (fee ${fee:.2f})")
                else:
                    # Allow short selling in paper mode
                    base_available -= base_size
                    usd_available += fill_value - fee
                    self.logger.info(f"PAPER FILL: SHORT SELL {base_size:.6f} {base_currency} for ${fill_value:.2f}")

            self.balances["USD"]["available"] = f"{usd_available:.2f}"
            self.balances[base_currency]["available"] = f"{base_available:.8f}"

            simulated_order = {
                "order_id": order_id,
                "product_id": product_id,
                "side": side,
                "order_configuration": config,
                "status": "FILLED",
                "filled_size": str(base_size),
                "filled_value": f"{fill_value:.2f}",
                "created_time": datetime.now().isoformat(),
                "completion_percentage": "100",
                "fee": f"{fee:.2f}"
            }

            self.orders[order_id] = simulated_order
            return {"order": simulated_order, "success": True}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get simulated order"""
        with self._lock:
            if order_id in self.orders:
                return {"order": self.orders[order_id]}
            return {"error": "Order not found"}

    def list_orders(self) -> Dict[str, Any]:
        """List all simulated orders"""
        with self._lock:
            return {"orders": list(self.orders.values())}

    def get_accounts(self) -> Dict[str, Any]:
        """Get simulated account balances"""
        with self._lock:
            accounts = []
            for currency, balance in self.balances.items():
                accounts.append({
                    "uuid": f"account_{currency.lower()}",
                    "name": f"{currency} Wallet",
                    "currency": currency,
                    "available_balance": {"value": balance["available"], "currency": currency},
                    "default": True,
                    "active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": datetime.now().isoformat(),
                    "deleted_at": None,
                    "type": "ACCOUNT_TYPE_CRYPTO" if currency != "USD" else "ACCOUNT_TYPE_FIAT",
                    "ready": True,
                    "hold": {"value": balance["hold"], "currency": currency}
                })
            return {"accounts": accounts}


class EnhancedCoinbaseClient:
    """
    Production-ready Coinbase Advanced Trade API client.

    Features:
    - Comprehensive error handling with retry logic
    - Rate limiting with token bucket algorithm
    - Circuit breaker pattern for reliability
    - Paper trading simulation mode
    - Request/response logging
    - Authentication handling
    - Data validation
    """

    def __init__(self, credentials: CoinbaseCredentials, logger: Optional[logging.Logger] = None,
                 retry_config: Optional[RetryConfig] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None,
                 paper_trading: bool = True):
        """
        Initialize Coinbase client.

        Args:
            credentials: API credentials
            logger: Optional logger instance
            retry_config: Retry configuration
            rate_limit_config: Rate limiting configuration
            paper_trading: Enable paper trading mode
        """

        self.credentials = credentials
        self.logger = logger or logging.getLogger(__name__)
        self.paper_trading = paper_trading

        # API endpoints
        if credentials.sandbox:
            self.base_url = "https://api-public.sandbox.exchange.coinbase.com"
        else:
            self.base_url = "https://api.coinbase.com"

        # Initialize components
        self.retry_config = retry_config or RetryConfig()
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.circuit_breaker = CircuitBreaker()

        # Paper trading simulator
        self.paper_trader = PaperTradingSimulator() if paper_trading else None

        # Initialize CDP JWT auth for Cloud API keys (UUID or organizations/ format)
        self.ccxt_exchange = None
        self._cdp_auth = None
        is_cloud_key = (
            credentials.api_key.startswith("organizations/")
            or (len(credentials.api_key) == 36 and credentials.api_key.count('-') == 4)  # UUID format
        )
        if is_cloud_key and credentials.api_key and credentials.api_secret:
            try:
                self._cdp_auth = _CDPAuth(credentials.api_key, credentials.api_secret, self.logger)
                self.logger.info("Initialized CDP JWT auth (key: %s...)", credentials.api_key[:12])
            except Exception as e:
                self.logger.error("Failed to initialize CDP auth: %s", e)

        # Request session setup
        if REQUESTS_AVAILABLE:
            self.session = requests.Session()
            if HTTPAdapter:
                adapter = HTTPAdapter(max_retries=0)  # We handle retries ourselves
                self.session.mount("https://", adapter)
                self.session.mount("https://", adapter)
        else:
            self.session = None
            self.logger.warning("Requests library not available - client will not work")

        # Statistics
        self.stats = {
            "requests_made": 0,
            "requests_failed": 0,
            "rate_limit_hits": 0,
            "circuit_breaker_trips": 0,
            "paper_trades": 0
        }
        self._stats_lock = threading.Lock()

        self.logger.info(
            f"Coinbase client initialized - Paper trading: {paper_trading}, Sandbox: {credentials.sandbox}")

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate CB-ACCESS-SIGN header (Legacy HMAC or v3 JWT)"""
        # Detect v3 Cloud API Key
        if self.credentials.api_key.startswith("organizations/"):
            return self._generate_v3_jwt(method, path)
            
        message = timestamp + method + path + body
        signature = hmac.new(
            base64.b64decode(self.credentials.api_secret),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        return base64.b64encode(signature).decode('utf-8')

    def _generate_v3_jwt(self, method: str, path: str) -> str:
        """
        Generate JWT for Coinbase Advanced Trade v3 (Cloud API Keys).
        Uses ES256 algorithm.
        """
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            import secrets
            import json
            import base64
            import time

            # Strip possible quotes and fix newlines in secret
            secret = self.credentials.api_secret.strip("'").strip('"').replace('\\n', '\n')
            
            # 1. Prepare Header
            header = {"alg": "ES256", "typ": "JWT", "kid": self.credentials.api_key}
            
            # 2. Prepare Payload
            # v3 expects 'uri' in the format "METHOD {path}" without query params usually, 
            # but officially it's "METHOD hostname path"
            # Actually for Advanced Trade v3 Cloud API keys:
            # { "iss": "coinbase", "nbf": ..., "exp": ..., "sub": "key_id", "uri": "METHOD hostname path" }
            
            # hostname is usually api.coinbase.com
            hostname = "api.coinbase.com"
            
            # For some v3 implementations, the 'uri' should just be the path without hostname
            # But officially for Cloud API keys it includes it.
            # Let's try WITHOUT hostname if the previous failed (already tried with)
            # Actually, standard is "METHOD hostname path"
            base_path = path.split('?')[0]
            if not base_path.startswith('/'):
                base_path = '/' + base_path
            
            uri = f"{method.upper()} {hostname}{base_path}"
            
            # Log URI for debugging (carefully)
            # self.logger.debug(f"JWT URI: {uri}")
            
            now = int(time.time())
            payload = {
                "iss": "coinbase",
                "nbf": now - 10,
                "exp": now + 60,
                "sub": self.credentials.api_key,
                "uri": uri,
                "aud": ["brokerage"]
            }
            
            def b64_encode(data: dict) -> str:
                return base64.urlsafe_b64encode(json.dumps(data, separators=(',', ':')).encode()).decode().rstrip("=")

            # 3. Create unsigned JWT
            unsigned_jwt = f"{b64_encode(header)}.{b64_encode(payload)}"
            
            # 4. Sign with EC private key
            private_key = serialization.load_pem_private_key(
                secret.encode(),
                password=None
            )
            
            signature = private_key.sign(
                unsigned_jwt.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            
            # 5. Convert DER signature to Raw R|S for JWT ES256
            # ES256 signatures are exactly 64 bytes (32 for R, 32 for S)
            from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
            r, s = decode_dss_signature(signature)
            
            def int_to_bytes(i: int) -> bytes:
                # Ensure exactly 32 bytes
                return i.to_bytes(32, byteorder='big')
            
            raw_signature = int_to_bytes(r) + int_to_bytes(s)
            
            # Use base64.urlsafe_b64encode and remove padding
            encoded_signature = base64.urlsafe_b64encode(raw_signature).decode('utf-8').rstrip('=')
            
            return f"{unsigned_jwt}.{encoded_signature}"
            
        except ImportError:
            self.logger.error("Cryptography library missing. Required for v3 JWT.")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to generate v3 JWT: {e}")
            return ""

    def _get_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        """Generate authentication headers"""
        timestamp = str(int(time.time()))
        
        # v3 Cloud API Keys use JWT in Authorization header
        if self.credentials.api_key.startswith("organizations/"):
            jwt_token = self._generate_v3_jwt(method, path)
            return {
                "Authorization": f"Bearer {jwt_token}",
                "Content-Type": "application/json",
                "User-Agent": "Enhanced-Trading-Bot/1.0"
            }

        headers = {
            "CB-ACCESS-KEY": self.credentials.api_key,
            "CB-ACCESS-SIGN": self._generate_signature(timestamp, method, path, body),
            "CB-ACCESS-TIMESTAMP": timestamp,
            "Content-Type": "application/json",
            "User-Agent": "Enhanced-Trading-Bot/1.0"
        }
        
        # Include passphrase only if provided (legacy keys)
        if self.credentials.api_passphrase:
            headers["CB-ACCESS-PASSPHRASE"] = self.credentials.api_passphrase
            
        return headers

    def _update_stats(self, stat_name: str, increment: int = 1):
        """Update statistics"""
        with self._stats_lock:
            if stat_name in self.stats:
                self.stats[stat_name] += increment

    @contextmanager
    def _rate_limit_context(self):
        """Context manager for rate limiting"""
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                self._update_stats("rate_limit_hits")
                self.logger.warning(f"Rate limit hit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)

                if not self.rate_limiter.acquire():
                    raise RateLimitExceededError("Could not acquire rate limit token")

        yield

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request with retry logic and error handling"""

        if not REQUESTS_AVAILABLE or not self.session:
            raise RuntimeError("Requests library not available")

        # Use CDP JWT auth for Cloud API keys
        if self._cdp_auth:
            try:
                self._update_stats("requests_made")
                params = data if method == "GET" else None
                body_data = data if method != "GET" else None
                response = self._cdp_auth.request(method, endpoint, params=params, body=body_data)
                return response
            except Exception as e:
                self._update_stats("requests_failed")
                self.logger.error(f"CDP request failed: {e}")
                raise e

        url = f"{self.base_url}{endpoint}"
        body = ""

        if data:
            import json
            body = json.dumps(data)

        headers = self._get_headers(method, endpoint, body)

        def _execute_request():
            with self._rate_limit_context():
                self._update_stats("requests_made")

                try:
                    if method.upper() == "GET":
                        response = self.session.get(url, headers=headers, params=data, timeout=30)
                    elif method.upper() == "POST":
                        response = self.session.post(url, headers=headers, data=body, timeout=30)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")

                    # Handle rate limiting — sleep and let outer retry loop re-attempt
                    if response.status_code == 429:
                        self._update_stats("rate_limit_hits")
                        retry_after = int(response.headers.get("Retry-After", 5))
                        retry_after = min(retry_after, 120)  # Cap at 2 minutes
                        self.logger.warning(f"Rate limited (429). Sleeping {retry_after}s before retry.")
                        time.sleep(retry_after)
                        raise RateLimitExceededError(f"Rate limited, retried after {retry_after} seconds")

                    # Check for HTTP errors
                    response.raise_for_status()

                    return response.json()

                except Exception as request_error:
                    self._update_stats("requests_failed")
                    self.logger.error(f"Request failed: {request_error}")
                    raise request_error

        # Execute with circuit breaker and retry logic
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return self.circuit_breaker.call(_execute_request)

            except CircuitBreakerError:
                self._update_stats("circuit_breaker_trips")
                raise

            except Exception as attempt_error:
                if attempt == self.retry_config.max_retries:
                    self.logger.error(f"All retry attempts failed: {attempt_error}")
                    raise attempt_error

                # Check if we should retry this error
                should_retry = False
                if isinstance(attempt_error, RateLimitExceededError):
                    should_retry = True  # Already slept in handler; retry immediately
                elif hasattr(attempt_error, 'response') and attempt_error.response:
                    should_retry = attempt_error.response.status_code in self.retry_config.retry_on_status
                elif isinstance(attempt_error, (ConnectionError, TimeoutError)):
                    should_retry = True

                if not should_retry:
                    raise attempt_error

                # Calculate delay with jitter to prevent thundering herd
                delay = min(
                    self.retry_config.base_delay * (self.retry_config.backoff_multiplier ** attempt),
                    self.retry_config.max_delay
                )
                delay *= random.uniform(0.5, 1.5)

                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {attempt_error}")
                time.sleep(delay)

        raise RuntimeError("Unexpected error in retry logic")

    def get_accounts(self) -> Dict[str, Any]:
        """Get account information"""
        if self.paper_trading and self.paper_trader:
            return self.paper_trader.get_accounts()

        if self._cdp_auth:
            try:
                return self._make_request("GET", "/api/v3/brokerage/accounts")
            except Exception as e:
                self.logger.error(f"CDP fetch accounts failed: {e}")
                return {"error": str(e), "accounts": []}

        try:
            return self._make_request("GET", "/api/v3/brokerage/accounts")
        except Exception as account_error:
            self.logger.error(f"Failed to get accounts: {account_error}")
            return {"error": str(account_error), "accounts": []}

    def get_account(self, account_uuid: str) -> Dict[str, Any]:
        """Get specific account information"""
        if self.paper_trading:
            # For paper trading, return mock account data
            return {
                "account": {
                    "uuid": account_uuid,
                    "name": "Paper Trading Account",
                    "currency": "USD",
                    "available_balance": {"value": "10000.00", "currency": "USD"},
                    "default": True,
                    "active": True,
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": datetime.now().isoformat(),
                    "type": "ACCOUNT_TYPE_FIAT",
                    "ready": True,
                    "hold": {"value": "0.00", "currency": "USD"}
                }
            }

        try:
            return self._make_request("GET", f"/api/v3/brokerage/accounts/{account_uuid}")
        except Exception as account_error:
            self.logger.error(f"Failed to get account {account_uuid}: {account_error}")
            return {"error": str(account_error)}

    def list_orders(self, product_id: Optional[str] = None,
                    order_status: Optional[List[str]] = None,
                    limit: int = 100) -> Dict[str, Any]:
        """List orders with optional filtering"""
        if self.paper_trading and self.paper_trader:
            return self.paper_trader.list_orders()

        try:
            params = {"limit": limit}
            if product_id:
                params["product_id"] = product_id
            if order_status:
                params["order_status"] = order_status

            return self._make_request("GET", "/api/v3/brokerage/orders/historical/batch", params)
        except Exception as orders_error:
            self.logger.error(f"Failed to list orders: {orders_error}")
            return {"error": str(orders_error), "orders": []}

    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get specific order information"""
        if self.paper_trading and self.paper_trader:
            return self.paper_trader.get_order(order_id)

        try:
            return self._make_request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
        except Exception as order_error:
            self.logger.error(f"Failed to get order {order_id}: {order_error}")
            return {"error": str(order_error)}

    def create_order(self, product_id: str, side: str, order_configuration: Dict[str, Any],
                     client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new order"""

        # Validate inputs
        if side not in ["BUY", "SELL"]:
            return {"error": "Invalid side. Must be 'BUY' or 'SELL'"}

        if not order_configuration:
            return {"error": "Order configuration is required"}

        order_data = {
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "product_id": product_id,
            "side": side,
            "order_configuration": order_configuration
        }

        if self.paper_trading and self.paper_trader:
            self._update_stats("paper_trades")
            self.logger.info(f"Paper trading order: {side} {product_id}")
            return self.paper_trader.create_order(order_data)

        try:
            result = self._make_request("POST", "/api/v3/brokerage/orders", order_data)
            self.logger.info(f"Order created: {result.get('order', {}).get('order_id', 'Unknown')}")
            return result
        except Exception as create_error:
            self.logger.error(f"Failed to create order: {create_error}")
            return {"error": str(create_error)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        if self.paper_trading:
            return {"success": True, "order_id": order_id, "message": "Paper trading order cancelled"}

        try:
            return self._make_request("POST", "/api/v3/brokerage/orders/batch_cancel",
                                      {"order_ids": [order_id]})
        except Exception as cancel_error:
            self.logger.error(f"Failed to cancel order {order_id}: {cancel_error}")
            return {"error": str(cancel_error)}

    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get product information"""
        try:
            return self._make_request("GET", f"/api/v3/brokerage/products/{product_id}")
        except Exception as product_error:
            self.logger.error(f"Failed to get product {product_id}: {product_error}")
            return {"error": str(product_error)}

    def get_product_book(self, product_id: str, limit: int = 10) -> Dict[str, Any]:
        """Get product order book (best-effort)."""
        if self._cdp_auth:
            try:
                params = {"product_id": product_id, "limit": limit}
                return self._make_request("GET", "/api/v3/brokerage/product_book", params)
            except Exception as e:
                self.logger.error(f"CDP fetch_order_book failed: {e}")
                return {"error": str(e), "bids": [], "asks": []}

        try:
            params = {"product_id": product_id, "limit": limit}
            return self._make_request("GET", "/api/v3/brokerage/product_book", params)
        except Exception as book_error:
            self.logger.error(f"Failed to get product book for {product_id}: {book_error}")
            return {"error": str(book_error), "bids": [], "asks": []}

    def get_product_candles(self, product_id: str, start: Optional[str] = None,
                            end: Optional[str] = None, granularity: str = "ONE_HOUR") -> Dict[str, Any]:
        """Get historical candle data"""
        if self._cdp_auth:
            try:
                params = {"granularity": granularity}
                if start:
                    params["start"] = start
                if end:
                    params["end"] = end
                resp = self._cdp_auth.request("GET", f"/api/v3/brokerage/products/{product_id}/candles", params=params)
                candles = resp.get("candles", [])
                response = {"candles": candles}
                if PANDAS_AVAILABLE and pd is not None and candles:
                    candles_data = []
                    for c in candles:
                        candles_data.append({
                            "timestamp": datetime.fromtimestamp(int(c["start"])),
                            "open": float(c["open"]),
                            "high": float(c["high"]),
                            "low": float(c["low"]),
                            "close": float(c["close"]),
                            "volume": float(c["volume"])
                        })
                    response["dataframe"] = pd.DataFrame(candles_data)
                return response
            except Exception as e:
                self.logger.error(f"CDP fetch candles failed: {e}")
                return {"error": str(e), "candles": []}

        try:
            params = {
                "product_id": product_id,
                "granularity": granularity
            }

            if start:
                params["start"] = start
            if end:
                params["end"] = end

            response = self._make_request("GET", "/api/v3/brokerage/products/{}/candles".format(product_id), params)

            # Convert to DataFrame if pandas is available
            if PANDAS_AVAILABLE and pd is not None and "candles" in response:
                candles_data = []
                for candle in response["candles"]:
                    candles_data.append({
                        "timestamp": datetime.fromtimestamp(int(candle["start"])),
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle["volume"])
                    })

                response["dataframe"] = pd.DataFrame(candles_data)

            return response

        except Exception as candles_error:
            self.logger.error(f"Failed to get candles for {product_id}: {candles_error}")
            return {"error": str(candles_error), "candles": []}

    def get_market_trades(self, product_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get recent market trades"""
        if self._cdp_auth:
            try:
                product = self._make_request("GET", f"/api/v3/brokerage/products/{product_id}")
                return {
                    "price": str(product.get("price", 0)),
                    "best_bid": str(product.get("bid_price", product.get("price", 0))),
                    "best_ask": str(product.get("ask_price", product.get("price", 0))),
                    "volume_24h": str(product.get("volume_24h", 0)),
                    "trades": []
                }
            except Exception as e:
                self.logger.error(f"CDP fetch ticker failed: {e}")
                return {"error": str(e), "trades": []}

        try:
            params = {"product_id": product_id, "limit": limit}
            return self._make_request("GET", "/api/v3/brokerage/products/{}/ticker".format(product_id), params)
        except Exception as trades_error:
            self.logger.error(f"Failed to get market trades for {product_id}: {trades_error}")
            return {"error": str(trades_error), "trades": []}

    def create_market_order(self, product_id: str, side: str, funds: Optional[float] = None,
                            size: Optional[float] = None,
                            client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a market order"""

        if not funds and not size:
            return {"error": "Either funds or size must be specified"}

        order_config = {"market_market_ioc": {}}

        if side == "BUY":
            if funds:
                order_config["market_market_ioc"]["quote_size"] = str(funds)
            else:
                order_config["market_market_ioc"]["base_size"] = str(size)
        else:  # SELL
            if size:
                order_config["market_market_ioc"]["base_size"] = str(size)
            else:
                return {"error": "Size must be specified for sell orders"}

        return self.create_order(product_id, side, order_config, client_order_id=client_order_id)

    def create_limit_order(self, product_id: str, side: str, size: float,
                           price: float, post_only: bool = False,
                           client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a limit order"""

        order_config = {
            "limit_limit_gtc": {
                "base_size": str(size),
                "limit_price": str(price),
                "post_only": post_only
            }
        }

        return self.create_order(product_id, side, order_config, client_order_id=client_order_id)

    def get_fills(self, product_id: Optional[str] = None, order_id: Optional[str] = None) -> Dict[str, Any]:
        """Get fill history"""
        if self.paper_trading:
            return {"fills": [], "message": "Paper trading - no real fills"}

        try:
            params = {}
            if product_id:
                params["product_id"] = product_id
            if order_id:
                params["order_id"] = order_id

            return self._make_request("GET", "/api/v3/brokerage/orders/historical/fills", params)
        except Exception as fills_error:
            self.logger.error(f"Failed to get fills: {fills_error}")
            return {"error": str(fills_error), "fills": []}

    def get_portfolio_breakdown(self) -> Dict[str, Any]:
        """Get portfolio breakdown with balances"""
        try:
            accounts_response = self.get_accounts()

            if "error" in accounts_response:
                return accounts_response

            portfolio = {
                "total_balance_usd": 0.0,
                "balances": {},
                "allocation": {}
            }

            for account in accounts_response.get("accounts", []):
                currency = account.get("currency", "")
                available_balance = float(account.get("available_balance", {}).get("value", 0))

                portfolio["balances"][currency] = {
                    "available": available_balance,
                    "currency": currency,
                    "account_id": account.get("uuid", "")
                }

                # For USD, add directly to total
                if currency == "USD":
                    portfolio["total_balance_usd"] += available_balance
                # For crypto, would need current prices to calculate USD value
                # This is simplified, for example

            return portfolio

        except Exception as portfolio_error:
            self.logger.error(f"Failed to get portfolio breakdown: {portfolio_error}")
            return {"error": str(portfolio_error)}

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the client"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "paper_trading": self.paper_trading,
            "sandbox_mode": self.credentials.sandbox,
            "circuit_breaker_state": self.circuit_breaker.state,
            "statistics": self.stats.copy(),
            "checks": {}
        }

        # Test basic connectivity
        try:
            if self.paper_trading:
                health_status["checks"]["connectivity"] = "skipped_paper_trading"
            else:
                accounts = self.get_accounts()
                if "error" not in accounts:
                    health_status["checks"]["connectivity"] = "ok"
                else:
                    health_status["checks"]["connectivity"] = f"failed: {accounts['error']}"
                    health_status["status"] = "degraded"
        except Exception as health_error:
            health_status["checks"]["connectivity"] = f"error: {health_error}"
            health_status["status"] = "unhealthy"

        # Check rate limiter
        if self.rate_limiter.tokens > 0:
            health_status["checks"]["rate_limiter"] = "ok"
        else:
            health_status["checks"]["rate_limiter"] = "throttled"

        return health_status

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        with self._stats_lock:
            stats_copy = self.stats.copy()

        # Calculate success rate
        total_requests = stats_copy["requests_made"]
        if total_requests > 0:
            stats_copy["success_rate"] = (total_requests - stats_copy["requests_failed"]) / total_requests
        else:
            stats_copy["success_rate"] = 0.0

        stats_copy["circuit_breaker_state"] = self.circuit_breaker.state
        stats_copy["available_tokens"] = self.rate_limiter.tokens

        return stats_copy

    def reset_stats(self):
        """Reset statistics"""
        with self._stats_lock:
            for key in self.stats:
                self.stats[key] = 0

        self.logger.info("Client statistics reset")


# Utility functions
def create_client_from_config(config_dict: Dict[str, Any],
                              logger: Optional[logging.Logger] = None) -> EnhancedCoinbaseClient:
    """Create client from configuration dictionary"""

    credentials = CoinbaseCredentials(
        api_key=config_dict.get("COINBASE_API_KEY", ""),
        api_secret=config_dict.get("COINBASE_API_SECRET", ""),
        api_passphrase=config_dict.get("COINBASE_API_PASSPHRASE", ""),
        sandbox=config_dict.get("SANDBOX_MODE", True)
    )

    retry_config = RetryConfig(
        max_retries=config_dict.get("MAX_RETRIES", 3),
        base_delay=config_dict.get("RETRY_BASE_DELAY", 1.0),
        max_delay=config_dict.get("RETRY_MAX_DELAY", 60.0)
    )

    rate_limit_config = RateLimitConfig(
        requests_per_second=config_dict.get("RATE_LIMIT_RPS", 10.0),
        burst_limit=config_dict.get("RATE_LIMIT_BURST", 20)
    )

    return EnhancedCoinbaseClient(
        credentials=credentials,
        logger=logger,
        retry_config=retry_config,
        rate_limit_config=rate_limit_config,
        paper_trading=config_dict.get("PAPER_TRADING", True)
    )


if __name__ == "__main__":
    # Client testing
    print("Enhanced Coinbase Client Test")
    print("=" * 50)

    # Test configuration
    test_config = {
        "COINBASE_API_KEY": "test_key",
        "COINBASE_API_SECRET": "test_secret",
        "COINBASE_API_PASSPHRASE": "test_passphrase",
        "SANDBOX_MODE": True,
        "PAPER_TRADING": True
    }

    try:
        # Initialize client
        test_logger = logging.getLogger("test_client")
        test_logger.setLevel(logging.INFO)

        client = create_client_from_config(test_config, test_logger)
        print("✅ Client initialized successfully")

        # Test paper trading operations
        print("\n🧪 Testing paper trading operations...")

        # Test account retrieval
        accounts = client.get_accounts()
        print(f"✅ Accounts retrieved: {len(accounts.get('accounts', []))} accounts")

        # Test order creation
        order_result = client.create_market_order("BTC-USD", "BUY", funds=100.0)
        print(f"✅ Paper order created: {order_result.get('order', {}).get('order_id', 'N/A')}")

        # Test health check
        health = client.health_check()
        print(f"✅ Health check: {health['status']}")

        # Test statistics
        stats = client.get_stats()
        print(f"✅ Statistics: {stats['requests_made']} requests, {stats['paper_trades']} paper trades")

        print("\n🎉 All client tests passed!")

    except Exception as test_error:
        print(f"❌ Client test failed: {test_error}")
        raise
