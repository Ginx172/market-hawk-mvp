"""
SCRIPT NAME: broker_auth.py
====================================
Execution Location: market-hawk-mvp/executor/
Purpose: Broker authentication framework — credential management, HMAC signing,
         key rotation, TLS enforcement for IB and Alpaca integrations.
Creation Date: 2026-03-07

Credentials are read exclusively from environment variables (never hardcoded).
Supports:
    - Interactive Brokers (TWS API): host/port/client_id
    - Alpaca: API key/secret + base URL
    - HMAC-SHA256 request signing
    - API key expiry tracking with 7-day advance alerts
    - TLS 1.2+ enforcement on all outbound connections
"""

import hashlib
import hmac
import logging
import os
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger("market_hawk.broker_auth")


# ============================================================
# CREDENTIAL CONTAINERS
# ============================================================

@dataclass
class IBCredentials:
    """Interactive Brokers TWS/Gateway connection parameters."""
    host: str
    port: int
    client_id: int


@dataclass
class AlpacaCredentials:
    """Alpaca API credentials."""
    api_key: str
    api_secret: str
    base_url: str


@dataclass
class KeyRotationInfo:
    """Tracks API key creation and expiry for rotation alerts."""
    broker: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    alert_days_before: int = 7


# ============================================================
# BROKER AUTH MANAGER
# ============================================================

class BrokerAuthManager:
    """Centralized broker credential management with security features.

    Features:
        - Credentials from env vars only (no config files)
        - HMAC-SHA256 request signing
        - TLS 1.2+ enforcement
        - Key expiry tracking with rotation alerts

    Usage:
        auth = BrokerAuthManager()
        ib = auth.get_credentials("ib")
        alpaca = auth.get_credentials("alpaca")

        signed = auth.sign_request(body, secret)
        ctx = auth.get_tls_context()
    """

    _WARN_DAYS = 7  # Alerteaza cu 7 zile inainte de expirare

    def __init__(self) -> None:
        self._rotation_registry: Dict[str, KeyRotationInfo] = {}
        self._ib: Optional[IBCredentials] = None
        self._alpaca: Optional[AlpacaCredentials] = None
        self._tls_context: Optional[ssl.SSLContext] = None

    # ---------- CREDENTIALS ----------

    def get_credentials(self, broker: str) -> Optional[object]:
        """Retrieve credentials for a broker from environment variables.

        Args:
            broker: One of 'ib' or 'alpaca'.

        Returns:
            IBCredentials or AlpacaCredentials, or None if not configured.
        """
        broker_lower = broker.lower()
        if broker_lower == "ib":
            return self._get_ib_credentials()
        elif broker_lower == "alpaca":
            return self._get_alpaca_credentials()
        else:
            logger.warning("Unknown broker: %r", broker)
            return None

    def _get_ib_credentials(self) -> Optional[IBCredentials]:
        """Load IB credentials from env vars (cached)."""
        if self._ib is not None:
            return self._ib

        host = os.environ.get("MH_IB_HOST", "")
        port_str = os.environ.get("MH_IB_PORT", "")
        client_id_str = os.environ.get("MH_IB_CLIENT_ID", "")

        if not host:
            logger.debug("MH_IB_HOST not set — IB broker not configured")
            return None

        try:
            port = int(port_str) if port_str else 7497
            client_id = int(client_id_str) if client_id_str else 1
        except ValueError as e:
            logger.error("Invalid IB port/client_id: %s", e)
            return None

        self._ib = IBCredentials(host=host, port=port, client_id=client_id)
        logger.info("IB credentials loaded: %s:%d (client %d)", host, port, client_id)
        return self._ib

    def _get_alpaca_credentials(self) -> Optional[AlpacaCredentials]:
        """Load Alpaca credentials from env vars (cached)."""
        if self._alpaca is not None:
            return self._alpaca

        api_key = os.environ.get("MH_ALPACA_KEY", "")
        api_secret = os.environ.get("MH_ALPACA_SECRET", "")
        base_url = os.environ.get(
            "MH_ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        )

        if not api_key or not api_secret:
            logger.debug("MH_ALPACA_KEY/SECRET not set — Alpaca not configured")
            return None

        self._alpaca = AlpacaCredentials(
            api_key=api_key, api_secret=api_secret, base_url=base_url,
        )
        logger.info("Alpaca credentials loaded (base: %s)", base_url)
        return self._alpaca

    # ---------- HMAC REQUEST SIGNING ----------

    @staticmethod
    def sign_request(body: bytes, secret: str, algorithm: str = "sha256") -> str:
        """Create HMAC signature for an API request body.

        Args:
            body: Raw request body bytes.
            secret: API secret key.
            algorithm: Hash algorithm (default sha256).

        Returns:
            Hex-encoded HMAC signature string.
        """
        if algorithm not in ("sha256", "sha384", "sha512"):
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")

        h = hmac.new(secret.encode("utf-8"), body, getattr(hashlib, algorithm))
        return h.hexdigest()

    @staticmethod
    def verify_signature(body: bytes, secret: str, signature: str,
                         algorithm: str = "sha256") -> bool:
        """Verify an HMAC signature (constant-time comparison)."""
        expected = BrokerAuthManager.sign_request(body, secret, algorithm)
        return hmac.compare_digest(expected, signature)

    # ---------- TLS ENFORCEMENT ----------

    def get_tls_context(self) -> ssl.SSLContext:
        """Create or return a cached TLS 1.2+ SSL context.

        Enforces minimum TLS 1.2, disables older protocols,
        requires certificate verification.
        """
        if self._tls_context is not None:
            return self._tls_context

        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
        ctx.load_default_certs()

        # Dezactiveaza protocoale vechi explicit
        ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3 | ssl.OP_NO_TLSv1 | ssl.OP_NO_TLSv1_1

        self._tls_context = ctx
        logger.info("TLS context created (minimum TLS 1.2)")
        return ctx

    def validate_connection(self, broker: str) -> Dict[str, object]:
        """Validate that broker credentials are present and TLS is available.

        Returns:
            Dict with 'valid' (bool), 'broker', 'tls_version', 'message'.
        """
        creds = self.get_credentials(broker)
        if creds is None:
            return {
                "valid": False,
                "broker": broker,
                "tls_version": None,
                "message": f"No credentials configured for {broker}",
            }

        tls_ctx = self.get_tls_context()
        return {
            "valid": True,
            "broker": broker,
            "tls_version": f">= TLS {tls_ctx.minimum_version.name}",
            "message": f"{broker} credentials OK, TLS enforced",
        }

    # ---------- KEY ROTATION ----------

    def register_key(self, broker: str, created_at: datetime,
                     expires_at: Optional[datetime] = None) -> None:
        """Register an API key's lifecycle for rotation tracking.

        Args:
            broker: Broker identifier (e.g., 'alpaca').
            created_at: When the key was created.
            expires_at: When the key expires (None = no expiry).
        """
        self._rotation_registry[broker] = KeyRotationInfo(
            broker=broker,
            created_at=created_at,
            expires_at=expires_at,
        )
        logger.info("Registered key rotation for %s (expires: %s)",
                     broker, expires_at or "never")

    def check_key_expiry(self) -> Dict[str, Dict]:
        """Check all registered keys for upcoming expiry.

        Returns:
            Dict mapping broker to status info:
            {'expired': bool, 'days_remaining': int or None, 'alert': bool}
        """
        now = datetime.now(timezone.utc)
        results = {}

        for broker, info in self._rotation_registry.items():
            if info.expires_at is None:
                results[broker] = {
                    "expired": False,
                    "days_remaining": None,
                    "alert": False,
                    "message": "No expiry set",
                }
                continue

            expires_utc = info.expires_at
            if expires_utc.tzinfo is None:
                expires_utc = expires_utc.replace(tzinfo=timezone.utc)

            remaining = (expires_utc - now).days
            expired = remaining <= 0
            alert = remaining <= self._WARN_DAYS and not expired

            if expired:
                logger.error("API key for %s has EXPIRED (%d days ago)", broker, -remaining)
            elif alert:
                logger.warning("API key for %s expires in %d days", broker, remaining)

            results[broker] = {
                "expired": expired,
                "days_remaining": remaining,
                "alert": alert,
                "message": "EXPIRED" if expired else (
                    f"Expires in {remaining} days" + (" — ROTATE SOON" if alert else "")
                ),
            }

        return results

    def rotate_key(self, broker: str) -> bool:
        """Signal that a key rotation is needed for a broker.

        In production, this would trigger the key rotation flow
        (generate new key via broker API, update env vars, etc.).
        For now, it logs the event and clears cached credentials.

        Returns:
            True if rotation was initiated.
        """
        logger.warning("Key rotation requested for %s — clear cached credentials", broker)

        broker_lower = broker.lower()
        if broker_lower == "ib":
            self._ib = None
        elif broker_lower == "alpaca":
            self._alpaca = None
        else:
            logger.warning("Unknown broker for rotation: %s", broker)
            return False

        if broker in self._rotation_registry:
            del self._rotation_registry[broker]

        return True


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    auth = BrokerAuthManager()

    # Test HMAC signing
    body = b'{"symbol": "AAPL", "qty": 10, "side": "buy"}'
    sig = auth.sign_request(body, "test-secret-key")
    print(f"HMAC signature: {sig}")
    print(f"Verify OK: {auth.verify_signature(body, 'test-secret-key', sig)}")
    print(f"Verify bad: {auth.verify_signature(body, 'wrong-key', sig)}")

    # Test TLS context
    ctx = auth.get_tls_context()
    print(f"TLS min version: {ctx.minimum_version.name}")

    # Test connection validation
    for b in ["ib", "alpaca"]:
        result = auth.validate_connection(b)
        print(f"{b}: {result}")

    # Test key rotation
    auth.register_key("alpaca",
                       created_at=datetime.now(timezone.utc) - timedelta(days=85),
                       expires_at=datetime.now(timezone.utc) + timedelta(days=5))
    expiry = auth.check_key_expiry()
    for broker, info in expiry.items():
        print(f"Key {broker}: {info}")
