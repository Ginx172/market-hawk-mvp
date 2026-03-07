"""
SCRIPT NAME: session.py
====================================
Execution Location: market-hawk-mvp/config/
Purpose: Secure session management for Market Hawk dashboard and API.
Creation Date: 2026-03-07

Provides:
    - Token-based sessions (secrets.token_urlsafe)
    - Configurable timeout (MH_SESSION_TIMEOUT_MINUTES, default 30)
    - In-memory session store with automatic cleanup
    - Thread-safe operations

Usage:
    from config.session import SessionManager

    manager = SessionManager()
    token = manager.create_session(user_id="admin")
    session = manager.get_session(token)
    if session and session.is_valid():
        ...
    manager.revoke(token)
"""

import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("market_hawk.session")


# ============================================================
# SESSION DATA
# ============================================================

@dataclass
class SecureSession:
    """A single authenticated session."""
    token: str
    user_id: str
    role: str
    created_at: float          # time.time() epoch
    last_activity: float       # time.time() epoch
    timeout_seconds: int       # Inactivity timeout
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if session has not expired due to inactivity."""
        return (time.time() - self.last_activity) < self.timeout_seconds

    def refresh(self) -> None:
        """Update last activity timestamp (extend session)."""
        self.last_activity = time.time()

    @property
    def age_seconds(self) -> float:
        """Total seconds since session creation."""
        return time.time() - self.created_at

    @property
    def idle_seconds(self) -> float:
        """Seconds since last activity."""
        return time.time() - self.last_activity

    @property
    def expires_at(self) -> float:
        """Epoch timestamp when session will expire if idle."""
        return self.last_activity + self.timeout_seconds


# ============================================================
# SESSION MANAGER
# ============================================================

class SessionManager:
    """Thread-safe in-memory session store with automatic cleanup.

    Args:
        timeout_minutes: Session inactivity timeout.
            Read from MH_SESSION_TIMEOUT_MINUTES env var, default 30.
        cleanup_interval: Seconds between automatic cleanup sweeps.
        token_bytes: Number of random bytes for token generation.
    """

    def __init__(self, timeout_minutes: Optional[int] = None,
                 cleanup_interval: int = 300,
                 token_bytes: int = 32) -> None:
        env_timeout = os.environ.get("MH_SESSION_TIMEOUT_MINUTES", "")
        if timeout_minutes is not None:
            self._timeout_seconds = timeout_minutes * 60
        elif env_timeout:
            try:
                self._timeout_seconds = int(env_timeout) * 60
            except ValueError:
                logger.warning("Invalid MH_SESSION_TIMEOUT_MINUTES=%r, using 30",
                               env_timeout)
                self._timeout_seconds = 30 * 60
        else:
            self._timeout_seconds = 30 * 60

        self._token_bytes = token_bytes
        self._cleanup_interval = cleanup_interval
        self._sessions: Dict[str, SecureSession] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

        logger.info("SessionManager initialized (timeout=%d min, cleanup=%ds)",
                     self._timeout_seconds // 60, cleanup_interval)

    # ---------- CRUD ----------

    def create_session(self, user_id: str, role: str = "VIEWER",
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session and return its token.

        Args:
            user_id: Identifier for the user.
            role: User role string (from RBAC).
            metadata: Optional extra data to store with session.

        Returns:
            URL-safe token string.
        """
        token = secrets.token_urlsafe(self._token_bytes)
        now = time.time()

        session = SecureSession(
            token=token,
            user_id=user_id,
            role=role,
            created_at=now,
            last_activity=now,
            timeout_seconds=self._timeout_seconds,
            metadata=metadata or {},
        )

        with self._lock:
            self._sessions[token] = session
            self._maybe_cleanup()

        logger.info("Session created for user=%s role=%s (timeout=%dm)",
                     user_id, role, self._timeout_seconds // 60)
        return token

    def get_session(self, token: str) -> Optional[SecureSession]:
        """Retrieve a session by token. Returns None if not found or expired.

        Automatically refreshes the session on access if valid.
        """
        with self._lock:
            session = self._sessions.get(token)

        if session is None:
            return None

        if not session.is_valid():
            self.revoke(token)
            logger.debug("Session expired for user=%s", session.user_id)
            return None

        session.refresh()
        return session

    def revoke(self, token: str) -> bool:
        """Revoke (delete) a session.

        Returns:
            True if session existed and was revoked.
        """
        with self._lock:
            session = self._sessions.pop(token, None)

        if session:
            logger.info("Session revoked for user=%s", session.user_id)
            return True
        return False

    def revoke_user(self, user_id: str) -> int:
        """Revoke all sessions for a user.

        Returns:
            Number of sessions revoked.
        """
        count = 0
        with self._lock:
            to_remove = [
                t for t, s in self._sessions.items() if s.user_id == user_id
            ]
            for token in to_remove:
                del self._sessions[token]
                count += 1

        if count:
            logger.info("Revoked %d sessions for user=%s", count, user_id)
        return count

    # ---------- CLEANUP ----------

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed since last sweep.

        Caller must hold self._lock.
        """
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now

        expired = [
            t for t, s in self._sessions.items() if not s.is_valid()
        ]
        for token in expired:
            del self._sessions[token]

        if expired:
            logger.info("Cleanup: removed %d expired sessions", len(expired))

    def cleanup(self) -> int:
        """Force cleanup of all expired sessions.

        Returns:
            Number of sessions removed.
        """
        with self._lock:
            expired = [
                t for t, s in self._sessions.items() if not s.is_valid()
            ]
            for token in expired:
                del self._sessions[token]
            self._last_cleanup = time.time()

        if expired:
            logger.info("Force cleanup: removed %d expired sessions", len(expired))
        return len(expired)

    @property
    def active_count(self) -> int:
        """Number of currently active (non-expired) sessions."""
        with self._lock:
            return sum(1 for s in self._sessions.values() if s.is_valid())


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    mgr = SessionManager(timeout_minutes=1)

    # Create session
    token = mgr.create_session("test_user", role="ADMIN")
    print(f"Token: {token[:20]}...")

    # Retrieve
    session = mgr.get_session(token)
    if session:
        print(f"User: {session.user_id}, Role: {session.role}, "
              f"Valid: {session.is_valid()}, Age: {session.age_seconds:.1f}s")

    # Active count
    print(f"Active sessions: {mgr.active_count}")

    # Revoke
    mgr.revoke(token)
    print(f"After revoke: {mgr.get_session(token)}")
    print(f"Active sessions: {mgr.active_count}")
