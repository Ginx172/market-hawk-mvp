"""
SCRIPT NAME: crypto_log.py
====================================
Execution Location: market-hawk-mvp/trading/
Purpose: Optional Fernet encryption/decryption for JSONL trade logs.
Creation Date: 2026-03-07

If MH_ENCRYPTION_KEY is set in the environment, trade log lines are
encrypted before writing and decrypted on read.  If the key is absent,
all operations fall through to plain text (backward compatible).

Usage:
    from trading.crypto_log import write_log_line, read_log_lines

    # Write (encrypts if key is set)
    write_log_line(path, json_string)

    # Read (decrypts if key is set)
    for line in read_log_lines(path):
        data = json.loads(line)
"""

import os
import logging
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("market_hawk.crypto_log")

_fernet: Optional[object] = None
_initialized: bool = False


def _init_fernet() -> None:
    """Lazy-init Fernet cipher from MH_ENCRYPTION_KEY env var."""
    global _fernet, _initialized
    if _initialized:
        return
    _initialized = True

    key = os.environ.get("MH_ENCRYPTION_KEY", "")
    if not key:
        logger.debug("MH_ENCRYPTION_KEY not set — trade logs will be plain text")
        return

    try:
        from cryptography.fernet import Fernet
        # Validare: Fernet accepta doar 32-byte url-safe base64 keys
        _fernet = Fernet(key.encode("utf-8"))
        logger.info("Trade log encryption enabled (Fernet)")
    except ImportError:
        logger.warning("cryptography package not installed — trade logs will be plain text")
    except Exception:
        logger.exception("Invalid MH_ENCRYPTION_KEY — trade logs will be plain text")


def is_encryption_enabled() -> bool:
    """Return True if Fernet encryption is active."""
    _init_fernet()
    return _fernet is not None


def write_log_line(filepath: Path, line: str) -> None:
    """Append a single line to a log file, encrypting if key is available.

    Args:
        filepath: Path to the .jsonl log file.
        line: A single JSON string (no trailing newline).
    """
    _init_fernet()
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if _fernet is not None:
        encrypted = _fernet.encrypt(line.encode("utf-8"))  # type: ignore[union-attr]
        with open(filepath, "ab") as f:
            f.write(encrypted + b"\n")
    else:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_log_lines(filepath: Path) -> Iterator[str]:
    """Read all lines from a log file, decrypting if needed.

    Yields:
        Decrypted/plain JSON strings, one per line.
    """
    _init_fernet()

    if not filepath.exists():
        return

    if _fernet is not None:
        with open(filepath, "rb") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    decrypted = _fernet.decrypt(raw_line)  # type: ignore[union-attr]
                    yield decrypted.decode("utf-8")
                except Exception as e:
                    # Linia poate fi plain text dintr-un log vechi pre-encryption
                    try:
                        yield raw_line.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.warning("Skipping unreadable log line: %s", e)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
