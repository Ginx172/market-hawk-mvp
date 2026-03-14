"""
Query cache for Knowledge Advisor.
Caches retrieval results to avoid redundant ChromaDB queries.
Uses an LRU cache with configurable TTL and max size.
"""
import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class CacheEntry:
    """A single entry stored in the query cache."""

    results: List[Any]
    timestamp: float
    hit_count: int = 0


class QueryCache:
    """LRU cache for RAG query results with TTL expiry."""

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0) -> None:
        """
        Initialize the query cache.

        Args:
            max_size: Maximum number of entries before LRU eviction.
            ttl_seconds: Seconds before a cached entry is considered stale.
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, query: str, n_results: int) -> str:
        """Produce a short, stable cache key from *query* and *n_results*."""
        normalized = query.strip().lower()
        return hashlib.sha256(f"{normalized}:{n_results}".encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str, n_results: int) -> Optional[List[Any]]:
        """
        Return cached results for *query*/*n_results*, or ``None`` on miss/expiry.

        Args:
            query: The retrieval question.
            n_results: Number of results that was requested.

        Returns:
            Cached list of results, or ``None`` if not found / expired.
        """
        key = self._make_key(query, n_results)
        if key in self._cache:
            entry = self._cache[key]
            if (time.time() - entry.timestamp) < self._ttl:
                self._hits += 1
                entry.hit_count += 1
                self._cache.move_to_end(key)
                return entry.results
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, query: str, n_results: int, results: List[Any]) -> None:
        """
        Store *results* in the cache under *query*/*n_results*.

        Evicts the least-recently-used entry when the cache is full.

        Args:
            query: The retrieval question.
            n_results: Number of results returned.
            results: The list of results to cache.
        """
        key = self._make_key(query, n_results)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(results=results, timestamp=time.time())
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(results=results, timestamp=time.time())

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._cache.clear()

    def stats(self) -> dict:
        """
        Return cache performance statistics.

        Returns:
            Dict with keys: size, max_size, hits, misses, hit_rate, ttl_seconds.
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "ttl_seconds": self._ttl,
        }
