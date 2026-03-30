"""
Bounded in-memory TTL cache with LRU eviction and metadata.

Replaces the unbounded _yf_news_cache and _news_cache dicts in api.py.
Thread-safe via threading.Lock. Uses OrderedDict for O(1) LRU eviction.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CacheMetadata:
    """Metadata about a cached entry."""
    stored_at: float      # time.monotonic() when stored
    ttl: int              # seconds
    key: str

    @property
    def ttl_remaining(self) -> int:
        """Seconds remaining before expiry. 0 if expired."""
        elapsed = time.monotonic() - self.stored_at
        return max(0, int(self.ttl - elapsed))

    @property
    def is_expired(self) -> bool:
        return self.ttl_remaining <= 0


class TTLCache:
    """
    Thread-safe bounded TTL cache with LRU eviction.

    - Entries expire after their TTL
    - When max_size is reached, oldest entries are evicted (LRU)
    - get() returns (value, CacheMetadata) or None
    - Thread-safe for concurrent FastAPI request handlers
    """

    def __init__(self, max_size: int = 256, default_ttl: int = 300):
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, tuple[Any, CacheMetadata]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> tuple[Any, CacheMetadata] | None:
        """
        Get a cached value by key.

        Returns (value, CacheMetadata) if found and not expired.
        Returns None if not found or expired.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, meta = entry
            if meta.is_expired:
                del self._store[key]
                self._misses += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            return value, meta

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value with optional custom TTL."""
        with self._lock:
            # Remove existing entry if present
            if key in self._store:
                del self._store[key]
            # Evict oldest if at capacity
            while len(self._store) >= self._max_size:
                self._store.popitem(last=False)
                self._evictions += 1
            meta = CacheMetadata(
                stored_at=time.monotonic(),
                ttl=ttl if ttl is not None else self._default_ttl,
                key=key,
            )
            self._store[key] = (value, meta)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key. Returns True if it was present."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        """Return cache performance statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
                "evictions": self._evictions,
                "size": len(self._store),
                "max_size": self._max_size,
                "default_ttl": self._default_ttl,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ── Pre-configured cache instances ──────────────────────────────────────
# Each has a bounded size and TTL appropriate for its data type.

# Stock quotes: small TTL (60s), frequent access
stock_cache = TTLCache(max_size=200, default_ttl=60)

# yfinance news per ticker: medium TTL (5 min), moderate size
yf_news_cache = TTLCache(max_size=200, default_ttl=300)

# Full company detail responses: medium TTL (2 min), large payloads
company_cache = TTLCache(max_size=100, default_ttl=120)

# Earnings calendar: long TTL (10 min), slow to compute
earnings_cache = TTLCache(max_size=50, default_ttl=600)

# Paginated news API responses: medium TTL (2 min)
news_api_cache = TTLCache(max_size=100, default_ttl=120)
