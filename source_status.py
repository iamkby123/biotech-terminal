"""
Source status and response metadata for API provenance tracking.

Every data-fetching function returns a SourceStatus alongside its data.
API responses include a 'meta' block with aggregated source status.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class SourceStatus:
    """Status of a single data source fetch."""
    source: str              # "yfinance", "clinicaltrials.gov", "rss:STAT News", "db"
    status: str              # "ok", "error", "timeout", "rate_limited", "partial"
    fetched_at: str          # ISO 8601
    record_count: int = 0
    error_message: Optional[str] = None
    data_mode: str = "live"  # "live", "cached", "snapshot"
    cache_hit: bool = False
    ttl_remaining: Optional[int] = None  # seconds remaining in cache

    def to_dict(self) -> dict:
        d = {
            "source": self.source,
            "status": self.status,
            "fetched_at": self.fetched_at,
            "record_count": self.record_count,
            "data_mode": self.data_mode,
        }
        if self.error_message:
            d["error_message"] = self.error_message
        if self.cache_hit:
            d["cache_hit"] = True
            if self.ttl_remaining is not None:
                d["ttl_remaining"] = self.ttl_remaining
        return d

    @classmethod
    def ok(
        cls,
        source: str,
        count: int,
        data_mode: str = "live",
        cache_hit: bool = False,
        ttl_remaining: int | None = None,
    ) -> SourceStatus:
        return cls(
            source=source,
            status="ok",
            fetched_at=_now_iso(),
            record_count=count,
            data_mode=data_mode,
            cache_hit=cache_hit,
            ttl_remaining=ttl_remaining,
        )

    @classmethod
    def error(cls, source: str, message: str, data_mode: str = "live") -> SourceStatus:
        return cls(
            source=source,
            status="error",
            fetched_at=_now_iso(),
            error_message=message,
            data_mode=data_mode,
        )

    @classmethod
    def empty(cls, source: str, data_mode: str = "snapshot") -> SourceStatus:
        """Source returned successfully but had no data."""
        return cls(
            source=source,
            status="ok",
            fetched_at=_now_iso(),
            record_count=0,
            data_mode=data_mode,
        )


@dataclass
class ResponseMeta:
    """Metadata block included in API responses."""
    as_of: str                             # ISO 8601
    data_mode: str                         # "live", "cached", "hybrid", "snapshot"
    source_status: list[SourceStatus] = field(default_factory=list)
    completeness_score: float = 1.0        # 0.0 to 1.0

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "data_mode": self.data_mode,
            "source_status": [s.to_dict() for s in self.source_status],
            "completeness_score": self.completeness_score,
        }


def compute_completeness(statuses: list[SourceStatus]) -> float:
    """Score between 0.0 and 1.0 based on how many sources returned ok."""
    if not statuses:
        return 0.0
    ok_count = sum(1 for s in statuses if s.status == "ok")
    return round(ok_count / len(statuses), 2)


def build_response_meta(statuses: list[SourceStatus]) -> ResponseMeta:
    """Build ResponseMeta from a list of SourceStatus objects."""
    modes = {s.data_mode for s in statuses} if statuses else {"live"}
    if len(modes) == 1:
        data_mode = modes.pop()
    elif "live" in modes and ("cached" in modes or "snapshot" in modes):
        data_mode = "hybrid"
    else:
        data_mode = "mixed"
    return ResponseMeta(
        as_of=_now_iso(),
        data_mode=data_mode,
        source_status=statuses,
        completeness_score=compute_completeness(statuses),
    )


def wrap_response_flat(data: dict, statuses: list[SourceStatus]) -> dict:
    """
    Add 'meta' to an API response dict without nesting 'data'.

    Backward compatible: the frontend reads top-level keys like
    response.trials, response.news, etc. The 'meta' key is additive —
    the frontend ignores unknown keys.
    """
    meta = build_response_meta(statuses)
    return {**data, "meta": meta.to_dict()}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
