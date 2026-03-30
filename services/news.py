"""
News service — paginated queries, stats, and sources.

Extracts /api/news, /api/news/stats, /api/news/sources from api.py.
Uses bounded TTL cache.
"""
from __future__ import annotations

import logging
from datetime import datetime

from cache import news_api_cache
from models.ticker_map import get_connection
from source_status import SourceStatus

logger = logging.getLogger(__name__)


def _cache_key(prefix: str, **kwargs) -> str:
    return prefix + "|" + "&".join(
        f"{k}={v}" for k, v in sorted(kwargs.items()) if v is not None
    )


def query_news(
    page: int = 1,
    per_page: int = 20,
    category: str | None = None,
    source: str | None = None,
    ticker: str | None = None,
    catalyst: str | None = None,
    sentiment: str | None = None,
    search: str | None = None,
) -> tuple[dict, SourceStatus]:
    """
    Paginated news query with filtering.
    Returns (result_dict, SourceStatus).
    """
    ck = _cache_key(
        "news",
        page=page,
        per_page=per_page,
        category=category,
        source=source,
        ticker=ticker,
        catalyst=catalyst,
        sentiment=sentiment,
        search=search,
    )

    cached = news_api_cache.get(ck)
    if cached is not None:
        value, meta = cached
        return value, SourceStatus.ok(
            "db:news", value.get("total", 0),
            data_mode="cached",
            cache_hit=True,
            ttl_remaining=meta.ttl_remaining,
        )

    per_page = min(per_page, 100)
    offset = (max(page, 1) - 1) * per_page

    conn = get_connection()
    cur = conn.cursor()
    try:
        where, params = [], []
        if category:
            where.append("category = %s")
            params.append(category)
        if source:
            where.append("source ILIKE %s")
            params.append(f"%{source}%")
        if ticker:
            where.append("tickers ILIKE %s")
            params.append(f"%{ticker.upper()}%")
        if catalyst:
            where.append("catalyst_type = %s")
            params.append(catalyst)
        if sentiment:
            where.append("sentiment = %s")
            params.append(sentiment)
        if search:
            where.append("(title ILIKE %s OR summary ILIKE %s)")
            params.extend([f"%{search}%", f"%{search}%"])

        clause = ("WHERE " + " AND ".join(where)) if where else ""

        # Count total
        cur.execute(f"SELECT COUNT(*) AS total FROM news {clause}", params)
        total = cur.fetchone()["total"]

        # Fetch page
        cur.execute(
            f"""
            SELECT id, source, category, title, summary, url, published_at, tickers,
                   nct_id, ingested_at, sentiment, sentiment_score, catalyst_type, image_url
            FROM news {clause}
            ORDER BY published_at DESC NULLS LAST, ingested_at DESC
            LIMIT %s OFFSET %s
        """,
            params + [per_page, offset],
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get("ingested_at"):
                r["ingested_at"] = str(r["ingested_at"])
            if r.get("sentiment_score") is not None:
                r["sentiment_score"] = float(r["sentiment_score"])

        cur.close()
        result = {
            "items": rows,
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": max(1, -(-total // per_page)),
        }
        news_api_cache.set(ck, result)
        return result, SourceStatus.ok("db:news", total, data_mode="snapshot")
    except Exception as e:
        logger.error(f"query_news failed: {e}")
        return {"items": [], "total": 0, "page": 1, "per_page": per_page, "pages": 1}, SourceStatus.error("db:news", str(e))
    finally:
        conn.close()


def get_news_stats() -> tuple[dict, SourceStatus]:
    """Get filter counts for the news sidebar."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE sentiment = 'positive') AS positive,
                COUNT(*) FILTER (WHERE sentiment = 'negative') AS negative,
                COUNT(*) FILTER (WHERE sentiment = 'neutral') AS neutral
            FROM news
        """)
        sentiment_counts = dict(cur.fetchone())

        cur.execute("""
            SELECT catalyst_type, COUNT(*) AS count
            FROM news WHERE catalyst_type IS NOT NULL
            GROUP BY catalyst_type ORDER BY count DESC
        """)
        catalyst_counts = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT source, COUNT(*) AS count
            FROM news GROUP BY source ORDER BY count DESC
        """)
        source_counts = [dict(r) for r in cur.fetchall()]

        cur.close()
        return (
            {
                "sentiment": sentiment_counts,
                "catalysts": catalyst_counts,
                "sources": source_counts,
            },
            SourceStatus.ok("db:news", sentiment_counts.get("total", 0), data_mode="snapshot"),
        )
    except Exception as e:
        logger.error(f"get_news_stats failed: {e}")
        return {}, SourceStatus.error("db:news", str(e))
    finally:
        conn.close()


def get_news_sources() -> tuple[list[dict], SourceStatus]:
    """Get news source breakdown with counts."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT source, category, COUNT(*) AS count,
                   MAX(published_at) AS latest
            FROM news
            GROUP BY source, category
            ORDER BY count DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows, SourceStatus.ok("db:news", len(rows), data_mode="snapshot")
    except Exception as e:
        logger.error(f"get_news_sources failed: {e}")
        return [], SourceStatus.error("db:news", str(e))
    finally:
        conn.close()
