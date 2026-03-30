"""
Clinical trials service — queries, stats, and pipeline.

Extracts trial-related endpoints from api.py.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from models.ticker_map import get_connection, release_connection
from source_status import SourceStatus

logger = logging.getLogger(__name__)

# Simple in-memory cache for trial queries
_trials_cache = {}  # key -> (timestamp, data)
_CACHE_TTL = 300  # 5 minutes


def query_trials(
    limit: int = 100,
    phase: str | None = None,
    status: str | None = None,
    sponsor: str | None = None,
) -> tuple[list[dict], SourceStatus]:
    """Query trials with optional filters. Cached for 5 minutes."""
    cache_key = f"{limit}:{phase}:{status}:{sponsor}"
    now = time.time()

    # Return cached result if fresh
    if cache_key in _trials_cache:
        ts, cached_rows = _trials_cache[cache_key]
        if now - ts < _CACHE_TTL:
            return cached_rows, SourceStatus.ok("db:trials:cached", len(cached_rows), data_mode="snapshot")

    conn = get_connection()
    cur = conn.cursor()
    try:
        where, params = [], []
        if phase:
            where.append("phase ILIKE %s")
            params.append(f"%{phase}%")
        if status:
            where.append("status ILIKE %s")
            params.append(f"%{status}%")
        if sponsor:
            where.append("sponsor ILIKE %s")
            params.append(f"%{sponsor}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""
            SELECT nct_id, title, sponsor, phase, status, condition,
                   enrollment, primary_completion_date, results_posted
            FROM trials {clause}
            ORDER BY phase DESC NULLS LAST, status
            LIMIT %s
        """,
            params + [limit],
        )
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()

        # Cache the result
        _trials_cache[cache_key] = (now, rows)

        return rows, SourceStatus.ok("db:trials", len(rows), data_mode="snapshot")
    except Exception as e:
        logger.error(f"query_trials failed: {e}")
        return [], SourceStatus.error("db:trials", str(e))
    finally:
        release_connection(conn)


_stats_cache = None
_stats_cache_time = 0

def get_trials_stats() -> tuple[dict, SourceStatus]:
    """Get trial statistics grouped by phase/status and top sponsors. Cached 5 min."""
    global _stats_cache, _stats_cache_time
    now = time.time()
    if _stats_cache and now - _stats_cache_time < _CACHE_TTL:
        return _stats_cache, SourceStatus.ok("db:trials:cached", 0, data_mode="snapshot")

    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT phase, status, COUNT(*) AS cnt
            FROM trials GROUP BY phase, status
            ORDER BY phase DESC NULLS LAST, cnt DESC
        """)
        phase_status = [dict(r) for r in cur.fetchall()]

        cur.execute("""
            SELECT sponsor, COUNT(*) AS total,
                   SUM(CASE WHEN status = 'RECRUITING' THEN 1 ELSE 0 END) AS recruiting,
                   SUM(CASE WHEN results_posted = 1 THEN 1 ELSE 0 END) AS with_results
            FROM trials WHERE sponsor IS NOT NULL
            GROUP BY sponsor ORDER BY total DESC LIMIT 20
        """)
        sponsors = [dict(r) for r in cur.fetchall()]

        cur.close()
        result = {"phase_status": phase_status, "sponsors": sponsors}
        _stats_cache = result
        _stats_cache_time = now
        return (
            result,
            SourceStatus.ok("db:trials", len(phase_status), data_mode="snapshot"),
        )
    except Exception as e:
        logger.error(f"get_trials_stats failed: {e}")
        return {}, SourceStatus.error("db:trials", str(e))
    finally:
        release_connection(conn)


def get_pipeline(ticker: str) -> tuple[dict, list[SourceStatus]]:
    """Get full pipeline view for a ticker (trials, filings, catalysts)."""
    ticker = ticker.upper()
    statuses: list[SourceStatus] = []
    conn = get_connection()

    try:
        import psycopg2.extensions

        conn.set_session(readonly=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = 5000")
        cur.close()

        cur = conn.cursor()

        # Lookup ticker_map
        cur.execute("SELECT * FROM ticker_map WHERE ticker = %s", (ticker,))
        info = cur.fetchone()
        if not info:
            cur.close()
            return (
                {"error": f"{ticker} not found"},
                [SourceStatus.error("db:ticker_map", f"{ticker} not found")],
            )
        info = dict(info)

        # Trials — try ticker match first, then company name
        cur.execute(
            """
            SELECT nct_id, title, phase, status, condition, intervention,
                   primary_completion_date, results_posted, enrollment
            FROM trials WHERE sponsor ILIKE %s
            ORDER BY phase DESC NULLS LAST, status
            LIMIT 100
        """,
            (f"%{ticker}%",),
        )
        trials = [dict(r) for r in cur.fetchall()]

        if not trials and info.get("company_name"):
            first_word = info["company_name"].split()[0]
            cur.execute(
                """
                SELECT nct_id, title, phase, status, condition, intervention,
                       primary_completion_date, results_posted, enrollment
                FROM trials WHERE sponsor ILIKE %s
                ORDER BY phase DESC NULLS LAST, status
                LIMIT 100
            """,
                (f"%{first_word}%",),
            )
            trials = [dict(r) for r in cur.fetchall()]

        statuses.append(SourceStatus.ok("db:trials", len(trials), data_mode="snapshot"))

        # Filings
        cur.execute(
            """
            SELECT form_type, filed_date, filing_url FROM filings
            WHERE ticker = %s ORDER BY filed_date DESC LIMIT 10
        """,
            (ticker,),
        )
        filings = [dict(r) for r in cur.fetchall()]
        statuses.append(SourceStatus.ok("db:filings", len(filings), data_mode="snapshot"))

        # Catalysts
        cur.execute(
            """
            SELECT event_date, event_type, drug_name, indication FROM catalysts
            WHERE ticker = %s ORDER BY event_date
        """,
            (ticker,),
        )
        cats = [dict(r) for r in cur.fetchall()]
        statuses.append(SourceStatus.ok("db:catalysts", len(cats), data_mode="snapshot"))

        cur.close()
        return (
            {"info": info, "trials": trials, "filings": filings, "catalysts": cats},
            statuses,
        )
    except Exception as e:
        logger.error(f"get_pipeline failed [{ticker}]: {e}")
        return {"error": str(e)}, [SourceStatus.error("db", str(e))]
    finally:
        release_connection(conn)


def get_trial_detail(nct_id: str) -> tuple[dict | None, SourceStatus]:
    """Get detailed trial information by NCT ID."""
    nct_id = nct_id.upper()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT nct_id, title, sponsor, phase, status, condition, intervention,
                   enrollment, start_date, primary_completion_date, results_posted,
                   brief_summary, detailed_description, locations, principal_investigator,
                   eligibility_criteria, study_type, study_design, why_stopped
            FROM trials WHERE nct_id = %s
        """,
            (nct_id,),
        )
        trial = cur.fetchone()
        cur.close()

        if not trial:
            return None, SourceStatus.empty("db:trials")
        return dict(trial), SourceStatus.ok("db:trials", 1, data_mode="snapshot")
    except Exception as e:
        logger.error(f"get_trial_detail failed [{nct_id}]: {e}")
        return None, SourceStatus.error("db:trials", str(e))
    finally:
        release_connection(conn)
