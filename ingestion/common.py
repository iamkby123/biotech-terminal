"""
Shared ingestion utilities.

Deduplicates the _log_ingestion function that was copy-pasted
across clinicaltrials.py, edgar.py, fda_calendar.py, and pubmed.py.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from models.ticker_map import get_connection

logger = logging.getLogger(__name__)


def log_ingestion(
    source: str,
    ticker: str,
    fetched: int,
    upserted: int,
    status: str,
    error: str | None,
    started: str | datetime,
) -> None:
    """
    Write an ingestion log entry to the database.

    Args:
        source: Data source identifier (e.g., "clinicaltrials", "edgar")
        ticker: Ticker or identifier being ingested
        fetched: Number of rows fetched from source
        upserted: Number of rows upserted to database
        status: "ok" or "error"
        error: Error message if status is "error", else None
        started: ISO timestamp or datetime of when ingestion started
    """
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO ingestion_log
               (source, ticker, rows_fetched, rows_upserted, status, error_message, started_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (source, ticker, fetched, upserted, status, error, started),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log ingestion [{source}/{ticker}]: {e}")
