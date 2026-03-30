"""
Ticker map database operations.

Infrastructure: get_connection, init_db, build_ticker_map, get_ticker_info, etc.
Entity resolution is delegated to models.entity_resolver.EntityResolver.

Legacy functions (normalize_name, fuzzy_match_sponsor, resolve_sponsor_to_ticker)
are kept as thin adapters for backward compatibility but delegate to EntityResolver.
"""
import httpx
import time
import logging
import psycopg2
import psycopg2.extras
from datetime import datetime
from config import (
    EDGAR_COMPANY_TICKERS,
    SEC_USER_AGENT,
    DATABASE_URL,
    TARGET_TICKERS,
)
from normalize import normalize_sponsor

logger = logging.getLogger(__name__)


def get_connection() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, "r") as f:
        schema = f.read()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(schema)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database initialized on Neon.")


def fetch_sec_ticker_map() -> dict[str, dict]:
    headers = {"User-Agent": SEC_USER_AGENT}
    with httpx.Client(timeout=30) as client:
        r = client.get(EDGAR_COMPANY_TICKERS, headers=headers)
        r.raise_for_status()
    raw = r.json()
    result = {}
    for entry in raw.values():
        ticker = entry.get("ticker", "").upper()
        cik = str(entry.get("cik_str", "")).zfill(10)
        name = entry.get("title", "")
        if ticker:
            result[ticker] = {"cik": cik, "company_name": name}
    return result


# ── Legacy adapters (delegate to EntityResolver) ──────────────────────


def normalize_name(name: str) -> str:
    """DEPRECATED: Use normalize.normalize_sponsor() instead."""
    return normalize_sponsor(name) or name.strip().lower()


def fuzzy_match_sponsor(sponsor: str, name_to_ticker: dict[str, str], cutoff=0.75) -> str | None:
    """DEPRECATED: Use EntityResolver.resolve_sponsor() instead."""
    from models.entity_resolver import resolver
    if resolver.is_loaded:
        result = resolver.resolve_sponsor(sponsor)
        if result.is_match:
            return result.ticker
    # Fallback to old difflib approach if resolver not loaded
    import difflib
    norm_sponsor = normalize_sponsor(sponsor) or sponsor.strip().lower()
    candidates = list(name_to_ticker.keys())
    matches = difflib.get_close_matches(norm_sponsor, candidates, n=1, cutoff=cutoff)
    if matches:
        return name_to_ticker[matches[0]]
    return None


def resolve_sponsor_to_ticker(sponsor: str) -> str | None:
    """DEPRECATED: Use EntityResolver.resolve_sponsor() instead."""
    from models.entity_resolver import resolver
    if resolver.is_loaded:
        result = resolver.resolve_sponsor(sponsor)
        if result.is_match:
            return result.ticker
        return None
    # Fallback: old DB-based resolution if resolver not loaded
    from config import MANUAL_TICKER_OVERRIDES
    for override_name, ticker in MANUAL_TICKER_OVERRIDES.items():
        if override_name.lower() in sponsor.lower():
            return ticker
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ticker, company_name, ctgov_sponsor_name FROM ticker_map")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    name_to_ticker: dict[str, str] = {}
    for row in rows:
        if row["company_name"]:
            name_to_ticker[normalize_name(row["company_name"])] = row["ticker"]
        if row["ctgov_sponsor_name"]:
            name_to_ticker[normalize_name(row["ctgov_sponsor_name"])] = row["ticker"]
    return fuzzy_match_sponsor(sponsor, name_to_ticker)


# ── Active infrastructure ─────────────────────────────────────────────


def build_ticker_map(tickers: list[str] | None = None) -> int:
    tickers = tickers or TARGET_TICKERS
    logger.info("Fetching SEC company ticker map...")
    sec_map = fetch_sec_ticker_map()

    conn = get_connection()
    cur = conn.cursor()
    upserted = 0

    for ticker in tickers:
        ticker = ticker.upper()
        info = sec_map.get(ticker)
        if not info:
            logger.warning(f"Ticker {ticker} not found in SEC map.")
            cur.execute(
                """INSERT INTO ticker_map (ticker, updated_at)
                   VALUES (%s, %s)
                   ON CONFLICT (ticker) DO UPDATE SET updated_at = EXCLUDED.updated_at""",
                (ticker, datetime.utcnow()),
            )
            continue

        cur.execute(
            """INSERT INTO ticker_map (ticker, company_name, cik, updated_at)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (ticker) DO UPDATE SET
                   company_name = EXCLUDED.company_name,
                   cik = EXCLUDED.cik,
                   updated_at = EXCLUDED.updated_at""",
            (ticker, info["company_name"], info["cik"], datetime.utcnow()),
        )
        upserted += 1
        time.sleep(0.01)

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Ticker map built: {upserted}/{len(tickers)} tickers resolved.")
    return upserted


def update_ctgov_sponsor(ticker: str, ctgov_sponsor_name: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE ticker_map SET ctgov_sponsor_name = %s WHERE ticker = %s",
        (ctgov_sponsor_name, ticker),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_ticker_info(ticker: str) -> dict | None:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ticker_map WHERE ticker = %s", (ticker.upper(),))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return dict(row) if row else None


def get_all_tickers() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM ticker_map")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]
