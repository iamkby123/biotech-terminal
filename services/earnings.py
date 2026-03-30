"""
Earnings calendar service.

Extracts /api/earnings logic from api.py.
Uses bounded TTL cache for expensive yfinance calendar lookups.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import yfinance as yf

from cache import earnings_cache
from models.ticker_map import get_connection
from normalize import safe_float
from source_status import SourceStatus

logger = logging.getLogger(__name__)


def fetch_earnings(
    days: int = 90, ticker: str | None = None
) -> tuple[list[dict], list[SourceStatus]]:
    """
    Get upcoming earnings calendar for biotech companies.
    Returns (list[earnings_entry], list[SourceStatus]).
    """
    cache_key = f"earnings:{ticker or 'all'}:{days}"
    cached = earnings_cache.get(cache_key)
    if cached is not None:
        value, meta = cached
        return value, [
            SourceStatus.ok(
                "yfinance:earnings",
                len(value),
                data_mode="cached",
                cache_hit=True,
                ttl_remaining=meta.ttl_remaining,
            )
        ]

    statuses: list[SourceStatus] = []

    conn = get_connection()
    cur = conn.cursor()
    try:
        if ticker:
            cur.execute(
                """
                SELECT DISTINCT ticker, company_name FROM ticker_map
                WHERE ticker = %s LIMIT 1
            """,
                (ticker.upper(),),
            )
        else:
            cur.execute(
                "SELECT DISTINCT ticker, company_name FROM ticker_map ORDER BY ticker LIMIT 50"
            )

        companies = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.error(f"fetch_earnings DB failed: {e}")
        return [], [SourceStatus.error("db:ticker_map", str(e))]
    finally:
        conn.close()

    statuses.append(SourceStatus.ok("db:ticker_map", len(companies), data_mode="snapshot"))

    earnings: list[dict] = []
    today = datetime.now()

    for company in companies:
        ticker_sym = company["ticker"]
        company_name = company["company_name"]

        try:
            stock = yf.Ticker(ticker_sym)

            # Try to get earnings dates from yfinance calendar
            cal = stock.calendar
            earnings_date = None
            eps_estimate = None
            revenue_estimate = None

            if cal is not None and isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and len(ed) > 0:
                        earnings_date = ed[0]
                    elif hasattr(ed, "strftime"):
                        earnings_date = ed
                eps_est = cal.get("Earnings Average") or cal.get("EPS Estimate")
                if eps_est is not None:
                    eps_estimate = (
                        float(eps_est)
                        if not isinstance(eps_est, list)
                        else float(eps_est[0]) if eps_est else None
                    )
                rev_est = cal.get("Revenue Average") or cal.get("Revenue Estimate")
                if rev_est is not None:
                    revenue_estimate = (
                        float(rev_est)
                        if not isinstance(rev_est, list)
                        else float(rev_est[0]) if rev_est else None
                    )

            # Fallback: earnings_dates property
            if earnings_date is None:
                try:
                    ed_df = stock.earnings_dates
                    if ed_df is not None and len(ed_df) > 0:
                        future_dates = [
                            d
                            for d in ed_df.index
                            if d.to_pydatetime().replace(tzinfo=None) >= today
                        ]
                        if future_dates:
                            earnings_date = min(future_dates).to_pydatetime()
                            row = ed_df.loc[min(future_dates)]
                            if eps_estimate is None and "EPS Estimate" in ed_df.columns:
                                val = row.get("EPS Estimate")
                                if val is not None and str(val) != "nan":
                                    eps_estimate = float(val)
                except Exception:
                    pass

            if earnings_date is not None:
                if hasattr(earnings_date, "to_pydatetime"):
                    earnings_date = earnings_date.to_pydatetime()
                if hasattr(earnings_date, "tzinfo") and earnings_date.tzinfo is not None:
                    earnings_date = earnings_date.replace(tzinfo=None)
                if type(earnings_date) is date and not isinstance(earnings_date, datetime):
                    earnings_date = datetime(
                        earnings_date.year, earnings_date.month, earnings_date.day
                    )

                days_away = (earnings_date - today).days
                if 0 <= days_away <= days:
                    month = earnings_date.month
                    if month <= 3:
                        quarter = f"Q4 {earnings_date.year - 1}"
                    elif month <= 6:
                        quarter = f"Q1 {earnings_date.year}"
                    elif month <= 9:
                        quarter = f"Q2 {earnings_date.year}"
                    else:
                        quarter = f"Q3 {earnings_date.year}"

                    entry = {
                        "ticker": ticker_sym,
                        "company_name": company_name,
                        "quarter": quarter,
                        "earnings_date": earnings_date.strftime("%Y-%m-%d"),
                        "days_away": days_away,
                        "status": "confirmed",
                    }
                    if eps_estimate is not None:
                        entry["eps_estimate"] = round(eps_estimate, 2)
                    if revenue_estimate is not None:
                        entry["revenue_estimate"] = round(revenue_estimate, 2)

                    earnings.append(entry)
        except Exception:
            pass

    earnings.sort(key=lambda x: x["earnings_date"])
    statuses.append(SourceStatus.ok("yfinance:earnings", len(earnings)))

    earnings_cache.set(cache_key, earnings)
    return earnings, statuses
