"""
Company detail service — aggregates all data for /api/company/{ticker}.

Extracts the 235-line company_detail endpoint from api.py into
focused sub-functions. Uses news_ranking for deterministic news ordering.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import yfinance as yf

from cache import company_cache
from models.canonical import NewsArticle, SourceMeta
from models.ticker_map import get_connection
from news_ranking import rank_news
from normalize import safe_float
from services.stock import fetch_yf_news
from source_status import SourceStatus

logger = logging.getLogger(__name__)


def get_company_detail(ticker: str) -> tuple[dict, list[SourceStatus]]:
    """
    Comprehensive company data: info, trials, filings, news (ranked),
    drugs, insider trades, holdings, competitors, management.

    Returns (response_dict, list[SourceStatus]).
    """
    ticker = ticker.upper()
    statuses: list[SourceStatus] = []

    # Check cache
    cached = company_cache.get(ticker)
    if cached is not None:
        value, meta = cached
        return value, [
            SourceStatus.ok(
                "company_cache",
                1,
                data_mode="cached",
                cache_hit=True,
                ttl_remaining=meta.ttl_remaining,
            )
        ]

    conn = get_connection()
    try:
        cur = conn.cursor()
        stock = yf.Ticker(ticker)

        # ── Ticker map info ──
        cur.execute("SELECT * FROM ticker_map WHERE ticker = %s", (ticker,))
        info = cur.fetchone()
        if not info:
            cur.close()
            return (
                {"error": f"{ticker} not found"},
                [SourceStatus.error("db:ticker_map", f"{ticker} not found")],
            )
        info = dict(info)
        statuses.append(SourceStatus.ok("db:ticker_map", 1, data_mode="snapshot"))

        # ── Trials (pipeline) ──
        sponsor_name = info.get("ctgov_sponsor_name") or info.get("company_name") or ticker
        # Try exact match first, then fall back to first-word fuzzy match
        cur.execute(
            """
            SELECT nct_id, title, phase, status, condition, intervention,
                   primary_completion_date, results_posted
            FROM trials WHERE sponsor ILIKE %s
            ORDER BY phase DESC NULLS LAST
            LIMIT 20
        """,
            (f"%{sponsor_name}%",),
        )
        trials = [dict(r) for r in cur.fetchall()]
        # Fallback: if exact match found nothing, try first word
        if not trials:
            first_word = sponsor_name.split()[0]
            cur.execute(
                """
                SELECT nct_id, title, phase, status, condition, intervention,
                       primary_completion_date, results_posted
                FROM trials WHERE sponsor ILIKE %s
                ORDER BY phase DESC NULLS LAST
                LIMIT 20
            """,
                (f"%{first_word}%",),
            )
            trials = [dict(r) for r in cur.fetchall()]
        statuses.append(SourceStatus.ok("db:trials", len(trials), data_mode="snapshot"))

        # ── Filings (deduplicated) ──
        cur.execute(
            """
            SELECT DISTINCT ON (form_type, filed_date) form_type, filed_date, filing_url
            FROM filings WHERE ticker = %s
            ORDER BY form_type, filed_date DESC
            LIMIT 10
        """,
            (ticker,),
        )
        filings = sorted(
            [dict(r) for r in cur.fetchall()],
            key=lambda f: f.get("filed_date", ""),
            reverse=True,
        )
        statuses.append(SourceStatus.ok("db:filings", len(filings), data_mode="snapshot"))

        # ── News: hybrid yfinance (live) + DB (Finnhub/RSS), ranked ──
        company_name = info.get("company_name") or info.get("ctgov_sponsor_name") or ""
        yf_articles, yf_status = fetch_yf_news(ticker, company_name)
        statuses.append(yf_status)

        cur.execute(
            """
            SELECT title, source, published_at, url, summary, sentiment,
                   sentiment_score, catalyst_type, image_url, tickers
            FROM news WHERE tickers ILIKE %s
            ORDER BY published_at DESC LIMIT 15
        """,
            (f"%{ticker}%",),
        )
        db_rows = cur.fetchall()
        db_articles = [
            NewsArticle(
                title=r["title"],
                source=r.get("source"),
                published_at=r.get("published_at"),
                url=r.get("url"),
                summary=r.get("summary"),
                sentiment=r.get("sentiment"),
                sentiment_score=float(r["sentiment_score"]) if r.get("sentiment_score") is not None else None,
                catalyst_type=r.get("catalyst_type"),
                image_url=r.get("image_url"),
                tickers=r.get("tickers"),
                origin="db",
                _meta=SourceMeta(
                    source_name="db",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    data_mode="snapshot",
                ),
            )
            for r in db_rows
        ]
        statuses.append(SourceStatus.ok("db:news", len(db_articles), data_mode="snapshot"))

        # Rank and deduplicate all articles
        all_articles = yf_articles + db_articles
        ranked = rank_news(all_articles, ticker, limit=15)
        news = [a.to_api_dict() for a in ranked]

        # ── Drugs (from trial interventions) ──
        drugs = _extract_drugs(trials)

        # ── Insider trades (yfinance live) ──
        insider_trades, insider_status = _fetch_insider_trades(stock)
        statuses.append(insider_status)

        # ── Institutional holdings (yfinance live) ──
        holdings, holdings_status = _fetch_institutional_holdings(stock)
        statuses.append(holdings_status)

        # ── Drug competitors ──
        drug_competitors = _fetch_drug_competitors(cur, drugs)

        # ── Company competitors ──
        company_competitors = _fetch_company_competitors(info)

        # ── Management ──
        management, mgmt_status = _fetch_management(stock)
        statuses.append(mgmt_status)

        cur.close()

        result = {
            "info": info,
            "trials": trials,
            "filings": filings,
            "news": news,
            "drugs": drugs[:10],
            "insider_trades": insider_trades,
            "institutional_holdings": holdings,
            "competitors": company_competitors,
            "management": management,
        }

        # Enrich: fill missing fields in trials and drugs (deterministic, no LLM)
        from services.enrich import enrich_company_data
        result = enrich_company_data(result)

        company_cache.set(ticker, result)
        return result, statuses

    except Exception as e:
        logger.error(f"get_company_detail failed [{ticker}]: {e}")
        return {"error": str(e)}, [SourceStatus.error("company", str(e))]
    finally:
        conn.close()


# ── Sub-functions ──────────────────────────────────────────────────────


_DRUG_FILLER = {
    "placebo", "vehicle", "sham", "control", "saline", "standard of care",
    "normal saline", "dummy", "comparator", "background", "observation",
    "no intervention", "best supportive care", "routine care", "usual care",
    "watchful waiting", "active control", "matching placebo", "sugar pill",
}


def _is_filler_drug(name: str) -> bool:
    """Return True if the name is a placebo/control/non-drug term."""
    lower = name.lower().strip()
    if lower in _DRUG_FILLER:
        return True
    # Catch partial matches like "Placebo (saline)" or "Vehicle control"
    for filler in _DRUG_FILLER:
        if filler in lower:
            return True
    return False


def _extract_drugs(trials: list[dict]) -> list[dict]:
    """Extract unique drug names from trial interventions, filtering out placebos/controls."""
    drugs = []
    drug_names: set[str] = set()
    for trial in trials:
        if trial.get("intervention"):
            for drug in (
                str(trial["intervention"]).replace(",", "|").replace(";", "|").split("|")
            ):
                drug = drug.strip()
                if (
                    drug
                    and len(drug) > 2
                    and drug not in drug_names
                    and not _is_filler_drug(drug)
                ):
                    drug_names.add(drug)
                    drugs.append(
                        {
                            "name": drug,
                            "status": trial.get("status", "Unknown"),
                            "phase": trial.get("phase", "Unknown"),
                            "condition": trial.get("condition", "Various"),
                        }
                    )
    return drugs


def _fetch_insider_trades(stock: yf.Ticker) -> tuple[list[dict], SourceStatus]:
    """Fetch insider trades from yfinance."""
    try:
        it_df = stock.insider_transactions
        if it_df is None or len(it_df) == 0:
            return [], SourceStatus.empty("yfinance:insider")

        trades = []
        for _, row in it_df.head(10).iterrows():
            text = str(row.get("Text", "")).lower()
            # Match whole-word patterns to avoid "Sale Manager" false positives
            if any(w in text for w in ("sale of", "sold", "automatic sale", "10b5-1 sale")):
                trade_type = "Sell"
            elif "sell" in text and "manager" not in text:
                trade_type = "Sell"
            elif any(w in text for w in ("purchase", "buy", "acquisition")):
                trade_type = "Buy"
            elif any(w in text for w in ("award", "grant", "vest", "exercise")):
                trade_type = "Grant"
            elif "gift" in text or "donate" in text:
                trade_type = "Gift"
            else:
                trade_type = "Other"

            shares = int(safe_float(row.get("Shares")))
            value = safe_float(row.get("Value"))
            insider_name = str(row.get("Insider", "Unknown"))
            position = str(row.get("Position", ""))
            if position:
                insider_name = f"{position} - {insider_name}"
            start_date = row.get("Start Date")
            if hasattr(start_date, "strftime"):
                date_str = start_date.strftime("%Y-%m-%d")
            else:
                date_str = str(start_date)[:10] if start_date else "N/A"

            trades.append(
                {
                    "date": date_str,
                    "insider": insider_name,
                    "type": trade_type,
                    "shares": shares,
                    "value": value,
                }
            )

        return trades, SourceStatus.ok("yfinance:insider", len(trades))
    except Exception as e:
        logger.debug(f"Insider trades failed: {e}")
        return [], SourceStatus.error("yfinance:insider", str(e))


def _fetch_institutional_holdings(stock: yf.Ticker) -> tuple[list[dict], SourceStatus]:
    """Fetch institutional holdings from yfinance."""
    try:
        ih_df = stock.institutional_holders
        if ih_df is None or len(ih_df) == 0:
            return [], SourceStatus.empty("yfinance:holdings")

        holdings = []
        for _, row in ih_df.head(8).iterrows():
            shares = int(safe_float(row.get("Shares")))
            pct = safe_float(row.get("pctHeld"))
            value = safe_float(row.get("Value"))
            holdings.append(
                {
                    "holder": str(row.get("Holder", "Unknown")),
                    "shares": f"{shares:,}",
                    "percent": f"{pct * 100:.2f}%" if pct else "N/A",
                    "value": f"${value / 1e9:.2f}B"
                    if value >= 1e9
                    else f"${value / 1e6:.0f}M",
                }
            )

        return holdings, SourceStatus.ok("yfinance:holdings", len(holdings))
    except Exception as e:
        logger.debug(f"Institutional holdings failed: {e}")
        return [], SourceStatus.error("yfinance:holdings", str(e))


def _fetch_drug_competitors(cur, drugs: list[dict]) -> list[dict]:
    """Find competing drugs for each drug (same condition, different intervention)."""
    drug_competitors = []
    if not drugs:
        return drug_competitors

    filler = {
        "placebo", "vehicle", "sham", "control", "saline",
        "standard of care", "normal saline", "dummy", "comparator", "background",
    }

    for drug in drugs[:5]:
        drug_name = drug.get("name", "")
        condition = drug.get("condition", "")
        if not condition:
            continue

        cur.execute(
            """
            SELECT DISTINCT intervention, phase, status, condition, sponsor
            FROM trials
            WHERE condition ILIKE %s
              AND intervention IS NOT NULL
              AND intervention NOT ILIKE %s
            ORDER BY phase DESC NULLS LAST
            LIMIT 10
        """,
            (f"%{condition}%", f"%{drug_name}%"),
        )

        competing_drugs = []
        seen_drugs: set[str] = set()
        for row in cur.fetchall():
            interventions = (
                str(row["intervention"]).replace(",", "|").replace(";", "|").split("|")
            )
            for comp_drug_name in interventions:
                comp_drug_name = comp_drug_name.strip()
                if (
                    comp_drug_name
                    and len(comp_drug_name) > 2
                    and comp_drug_name.lower() not in drug_name.lower()
                    and comp_drug_name not in seen_drugs
                    and not any(f in comp_drug_name.lower() for f in filler)
                ):
                    seen_drugs.add(comp_drug_name)
                    competing_drugs.append(
                        {
                            "name": comp_drug_name[:50],
                            "phase": row["phase"] or "Unknown",
                            "status": row["status"] or "Unknown",
                            "condition": row["condition"] or condition,
                            "sponsor": row["sponsor"] or "Unknown",
                        }
                    )
                    if len(competing_drugs) >= 5:
                        break
            if len(competing_drugs) >= 5:
                break

        if competing_drugs:
            drug_competitors.append(
                {
                    "drug_name": drug_name,
                    "condition": condition,
                    "phase": drug.get("phase", "Unknown"),
                    "competing_drugs": competing_drugs,
                }
            )

    return drug_competitors


def _fetch_company_competitors(info: dict) -> list[dict]:
    """Fetch company competitor quotes from yfinance."""
    company_competitors = []
    competitor_tickers = info.get("competitors") or []
    if not competitor_tickers:
        return company_competitors

    for comp_ticker in competitor_tickers[:3]:
        try:
            comp_stock = yf.Ticker(comp_ticker)
            comp_info = comp_stock.info
            company_competitors.append(
                {
                    "ticker": comp_ticker,
                    "name": comp_info.get("longName") or comp_ticker,
                    "price": round(
                        safe_float(
                            comp_info.get("currentPrice")
                            or comp_info.get("regularMarketPrice")
                        ),
                        2,
                    ),
                }
            )
        except Exception:
            pass

    return company_competitors


def _fetch_management(stock: yf.Ticker) -> tuple[list[dict], SourceStatus]:
    """Fetch management team from yfinance."""
    try:
        officers = stock.info.get("companyOfficers", [])
        management = []
        for officer in officers[:6]:
            fiscal_year = officer.get("fiscalYear")
            since = str(fiscal_year) if fiscal_year else "N/A"
            total_pay = officer.get("totalPay")
            exercised = officer.get("exercisedValue")
            unexercised = officer.get("unexercisedValue")
            management.append(
                {
                    "name": officer.get("name", "Unknown"),
                    "title": officer.get("title", "Unknown"),
                    "since": since,
                    "age": officer.get("age"),
                    "total_pay": int(total_pay) if total_pay else None,
                    "exercised_value": int(exercised) if exercised else None,
                    "unexercised_value": int(unexercised) if unexercised else None,
                }
            )
        return management, SourceStatus.ok("yfinance:management", len(management))
    except Exception as e:
        logger.debug(f"Management fetch failed: {e}")
        return [], SourceStatus.error("yfinance:management", str(e))
