import httpx
import time
import logging
from datetime import datetime
from config import EDGAR_SUBMISSIONS, SEC_USER_AGENT, EDGAR_RATE_LIMIT
from models.ticker_map import get_connection, get_ticker_info

logger = logging.getLogger(__name__)

PIPELINE_KEYWORDS = [
    "pipeline", "clinical trial", "phase 1", "phase 2", "phase 3",
    "IND ", "NDA ", "BLA ", "PDUFA", "FDA approval", "primary endpoint",
    "investigational", "drug candidate",
]


def _get_headers() -> dict:
    return {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def fetch_company_filings(cik: str, form_types: list[str] = None) -> list[dict]:
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    url = EDGAR_SUBMISSIONS.format(cik=cik.lstrip("0").zfill(10))
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(url, headers=_get_headers())
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(f"EDGAR submissions fetch error CIK={cik}: {e}")
        return []

    data = r.json()
    filings_data = data.get("filings", {}).get("recent", {})

    accession_numbers = filings_data.get("accessionNumber", [])
    form_type_list = filings_data.get("form", [])
    filed_dates = filings_data.get("filingDate", [])
    periods = filings_data.get("reportDate", [])

    results = []
    for i, acc in enumerate(accession_numbers):
        form = form_type_list[i] if i < len(form_type_list) else ""
        if form not in form_types:
            continue

        results.append({
            "accession_number": acc,
            "form_type": form,
            "filed_date": filed_dates[i] if i < len(filed_dates) else None,
            "period_of_report": periods[i] if i < len(periods) else None,
            "filing_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form}&dateb=&owner=include&count=40",
        })

    return results


def _upsert_filing(cur, filing: dict):
    cur.execute(
        """INSERT INTO filings
           (accession_number, cik, ticker, form_type, filed_date, period_of_report, filing_url, pipeline_text)
           VALUES (%(accession_number)s, %(cik)s, %(ticker)s, %(form_type)s, %(filed_date)s,
                   %(period_of_report)s, %(filing_url)s, %(pipeline_text)s)
           ON CONFLICT (accession_number) DO UPDATE SET
               pipeline_text = EXCLUDED.pipeline_text,
               filing_url = EXCLUDED.filing_url""",
        filing,
    )


def ingest_for_tickers(tickers: list[str], form_types: list[str] = None, max_per_ticker: int = 10) -> dict[str, int]:
    if form_types is None:
        form_types = ["10-K", "8-K"]

    results = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        if not info or not info.get("cik"):
            logger.warning(f"No CIK for {ticker}, skipping EDGAR.")
            results[ticker] = 0
            continue

        cik = info["cik"]
        started = datetime.utcnow().isoformat()
        filings = fetch_company_filings(cik, form_types)
        filings = filings[:max_per_ticker]

        conn = get_connection()
        cur = conn.cursor()
        upserted = 0
        for f in filings:
            record = {
                "accession_number": f["accession_number"],
                "cik": cik,
                "ticker": ticker,
                "form_type": f["form_type"],
                "filed_date": f["filed_date"],
                "period_of_report": f["period_of_report"],
                "filing_url": f["filing_url"],
                "pipeline_text": None,
            }
            _upsert_filing(cur, record)
            upserted += 1

        conn.commit()
        cur.close()
        conn.close()

        _log_ingestion("edgar", ticker, len(filings), upserted, "ok", None, started)
        results[ticker] = upserted
        logger.info(f"EDGAR [{ticker}]: {upserted} filings upserted.")
        time.sleep(EDGAR_RATE_LIMIT)

    return results


def _log_ingestion(source, ticker, fetched, upserted, status, error, started):
    """DEPRECATED: Use ingestion.common.log_ingestion instead."""
    from ingestion.common import log_ingestion
    log_ingestion(source, ticker, fetched, upserted, status, error, started)
