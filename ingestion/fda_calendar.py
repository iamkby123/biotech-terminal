import httpx
import logging
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from config import FDA_PDUFA_URL, FDA_ADCOM_URL
from models.ticker_map import get_connection, resolve_sponsor_to_ticker

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

DATE_PATTERNS = [
    r"\b(\w+ \d{1,2},?\s?\d{4})\b",
    r"\b(\d{1,2}/\d{1,2}/\d{4})\b",
    r"\b(\d{4}-\d{2}-\d{2})\b",
]


def _parse_date(raw: str) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    for pattern in DATE_PATTERNS:
        m = re.search(pattern, raw)
        if m:
            try:
                for fmt in ("%B %d %Y", "%m/%d/%Y", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(m.group(1).replace(",", ""), fmt).strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            except Exception:
                pass
    return None


def _upsert_catalyst(cur, record: dict):
    cur.execute(
        """INSERT INTO catalysts
           (company, drug_name, indication, event_type, event_date, ticker, source_url)
           VALUES (%(company)s, %(drug_name)s, %(indication)s, %(event_type)s,
                   %(event_date)s, %(ticker)s, %(source_url)s)
           ON CONFLICT (company, drug_name, event_type, event_date) DO UPDATE SET
               indication = EXCLUDED.indication,
               ticker = EXCLUDED.ticker""",
        record,
    )


def scrape_pdufa_calendar() -> int:
    logger.info("Scraping FDA PDUFA calendar...")
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            r = client.get(FDA_PDUFA_URL, headers=HEADERS)
            r.raise_for_status()
    except Exception as e:
        logger.error(f"FDA PDUFA fetch error: {e}")
        return 0

    soup = BeautifulSoup(r.text, "html.parser")
    upserted = 0
    conn = get_connection()
    cur = conn.cursor()
    started = datetime.utcnow().isoformat()

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        headers_row = rows[0].find_all(["th", "td"]) if rows else []
        header_texts = [h.get_text(strip=True).lower() for h in headers_row]

        col_map = {}
        for i, h in enumerate(header_texts):
            if any(w in h for w in ["drug", "application", "product"]):
                col_map["drug"] = i
            elif any(w in h for w in ["company", "sponsor", "applicant"]):
                col_map["company"] = i
            elif any(w in h for w in ["date", "action"]):
                col_map["date"] = i
            elif any(w in h for w in ["indication", "use", "disease"]):
                col_map["indication"] = i

        if "date" not in col_map:
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            def get_cell(key):
                idx = col_map.get(key)
                if idx is not None and idx < len(cells):
                    return cells[idx].get_text(strip=True)
                return None

            company = get_cell("company") or ""
            drug = get_cell("drug") or ""
            date_raw = get_cell("date") or ""
            indication = get_cell("indication") or ""

            if not date_raw and not company:
                continue

            event_date = _parse_date(date_raw)
            if not event_date:
                continue

            ticker = resolve_sponsor_to_ticker(company) if company else None

            record = {
                "company": company,
                "drug_name": drug,
                "indication": indication,
                "event_type": "PDUFA",
                "event_date": event_date,
                "ticker": ticker,
                "source_url": FDA_PDUFA_URL,
            }
            try:
                _upsert_catalyst(cur, record)
                upserted += 1
            except Exception:
                pass

    if upserted == 0:
        upserted = _scrape_pdufa_text_fallback(soup, cur)

    conn.commit()
    cur.close()
    conn.close()
    _log_ingestion("fda_pdufa", None, upserted, upserted, "ok", None, started)
    logger.info(f"FDA PDUFA: {upserted} catalysts upserted.")
    return upserted


def _scrape_pdufa_text_fallback(soup: BeautifulSoup, cur) -> int:
    upserted = 0
    text = soup.get_text(separator="\n")
    lines = text.split("\n")

    date_pattern = re.compile(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}", re.IGNORECASE
    )

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        m = date_pattern.search(line)
        if m:
            event_date = _parse_date(m.group(0))
            if not event_date:
                continue

            record = {
                "company": "",
                "drug_name": line[:200],
                "indication": "",
                "event_type": "PDUFA",
                "event_date": event_date,
                "ticker": None,
                "source_url": FDA_PDUFA_URL,
            }
            try:
                _upsert_catalyst(cur, record)
                upserted += 1
            except Exception:
                pass

    return upserted


def scrape_adcom_calendar() -> int:
    logger.info("Scraping FDA AdCom calendar...")
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            r = client.get(FDA_ADCOM_URL, headers=HEADERS)
            r.raise_for_status()
    except Exception as e:
        logger.error(f"FDA AdCom fetch error: {e}")
        return 0

    soup = BeautifulSoup(r.text, "html.parser")
    upserted = 0
    conn = get_connection()
    cur = conn.cursor()
    started = datetime.utcnow().isoformat()

    tables = soup.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            texts = [c.get_text(strip=True) for c in cells]
            event_date = None
            for t in texts:
                event_date = _parse_date(t)
                if event_date:
                    break
            if not event_date:
                continue

            record = {
                "company": "",
                "drug_name": texts[1] if len(texts) > 1 else "",
                "indication": texts[2] if len(texts) > 2 else "",
                "event_type": "AdCom",
                "event_date": event_date,
                "ticker": None,
                "source_url": FDA_ADCOM_URL,
            }
            try:
                _upsert_catalyst(cur, record)
                upserted += 1
            except Exception:
                pass

    conn.commit()
    cur.close()
    conn.close()
    _log_ingestion("fda_adcom", None, upserted, upserted, "ok", None, started)
    logger.info(f"FDA AdCom: {upserted} catalysts upserted.")
    return upserted


def ingest_all() -> int:
    total = scrape_pdufa_calendar()
    time.sleep(1)
    total += scrape_adcom_calendar()
    return total


def _log_ingestion(source, ticker, fetched, upserted, status, error, started):
    """DEPRECATED: Use ingestion.common.log_ingestion instead."""
    from ingestion.common import log_ingestion
    log_ingestion(source, ticker, fetched, upserted, status, error, started)
