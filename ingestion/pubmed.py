import httpx
import time
import logging
from datetime import datetime
from config import PUBMED_SEARCH, PUBMED_FETCH, NCBI_API_KEY, PUBMED_RATE_LIMIT
from models.ticker_map import get_connection

logger = logging.getLogger(__name__)


def _base_params() -> dict:
    params = {"retmode": "json", "retmax": 100}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


def search_pmids_by_nct(nct_id: str) -> list[str]:
    params = _base_params()
    params.update({"db": "pubmed", "term": f"{nct_id}[si]"})
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(PUBMED_SEARCH, params=params)
            r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed search error for {nct_id}: {e}")
        return []


def search_pmids_by_term(term: str, max_results: int = 20) -> list[str]:
    params = _base_params()
    params.update({
        "db": "pubmed",
        "term": term,
        "retmax": max_results,
        "sort": "relevance",
    })
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(PUBMED_SEARCH, params=params)
            r.raise_for_status()
        data = r.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        logger.warning(f"PubMed search error for term={term}: {e}")
        return []


def fetch_pubmed_records(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []

    params = _base_params()
    params.update({
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    })

    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(PUBMED_FETCH, params=params)
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"PubMed fetch error: {e}")
        return []

    from xml.etree import ElementTree as ET
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError as e:
        logger.warning(f"PubMed XML parse error: {e}")
        return []

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        if not pmid:
            continue

        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else None

        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            (el.get("Label", "") + ": " + (el.text or "")).strip()
            for el in abstract_parts
            if el.text
        ).strip() or None

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else None

        year_el = article.find(".//PubDate/Year")
        month_el = article.find(".//PubDate/Month")
        year = year_el.text if year_el is not None else None
        month = month_el.text if month_el is not None else "01"

        pub_date = None
        if year:
            try:
                if month and not month.isdigit():
                    month = datetime.strptime(month, "%b").strftime("%m")
                pub_date = f"{year}-{month.zfill(2)}-01"
            except Exception:
                pub_date = f"{year}-01-01"

        nct_ids = []
        for ref in article.findall(".//AccessionNumber"):
            if ref.text and ref.text.startswith("NCT"):
                nct_ids.append(ref.text)

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "pub_date": pub_date,
            "nct_id": nct_ids[0] if nct_ids else None,
        })

    return records


def _upsert_publication(cur, record: dict):
    cur.execute(
        """INSERT INTO publications
           (pmid, title, abstract, journal, pub_date, nct_id, sponsor_match)
           VALUES (%(pmid)s, %(title)s, %(abstract)s, %(journal)s, %(pub_date)s,
                   %(nct_id)s, %(sponsor_match)s)
           ON CONFLICT (pmid) DO UPDATE SET
               title = EXCLUDED.title,
               abstract = EXCLUDED.abstract,
               journal = EXCLUDED.journal,
               pub_date = EXCLUDED.pub_date,
               nct_id = COALESCE(EXCLUDED.nct_id, publications.nct_id)""",
        record,
    )


def ingest_for_nct_ids(nct_ids: list[str]) -> int:
    upserted = 0
    started = datetime.utcnow().isoformat()
    conn = get_connection()
    cur = conn.cursor()

    for nct_id in nct_ids:
        pmids = search_pmids_by_nct(nct_id)
        time.sleep(PUBMED_RATE_LIMIT)
        if not pmids:
            continue

        records = fetch_pubmed_records(pmids)
        time.sleep(PUBMED_RATE_LIMIT)

        for rec in records:
            if not rec.get("nct_id"):
                rec["nct_id"] = nct_id
            rec["sponsor_match"] = None
            _upsert_publication(cur, rec)
            upserted += 1

        conn.commit()
        logger.debug(f"PubMed [{nct_id}]: {len(records)} publications.")

    cur.close()
    conn.close()
    _log_ingestion("pubmed", None, upserted, upserted, "ok", None, started)
    logger.info(f"PubMed NCT ingest: {upserted} publications upserted.")
    return upserted


def ingest_for_tickers(tickers: list[str], max_per_company: int = 20) -> dict[str, int]:
    from models.ticker_map import get_ticker_info

    results = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        if not info or not info.get("company_name"):
            results[ticker] = 0
            continue

        company_name = info["company_name"]
        started = datetime.utcnow().isoformat()

        term = f'"{company_name}"[Affiliation] OR "{company_name}"[Corporate Author]'
        pmids = search_pmids_by_term(term, max_results=max_per_company)
        time.sleep(PUBMED_RATE_LIMIT)

        if not pmids:
            results[ticker] = 0
            continue

        records = fetch_pubmed_records(pmids)
        time.sleep(PUBMED_RATE_LIMIT)

        conn = get_connection()
        cur = conn.cursor()
        upserted = 0
        for rec in records:
            rec["sponsor_match"] = ticker
            _upsert_publication(cur, rec)
            upserted += 1
        conn.commit()
        cur.close()
        conn.close()

        _log_ingestion("pubmed", ticker, len(records), upserted, "ok", None, started)
        results[ticker] = upserted
        logger.info(f"PubMed [{ticker}]: {upserted} publications upserted.")

    return results


def ingest_from_trials_table() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT nct_id FROM trials")
    nct_ids = [r["nct_id"] for r in cur.fetchall()]
    cur.close()
    conn.close()

    if not nct_ids:
        logger.info("No trials in DB to cross-reference with PubMed.")
        return 0

    logger.info(f"Fetching PubMed publications for {len(nct_ids)} NCT IDs...")
    return ingest_for_nct_ids(nct_ids)


def _log_ingestion(source, ticker, fetched, upserted, status, error, started):
    """DEPRECATED: Use ingestion.common.log_ingestion instead."""
    from ingestion.common import log_ingestion
    log_ingestion(source, ticker, fetched, upserted, status, error, started)
