import os
import sys
import time
import logging
import httpx
import psycopg2
import psycopg2.extras
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

BIOTECH_SIC_CODES = [
    "2836",  # Pharmaceutical Preparations
    "2835",  # In Vitro Diagnostic Substances
    "2834",  # Pharmaceutical Preparations (alt)
    "2833",  # Medicinal Chemicals & Botanical Products
    "2830",  # Drugs
    "8731",  # Commercial Physical & Biological Research
    "2860",  # Industrial Chemicals (includes some biotech tools)
    "5047",  # Medical & Hospital Equipment (biotech tools)
]

SEC_USER_AGENT = os.environ.get("SEC_USER_AGENT", "BiotechTerminal contact@example.com")
HEADERS = {"User-Agent": SEC_USER_AGENT}


def get_connection():
    return psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)


def fetch_all_biotech_companies() -> list[dict]:
    logger.info("Fetching full company list from SEC EDGAR...")
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    with httpx.Client(timeout=60) as client:
        r = client.get(url, headers=HEADERS)
        r.raise_for_status()
    data = r.json()

    fields = data.get("fields", [])
    rows = data.get("data", [])
    cik_idx = fields.index("cik")
    name_idx = fields.index("name")
    ticker_idx = fields.index("ticker")
    exchange_idx = fields.index("exchange")

    logger.info(f"Total SEC-listed companies: {len(rows)}")

    url2 = "https://www.sec.gov/files/company_tickers.json"
    with httpx.Client(timeout=60) as client:
        r2 = client.get(url2, headers=HEADERS)
        r2.raise_for_status()
    simple_map = r2.json()

    cik_to_sic: dict[str, str] = {}
    logger.info("Fetching SIC codes for all companies (this takes a moment)...")

    url3 = "https://efts.sec.gov/LATEST/search-index?q=%22%22&forms=10-K&dateRange=custom&startdt=2022-01-01&enddt=2026-12-31"

    biotech_from_exchange = []
    for row in rows:
        ticker = row[ticker_idx]
        name = row[name_idx]
        cik = str(row[cik_idx]).zfill(10)
        exchange = row[exchange_idx] or ""
        if exchange in ("Nasdaq", "NYSE", "NYSE MKT", "NYSE ARCA", "NASDAQ"):
            biotech_from_exchange.append({
                "ticker": ticker.upper(),
                "company_name": name,
                "cik": cik,
                "exchange": exchange,
            })

    return biotech_from_exchange


def fetch_sic_company_list(sic_code: str) -> list[dict]:
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22%22&dateRange=custom&startdt=2023-01-01&forms=10-K&_source=hits.hits._source.period_of_report,hits.hits._source.file_date,hits.hits._source.entity_name,hits.hits._source.file_num,hits.hits._source.period_of_report&category=form-type&location=US&locationCode=US"

    url_browse = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&SIC={sic_code}&dateb=&owner=include&count=100&search_text=&action=getcompany&output=atom"

    companies = []
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            r = client.get(
                f"https://efts.sec.gov/LATEST/search-index?q=%22%22&dateRange=custom&startdt=2022-01-01&forms=10-K&category=form-type",
                params={"_source": "entity_name,file_num", "dateRange": "custom"},
                headers=HEADERS,
            )
    except Exception as e:
        logger.warning(f"SIC browse error: {e}")

    return companies


def fetch_biotech_tickers_by_sic() -> list[dict]:
    logger.info("Fetching biotech tickers by SIC code from SEC EDGAR full-text search...")
    results = []
    seen_ciks = set()

    url = "https://efts.sec.gov/LATEST/search-index"

    for sic in BIOTECH_SIC_CODES:
        offset = 0
        page_size = 100
        while True:
            try:
                with httpx.Client(timeout=30) as client:
                    r = client.get(
                        "https://efts.sec.gov/LATEST/search-index",
                        params={
                            "q": f'"{sic}"',
                            "dateRange": "custom",
                            "startdt": "2021-01-01",
                            "forms": "10-K",
                            "from": offset,
                            "size": page_size,
                        },
                        headers=HEADERS,
                    )
                    r.raise_for_status()
                data = r.json()
                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for h in hits:
                    src = h.get("_source", {})
                    entity = src.get("entity_name", "")
                    file_num = src.get("file_num", "")
                    cik = src.get("file_num", "")
                    if entity:
                        results.append({"company_name": entity, "sic": sic})
                offset += page_size
                if offset >= min(data.get("hits", {}).get("total", {}).get("value", 0), 1000):
                    break
                time.sleep(0.2)
            except Exception as e:
                logger.warning(f"SIC {sic} fetch error at offset {offset}: {e}")
                break

    return results


def load_tickers_from_sec_exchange() -> dict[str, dict]:
    logger.info("Loading exchange-listed companies from SEC...")
    url = "https://www.sec.gov/files/company_tickers_exchange.json"
    with httpx.Client(timeout=60) as client:
        r = client.get(url, headers=HEADERS)
        r.raise_for_status()
    data = r.json()
    fields = data.get("fields", [])
    rows = data.get("data", [])
    cik_idx = fields.index("cik")
    name_idx = fields.index("name")
    ticker_idx = fields.index("ticker")
    exchange_idx = fields.index("exchange")

    result = {}
    for row in rows:
        ticker = (row[ticker_idx] or "").upper()
        name = row[name_idx] or ""
        cik = str(row[cik_idx]).zfill(10)
        exchange = row[exchange_idx] or ""
        if ticker and exchange in ("Nasdaq", "NYSE", "NYSE MKT", "NYSE ARCA"):
            result[ticker] = {"cik": cik, "company_name": name, "exchange": exchange}
    return result


def fetch_submissions_for_cik(cik: str) -> dict | None:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        with httpx.Client(timeout=20) as client:
            r = client.get(url, headers=HEADERS)
            r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_all_biotech_tickers() -> list[dict]:
    exchange_map = load_tickers_from_sec_exchange()
    logger.info(f"Exchange-listed tickers total: {len(exchange_map)}")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM ticker_map")
    existing = {r["ticker"] for r in cur.fetchall()}
    cur.close()
    conn.close()

    biotech_keywords = [
        "pharma", "bio", "therapeut", "oncol", "genomic", "gene", "cell",
        "immun", "medic", "health", "scienc", "laborator", "diagnost",
        "clinic", "drug", "vaccine", "protein", "enzyme", "antibod",
        "peptide", "molecular", "neuroscien", "cardiovasc", "ophthalm",
        "dermatol", "hematol", "oncolog", "crispr", "rna", "dna", "mRNA",
        "regenerat", "stem cell", "biologic", "biosimilar", "agen", "corp",
    ]

    biotech_tickers = {}
    for ticker, info in exchange_map.items():
        name_lower = info["company_name"].lower()
        if any(kw in name_lower for kw in biotech_keywords):
            biotech_tickers[ticker] = info

    logger.info(f"Keyword-filtered biotech tickers: {len(biotech_tickers)}")
    new_tickers = {k: v for k, v in biotech_tickers.items() if k not in existing}
    logger.info(f"New tickers to add: {len(new_tickers)}")
    return list(new_tickers.values()) + [{"ticker": t} for t in existing]


def upsert_tickers(tickers: list[dict]) -> int:
    conn = get_connection()
    cur = conn.cursor()
    upserted = 0
    for t in tickers:
        if not t.get("ticker"):
            continue
        cur.execute(
            """INSERT INTO ticker_map (ticker, company_name, cik, updated_at)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (ticker) DO UPDATE SET
                   company_name = COALESCE(EXCLUDED.company_name, ticker_map.company_name),
                   cik = COALESCE(EXCLUDED.cik, ticker_map.cik),
                   updated_at = EXCLUDED.updated_at""",
            (t["ticker"], t.get("company_name"), t.get("cik"), datetime.utcnow()),
        )
        upserted += 1
    conn.commit()
    cur.close()
    conn.close()
    return upserted


def run_edgar_for_new_tickers(tickers: list[str], batch_size: int = 50):
    from ingestion.edgar import ingest_for_tickers
    logger.info(f"Running EDGAR ingestion for {len(tickers)} tickers...")
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info(f"  EDGAR batch {i//batch_size + 1}: {batch[0]} ... {batch[-1]}")
        results = ingest_for_tickers(batch, form_types=["10-K", "8-K"], max_per_ticker=5)
        total = sum(results.values())
        logger.info(f"  Batch done: {total} filings")


def run_ctgov_broad():
    from ingestion.clinicaltrials import fetch_trials_by_condition
    CONDITIONS = [
        "cancer", "leukemia", "lymphoma", "melanoma", "lung cancer", "breast cancer",
        "colorectal cancer", "prostate cancer", "pancreatic cancer", "ovarian cancer",
        "glioblastoma", "multiple myeloma", "solid tumor",
        "Alzheimer", "Parkinson", "ALS", "multiple sclerosis", "epilepsy",
        "schizophrenia", "depression", "bipolar disorder", "ADHD",
        "diabetes", "obesity", "NASH", "NAFLD", "heart failure",
        "atrial fibrillation", "hypertension", "hypercholesterolemia",
        "HIV", "hepatitis B", "hepatitis C", "COVID-19", "influenza",
        "asthma", "COPD", "cystic fibrosis", "lupus", "rheumatoid arthritis",
        "Crohn", "ulcerative colitis", "psoriasis", "atopic dermatitis",
        "hemophilia", "sickle cell", "beta thalassemia", "rare disease",
        "macular degeneration", "glaucoma", "dry eye",
    ]
    logger.info(f"Running broad CTGOV ingest across {len(CONDITIONS)} conditions...")
    total = 0
    for cond in CONDITIONS:
        try:
            n = fetch_trials_by_condition(cond, phase=["PHASE2", "PHASE3", "PHASE4"])
            total += n
            logger.info(f"  [{cond}] → {n} trials")
            time.sleep(0.2)
        except Exception as e:
            logger.warning(f"  [{cond}] error: {e}")
    logger.info(f"Broad CTGOV done: {total} total upserted.")
    return total


def run_ctgov_for_all_tickers():
    from ingestion.clinicaltrials import ingest_for_tickers
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ticker, company_name FROM ticker_map WHERE company_name IS NOT NULL ORDER BY ticker")
    tickers = [r["ticker"] for r in cur.fetchall()]
    cur.close()
    conn.close()

    logger.info(f"Running per-sponsor CTGOV ingest for {len(tickers)} tickers...")
    results = ingest_for_tickers(tickers)
    total = sum(results.values())
    logger.info(f"Per-sponsor CTGOV done: {total} total trials")
    return total


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("tickers", "all"):
        logger.info("=== STEP 1: Expanding ticker universe ===")
        exchange_map = load_tickers_from_sec_exchange()
        biotech_keywords = [
            "pharma", "bio", "therapeut", "oncol", "genomic", "gene therapy",
            "immun", "vaccine", "biolog", "biosimilar", "crispr", "rna ",
            " rna", "dna", "mrna", "regenerat", "antibod", "peptide",
            "molecular", "neuroscien", "cell therap", "stem cell",
        ]
        biotech_map = {
            t: info for t, info in exchange_map.items()
            if any(kw in info["company_name"].lower() for kw in biotech_keywords)
        }
        logger.info(f"Strict biotech keyword match: {len(biotech_map)} tickers")

        broad_keywords = [
            "pharma", "bio", "therapeut", "oncol", "genomic", "gene",
            "immun", "medic", "scienc", "laborator", "diagnost",
            "clinic", "drug", "vaccine", "protein", "enzyme",
            "peptide", "molecular", "neuroscien", "dermatol",
            "hematol", "crispr", "rna", "dna", "regenerat",
        ]
        broad_map = {
            t: info for t, info in exchange_map.items()
            if any(kw in info["company_name"].lower() for kw in broad_keywords)
        }
        logger.info(f"Broad keyword match: {len(broad_map)} tickers")

        upsert_list = [{"ticker": t, "company_name": v["company_name"], "cik": v["cik"]} for t, v in broad_map.items()]
        n = upsert_tickers(upsert_list)
        logger.info(f"Upserted {n} tickers into ticker_map")

    if mode in ("edgar", "all"):
        logger.info("=== STEP 2: EDGAR ingestion for all tickers ===")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT ticker FROM ticker_map WHERE cik IS NOT NULL ORDER BY ticker")
        all_tickers = [r["ticker"] for r in cur.fetchall()]
        cur.close()
        conn.close()
        logger.info(f"Running EDGAR for {len(all_tickers)} tickers")
        run_edgar_for_new_tickers(all_tickers)

    if mode in ("ctgov", "all"):
        logger.info("=== STEP 3: ClinicalTrials.gov broad condition sweep ===")
        run_ctgov_broad()
        logger.info("=== STEP 4: ClinicalTrials.gov per-sponsor sweep ===")
        run_ctgov_for_all_tickers()

    logger.info("=== Expansion complete ===")
