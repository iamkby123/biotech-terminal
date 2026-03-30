import httpx
import time
import logging
import re
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from models.ticker_map import get_connection
from ingestion.finnhub_news import classify_catalyst, score_sentiment

logger = logging.getLogger(__name__)

RSS_SOURCES = [
    {
        "name": "STAT News",
        "url": "https://www.statnews.com/feed/",
        "category": "news",
    },
    {
        "name": "BioPharma Dive",
        "url": "https://www.biopharmadive.com/feeds/news/",
        "category": "news",
    },
    {
        "name": "FierceBiotech",
        "url": "https://www.fiercebiotech.com/rss/xml",
        "category": "news",
    },
    {
        "name": "BioSpace",
        "url": "https://www.biospace.com/news/rss/all/",
        "category": "news",
    },
    {
        "name": "MedCity News",
        "url": "https://medcitynews.com/category/health-tech/feed/",
        "category": "news",
    },
    {
        "name": "FDA Press Releases",
        "url": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
        "category": "regulatory",
    },
    {
        "name": "Endpoints News",
        "url": "https://endpts.com/feed/",
        "category": "news",
    },
    {
        "name": "Reuters Health",
        "url": "https://feeds.reuters.com/reuters/healthNews",
        "category": "news",
    },
    {
        "name": "NIH News",
        "url": "https://www.nih.gov/news-events/news-releases/feed",
        "category": "regulatory",
    },
    {
        "name": "PharmaLive",
        "url": "https://www.pharmalive.com/feed/",
        "category": "news",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, text/xml, */*",
}

NCT_RE = re.compile(r"NCT\d{8}", re.IGNORECASE)

BIOTECH_KEYWORDS = [
    "pharma", "biotech", "drug", "fda", "approval", "trial", "phase",
    "cancer", "oncology", "therapy", "therapeutic", "clinical", "nda",
    "bla", "pdufa", "adcom", "ipo", "acquisition", "merger", "data",
    "readout", "endpoint", "efficacy", "safety", "pipeline", "ceo",
    "fundrais", "series", "biologics", "gene therapy", "cell therapy",
    "antibody", "immunotherapy", "crispr", "mrna", "rna", "rare disease",
]


def _parse_date(raw: str) -> str | None:
    if not raw:
        return None
    raw = raw.strip()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    # Try ISO format as last resort
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        pass
    return None


def _clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:800]


# Ticker symbols that are also common words in biotech/pharma articles
# These should only match via company name, never by ticker symbol alone
_AMBIGUOUS_TICKERS = {
    "MRNA", "RNA", "DNA", "DRUG", "RARE", "EDIT", "FATE", "BEAM", "SAGE",
    "FOLD", "BLUE", "BOLD", "DAWN", "IRON", "GENE", "STEM", "CELL", "FORM",
    "NEXT", "PLUS", "PROS", "CASH", "FAST", "LIVE", "OPEN", "PLAN", "SAFE",
    "TELL", "TRUE", "WELL", "MASS", "GOLD", "REAL", "SELF", "UNIT", "TURN",
    "HOOK", "BAND", "EVER", "MIND", "TALK",
}


def _extract_tickers(text: str, known_tickers: set[str], name_to_ticker: dict[str, str] = None) -> str | None:
    found = set()

    # First pass: match by company name (most reliable)
    if name_to_ticker:
        text_lower = text.lower()
        for name, ticker in name_to_ticker.items():
            if name in text_lower:
                found.add(ticker)

    # Second pass: match by ticker symbol, but only non-ambiguous ones
    text_upper = text.upper()
    for ticker in known_tickers:
        if len(ticker) < 2:
            continue
        if ticker in _AMBIGUOUS_TICKERS:
            continue
        # Require word boundaries and at least some context (e.g., "$LLY" or "(LLY)")
        # For short tickers (2-3 chars), require them to appear near stock/financial context
        if len(ticker) <= 3:
            # Only match short tickers if preceded by $ or in parentheses
            pattern = r"(?:\$|[(])" + re.escape(ticker) + r"\b"
            if re.search(pattern, text_upper):
                found.add(ticker)
        else:
            pattern = r"\b" + re.escape(ticker) + r"\b"
            if re.search(pattern, text_upper):
                found.add(ticker)

    return ",".join(sorted(found)[:5]) if found else None


def _extract_nct(text: str) -> str | None:
    m = NCT_RE.search(text or "")
    return m.group(0).upper() if m else None


def _is_biotech_relevant(title: str, summary: str) -> bool:
    combined = (title + " " + summary).lower()
    return any(kw in combined for kw in BIOTECH_KEYWORDS)


def _fetch_rss(source: dict, known_tickers: set[str], name_to_ticker: dict[str, str] = None) -> list[dict]:
    items = []
    try:
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            r = client.get(source["url"], headers=HEADERS)
            r.raise_for_status()
    except Exception as e:
        logger.warning(f"RSS fetch failed [{source['name']}]: {e}")
        return []

    try:
        root = ET.fromstring(r.content)
    except ET.ParseError as e:
        logger.warning(f"RSS parse error [{source['name']}]: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}

    # Handle both RSS 2.0 and Atom feeds
    entries = root.findall(".//item") or root.findall(".//atom:entry", ns) or root.findall(".//entry")

    for entry in entries:
        def tag(name, alt=None):
            el = entry.find(name) or (entry.find(alt) if alt else None)
            return (el.text or "").strip() if el is not None else ""

        title = (
            tag("title") or
            tag("atom:title", "title")
        )
        if not title:
            continue

        link_el = entry.find("link")
        if link_el is not None:
            url = link_el.get("href") or link_el.text or ""
        else:
            url = tag("{http://www.w3.org/2005/Atom}link")
        url = url.strip()

        summary_raw = (
            tag("description") or
            tag("summary") or
            tag("{http://www.w3.org/2005/Atom}summary") or
            tag("content") or ""
        )
        summary = _clean_html(summary_raw)

        pub_raw = (
            tag("pubDate") or
            tag("published") or
            tag("{http://www.w3.org/2005/Atom}published") or
            tag("updated") or ""
        )
        published_at = _parse_date(pub_raw)

        combined_text = title + " " + summary
        if not _is_biotech_relevant(title, summary):
            continue

        tickers = _extract_tickers(combined_text, known_tickers, name_to_ticker)
        nct_id = _extract_nct(combined_text)

        catalyst = classify_catalyst(title, summary)
        sentiment, sent_score = score_sentiment(title, summary)

        items.append({
            "source": source["name"],
            "category": source["category"],
            "title": title[:500],
            "summary": summary,
            "url": url or None,
            "published_at": published_at,
            "tickers": tickers,
            "nct_id": nct_id,
            "catalyst_type": catalyst,
            "sentiment": sentiment,
            "sentiment_score": sent_score,
        })

    return items


def _upsert_news(cur, item: dict):
    cur.execute(
        """INSERT INTO news (source, category, title, summary, url, published_at, tickers, nct_id,
                            catalyst_type, sentiment, sentiment_score)
           VALUES (%(source)s, %(category)s, %(title)s, %(summary)s, %(url)s,
                   %(published_at)s, %(tickers)s, %(nct_id)s,
                   %(catalyst_type)s, %(sentiment)s, %(sentiment_score)s)
           ON CONFLICT (url) DO UPDATE SET
               title = EXCLUDED.title,
               summary = EXCLUDED.summary,
               published_at = COALESCE(EXCLUDED.published_at, news.published_at),
               tickers = COALESCE(EXCLUDED.tickers, news.tickers),
               nct_id = COALESCE(EXCLUDED.nct_id, news.nct_id),
               catalyst_type = COALESCE(EXCLUDED.catalyst_type, news.catalyst_type),
               sentiment = COALESCE(EXCLUDED.sentiment, news.sentiment),
               sentiment_score = COALESCE(EXCLUDED.sentiment_score, news.sentiment_score)""",
        item,
    )


def _build_name_to_ticker(cur) -> dict[str, str]:
    """Build a mapping of company name variants to ticker symbols."""
    cur.execute("SELECT ticker, company_name, ctgov_sponsor_name FROM ticker_map WHERE ticker IS NOT NULL")
    name_to_ticker = {}
    for row in cur.fetchall():
        ticker = row["ticker"]
        for name_field in ["company_name", "ctgov_sponsor_name"]:
            name = row.get(name_field)
            if not name or len(name) < 4:
                continue
            name_lower = name.lower()
            name_to_ticker[name_lower] = ticker
            # Also add first word if it's long enough (e.g., "Moderna" from "Moderna, Inc.")
            first_word = name.split()[0].rstrip(",").lower()
            if len(first_word) >= 5:
                name_to_ticker[first_word] = ticker
            # Add name without suffix (Inc., Corp., etc.)
            clean = re.sub(r',?\s*(Inc\.?|Corp\.?|Ltd\.?|plc|S\.A\.|SE|N\.V\.|AG|Co\.?|Company|Therapeutics|Pharmaceuticals|Biosciences|Sciences)$', '', name, flags=re.IGNORECASE).strip().lower()
            if len(clean) >= 4 and clean != name_lower:
                name_to_ticker[clean] = ticker
    return name_to_ticker


def fetch_rss_news() -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM ticker_map WHERE ticker IS NOT NULL")
    known_tickers = {r["ticker"] for r in cur.fetchall()}
    name_to_ticker = _build_name_to_ticker(cur)

    upserted = 0
    for source in RSS_SOURCES:
        items = _fetch_rss(source, known_tickers, name_to_ticker)
        for item in items:
            try:
                _upsert_news(cur, item)
                upserted += 1
            except Exception as e:
                logger.debug(f"News upsert skip [{item.get('title','')[:40]}]: {e}")
                conn.rollback()
                continue
        conn.commit()
        logger.info(f"News [{source['name']}]: {len(items)} items fetched, source done.")
        time.sleep(0.5)

    # Re-tag existing untagged news with company names
    retagged = _retag_news(cur, known_tickers, name_to_ticker)
    conn.commit()

    cur.close()
    conn.close()
    logger.info(f"RSS news ingest done: {upserted} items upserted, {retagged} re-tagged.")
    return upserted


def _retag_news(cur, known_tickers: set[str], name_to_ticker: dict[str, str]) -> int:
    """Re-tag existing news that have no tickers assigned."""
    cur.execute("SELECT id, title, summary FROM news WHERE tickers IS NULL")
    rows = cur.fetchall()
    retagged = 0
    for row in rows:
        combined = (row.get("title") or "") + " " + (row.get("summary") or "")
        tickers = _extract_tickers(combined, known_tickers, name_to_ticker)
        if tickers:
            cur.execute("UPDATE news SET tickers = %s WHERE id = %s", (tickers, row["id"]))
            retagged += 1
    logger.info(f"Re-tagged {retagged} news items with tickers.")
    return retagged


def fetch_trial_updates(days_back: int = 7) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT nct_id, title, sponsor, phase, status, condition,
               primary_completion_date, results_posted, last_updated
        FROM trials
        WHERE last_updated >= (NOW() - INTERVAL '7 days')::TEXT::DATE::TEXT
           OR results_posted = 1
        ORDER BY last_updated DESC NULLS LAST
        LIMIT 500
    """)
    recent_trials = cur.fetchall()

    cur.execute("SELECT ticker, company_name FROM ticker_map WHERE company_name IS NOT NULL")
    ticker_rows = cur.fetchall()
    company_to_ticker = {}
    for r in ticker_rows:
        if r["company_name"]:
            company_to_ticker[r["company_name"].lower()[:20]] = r["ticker"]

    upserted = 0
    for trial in recent_trials:
        nct_id = trial["nct_id"]
        title = trial["title"] or ""
        sponsor = trial["sponsor"] or ""
        status = trial["status"] or ""
        condition = trial["condition"] or ""
        phase = trial["phase"] or ""
        results = trial["results_posted"]
        last_updated = trial["last_updated"] or ""

        if results:
            headline = f"Results Posted: {title[:120]}"
            category = "trial_results"
        elif "RECRUITING" in status.upper():
            headline = f"Now Recruiting: {title[:120]}"
            category = "trial_update"
        elif "COMPLETED" in status.upper():
            headline = f"Trial Completed: {title[:120]}"
            category = "trial_update"
        elif "TERMINATED" in status.upper():
            headline = f"Trial Terminated: {title[:120]}"
            category = "trial_update"
        else:
            headline = f"Trial Update [{status}]: {title[:100]}"
            category = "trial_update"

        summary = (
            f"{phase} | {condition[:80]} | Sponsor: {sponsor[:60]} | "
            f"Status: {status} | Last updated: {last_updated}"
        )

        ticker = None
        sponsor_lower = sponsor.lower()
        for comp_key, tick in company_to_ticker.items():
            if comp_key in sponsor_lower or sponsor_lower[:15] in comp_key:
                ticker = tick
                break

        url = f"https://clinicaltrials.gov/study/{nct_id}"

        catalyst = classify_catalyst(headline, summary)
        sentiment, sent_score = score_sentiment(headline, summary)

        item = {
            "source": "ClinicalTrials.gov",
            "category": category,
            "title": headline,
            "summary": summary,
            "url": url,
            "published_at": last_updated or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers": ticker,
            "nct_id": nct_id,
            "catalyst_type": catalyst,
            "sentiment": sentiment,
            "sentiment_score": sent_score,
        }
        try:
            _upsert_news(cur, item)
            upserted += 1
        except Exception:
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Trial updates: {upserted} news items generated.")
    return upserted


def ingest_all_news(include_finnhub: bool = False) -> int:
    n1 = fetch_rss_news()
    n2 = fetch_trial_updates()
    n3 = 0
    if include_finnhub:
        from ingestion.finnhub_news import fetch_finnhub_news
        n3 = fetch_finnhub_news()
    return n1 + n2 + n3
