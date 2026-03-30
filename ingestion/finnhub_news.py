import httpx
import time
import re
import logging
from datetime import datetime, timezone
from models.ticker_map import get_connection
from config import TARGET_TICKERS

logger = logging.getLogger(__name__)

FINNHUB_API_KEY = "ct2qh1pr01qhb1amkn3gct2qh1pr01qhb1amkn40"
FINNHUB_BASE = "https://finnhub.io/api/v1"

# ── Catalyst classification ──────────────────────────────────────────────

CATALYST_PATTERNS = {
    "FDA Approval": [
        r"\bfda\s+approv", r"\bfda\s+clear", r"\bfda\s+grant", r"\bfda\s+accept",
        r"\bema\s+approv", r"\bapproved\b.*\bdrug\b", r"\bnda\s+approv",
        r"\bbla\s+approv", r"\bsupplemental.*approv",
    ],
    "Clinical Trial": [
        r"\bphase\s+[1-4ivx]+", r"\bclinical\s+trial", r"\bclinical\s+data",
        r"\btrial\s+result", r"\bpivotal\s+(study|trial)", r"\bopen-label",
        r"\brandomized", r"\bdouble-blind", r"\bplacebo-controlled",
        r"\benrollment", r"\bpatient\s+enroll",
    ],
    "Data Readout": [
        r"\bdata\s+readout", r"\btopline\s+(data|result)", r"\btop-line\s+(data|result)",
        r"\binterim\s+(data|result|analysis)", r"\bprimary\s+endpoint",
        r"\befficacy\s+data", r"\bsafety\s+data", r"\boverall\s+survival",
        r"\bresponse\s+rate", r"\bstatistically\s+significant",
    ],
    "PDUFA Date": [
        r"\bpdufa", r"\baction\s+date", r"\btarget\s+date.*fda",
        r"\bfda.*target\s+date", r"\bprescription\s+drug\s+user\s+fee",
    ],
    "IPO/Offering": [
        r"\bipo\b", r"\binitial\s+public\s+offering", r"\bpublic\s+offering",
        r"\bshelf\s+offering", r"\bsecondary\s+offering", r"\bfollow-on\s+offering",
        r"\bshare\s+offering", r"\bstock\s+offering", r"\bpriced\s+offering",
    ],
    "M&A": [
        r"\bacquisition", r"\bacquire[sd]?\b", r"\bmerger\b", r"\bbuyout\b",
        r"\btakeover\b", r"\bbid\s+for\b", r"\blicensing\s+deal",
        r"\blicense\s+agreement", r"\bcollaboration\s+agreement",
        r"\bpartnership\b.*\bdeal\b",
    ],
    "Earnings": [
        r"\bearnings", r"\brevenue\s+report", r"\bquarterly\s+result",
        r"\bfinancial\s+result", r"\bq[1-4]\s+\d{4}", r"\bbeat\s+estimate",
        r"\bmiss\s+estimate", r"\bguidance\b",
    ],
    "Regulatory": [
        r"\badcom\b", r"\badvisory\s+committee", r"\bcrl\b",
        r"\bcomplete\s+response\s+letter", r"\bbreakthrough\s+therapy",
        r"\bfast\s+track", r"\bpriority\s+review", r"\borphan\s+drug",
        r"\baccelerated\s+approval", r"\bfda\s+warning",
    ],
    "Pipeline": [
        r"\bpipeline\b", r"\bnew\s+indication", r"\bexpand.*indication",
        r"\bpreclinical", r"\bind\s+filing", r"\binvestigational\s+new\s+drug",
        r"\bfirst.in.human", r"\bfirst.in.class",
    ],
}

_COMPILED_CATALYST = {
    cat: [re.compile(p, re.IGNORECASE) for p in patterns]
    for cat, patterns in CATALYST_PATTERNS.items()
}


def classify_catalyst(title: str, summary: str = "") -> str | None:
    text = (title or "") + " " + (summary or "")
    best, best_count = None, 0
    for cat, patterns in _COMPILED_CATALYST.items():
        count = sum(1 for p in patterns if p.search(text))
        if count > best_count:
            best, best_count = cat, count
    return best if best_count >= 1 else None


# ── Sentiment scoring ────────────────────────────────────────────────────

_POSITIVE_WORDS = {
    "approved", "breakthrough", "positive", "success", "effective", "efficacy",
    "significant", "outperform", "beat", "exceeded", "upgrade", "promising",
    "granted", "clearance", "benefit", "improved", "favorable", "strong",
    "milestone", "accelerated", "innovative", "achieved", "surpass", "robust",
}

_NEGATIVE_WORDS = {
    "failed", "failure", "rejected", "denied", "terminated", "halted",
    "discontinued", "adverse", "death", "deaths", "recalled", "recall",
    "downgrade", "miss", "missed", "warning", "concern", "risk", "decline",
    "disappointing", "setback", "suspend", "suspended", "lawsuit", "fraud",
    "violation", "unsafe", "toxic", "toxicity", "delay", "delayed",
}


def score_sentiment(title: str, summary: str = "") -> tuple[str, float]:
    text = ((title or "") + " " + (summary or "")).lower()
    words = set(re.findall(r"\w+", text))
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return "neutral", 0.0
    score = (pos - neg) / total  # -1 to +1
    if score > 0.2:
        return "positive", round(score, 2)
    elif score < -0.2:
        return "negative", round(score, 2)
    return "neutral", round(score, 2)


# ── Finnhub fetcher ──────────────────────────────────────────────────────

def _fetch_ticker_news(ticker: str, from_date: str, to_date: str) -> list[dict]:
    """Fetch news for a single ticker from Finnhub."""
    try:
        r = httpx.get(
            f"{FINNHUB_BASE}/company-news",
            params={
                "symbol": ticker,
                "from": from_date,
                "to": to_date,
                "token": FINNHUB_API_KEY,
            },
            timeout=15,
        )
        if r.status_code == 429:
            logger.warning(f"Finnhub rate limited on {ticker}, sleeping 60s")
            time.sleep(60)
            return _fetch_ticker_news(ticker, from_date, to_date)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Finnhub fetch failed [{ticker}]: {e}")
        return []


def _upsert_finnhub_news(cur, item: dict):
    cur.execute(
        """INSERT INTO news (source, category, title, summary, url, published_at,
                            tickers, image_url, finnhub_id, sentiment, sentiment_score, catalyst_type)
           VALUES (%(source)s, %(category)s, %(title)s, %(summary)s, %(url)s,
                   %(published_at)s, %(tickers)s, %(image_url)s, %(finnhub_id)s,
                   %(sentiment)s, %(sentiment_score)s, %(catalyst_type)s)
           ON CONFLICT (url) DO UPDATE SET
               title = EXCLUDED.title,
               summary = EXCLUDED.summary,
               published_at = COALESCE(EXCLUDED.published_at, news.published_at),
               tickers = COALESCE(EXCLUDED.tickers, news.tickers),
               image_url = COALESCE(EXCLUDED.image_url, news.image_url),
               finnhub_id = COALESCE(EXCLUDED.finnhub_id, news.finnhub_id),
               sentiment = COALESCE(EXCLUDED.sentiment, news.sentiment),
               sentiment_score = COALESCE(EXCLUDED.sentiment_score, news.sentiment_score),
               catalyst_type = COALESCE(EXCLUDED.catalyst_type, news.catalyst_type)""",
        item,
    )


def fetch_finnhub_news(tickers: list[str] = None, days_back: int = 7) -> int:
    """Fetch news from Finnhub for all target tickers."""
    if not FINNHUB_API_KEY or FINNHUB_API_KEY.startswith("ct2qh1"):
        logger.info("Finnhub API key not configured or invalid, skipping Finnhub news ingestion.")
        return 0

    if tickers is None:
        tickers = TARGET_TICKERS

    conn = get_connection()
    cur = conn.cursor()

    to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    from_date = (datetime.now(timezone.utc) - __import__("datetime").timedelta(days=days_back)).strftime("%Y-%m-%d")

    upserted = 0
    for i, ticker in enumerate(tickers):
        articles = _fetch_ticker_news(ticker, from_date, to_date)
        for art in articles[:20]:  # Cap per ticker to avoid noise
            title = (art.get("headline") or "").strip()
            if not title:
                continue

            summary = (art.get("summary") or "").strip()[:800]
            pub_ts = art.get("datetime")
            published_at = (
                datetime.fromtimestamp(pub_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                if pub_ts else None
            )

            catalyst = classify_catalyst(title, summary)
            sentiment, sent_score = score_sentiment(title, summary)

            item = {
                "source": art.get("source", "Finnhub"),
                "category": catalyst or "news",
                "title": title[:500],
                "summary": summary,
                "url": art.get("url"),
                "published_at": published_at,
                "tickers": ticker.upper(),
                "image_url": art.get("image") or None,
                "finnhub_id": str(art.get("id", "")) or None,
                "sentiment": sentiment,
                "sentiment_score": sent_score,
                "catalyst_type": catalyst,
            }
            try:
                _upsert_finnhub_news(cur, item)
                upserted += 1
            except Exception as e:
                logger.debug(f"Finnhub upsert skip [{title[:40]}]: {e}")
                conn.rollback()
                continue

        conn.commit()
        # Rate limit: Finnhub free tier = 60 req/min
        if (i + 1) % 55 == 0:
            logger.info(f"Finnhub rate limit pause after {i+1} tickers...")
            time.sleep(62)
        else:
            time.sleep(1.1)

    # Backfill catalyst/sentiment on existing rows that are missing them
    backfilled = _backfill_catalyst_sentiment(cur)
    conn.commit()

    cur.close()
    conn.close()
    logger.info(f"Finnhub news ingest done: {upserted} upserted, {backfilled} backfilled.")
    return upserted


def _backfill_catalyst_sentiment(cur) -> int:
    """Backfill catalyst_type and sentiment on existing news rows missing them."""
    cur.execute("SELECT id, title, summary FROM news WHERE sentiment IS NULL OR catalyst_type IS NULL LIMIT 2000")
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        title = row.get("title") or ""
        summary = row.get("summary") or ""
        catalyst = classify_catalyst(title, summary)
        sentiment, sent_score = score_sentiment(title, summary)
        cur.execute(
            """UPDATE news SET
                catalyst_type = COALESCE(catalyst_type, %s),
                sentiment = COALESCE(sentiment, %s),
                sentiment_score = COALESCE(sentiment_score, %s)
               WHERE id = %s""",
            (catalyst, sentiment, sent_score, row["id"]),
        )
        updated += 1
    return updated
