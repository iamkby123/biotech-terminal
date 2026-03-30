"""
Deterministic news ranking and deduplication.

Replaces the ad-hoc yf_news + db_news concatenation in api.py.
Produces reproducible ranking based on weighted scoring.
"""
from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone
from typing import Optional

from models.canonical import NewsArticle, ScoredNewsArticle


# ── Source reliability weights (0.0 to 1.0, higher = more trusted) ───────

SOURCE_WEIGHTS: dict[str, float] = {
    # Premium biotech/pharma outlets
    "STAT News": 1.0,
    "BioPharma Dive": 0.95,
    "FierceBiotech": 0.90,
    "Endpoints News": 0.90,
    "BioSpace": 0.75,
    "MedCity News": 0.70,
    "PharmaLive": 0.65,
    # Regulatory/official
    "FDA Press Releases": 1.0,
    "NIH News": 0.90,
    "ClinicalTrials.gov": 0.80,
    # General financial
    "Reuters Health": 0.85,
    "Reuters": 0.85,
    "MarketWatch": 0.70,
    "Zacks": 0.65,
    "Simply Wall St.": 0.60,
    "Motley Fool": 0.55,
    "Yahoo Finance": 0.60,
    "Seeking Alpha": 0.60,
    "Benzinga": 0.55,
    "GuruFocus.com": 0.50,
    "StockStory": 0.50,
    "MarketBeat": 0.55,
    "Moneywise": 0.50,
    "InvestorPlace": 0.50,
    # Aggregators
    "Finnhub": 0.55,
}

# Default weight for unknown sources
_DEFAULT_SOURCE_WEIGHT = 0.40


# ── Catalyst priority (0.0 to 1.0, higher = more impactful) ─────────────

CATALYST_PRIORITY: dict[str, float] = {
    "FDA Approval": 1.0,
    "Data Readout": 0.9,
    "PDUFA Date": 0.85,
    "Regulatory": 0.8,
    "M&A": 0.75,
    "Clinical Trial": 0.7,
    "IPO/Offering": 0.6,
    "Earnings": 0.5,
    "Pipeline": 0.4,
}

# No catalyst = neutral priority
_DEFAULT_CATALYST_PRIORITY = 0.2


# ── Scoring functions ────────────────────────────────────────────────────

def source_weight(source: str | None) -> float:
    """Get reliability weight for a news source."""
    if not source:
        return _DEFAULT_SOURCE_WEIGHT
    # Try exact match first
    w = SOURCE_WEIGHTS.get(source)
    if w is not None:
        return w
    # Try case-insensitive partial match
    source_lower = source.lower()
    for name, weight in SOURCE_WEIGHTS.items():
        if name.lower() in source_lower or source_lower in name.lower():
            return weight
    return _DEFAULT_SOURCE_WEIGHT


def recency_score(published_at: str | None) -> float:
    """
    Score based on how recent the article is.
    Returns 1.0 for < 1 hour, decays exponentially, 0.0 for > 7 days.
    """
    if not published_at:
        return 0.1  # Unknown date = low score

    try:
        # Try parsing various formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                dt = datetime.strptime(published_at, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            try:
                dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return 0.1

        now = datetime.now(timezone.utc)
        age_hours = max(0, (now - dt).total_seconds() / 3600)

        if age_hours < 1:
            return 1.0
        elif age_hours < 6:
            return 0.9
        elif age_hours < 24:
            return 0.75
        elif age_hours < 48:
            return 0.6
        elif age_hours < 72:
            return 0.45
        elif age_hours < 168:  # 7 days
            return 0.3
        else:
            return 0.1

    except Exception:
        return 0.1


def relevance_score(article: NewsArticle, ticker: str) -> float:
    """
    Score how directly relevant an article is to a specific ticker.
    1.0 = ticker explicitly mentioned in title
    0.7 = ticker in summary
    0.3 = ticker only in tickers metadata field
    """
    if not ticker:
        return 0.5

    title = (article.title or "").upper()
    summary = (article.summary or "").upper()
    ticker_upper = ticker.upper()

    # Check title (strongest signal)
    if f"({ticker_upper})" in title or f"${ticker_upper}" in title:
        return 1.0
    if re.search(r'\b' + re.escape(ticker_upper) + r'\b', title):
        return 1.0

    # Check summary
    if f"({ticker_upper})" in summary or f"${ticker_upper}" in summary:
        return 0.7
    if re.search(r'\b' + re.escape(ticker_upper) + r'\b', summary):
        return 0.7

    # Check tickers metadata
    if article.tickers and ticker_upper in article.tickers.upper():
        return 0.3

    return 0.1


def catalyst_priority(catalyst_type: str | None) -> float:
    """Get priority weight for a catalyst type."""
    if not catalyst_type:
        return _DEFAULT_CATALYST_PRIORITY
    return CATALYST_PRIORITY.get(catalyst_type, _DEFAULT_CATALYST_PRIORITY)


# ── Deduplication ────────────────────────────────────────────────────────

def dedup_articles(articles: list[NewsArticle]) -> list[NewsArticle]:
    """
    Remove duplicate articles using two strategies:
    1. Exact URL match
    2. Title similarity > 90% (catches same article from different sources)
    """
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    result: list[NewsArticle] = []

    for article in articles:
        # Skip URL duplicates
        if article.url:
            if article.url in seen_urls:
                continue
            seen_urls.add(article.url)

        # Skip title-similarity duplicates
        title_lower = (article.title or "").lower().strip()
        if not title_lower:
            continue

        is_dup = False
        for prev_title in seen_titles:
            ratio = difflib.SequenceMatcher(
                None, title_lower, prev_title
            ).ratio()
            if ratio > 0.90:
                is_dup = True
                break

        if not is_dup:
            seen_titles.append(title_lower)
            result.append(article)

    return result


# ── Main ranking pipeline ───────────────────────────────────────────────

def rank_news(
    articles: list[NewsArticle],
    ticker: str,
    limit: int = 15,
) -> list[ScoredNewsArticle]:
    """
    Score, dedup, and rank news articles for a given ticker.

    Scoring weights:
        source reliability:  30%
        recency:             30%
        ticker relevance:    25%
        catalyst priority:   15%

    Returns top N articles sorted by score (descending), deterministic.
    """
    deduped = dedup_articles(articles)

    scored: list[ScoredNewsArticle] = []
    for article in deduped:
        sw = source_weight(article.source)
        rs = recency_score(article.published_at)
        rv = relevance_score(article, ticker)
        cp = catalyst_priority(article.catalyst_type)

        score = sw * 0.30 + rs * 0.30 + rv * 0.25 + cp * 0.15

        # Build ScoredNewsArticle from article fields
        scored_article = ScoredNewsArticle(
            title=article.title,
            url=article.url,
            summary=article.summary,
            published_at=article.published_at,
            source=article.source,
            image_url=article.image_url,
            sentiment=article.sentiment,
            sentiment_score=article.sentiment_score,
            catalyst_type=article.catalyst_type,
            tickers=article.tickers,
            nct_id=article.nct_id,
            category=article.category,
            origin=article.origin,
            _meta=article._meta,
            rank_score=round(score, 4),
            rank_components={
                "source": round(sw, 3),
                "recency": round(rs, 3),
                "relevance": round(rv, 3),
                "catalyst": round(cp, 3),
            },
        )
        scored.append(scored_article)

    # Sort by score descending, then by published_at descending for ties
    scored.sort(
        key=lambda a: (a.rank_score, a.published_at or ""),
        reverse=True,
    )

    return scored[:limit]
