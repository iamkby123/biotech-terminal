"""
Stock data service — yfinance quotes and news.

Extracts _fetch_yf_news, _is_relevant_to_ticker, and stock_data logic from api.py.
Uses bounded TTL cache and returns SourceStatus for provenance.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

import yfinance as yf

from cache import stock_cache, yf_news_cache
from models.canonical import NewsArticle, StockQuote, SourceMeta
from normalize import safe_float
from source_status import SourceStatus

logger = logging.getLogger(__name__)


# ── Relevance filtering ───────────────────────────────────────────────


def _is_relevant_to_ticker(
    title: str, summary: str, ticker: str, company_name: str = ""
) -> bool:
    """Check if an article is actually about the given ticker/company."""
    combined = title + " " + summary
    combined_lower = combined.lower()

    # Ticker in parentheses: (MRNA), (LLY)
    if f"({ticker})" in combined:
        return True
    # $TICKER
    if f"${ticker}" in combined.upper():
        return True
    # Ticker as word boundary
    if re.search(r"\b" + re.escape(ticker) + r"\b", combined, re.IGNORECASE):
        return True
    # Company name variants
    if company_name:
        name_lower = company_name.lower()
        if name_lower in combined_lower:
            return True
        first_word = name_lower.split()[0].rstrip(",") if name_lower else ""
        if first_word and len(first_word) >= 4 and first_word in combined_lower:
            return True
    return False


# ── Stock quote ────────────────────────────────────────────────────────


def fetch_stock_quote(ticker: str) -> tuple[StockQuote | None, SourceStatus]:
    """
    Fetch real-time stock quote from yfinance.
    Returns (StockQuote, SourceStatus).
    """
    ticker = ticker.upper()

    # Check cache
    cached = stock_cache.get(ticker)
    if cached is not None:
        value, meta = cached
        return value, SourceStatus.ok(
            "yfinance", 1,
            data_mode="cached",
            cache_hit=True,
            ttl_remaining=meta.ttl_remaining,
        )

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        prev_close = info.get("previousClose")
        change_pct = None
        change_amt = None
        if price and prev_close:
            change_pct = ((price - prev_close) / prev_close) * 100
            change_amt = price - prev_close

        quote = StockQuote(
            ticker=ticker,
            price=price,
            change_percent=change_pct,
            change_amount=change_amt,
            market_cap=info.get("marketCap"),
            volume=info.get("volume") or info.get("regularMarketVolume"),
            pe_ratio=info.get("trailingPE") or info.get("forwardPE"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            employees=info.get("fullTimeEmployees"),
            city=info.get("city"),
            state=info.get("state"),
            country=info.get("country"),
            _meta=SourceMeta(
                source_name="yfinance",
                fetched_at=datetime.now(timezone.utc).isoformat(),
                data_mode="live",
            ),
        )

        stock_cache.set(ticker, quote)
        return quote, SourceStatus.ok("yfinance", 1)

    except Exception as e:
        logger.warning(f"yfinance quote failed [{ticker}]: {e}")
        return None, SourceStatus.error("yfinance", str(e))


# ── yfinance news ──────────────────────────────────────────────────────


def fetch_yf_news(
    ticker: str, company_name: str = ""
) -> tuple[list[NewsArticle], SourceStatus]:
    """
    Fetch news from yfinance for a ticker.
    Returns (list[NewsArticle], SourceStatus).
    """
    ticker = ticker.upper()

    # Check cache
    cached = yf_news_cache.get(ticker)
    if cached is not None:
        value, meta = cached
        return value, SourceStatus.ok(
            "yfinance:news", len(value),
            data_mode="cached",
            cache_hit=True,
            ttl_remaining=meta.ttl_remaining,
        )

    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news or []

        # Get company name from yfinance if not provided
        if not company_name:
            try:
                company_name = stock.info.get("shortName") or stock.info.get("longName") or ""
            except Exception:
                company_name = ""
    except Exception as e:
        logger.warning(f"yfinance news failed [{ticker}]: {e}")
        return [], SourceStatus.error("yfinance:news", str(e))

    from ingestion.finnhub_news import classify_catalyst, score_sentiment

    articles: list[NewsArticle] = []
    for article in raw_news:
        c = article.get("content", {})
        title = c.get("title", "").strip()
        if not title:
            continue

        summary = c.get("summary", "").strip()[:800]

        # Filter: only keep articles that mention this ticker/company
        if not _is_relevant_to_ticker(title, summary, ticker, company_name):
            continue

        # Parse date
        pub_date = c.get("pubDate") or c.get("displayTime") or ""
        published_at = None
        if pub_date:
            try:
                dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                published_at = pub_date[:19]

        # Thumbnail
        thumb = c.get("thumbnail")
        image_url = None
        if thumb and thumb.get("resolutions"):
            resolutions = sorted(
                thumb["resolutions"], key=lambda r: r.get("width", 0), reverse=True
            )
            image_url = resolutions[0].get("url")

        # Source
        provider = c.get("provider", {})
        source = provider.get("displayName", "Yahoo Finance")

        # URL
        url = None
        click_through = c.get("clickThroughUrl", {})
        if click_through:
            url = click_through.get("url")
        if not url:
            canonical = c.get("canonicalUrl", {})
            url = canonical.get("url") if canonical else None

        catalyst = classify_catalyst(title, summary)
        sentiment, sent_score = score_sentiment(title, summary)

        articles.append(
            NewsArticle(
                title=title,
                summary=summary,
                source=source,
                url=url,
                published_at=published_at,
                image_url=image_url,
                sentiment=sentiment,
                sentiment_score=sent_score,
                catalyst_type=catalyst,
                tickers=ticker,
                origin="yfinance",
                _meta=SourceMeta(
                    source_name="yfinance",
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                    data_mode="live",
                ),
            )
        )

    yf_news_cache.set(ticker, articles)
    return articles, SourceStatus.ok("yfinance:news", len(articles))
