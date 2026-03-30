"""
Deterministic entity resolution with cached lookup tables.

Replaces the scattered resolution logic in ticker_map.py, news.py,
fda_calendar.py, and api.py. Preloads lookup tables on startup for
consistent, fast resolution with confidence scoring.
"""
from __future__ import annotations

import difflib
import logging
import re
import threading
from dataclasses import dataclass
from typing import Optional

from normalize import normalize_sponsor, normalize_sponsor_first_word
from config import MANUAL_TICKER_OVERRIDES

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Result of entity resolution with confidence metadata."""
    ticker: str | None
    confidence: float         # 0.0 to 1.0
    match_reason: str         # "exact_ticker", "manual_override", "company_name", etc.
    matched_term: str | None  # The string that matched

    @property
    def is_match(self) -> bool:
        return self.ticker is not None

    @classmethod
    def no_match(cls) -> MatchResult:
        return cls(ticker=None, confidence=0.0, match_reason="no_match", matched_term=None)


# Tickers that are common English/biotech words.
# These should only match via company name, never by ticker symbol alone.
AMBIGUOUS_TICKERS = {
    "MRNA", "RNA", "DNA", "DRUG", "RARE", "EDIT", "FATE", "BEAM", "SAGE",
    "FOLD", "BLUE", "BOLD", "DAWN", "IRON", "GENE", "STEM", "CELL", "FORM",
    "NEXT", "PLUS", "PROS", "CASH", "FAST", "LIVE", "OPEN", "PLAN", "SAFE",
    "TELL", "TRUE", "WELL", "MASS", "GOLD", "REAL", "SELF", "UNIT", "TURN",
    "HOOK", "BAND", "EVER", "MIND", "TALK", "ALT", "VIR",
}


class EntityResolver:
    """
    Thread-safe entity resolver with preloaded lookup tables.

    Usage:
        resolver = EntityResolver()
        resolver.load(db_connection)  # Call once at startup
        result = resolver.resolve_sponsor("Moderna, Inc.")
        # result.ticker == "MRNA", result.confidence == 1.0
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._ticker_to_company: dict[str, str] = {}       # MRNA -> "Moderna, Inc."
        self._normalized_company: dict[str, str] = {}       # "moderna" -> MRNA
        self._normalized_sponsor: dict[str, str] = {}       # "modernatx" -> MRNA
        self._first_word: dict[str, str] = {}               # "moderna" -> MRNA
        self._all_tickers: set[str] = set()
        self._manual_overrides: dict[str, str] = {}         # "Genentech" -> "RHHBY"
        self._loaded = False

    def load(self, db_connection) -> None:
        """
        Preload lookup tables from ticker_map table.
        Call once at application startup.
        """
        with self._lock:
            try:
                cur = db_connection.cursor()
                cur.execute(
                    "SELECT ticker, company_name, ctgov_sponsor_name "
                    "FROM ticker_map WHERE ticker IS NOT NULL"
                )
                rows = cur.fetchall()
                cur.close()
            except Exception as e:
                logger.error(f"Failed to load entity resolver: {e}")
                return

            self._ticker_to_company.clear()
            self._normalized_company.clear()
            self._normalized_sponsor.clear()
            self._first_word.clear()
            self._all_tickers.clear()

            for row in rows:
                ticker = row["ticker"]
                company_name = row.get("company_name") or ""
                ctgov_name = row.get("ctgov_sponsor_name") or ""

                self._all_tickers.add(ticker)
                if company_name:
                    self._ticker_to_company[ticker] = company_name

                # Normalized company name -> ticker
                norm_company = normalize_sponsor(company_name)
                if norm_company and len(norm_company) >= 3:
                    self._normalized_company[norm_company] = ticker

                # Normalized ctgov sponsor name -> ticker
                norm_ctgov = normalize_sponsor(ctgov_name)
                if norm_ctgov and len(norm_ctgov) >= 3:
                    self._normalized_sponsor[norm_ctgov] = ticker

                # First word of company name (for fuzzy matching)
                first = normalize_sponsor_first_word(company_name)
                if first:
                    self._first_word[first] = ticker
                first_ctgov = normalize_sponsor_first_word(ctgov_name)
                if first_ctgov:
                    self._first_word[first_ctgov] = ticker

            # Register manual overrides
            for name, ticker in MANUAL_TICKER_OVERRIDES.items():
                norm = normalize_sponsor(name)
                self._manual_overrides[norm] = ticker
                self._normalized_company[norm] = ticker

            self._loaded = True
            logger.info(
                f"EntityResolver loaded: {len(self._all_tickers)} tickers, "
                f"{len(self._normalized_company)} company names, "
                f"{len(self._normalized_sponsor)} ctgov names, "
                f"{len(self._manual_overrides)} manual overrides"
            )

    def resolve_sponsor(self, sponsor: str) -> MatchResult:
        """
        Resolve a sponsor/company name to a ticker with confidence scoring.

        Resolution priority:
        1. Manual overrides (config.py)     -> confidence 1.0
        2. Exact normalized company name    -> confidence 1.0
        3. Exact normalized ctgov name      -> confidence 0.95
        4. First-word match (word >= 5 ch)  -> confidence 0.80
        5. Fuzzy match (>= 0.75 ratio)      -> confidence = ratio
        6. No match                          -> confidence 0.0
        """
        if not sponsor:
            return MatchResult.no_match()

        norm = normalize_sponsor(sponsor)
        if not norm:
            return MatchResult.no_match()

        with self._lock:
            # 1. Manual overrides
            if norm in self._manual_overrides:
                return MatchResult(
                    ticker=self._manual_overrides[norm],
                    confidence=1.0,
                    match_reason="manual_override",
                    matched_term=sponsor,
                )

            # 2. Exact normalized company name
            if norm in self._normalized_company:
                return MatchResult(
                    ticker=self._normalized_company[norm],
                    confidence=1.0,
                    match_reason="company_name",
                    matched_term=norm,
                )

            # 3. Exact normalized ctgov sponsor name
            if norm in self._normalized_sponsor:
                return MatchResult(
                    ticker=self._normalized_sponsor[norm],
                    confidence=0.95,
                    match_reason="ctgov_name",
                    matched_term=norm,
                )

            # 4. First-word match
            first = normalize_sponsor_first_word(sponsor)
            if first and first in self._first_word:
                return MatchResult(
                    ticker=self._first_word[first],
                    confidence=0.80,
                    match_reason="first_word",
                    matched_term=first,
                )

            # 5. Fuzzy match
            all_names = list(self._normalized_company.keys())
            matches = difflib.get_close_matches(norm, all_names, n=1, cutoff=0.75)
            if matches:
                matched_name = matches[0]
                ratio = difflib.SequenceMatcher(None, norm, matched_name).ratio()
                return MatchResult(
                    ticker=self._normalized_company[matched_name],
                    confidence=round(ratio, 3),
                    match_reason="fuzzy",
                    matched_term=matched_name,
                )

            return MatchResult.no_match()

    def resolve_ticker(self, ticker: str) -> MatchResult:
        """Check if a ticker is known and return its company name."""
        with self._lock:
            ticker = ticker.upper()
            if ticker in self._all_tickers:
                return MatchResult(
                    ticker=ticker,
                    confidence=1.0,
                    match_reason="exact_ticker",
                    matched_term=ticker,
                )
            return MatchResult.no_match()

    def get_company_name(self, ticker: str) -> str | None:
        """Get company name for a known ticker."""
        with self._lock:
            return self._ticker_to_company.get(ticker.upper())

    def get_all_tickers(self) -> set[str]:
        """Return the set of all known tickers."""
        with self._lock:
            return set(self._all_tickers)

    def get_name_to_ticker_map(self) -> dict[str, str]:
        """
        Get a mapping of company name variants to tickers.
        Used by news ticker extraction (replaces _build_name_to_ticker in news.py).
        """
        with self._lock:
            result = {}
            result.update(self._normalized_company)
            result.update(self._normalized_sponsor)
            result.update(self._first_word)
            return result

    def is_ambiguous(self, ticker: str) -> bool:
        """Check if a ticker symbol is a common word that causes false matches."""
        return ticker.upper() in AMBIGUOUS_TICKERS

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# Singleton instance — loaded at application startup
resolver = EntityResolver()
