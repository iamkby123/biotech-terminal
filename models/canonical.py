"""
Canonical data models for the biotech terminal.

Every data record flowing through the system should be represented
by one of these models. Each has a to_api_dict() method that returns
exactly the field names the frontend expects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SourceMeta:
    """Provenance metadata attached to data records."""
    source_name: str          # "yfinance", "clinicaltrials.gov", "rss:STAT News"
    fetched_at: str           # ISO 8601 timestamp
    data_mode: str = "live"   # "live", "cached", "snapshot"


@dataclass
class NewsArticle:
    """Canonical news article model."""
    title: str
    url: Optional[str] = None
    summary: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None
    image_url: Optional[str] = None
    sentiment: Optional[str] = None         # "positive", "negative", "neutral"
    sentiment_score: Optional[float] = None  # -1.0 to +1.0
    catalyst_type: Optional[str] = None     # "FDA Approval", "Clinical Trial", etc.
    tickers: Optional[str] = None           # Comma-separated
    nct_id: Optional[str] = None
    category: Optional[str] = None          # "news", "regulatory", "trial_update"
    origin: Optional[str] = None            # "yfinance", "rss", "finnhub", "db"
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        """Serialize for API response. Field names match frontend expectations."""
        return {
            "title": self.title,
            "summary": self.summary,
            "image_url": self.image_url,
            "url": self.url,
            "published_at": self.published_at,
            "source": self.source,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "catalyst_type": self.catalyst_type,
            "tickers": self.tickers,
            "origin": self.origin,
        }


@dataclass
class ScoredNewsArticle(NewsArticle):
    """News article with ranking score attached."""
    rank_score: float = 0.0
    rank_components: Optional[dict] = None

    def to_api_dict(self) -> dict:
        d = super().to_api_dict()
        # rank_score available for debugging but not sent by default
        return d


@dataclass
class ClinicalTrial:
    """Canonical clinical trial model."""
    nct_id: str
    title: Optional[str] = None
    sponsor: Optional[str] = None
    phase: Optional[str] = None              # Raw from source
    phase_normalized: Optional[str] = None   # After normalize_phase()
    status: Optional[str] = None             # Raw from source
    status_normalized: Optional[str] = None  # After normalize_status()
    condition: Optional[str] = None          # Semicolon-separated
    intervention: Optional[str] = None       # Semicolon-separated drug names
    enrollment: Optional[int] = None
    start_date: Optional[str] = None
    primary_completion_date: Optional[str] = None
    primary_endpoint: Optional[str] = None
    results_posted: int = 0
    last_updated: Optional[str] = None
    brief_summary: Optional[str] = None
    detailed_description: Optional[str] = None
    locations: Optional[str] = None          # JSON string
    principal_investigator: Optional[str] = None
    eligibility_criteria: Optional[str] = None
    study_type: Optional[str] = None
    study_design: Optional[str] = None
    why_stopped: Optional[str] = None
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        """Fields the frontend reads from trials."""
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "phase": self.phase,
            "condition": self.condition,
            "status": self.status,
            "results_posted": self.results_posted,
            "primary_completion_date": self.primary_completion_date,
            "enrollment": self.enrollment,
            "sponsor": self.sponsor,
            "intervention": self.intervention,
        }


@dataclass
class StockQuote:
    """Real-time stock quote from yfinance."""
    ticker: str
    price: Optional[float] = None
    change_percent: Optional[float] = None
    change_amount: Optional[float] = None
    market_cap: Optional[float] = None
    volume: Optional[int] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    employees: Optional[int] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "price": self.price,
            "change_percent": self.change_percent,
            "change_amount": self.change_amount,
            "market_cap": self.market_cap,
            "volume": self.volume,
            "pe_ratio": self.pe_ratio,
            "fifty_two_week_high": self.fifty_two_week_high,
            "fifty_two_week_low": self.fifty_two_week_low,
            "sector": self.sector,
            "industry": self.industry,
            "employees": self.employees,
            "city": self.city,
            "state": self.state,
            "country": self.country,
        }


@dataclass
class Filing:
    """SEC filing record."""
    accession_number: str
    form_type: Optional[str] = None
    filed_date: Optional[str] = None
    filing_url: Optional[str] = None
    ticker: Optional[str] = None
    cik: Optional[str] = None
    period_of_report: Optional[str] = None
    pipeline_text: Optional[str] = None
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "form_type": self.form_type,
            "filed_date": self.filed_date,
            "filing_url": self.filing_url,
            "ticker": self.ticker,
        }


@dataclass
class CatalystEvent:
    """FDA/regulatory catalyst event."""
    company: str
    drug_name: Optional[str] = None
    indication: Optional[str] = None
    event_type: Optional[str] = None    # "PDUFA", "AdCom"
    event_date: Optional[str] = None
    ticker: Optional[str] = None
    source_url: Optional[str] = None
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "event_date": self.event_date,
            "event_type": self.event_type,
            "company": self.company,
            "drug_name": self.drug_name,
            "indication": self.indication,
            "ticker": self.ticker,
        }


@dataclass
class Publication:
    """PubMed publication."""
    pmid: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    nct_id: Optional[str] = None
    sponsor_match: Optional[str] = None
    _meta: Optional[SourceMeta] = None


@dataclass
class InsiderTrade:
    """Insider trading record from yfinance."""
    date: str
    insider: str
    type: str       # "Buy", "Sell", etc.
    shares: float
    value: float
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "date": self.date,
            "insider": self.insider,
            "type": self.type,
            "shares": self.shares,
            "value": self.value,
        }


@dataclass
class InstitutionalHolding:
    """Institutional holder record from yfinance."""
    holder: str
    shares: str
    percent: str
    value: str
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "holder": self.holder,
            "shares": self.shares,
            "percent": self.percent,
            "value": self.value,
        }


@dataclass
class ManagementOfficer:
    """Company officer from yfinance."""
    name: str
    title: str
    since: str
    _meta: Optional[SourceMeta] = None

    def to_api_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "since": self.since,
        }
