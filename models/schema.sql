-- trials is a VIEW over v2.trial (normalized schema with 565K rows).
-- Do NOT recreate as a table — the view is created in 001_create_schema.sql / migration.
-- CREATE TABLE IF NOT EXISTS trials (...) intentionally removed.

CREATE TABLE IF NOT EXISTS catalysts (
    id SERIAL PRIMARY KEY,
    company TEXT,
    drug_name TEXT,
    indication TEXT,
    event_type TEXT,
    event_date TEXT,
    ticker TEXT,
    source_url TEXT,
    ingested_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(company, drug_name, event_type, event_date)
);

CREATE TABLE IF NOT EXISTS filings (
    accession_number TEXT PRIMARY KEY,
    cik TEXT,
    ticker TEXT,
    form_type TEXT,
    filed_date TEXT,
    period_of_report TEXT,
    filing_url TEXT,
    pipeline_text TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS publications (
    pmid TEXT PRIMARY KEY,
    title TEXT,
    abstract TEXT,
    journal TEXT,
    pub_date TEXT,
    nct_id TEXT,  -- references v2.trial(nct_id) — no FK since trials is a view
    sponsor_match TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ticker_map (
    ticker TEXT PRIMARY KEY,
    company_name TEXT,
    cik TEXT,
    ctgov_sponsor_name TEXT,
    market_cap_bucket TEXT,
    competitors TEXT[],
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    ticker TEXT,
    rows_fetched INTEGER DEFAULT 0,
    rows_upserted INTEGER DEFAULT 0,
    status TEXT,
    error_message TEXT,
    started_at TEXT,
    finished_at TIMESTAMP DEFAULT NOW()
);

-- Indexes on trials removed — trials is a VIEW over v2.trial which has its own indexes.
CREATE INDEX IF NOT EXISTS idx_catalysts_date ON catalysts(event_date);
CREATE INDEX IF NOT EXISTS idx_catalysts_ticker ON catalysts(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker);
CREATE INDEX IF NOT EXISTS idx_filings_form ON filings(form_type);
CREATE INDEX IF NOT EXISTS idx_publications_nct ON publications(nct_id);

-- Enable pg_trgm extension for text search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Text search indexes on trials removed — trials is a VIEW; v2.trial has gin indexes already.

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    source TEXT,
    category TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    url TEXT UNIQUE,
    published_at TEXT,
    tickers TEXT,
    nct_id TEXT,
    image_url TEXT,
    finnhub_id TEXT,
    sentiment TEXT,
    sentiment_score REAL,
    catalyst_type TEXT,
    ingested_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_news_published ON news(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_source ON news(source);
CREATE INDEX IF NOT EXISTS idx_news_category ON news(category);
