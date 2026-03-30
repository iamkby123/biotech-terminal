-- Migration 001: Add provenance columns, missing news columns, and indexes.
-- All operations are IF NOT EXISTS / ADD COLUMN IF NOT EXISTS — safe to run repeatedly.

-- ── News table: add missing production columns ───────────────────────

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS image_url TEXT;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS finnhub_id TEXT;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS sentiment TEXT;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS sentiment_score REAL;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS catalyst_type TEXT;
EXCEPTION WHEN others THEN NULL; END $$;

-- ── Provenance: TIMESTAMPTZ columns alongside TEXT dates ─────────────
-- Non-destructive: TEXT columns kept for backward compatibility.

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS start_date_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS primary_completion_date_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS last_updated_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS fetched_at TIMESTAMPTZ DEFAULT NOW();
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS data_quality REAL;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE trials ADD COLUMN IF NOT EXISTS raw_payload JSONB;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE catalysts ADD COLUMN IF NOT EXISTS event_date_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE filings ADD COLUMN IF NOT EXISTS filed_date_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE news ADD COLUMN IF NOT EXISTS published_at_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE publications ADD COLUMN IF NOT EXISTS pub_date_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

-- ── Missing indexes ──────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_trials_intervention_trgm
    ON trials USING gin(intervention gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_news_tickers_trgm
    ON news USING gin(tickers gin_trgm_ops);

CREATE INDEX IF NOT EXISTS idx_filings_filed_date
    ON filings(filed_date DESC);

CREATE INDEX IF NOT EXISTS idx_news_sentiment
    ON news(sentiment);

CREATE INDEX IF NOT EXISTS idx_news_catalyst_type
    ON news(catalyst_type);

CREATE INDEX IF NOT EXISTS idx_catalysts_company
    ON catalysts(company);

CREATE INDEX IF NOT EXISTS idx_ingestion_log_finished
    ON ingestion_log(finished_at DESC);
