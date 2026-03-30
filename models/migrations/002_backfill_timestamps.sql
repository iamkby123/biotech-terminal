-- Migration 002: Backfill TIMESTAMPTZ columns from TEXT date columns.
-- Safe to run repeatedly — only updates rows where _ts column IS NULL.
-- Uses regex guard to skip unparseable strings.

-- ── Trials: start_date ──
UPDATE trials
SET start_date_ts = start_date::TIMESTAMPTZ
WHERE start_date IS NOT NULL
  AND start_date_ts IS NULL
  AND start_date ~ '^\d{4}';

-- ── Trials: primary_completion_date ──
UPDATE trials
SET primary_completion_date_ts = primary_completion_date::TIMESTAMPTZ
WHERE primary_completion_date IS NOT NULL
  AND primary_completion_date_ts IS NULL
  AND primary_completion_date ~ '^\d{4}';

-- ── Trials: last_updated ──
UPDATE trials
SET last_updated_ts = last_updated::TIMESTAMPTZ
WHERE last_updated IS NOT NULL
  AND last_updated_ts IS NULL
  AND last_updated ~ '^\d{4}';

-- ── Catalysts: event_date ──
UPDATE catalysts
SET event_date_ts = event_date::TIMESTAMPTZ
WHERE event_date IS NOT NULL
  AND event_date_ts IS NULL
  AND event_date ~ '^\d{4}';

-- ── Filings: filed_date ──
UPDATE filings
SET filed_date_ts = filed_date::TIMESTAMPTZ
WHERE filed_date IS NOT NULL
  AND filed_date_ts IS NULL
  AND filed_date ~ '^\d{4}';

-- ── News: published_at ──
UPDATE news
SET published_at_ts = published_at::TIMESTAMPTZ
WHERE published_at IS NOT NULL
  AND published_at_ts IS NULL
  AND published_at ~ '^\d{4}';

-- ── Publications: pub_date ──
UPDATE publications
SET pub_date_ts = pub_date::TIMESTAMPTZ
WHERE pub_date IS NOT NULL
  AND pub_date_ts IS NULL
  AND pub_date ~ '^\d{4}';

-- ── Fix ingestion_log.started_at TEXT → TIMESTAMPTZ ──
-- This is a new column; the old TEXT column is kept.
DO $$ BEGIN
    ALTER TABLE ingestion_log ADD COLUMN IF NOT EXISTS started_at_ts TIMESTAMPTZ;
EXCEPTION WHEN others THEN NULL; END $$;

UPDATE ingestion_log
SET started_at_ts = started_at::TIMESTAMPTZ
WHERE started_at IS NOT NULL
  AND started_at_ts IS NULL
  AND started_at ~ '^\d{4}';
