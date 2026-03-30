import os
import schedule
import time
import logging
from datetime import datetime
from config import TARGET_TICKERS

logger = logging.getLogger(__name__)


def run_daily_trials(tickers: list[str] = None):
    from ingestion.clinicaltrials import ingest_for_tickers
    tickers = tickers or TARGET_TICKERS
    logger.info(f"[Scheduler] Running daily ClinicalTrials ingest for {len(tickers)} tickers...")
    results = ingest_for_tickers(tickers)
    total = sum(results.values())
    logger.info(f"[Scheduler] ClinicalTrials done: {total} total trials upserted.")


def run_daily_date_range_update():
    """Daily update to fetch all trials updated in last 24 hours (not just company-specific)"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from daily_trial_update import daily_update
    logger.info("[Scheduler] Running daily date-range trial update...")
    count = daily_update()
    logger.info(f"[Scheduler] Daily update done: {count} trials processed.")


def run_daily_edgar(tickers: list[str] = None):
    from ingestion.edgar import ingest_for_tickers
    tickers = tickers or TARGET_TICKERS
    logger.info(f"[Scheduler] Running daily EDGAR 8-K ingest for {len(tickers)} tickers...")
    results = ingest_for_tickers(tickers, form_types=["8-K"], max_per_ticker=5)
    total = sum(results.values())
    logger.info(f"[Scheduler] EDGAR 8-K done: {total} total filings upserted.")


def run_weekly_edgar(tickers: list[str] = None):
    from ingestion.edgar import ingest_for_tickers
    tickers = tickers or TARGET_TICKERS
    logger.info(f"[Scheduler] Running weekly EDGAR 10-K/10-Q ingest for {len(tickers)} tickers...")
    results = ingest_for_tickers(tickers, form_types=["10-K", "10-Q"], max_per_ticker=5)
    total = sum(results.values())
    logger.info(f"[Scheduler] EDGAR 10-K/10-Q done: {total} total filings upserted.")


def run_weekly_fda():
    from ingestion.fda_calendar import ingest_all
    logger.info("[Scheduler] Running weekly FDA calendar ingest...")
    total = ingest_all()
    logger.info(f"[Scheduler] FDA calendar done: {total} events upserted.")


def run_weekly_pubmed(tickers: list[str] = None):
    from ingestion.pubmed import ingest_from_trials_table, ingest_for_tickers
    tickers = tickers or TARGET_TICKERS
    logger.info("[Scheduler] Running weekly PubMed ingest (NCT cross-reference + company)...")
    n1 = ingest_from_trials_table()
    n2 = sum(ingest_for_tickers(tickers).values())
    logger.info(f"[Scheduler] PubMed done: {n1 + n2} publications upserted.")


def run_all(tickers: list[str] = None):
    tickers = tickers or TARGET_TICKERS
    logger.info(f"[Scheduler] Full ingest for {len(tickers)} tickers...")
    run_daily_trials(tickers)
    run_weekly_edgar(tickers)
    run_weekly_fda()
    run_weekly_pubmed(tickers)
    logger.info("[Scheduler] Full ingest complete.")


def start_background_scheduler(tickers: list[str] = None):
    tickers = tickers or TARGET_TICKERS

    # Company-specific trial updates
    schedule.every().day.at("06:00").do(run_daily_trials, tickers=tickers)
    
    # Daily broad trial update (catches all biotech/pharma trials updated in last 24h)
    schedule.every().day.at("02:00").do(run_daily_date_range_update)
    
    schedule.every().day.at("06:30").do(run_daily_edgar, tickers=tickers)
    schedule.every().monday.at("07:00").do(run_weekly_edgar, tickers=tickers)
    schedule.every().monday.at("07:30").do(run_weekly_fda)
    schedule.every().monday.at("08:00").do(run_weekly_pubmed, tickers=tickers)

    logger.info("[Scheduler] Background scheduler started. Press Ctrl+C to stop.")
    logger.info("[Scheduler] Daily trial updates at 02:00 (broad) and 06:00 (company-specific)")
    while True:
        schedule.run_pending()
        time.sleep(60)
