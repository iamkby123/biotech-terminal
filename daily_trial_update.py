"""
Daily incremental trial update - fetches trials updated in last 24 hours
Run this daily to keep the database fresh
"""
import os
import sys
import logging
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.clinicaltrials import fetch_trials_by_date_range
from models.ticker_map import get_connection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def daily_update(study_type: str = "Interventional") -> int:
    """
    Fetch trials updated in the last 24 hours
    Run this daily to keep database current
    """
    # Calculate yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    today = datetime.now()
    
    start_date = yesterday.strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")
    
    logger.info(f"Starting daily update: {start_date} to {end_date}")
    
    # Get current trial count for comparison
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM trials")
    before_count = cur.fetchone()["count"]
    cur.close()
    conn.close()
    
    # Fetch updated trials
    imported = fetch_trials_by_date_range(
        start_date=start_date,
        end_date=end_date,
        study_type=study_type
    )
    
    # Get new count
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) as count FROM trials")
    after_count = cur.fetchone()["count"]
    cur.close()
    conn.close()
    
    new_trials = after_count - before_count
    updated_trials = imported - new_trials
    
    logger.info(f"Daily update complete:")
    logger.info(f"  - Trials processed: {imported}")
    logger.info(f"  - New trials added: {new_trials}")
    logger.info(f"  - Existing trials updated: {updated_trials}")
    logger.info(f"  - Total trials in database: {after_count:,}")
    
    return imported


def update_recent_trials(days: int = 7) -> int:
    """
    Update trials from last N days (useful for catching up)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Updating trials from last {days} days: {start_date.date()} to {end_date.date()}")
    
    imported = fetch_trials_by_date_range(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    
    logger.info(f"Updated {imported} trials from last {days} days")
    return imported


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily trial update")
    parser.add_argument("--days", type=int, help="Update last N days instead of just yesterday")
    parser.add_argument("--study-type", default="Interventional", help="Study type filter")
    
    args = parser.parse_args()
    
    if args.days:
        update_recent_trials(args.days)
    else:
        daily_update(args.study_type)
