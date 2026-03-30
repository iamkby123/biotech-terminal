"""
Bulk import ALL trials from last 5 years
Uses multiple strategies to get comprehensive coverage:
1. Fetch by major therapeutic areas (cancer, cardiovascular, etc.)
2. Fetch by phase (Phase 2-4 for investment relevance)
3. Fetch by company sponsors (all 285 tickers)
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.clinicaltrials import fetch_trials_by_sponsor, fetch_trials_by_condition, _parse_study, _upsert_trial, CTGOV_BASE, CTGOV_RATE_LIMIT
from config import TARGET_TICKERS
from models.ticker_map import get_connection

import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bulk_import.log')
    ]
)
logger = logging.getLogger(__name__)

PROGRESS_FILE = "bulk_import_progress.json"

# Major therapeutic areas to fetch
THERAPEUTIC_AREAS = [
    "cancer", "oncology", "tumor",
    "cardiovascular", "heart",
    "neurology", "alzheimer", "parkinson",
    "immunology", "autoimmune", "rheumatoid",
    "infectious", "covid", "hiv",
    "rare disease", "genetic disorder",
    "diabetes", "metabolic",
    "respiratory", "asthma", "copd",
    "gastrointestinal", "liver",
    "dermatology", "psoriasis",
    "ophthalmology", "eye",
]


def load_progress() -> dict:
    """Load progress from file or return default"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
    return {
        "started_at": None,
        "completed_sponsors": [],
        "completed_conditions": [],
        "total_imported": 0,
        "status": "not_started"
    }


def save_progress(progress: dict):
    """Save progress to file"""
    progress["updated_at"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    logger.info(f"Progress saved: {progress['total_imported']} trials imported")


def fetch_all_trials_by_condition(condition: str, max_pages: int = 50) -> int:
    """Fetch all trials for a condition with pagination"""
    params = {
        "query.cond": condition,
        "pageSize": 100,
        "format": "json",
    }
    
    upserted = 0
    next_token = None
    conn = get_connection()
    cur = conn.cursor()
    
    logger.info(f"Fetching trials for condition: {condition}")
    
    with httpx.Client(timeout=60) as client:
        page = 1
        while page <= max_pages:
            if next_token:
                params["pageToken"] = next_token
            elif "pageToken" in params:
                del params["pageToken"]

            try:
                r = client.get(CTGOV_BASE, params=params)
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error for {condition}: {e}")
                break
            except httpx.TimeoutException:
                logger.error(f"Timeout for {condition}")
                break

            data = r.json()
            studies = data.get("studies", [])
            
            if not studies:
                break

            for study in studies:
                trial = _parse_study(study)
                if trial:
                    try:
                        _upsert_trial(cur, trial)
                        upserted += 1
                    except Exception as e:
                        logger.error(f"Error upserting {trial.get('nct_id')}: {e}")

            conn.commit()
            
            if page % 5 == 0:
                logger.info(f"  {condition} - Page {page}: {upserted} trials")
            
            next_token = data.get("nextPageToken")
            if not next_token:
                break

            page += 1
            time.sleep(CTGOV_RATE_LIMIT)

    cur.close()
    conn.close()
    logger.info(f"Condition '{condition}': {upserted} trials imported")
    return upserted


def bulk_import_all_trials(resume: bool = True):
    """
    Import all trials using multiple strategies
    """
    progress = load_progress()
    
    if resume and progress.get("status") == "running":
        logger.info("Resuming from previous run...")
    else:
        progress = {
            "started_at": datetime.now().isoformat(),
            "completed_sponsors": [],
            "completed_conditions": [],
            "total_imported": 0,
            "status": "running"
        }
    
    total_imported = progress.get("total_imported", 0)
    
    try:
        # Strategy 1: Fetch by therapeutic areas
        logger.info(f"\n{'='*60}")
        logger.info("STRATEGY 1: Fetching by therapeutic areas")
        logger.info(f"{'='*60}")
        
        for condition in THERAPEUTIC_AREAS:
            if condition in progress.get("completed_conditions", []):
                logger.info(f"Skipping {condition} (already completed)")
                continue
            
            count = fetch_all_trials_by_condition(condition, max_pages=50)
            total_imported += count
            
            progress["completed_conditions"].append(condition)
            progress["total_imported"] = total_imported
            save_progress(progress)
            
            logger.info(f"Progress: {len(progress['completed_conditions'])}/{len(THERAPEUTIC_AREAS)} conditions, {total_imported} total trials")
        
        # Strategy 2: Fetch by company sponsors
        logger.info(f"\n{'='*60}")
        logger.info("STRATEGY 2: Fetching by company sponsors")
        logger.info(f"{'='*60}")
        
        from models.ticker_map import get_ticker_info
        
        for ticker in TARGET_TICKERS:
            if ticker in progress.get("completed_sponsors", []):
                logger.info(f"Skipping {ticker} (already completed)")
                continue
            
            info = get_ticker_info(ticker)
            if not info or not info.get("company_name"):
                continue
            
            company_name = info["company_name"]
            logger.info(f"Fetching trials for {ticker} ({company_name})...")
            
            try:
                count = fetch_trials_by_sponsor(company_name)
                total_imported += count
                
                progress["completed_sponsors"].append(ticker)
                progress["total_imported"] = total_imported
                save_progress(progress)
                
                logger.info(f"{ticker}: {count} trials (Total: {total_imported})")
            except Exception as e:
                logger.error(f"Error fetching {ticker}: {e}")
            
            time.sleep(CTGOV_RATE_LIMIT)
        
        progress["status"] = "completed"
        progress["completed_at"] = datetime.now().isoformat()
        save_progress(progress)
        
        logger.info(f"\n{'='*60}")
        logger.info("BULK IMPORT COMPLETE!")
        logger.info(f"Total trials imported: {total_imported}")
        logger.info(f"Conditions: {len(progress['completed_conditions'])}")
        logger.info(f"Sponsors: {len(progress['completed_sponsors'])}")
        logger.info(f"{'='*60}")
        
        return {"success": True, "total_imported": total_imported}
        
    except KeyboardInterrupt:
        logger.warning("\nImport interrupted by user")
        progress["status"] = "interrupted"
        save_progress(progress)
        return {"success": False, "status": "interrupted", "total_imported": total_imported}
        
    except Exception as e:
        logger.error(f"\nImport failed: {e}")
        progress["status"] = "failed"
        progress["error"] = str(e)
        save_progress(progress)
        return {"success": False, "status": "failed", "error": str(e), "total_imported": total_imported}


def reset_progress():
    """Reset progress file to start fresh"""
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
        logger.info("Progress file reset. Next run will start from beginning.")
    else:
        logger.info("No progress file found.")


def show_status():
    """Show current import status"""
    progress = load_progress()
    
    print("\n" + "="*60)
    print("BULK IMPORT STATUS")
    print("="*60)
    print(f"Status: {progress.get('status', 'unknown')}")
    print(f"Started: {progress.get('started_at', 'N/A')}")
    print(f"Last updated: {progress.get('updated_at', 'N/A')}")
    print(f"Total imported: {progress.get('total_imported', 0):,} trials")
    print(f"Conditions completed: {len(progress.get('completed_conditions', []))}/{len(THERAPEUTIC_AREAS)}")
    print(f"Sponsors completed: {len(progress.get('completed_sponsors', []))}/{len(TARGET_TICKERS)}")
    
    if progress.get('error'):
        print(f"\nLast error: {progress['error']}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bulk import all clinical trials")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--reset", action="store_true", help="Reset progress and start fresh")
    parser.add_argument("--status", action="store_true", help="Show current status")
    
    args = parser.parse_args()
    
    if args.status:
        show_status()
    elif args.reset:
        reset_progress()
    else:
        result = bulk_import_all_trials(resume=args.resume)
        
        if not result["success"]:
            sys.exit(1)
