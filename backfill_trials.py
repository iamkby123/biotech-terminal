"""
Backfill script to update existing trials with detailed data from ClinicalTrials.gov
"""
import httpx
import json
import time
import logging
from models.ticker_map import get_connection
from config import CTGOV_BASE, CTGOV_RATE_LIMIT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_trial_details(nct_id: str) -> dict | None:
    """Fetch detailed trial data from ClinicalTrials.gov API v2"""
    url = f"{CTGOV_BASE}/{nct_id}"
    params = {"format": "json"}
    
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(url, params=params)
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for {nct_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching {nct_id}: {e}")
        return None


def parse_detailed_fields(study: dict) -> dict:
    """Extract detailed fields from study data"""
    ps = study.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    status_mod = ps.get("statusModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    design_mod = ps.get("designModule", {})
    description_mod = ps.get("descriptionModule", {})
    eligibility_mod = ps.get("eligibilityModule", {})
    contacts_mod = ps.get("contactsLocationsModule", {})
    
    # Extract locations
    locations_list = []
    for loc in contacts_mod.get("locations", []):
        location_info = {
            "facility": loc.get("facility", ""),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "status": loc.get("status", "")
        }
        locations_list.append(location_info)
    
    # Extract principal investigator
    pi = ""
    responsible_party = sponsor_mod.get("responsibleParty", {})
    if responsible_party.get("type") == "Principal Investigator":
        pi = responsible_party.get("investigatorFullName", "")
    
    # Extract study design
    design_allocation = design_mod.get("designInfo", {}).get("allocation", "")
    design_intervention = design_mod.get("designInfo", {}).get("interventionModel", "")
    study_design = f"{design_allocation} {design_intervention}".strip() if design_allocation or design_intervention else None
    
    return {
        "nct_id": id_mod.get("nctId"),
        "brief_summary": description_mod.get("briefSummary"),
        "detailed_description": description_mod.get("detailedDescription"),
        "locations": json.dumps(locations_list) if locations_list else None,
        "principal_investigator": pi if pi else None,
        "eligibility_criteria": eligibility_mod.get("eligibilityCriteria"),
        "study_type": design_mod.get("studyType"),
        "study_design": study_design,
        "why_stopped": status_mod.get("whyStopped"),
    }


def update_trial_with_details(cur, details: dict):
    """Update trial record with detailed fields"""
    cur.execute("""
        UPDATE trials SET
            brief_summary = %(brief_summary)s,
            detailed_description = %(detailed_description)s,
            locations = %(locations)s::jsonb,
            principal_investigator = %(principal_investigator)s,
            eligibility_criteria = %(eligibility_criteria)s,
            study_type = %(study_type)s,
            study_design = %(study_design)s,
            why_stopped = %(why_stopped)s
        WHERE nct_id = %(nct_id)s
    """, details)


def _log_backfill(nct_id: str, status: str, error: str = None):
    """Log backfill activity to ingestion_log"""
    from datetime import datetime
    conn = get_connection()
    cur = conn.cursor()
    started = datetime.utcnow().isoformat()
    cur.execute(
        """INSERT INTO ingestion_log
           (source, ticker, rows_fetched, rows_upserted, status, error_message, started_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        ("clinicaltrials_backfill", nct_id, 1, 1 if status == "ok" else 0, status, error, started),
    )
    conn.commit()
    cur.close()
    conn.close()


def backfill_trial_details(batch_size: int = 50):
    """Backfill detailed fields for all existing trials"""
    conn = get_connection()
    cur = conn.cursor()
    
    # Get all trials that need backfilling (missing brief_summary)
    cur.execute("""
        SELECT nct_id FROM trials 
        WHERE brief_summary IS NULL 
        ORDER BY nct_id
    """)
    trials_to_update = [row["nct_id"] for row in cur.fetchall()]
    
    logger.info(f"Found {len(trials_to_update)} trials to backfill")
    
    updated = 0
    errors = 0
    
    for i, nct_id in enumerate(trials_to_update):
        logger.info(f"Processing {i+1}/{len(trials_to_update)}: {nct_id}")
        
        study_data = fetch_trial_details(nct_id)
        if study_data:
            try:
                details = parse_detailed_fields(study_data)
                update_trial_with_details(cur, details)
                conn.commit()
                updated += 1
                _log_backfill(nct_id, "ok")
                logger.info(f"✓ Updated {nct_id}")
            except Exception as e:
                conn.rollback()
                errors += 1
                _log_backfill(nct_id, "error", str(e))
                logger.error(f"✗ Error updating {nct_id}: {e}")
        else:
            errors += 1
            _log_backfill(nct_id, "error", "Failed to fetch data")
            logger.warning(f"✗ Could not fetch data for {nct_id}")
        
        # Rate limiting
        time.sleep(CTGOV_RATE_LIMIT)
        
        # Progress update every batch
        if (i + 1) % batch_size == 0:
            logger.info(f"Progress: {i+1}/{len(trials_to_update)} processed, {updated} updated, {errors} errors")
    
    cur.close()
    conn.close()
    
    logger.info(f"Backfill complete: {updated} updated, {errors} errors")
    return {"updated": updated, "errors": errors}


if __name__ == "__main__":
    result = backfill_trial_details()
    print(f"\nBackfill Results:")
    print(f"  Updated: {result['updated']}")
    print(f"  Errors: {result['errors']}")
