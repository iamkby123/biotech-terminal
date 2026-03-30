import httpx
import time
import logging
import json
from datetime import datetime
from config import CTGOV_BASE, CTGOV_PAGE_SIZE, CTGOV_RATE_LIMIT
from models.ticker_map import get_connection, resolve_sponsor_to_ticker, update_ctgov_sponsor

logger = logging.getLogger(__name__)


def _parse_study(study: dict) -> dict | None:
    ps = study.get("protocolSection", {})
    id_mod = ps.get("identificationModule", {})
    status_mod = ps.get("statusModule", {})
    sponsor_mod = ps.get("sponsorCollaboratorsModule", {})
    design_mod = ps.get("designModule", {})
    conditions_mod = ps.get("conditionsModule", {})
    arms_mod = ps.get("armsInterventionsModule", {})
    outcomes_mod = ps.get("outcomesModule", {})
    description_mod = ps.get("descriptionModule", {})
    eligibility_mod = ps.get("eligibilityModule", {})
    contacts_mod = ps.get("contactsLocationsModule", {})
    ipd_mod = ps.get("ipdSharingStatementModule", {})

    nct_id = id_mod.get("nctId")
    if not nct_id:
        return None

    interventions = arms_mod.get("interventions", [])
    intervention_names = "; ".join(
        i.get("name", "") for i in interventions if i.get("name")
    )

    primary_outcomes = outcomes_mod.get("primaryOutcomes", [])
    primary_endpoint = "; ".join(
        o.get("measure", "") for o in primary_outcomes if o.get("measure")
    )

    phases = design_mod.get("phases", [])
    phase = "; ".join(phases) if phases else None

    conditions = conditions_mod.get("conditions", [])
    condition = "; ".join(conditions) if conditions else None

    has_results = study.get("hasResults", False)
    
    # Extract detailed description
    brief_summary = description_mod.get("briefSummary")
    detailed_description = description_mod.get("detailedDescription")
    
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
    
    # Extract eligibility criteria
    eligibility_criteria = eligibility_mod.get("eligibilityCriteria")
    
    # Extract study type and design
    study_type = design_mod.get("studyType")
    design_allocation = design_mod.get("designInfo", {}).get("allocation", "")
    design_intervention = design_mod.get("designInfo", {}).get("interventionModel", "")
    study_design = f"{design_allocation} {design_intervention}".strip() if design_allocation or design_intervention else None
    
    # Extract why stopped
    why_stopped = status_mod.get("whyStopped")

    return {
        "nct_id": nct_id,
        "title": id_mod.get("briefTitle"),
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name"),
        "phase": phase,
        "status": status_mod.get("overallStatus"),
        "condition": condition,
        "intervention": intervention_names or None,
        "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
        "start_date": status_mod.get("startDateStruct", {}).get("date"),
        "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date"),
        "primary_endpoint": primary_endpoint or None,
        "results_posted": 1 if has_results else 0,
        "last_updated": status_mod.get("lastUpdatePostDateStruct", {}).get("date"),
        # Extended fields
        "brief_summary": brief_summary,
        "detailed_description": detailed_description,
        "locations": json.dumps(locations_list) if locations_list else None,
        "principal_investigator": pi if pi else None,
        "eligibility_criteria": eligibility_criteria,
        "study_type": study_type,
        "study_design": study_design,
        "why_stopped": why_stopped,
    }


def _upsert_trial(cur, trial: dict):
    cur.execute(
        """INSERT INTO trials
           (nct_id, title, sponsor, phase, status, condition, intervention,
            enrollment, start_date, primary_completion_date, primary_endpoint,
            results_posted, last_updated, brief_summary, detailed_description,
            locations, principal_investigator, eligibility_criteria, study_type,
            study_design, why_stopped)
           VALUES (%(nct_id)s, %(title)s, %(sponsor)s, %(phase)s, %(status)s,
                   %(condition)s, %(intervention)s, %(enrollment)s, %(start_date)s,
                   %(primary_completion_date)s, %(primary_endpoint)s,
                   %(results_posted)s, %(last_updated)s, %(brief_summary)s,
                   %(detailed_description)s, %(locations)s, %(principal_investigator)s,
                   %(eligibility_criteria)s, %(study_type)s, %(study_design)s,
                   %(why_stopped)s)
           ON CONFLICT (nct_id) DO UPDATE SET
               title = EXCLUDED.title,
               sponsor = EXCLUDED.sponsor,
               phase = EXCLUDED.phase,
               status = EXCLUDED.status,
               condition = EXCLUDED.condition,
               intervention = EXCLUDED.intervention,
               enrollment = EXCLUDED.enrollment,
               start_date = EXCLUDED.start_date,
               primary_completion_date = EXCLUDED.primary_completion_date,
               primary_endpoint = EXCLUDED.primary_endpoint,
               results_posted = EXCLUDED.results_posted,
               last_updated = EXCLUDED.last_updated,
               brief_summary = EXCLUDED.brief_summary,
               detailed_description = EXCLUDED.detailed_description,
               locations = EXCLUDED.locations,
               principal_investigator = EXCLUDED.principal_investigator,
               eligibility_criteria = EXCLUDED.eligibility_criteria,
               study_type = EXCLUDED.study_type,
               study_design = EXCLUDED.study_design,
               why_stopped = EXCLUDED.why_stopped""",
        trial,
    )


def fetch_trials_by_sponsor(sponsor_name: str, status_filter: list[str] | None = None) -> int:
    params = {
        "query.spons": sponsor_name,
        "pageSize": CTGOV_PAGE_SIZE,
        "format": "json",
    }
    if status_filter:
        params["filter.overallStatus"] = ",".join(status_filter)

    upserted = 0
    next_token = None
    conn = get_connection()
    cur = conn.cursor()

    with httpx.Client(timeout=30) as client:
        while True:
            if next_token:
                params["pageToken"] = next_token
            elif "pageToken" in params:
                del params["pageToken"]

            try:
                r = client.get(CTGOV_BASE, params=params)
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"CTGOV HTTP error for {sponsor_name}: {e}")
                break

            data = r.json()
            studies = data.get("studies", [])

            for study in studies:
                trial = _parse_study(study)
                if trial:
                    _upsert_trial(cur, trial)
                    upserted += 1

            conn.commit()
            next_token = data.get("nextPageToken")
            if not next_token:
                break

            time.sleep(CTGOV_RATE_LIMIT)

    cur.close()
    conn.close()
    logger.info(f"CTGOV [{sponsor_name}]: {upserted} trials upserted.")
    return upserted


def fetch_trials_by_condition(condition: str, phase: list[str] | None = None) -> int:
    params = {
        "query.cond": condition,
        "pageSize": CTGOV_PAGE_SIZE,
        "format": "json",
    }
    if phase:
        for p in phase:
            params.setdefault("filter.phase", p)

    upserted = 0
    next_token = None
    conn = get_connection()
    cur = conn.cursor()

    with httpx.Client(timeout=30) as client:
        while True:
            if next_token:
                params["pageToken"] = next_token
            elif "pageToken" in params:
                del params["pageToken"]

            try:
                r = client.get(CTGOV_BASE, params=params)
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"CTGOV HTTP error for condition={condition}: {e}")
                break

            data = r.json()
            studies = data.get("studies", [])

            for study in studies:
                trial = _parse_study(study)
                if trial:
                    _upsert_trial(cur, trial)
                    upserted += 1

            conn.commit()
            next_token = data.get("nextPageToken")
            if not next_token:
                break

            time.sleep(CTGOV_RATE_LIMIT)

    cur.close()
    conn.close()
    logger.info(f"CTGOV [condition={condition}]: {upserted} trials upserted.")
    return upserted


def fetch_trials_by_date_range(
    start_date: str,
    end_date: str,
    study_type: str = "Interventional",
    page_size: int = 100,
    max_pages: int = 10
) -> int:
    """
    Fetch trials by date range from ClinicalTrials.gov
    Uses a condition-based query to limit results, then filters by date.
    
    Note: The ClinicalTrials.gov API v2 does not support direct date range
    filtering. This function fetches recent studies and filters by date.
    For comprehensive date-based imports, use fetch_trials_by_sponsor or
    fetch_trials_by_condition instead.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        study_type: Filter by study type (Interventional, Observational, etc.)
        page_size: Number of results per page (max 100)
        max_pages: Maximum number of pages to fetch (default 10)
    
    Returns:
        Number of trials upserted
    """
    from datetime import datetime
    
    # Parse date range for filtering
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Use a broad condition query to get studies
    # This limits the dataset while still providing good coverage
    params = {
        "pageSize": page_size,
        "format": "json",
        "query.cond": "cancer",  # Use a common condition to limit results
    }

    upserted = 0
    next_token = None
    conn = get_connection()
    cur = conn.cursor()

    logger.info(f"Fetching trials from {start_date} to {end_date} (limited to {max_pages} pages)")

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
                logger.error(f"CTGOV HTTP error for date range {start_date}-{end_date}: {e}")
                break
            except httpx.TimeoutException:
                logger.error(f"Timeout fetching date range {start_date}-{end_date}")
                break

            data = r.json()
            studies = data.get("studies", [])
            
            if not studies:
                break

            for study in studies:
                trial = _parse_study(study)
                if trial:
                    # Filter by date range
                    last_updated = trial.get("last_updated")
                    if last_updated:
                        try:
                            trial_dt = datetime.strptime(last_updated, "%Y-%m-%d")
                            if start_dt <= trial_dt <= end_dt:
                                # Filter by study type if specified
                                if study_type:
                                    trial_study_type = trial.get("study_type")
                                    if trial_study_type != study_type:
                                        continue
                                try:
                                    _upsert_trial(cur, trial)
                                    upserted += 1
                                except Exception as e:
                                    logger.error(f"Error upserting trial {trial.get('nct_id')}: {e}")
                        except ValueError:
                            # Skip if date parsing fails
                            pass

            conn.commit()
            
            logger.info(f"  Page {page}: {upserted} trials upserted so far...")
            
            next_token = data.get("nextPageToken")
            if not next_token:
                break

            page += 1
            time.sleep(CTGOV_RATE_LIMIT)

    cur.close()
    conn.close()
    logger.info(f"CTGOV [{start_date} to {end_date}]: {upserted} trials upserted.")
    return upserted


def ingest_for_tickers(tickers: list[str]) -> dict[str, int]:
    from models.ticker_map import get_ticker_info

    results = {}
    for ticker in tickers:
        info = get_ticker_info(ticker)
        if not info or not info.get("company_name"):
            logger.warning(f"No company name for ticker {ticker}, skipping CTGOV.")
            results[ticker] = 0
            continue

        company_name = info["company_name"]
        started = datetime.utcnow().isoformat()
        count = fetch_trials_by_sponsor(company_name)

        if count > 0:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT DISTINCT sponsor FROM trials WHERE sponsor ILIKE %s LIMIT 1",
                (f"%{company_name.split()[0]}%",),
            )
            sponsor_row = cur.fetchone()
            cur.close()
            conn.close()
            if sponsor_row:
                update_ctgov_sponsor(ticker, sponsor_row["sponsor"])

        _log_ingestion("clinicaltrials", ticker, count, count, "ok", None, started)
        results[ticker] = count
        time.sleep(CTGOV_RATE_LIMIT)

    return results


def _log_ingestion(source, ticker, fetched, upserted, status, error, started):
    """DEPRECATED: Use ingestion.common.log_ingestion instead."""
    from ingestion.common import log_ingestion
    log_ingestion(source, ticker, fetched, upserted, status, error, started)
