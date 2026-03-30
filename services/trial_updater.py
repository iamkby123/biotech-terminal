"""
Trial Auto-Updater — pulls new/updated trials from ClinicalTrials.gov API daily.
Stores recent updates in the database and provides a feed of latest changes.
"""
import json
import logging
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"


def fetch_recent_trials(days_back: int = 7, max_results: int = 50) -> List[dict]:
    """Fetch trials updated in the last N days from ClinicalTrials.gov API."""
    since = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "format": "json",
        "pageSize": min(max_results, 100),
        "sort": "LastUpdatePostDate:desc",
        "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{since},MAX]",
    }

    try:
        url = CTGOV_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        trials = []
        for s in data.get("studies", []):
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod = proto.get("conditionsModule", {})
            design_mod = proto.get("designModule", {})
            desc_mod = proto.get("descriptionModule", {})
            enroll_mod = design_mod.get("enrollmentInfo", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            interventions = [i.get("name", "") for i in arms_mod.get("interventions", [])]

            trials.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status_mod.get("overallStatus", ""),
                "phase": ", ".join(design_mod.get("phases", [])),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                "condition": ", ".join(cond_mod.get("conditions", [])[:3]),
                "enrollment": enroll_mod.get("count"),
                "intervention": ", ".join(interventions[:3]),
                "brief_summary": (desc_mod.get("briefSummary") or "")[:300],
                "last_update": status_mod.get("lastUpdateSubmitDate", ""),
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                "has_results": s.get("hasResults", False),
                "study_type": design_mod.get("studyType", ""),
            })

        logger.info("Fetched %d recent trials from ClinicalTrials.gov (last %d days)", len(trials), days_back)
        return trials
    except Exception as e:
        logger.warning("Failed to fetch recent trials: %s", e)
        return []


def sync_to_database(trials: List[dict]) -> int:
    """Insert/update fetched trials into the local database."""
    if not trials:
        return 0

    try:
        from models.ticker_map import get_connection
        conn = get_connection()
        cur = conn.cursor()

        inserted = 0
        for t in trials:
            try:
                cur.execute("""
                    INSERT INTO trials (nct_id, title, phase, status, sponsor, condition,
                        enrollment, intervention, brief_summary, results_posted)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (nct_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        status = EXCLUDED.status,
                        phase = EXCLUDED.phase,
                        sponsor = EXCLUDED.sponsor,
                        condition = EXCLUDED.condition,
                        enrollment = EXCLUDED.enrollment,
                        intervention = EXCLUDED.intervention,
                        brief_summary = EXCLUDED.brief_summary,
                        results_posted = EXCLUDED.results_posted
                """, (
                    t["nct_id"], t["title"], t["phase"], t["status"],
                    t["sponsor"], t["condition"], t["enrollment"],
                    t["intervention"], t["brief_summary"],
                    1 if t.get("has_results") else 0,
                ))
                inserted += cur.rowcount
            except Exception as e:
                logger.debug("Skip trial %s: %s", t.get("nct_id"), e)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Synced %d trials to database", inserted)
        return inserted
    except Exception as e:
        logger.warning("Database sync failed: %s", e)
        return 0


def run_daily_update(days_back: int = 3, max_results: int = 50) -> dict:
    """Run the daily update: fetch from CTGOV + sync to DB."""
    t0 = time.time()
    trials = fetch_recent_trials(days_back=days_back, max_results=max_results)
    synced = sync_to_database(trials)
    elapsed = round(time.time() - t0, 2)

    return {
        "fetched": len(trials),
        "synced": synced,
        "elapsed": elapsed,
        "timestamp": datetime.now().isoformat(),
    }


def get_recently_updated(limit: int = 50) -> List[dict]:
    """Get recently updated trials from ClinicalTrials.gov API (live, not from DB)."""
    return fetch_recent_trials(days_back=7, max_results=limit)
