"""
Biotech Terminal API — thin routing layer.

All business logic lives in services/. This file defines routes,
wires up startup events, and serves static HTML pages.
"""
import os
import logging
import subprocess
import sys
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models.ticker_map import get_connection
from source_status import wrap_response_flat, SourceStatus

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

app = FastAPI(title="Biotech Terminal")

# CORS for Next.js frontend
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://localhost:3001", "tauri://localhost", "https://tauri.localhost"], allow_methods=["*"], allow_headers=["*"])


# ── Startup: load EntityResolver ──────────────────────────────────────

@app.on_event("startup")
def startup_load_resolver():
    """Load the EntityResolver lookup tables on startup."""
    try:
        from models.entity_resolver import resolver
        conn = get_connection()
        resolver.load(conn)
        conn.close()
        logger.info("EntityResolver loaded at startup.")
    except Exception as e:
        logger.error(f"Failed to load EntityResolver at startup: {e}")


@app.on_event("startup")
def startup_load_models():
    """Load RF V6 + fine-tuned Qwen V5 prediction models."""
    try:
        from services.orchestrator import init_all_models
        init_all_models()
    except Exception as e:
        logger.error(f"Failed to load prediction models at startup: {e}")


@app.on_event("startup")
def startup_daily_sync():
    """Start background daily trial sync from ClinicalTrials.gov."""
    import threading
    def daily_sync_loop():
        import time as _time
        # Initial sync on startup (after 30s delay to let server fully start)
        _time.sleep(30)
        try:
            from services.trial_updater import run_daily_update
            result = run_daily_update(days_back=3, max_results=50)
            logger.info("Startup trial sync: %s", result)
        except Exception as e:
            logger.warning("Startup trial sync failed: %s", e)
        # Then sync every 24 hours
        while True:
            _time.sleep(86400)  # 24 hours
            try:
                from services.trial_updater import run_daily_update
                result = run_daily_update(days_back=3, max_results=50)
                logger.info("Daily trial sync: %s", result)
            except Exception as e:
                logger.warning("Daily trial sync failed: %s", e)

    t = threading.Thread(target=daily_sync_loop, daemon=True)
    t.start()
    logger.info("Daily trial sync thread started")


# ── Stock ──────────────────────────────────────────────────────────────

@app.get("/api/stock/{ticker}")
def stock_data(ticker: str):
    """Get real-time stock data from Yahoo Finance."""
    from services.stock import fetch_stock_quote
    quote, status = fetch_stock_quote(ticker)
    if quote is None:
        return JSONResponse({"error": status.error_message or "Failed to fetch stock data"}, status_code=500)
    return wrap_response_flat(quote.to_api_dict(), [status])


@app.get("/api/options/{ticker}")
def options_data(ticker: str):
    """Get options chain analysis for a ticker."""
    from services.options import fetch_options_chain
    data, status = fetch_options_chain(ticker)
    return wrap_response_flat(data, [status])


def _sanitize_for_json(obj):
    """Recursively replace NaN/Inf floats with None for JSON serialization."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj

@app.get("/api/insider/{ticker}")
def insider_data(ticker: str):
    """Get insider trading activity for a ticker."""
    from services.options import fetch_insider_activity
    data, status = fetch_insider_activity(ticker)
    return JSONResponse(_sanitize_for_json(wrap_response_flat(data, [status])))


# ── Trials ─────────────────────────────────────────────────────────────

@app.get("/api/trials")
def trials(limit: int = 100, phase: str = None, status: str = None, sponsor: str = None):
    from services.trials import query_trials
    rows, src_status = query_trials(limit=limit, phase=phase, status=status, sponsor=sponsor)
    return rows


@app.get("/api/trials/stats")
def trials_stats():
    from services.trials import get_trials_stats
    data, src_status = get_trials_stats()
    return data


# ── Catalysts ──────────────────────────────────────────────────────────

@app.get("/api/catalysts")
def catalysts(days: int = 180, ticker: str = None):
    conn = get_connection()
    cur = conn.cursor()
    try:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        cutoff = (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")
        if ticker:
            cur.execute("""
                SELECT event_date, event_type, company, drug_name, indication, ticker
                FROM catalysts WHERE event_date >= %s AND event_date <= %s
                AND ticker ILIKE %s ORDER BY event_date ASC
            """, (today, cutoff, ticker.upper()))
        else:
            cur.execute("""
                SELECT event_date, event_type, company, drug_name, indication, ticker
                FROM catalysts WHERE event_date >= %s AND event_date <= %s
                ORDER BY event_date ASC
            """, (today, cutoff))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    finally:
        conn.close()


# ── Filings ────────────────────────────────────────────────────────────

@app.get("/api/filings")
def filings(limit: int = 50, ticker: str = None):
    conn = get_connection()
    cur = conn.cursor()
    try:
        if ticker:
            cur.execute("""
                SELECT ticker, form_type, filed_date, filing_url
                FROM filings WHERE ticker ILIKE %s
                ORDER BY filed_date DESC NULLS LAST LIMIT %s
            """, (ticker.upper(), limit))
        else:
            cur.execute("""
                SELECT ticker, form_type, filed_date, filing_url
                FROM filings ORDER BY filed_date DESC NULLS LAST LIMIT %s
            """, (limit,))
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    finally:
        conn.close()


# ── Pipeline ───────────────────────────────────────────────────────────

@app.get("/api/pipeline/{ticker}")
def pipeline(ticker: str):
    from services.trials import get_pipeline
    data, statuses = get_pipeline(ticker)
    if "error" in data and "not found" in str(data.get("error", "")):
        return JSONResponse(data, status_code=404)
    return wrap_response_flat(data, statuses)


# ── Tickers ────────────────────────────────────────────────────────────

@app.get("/api/tickers")
def tickers():
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT tm.ticker, tm.company_name,
                   COUNT(DISTINCT t.nct_id) AS trial_count,
                   COUNT(DISTINCT f.accession_number) AS filing_count
            FROM ticker_map tm
            LEFT JOIN trials t ON t.sponsor ILIKE CONCAT('%', SPLIT_PART(tm.company_name, ' ', 1), '%')
            LEFT JOIN filings f ON f.ticker = tm.ticker
            WHERE tm.company_name IS NOT NULL
            GROUP BY tm.ticker, tm.company_name
            ORDER BY trial_count DESC
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    finally:
        conn.close()


# ── News ───────────────────────────────────────────────────────────────

@app.get("/api/news")
def news(
    page: int = 1,
    per_page: int = 20,
    category: str = None,
    source: str = None,
    ticker: str = None,
    catalyst: str = None,
    sentiment: str = None,
    search: str = None,
):
    from services.news import query_news
    result, src_status = query_news(
        page=page, per_page=per_page, category=category,
        source=source, ticker=ticker, catalyst=catalyst,
        sentiment=sentiment, search=search,
    )
    return result


@app.get("/api/news/stats")
def news_stats():
    from services.news import get_news_stats
    data, src_status = get_news_stats()
    return data


@app.get("/api/news/ingest")
def trigger_news_ingest():
    from ingestion.news import ingest_all_news
    n = ingest_all_news()
    return {"upserted": n}


@app.get("/api/news/ingest/finnhub")
def trigger_finnhub_ingest():
    from ingestion.finnhub_news import fetch_finnhub_news
    n = fetch_finnhub_news()
    return {"upserted": n}


@app.get("/api/news/sources")
def news_sources():
    from services.news import get_news_sources
    rows, src_status = get_news_sources()
    return rows


# ── Competitors ────────────────────────────────────────────────────────

@app.get("/api/competitors/{ticker}")
def competitors(ticker: str, indication: str = None):
    """Get competitive landscape for a ticker."""
    ticker = ticker.upper()
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Get this ticker's sponsor name(s) and conditions
        cur.execute("""
            SELECT DISTINCT t.sponsor, t.condition
            FROM trials t
            JOIN ticker_map tm ON t.sponsor ILIKE CONCAT('%%', SPLIT_PART(tm.company_name, ' ', 1), '%%')
            WHERE tm.ticker = %s AND t.condition IS NOT NULL
            LIMIT 50
        """, (ticker,))
        own_rows = cur.fetchall()

        if not own_rows:
            cur.execute("""
                SELECT DISTINCT condition FROM trials
                WHERE sponsor ILIKE (SELECT CONCAT('%%', SPLIT_PART(COALESCE(ctgov_sponsor_name, company_name), ' ', 1), '%%')
                                     FROM ticker_map WHERE ticker = %s LIMIT 1)
                AND condition IS NOT NULL LIMIT 50
            """, (ticker,))
            own_conditions = [r["condition"] for r in cur.fetchall()]
            own_sponsors = []
        else:
            own_conditions = list({r["condition"] for r in own_rows})
            own_sponsors = list({r["sponsor"] for r in own_rows})

        if not own_conditions:
            cur.close()
            return {"ticker": ticker, "indications": [], "competitors": []}

        # Filter to specific indication if requested
        if indication:
            target_conditions = [c for c in own_conditions if indication.lower() in c.lower()]
            if not target_conditions:
                target_conditions = own_conditions
        else:
            target_conditions = own_conditions

        # Find all sponsors in the same conditions
        placeholders = ",".join(["%s"] * len(target_conditions))
        cur.execute(f"""
            SELECT
                sponsor,
                COUNT(*) AS trial_count,
                MAX(phase) AS best_phase,
                SUM(CASE WHEN status IN ('RECRUITING','ACTIVE_NOT_RECRUITING','NOT_YET_RECRUITING','ENROLLING_BY_INVITATION') THEN 1 ELSE 0 END) AS active_count,
                SUM(CASE WHEN results_posted = 1 THEN 1 ELSE 0 END) AS results_count,
                SUM(CASE WHEN status = 'COMPLETED' THEN 1 ELSE 0 END) AS completed_count,
                STRING_AGG(DISTINCT condition, ' | ' ORDER BY condition) AS conditions,
                STRING_AGG(DISTINCT COALESCE(intervention,''), '; ' ORDER BY COALESCE(intervention,'')) AS drugs
            FROM trials
            WHERE condition IN ({placeholders})
            AND sponsor IS NOT NULL
            GROUP BY sponsor
            ORDER BY
                MAX(CASE phase
                    WHEN 'PHASE4' THEN 7 WHEN 'PHASE3' THEN 6 WHEN 'PHASE2; PHASE3' THEN 5
                    WHEN 'PHASE2' THEN 4 WHEN 'PHASE1; PHASE2' THEN 3 WHEN 'PHASE1' THEN 2
                    WHEN 'EARLY_PHASE1' THEN 1 ELSE 0 END) DESC,
                COUNT(*) DESC
            LIMIT 40
        """, target_conditions)
        comp_rows = cur.fetchall()

        # Map sponsors to tickers
        cur.execute("SELECT ticker, company_name, ctgov_sponsor_name FROM ticker_map WHERE company_name IS NOT NULL")
        tm_rows = cur.fetchall()
        cur.close()

        sponsor_to_ticker = {}
        for tm in tm_rows:
            name = (tm["company_name"] or "").lower()
            ctgov = (tm["ctgov_sponsor_name"] or "").lower()
            first = name.split()[0] if name else ""
            sponsor_to_ticker[first] = tm["ticker"]
            if ctgov:
                cfirst = ctgov.split()[0]
                sponsor_to_ticker[cfirst] = tm["ticker"]

        def resolve_ticker(sponsor):
            if not sponsor:
                return None
            first = sponsor.lower().split()[0]
            return sponsor_to_ticker.get(first)

        def clean_drugs(drugs_str):
            if not drugs_str:
                return []
            filler = {'placebo', 'vehicle', 'sham', 'control', 'saline', 'standard of care',
                       'normal saline', 'dummy', 'comparator', 'background'}
            seen, out = set(), []
            for d in drugs_str.split(';'):
                d = d.strip()
                if d and not any(f in d.lower() for f in filler) and d not in seen:
                    seen.add(d)
                    out.append(d)
            return out[:4]

        results = []
        for r in comp_rows:
            t = resolve_ticker(r["sponsor"])
            is_self = t == ticker or (r["sponsor"] in own_sponsors)
            results.append({
                "sponsor": r["sponsor"],
                "ticker": t,
                "is_self": is_self,
                "trial_count": r["trial_count"],
                "best_phase": r["best_phase"],
                "active_count": r["active_count"],
                "results_count": r["results_count"],
                "completed_count": r["completed_count"],
                "conditions": (r["conditions"] or "")[:120],
                "drugs": clean_drugs(r["drugs"] or ""),
            })

        all_indications = []
        seen_ind = set()
        for cond in own_conditions:
            for part in cond.split(";"):
                p = part.strip()
                if p and p not in seen_ind:
                    seen_ind.add(p)
                    all_indications.append(p)

        return {
            "ticker": ticker,
            "indications": all_indications[:10],
            "target_conditions": target_conditions[:5],
            "competitors": results,
        }
    finally:
        conn.close()


# ── SIC descriptions (used by entity endpoints) ───────────────────────

SIC_DESCRIPTIONS = {
    "2836": "Pharmaceutical Preparations", "2835": "In Vitro Diagnostic Substances",
    "2834": "Pharmaceutical Preparations", "2833": "Medicinal Chemicals & Botanical Products",
    "2830": "Drugs", "8731": "Commercial Physical & Biological Research",
    "5047": "Medical & Hospital Equipment", "2860": "Industrial Chemicals",
    "3841": "Surgical & Medical Instruments", "3826": "Laboratory Analytical Instruments",
    "7372": "Prepackaged Software", "3825": "Instruments for Measuring",
    "3827": "Optical Instruments", "2819": "Industrial Inorganic Chemicals",
}


# ── Entity endpoints (fixed cursor-after-close bugs) ──────────────────

@app.get("/api/entity/company/{ticker}")
def entity_company(ticker: str):
    """Get company entity data for dashboard pinning."""
    ticker = ticker.upper()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM ticker_map WHERE ticker = %s", (ticker,))
        info = cur.fetchone()
        if not info:
            cur.close()
            return JSONResponse({"error": f"Company {ticker} not found"}, status_code=404)
        info = dict(info)

        cur.execute("""
            SELECT COUNT(*) as cnt FROM trials
            WHERE sponsor ILIKE %s
        """, (f"%{info.get('company_name', '').split()[0]}%",))
        trial_count = cur.fetchone()["cnt"]

        cur.execute("""
            SELECT COUNT(*) as cnt FROM filings WHERE ticker = %s
        """, (ticker,))
        filing_count = cur.fetchone()["cnt"]

        cur.close()
        return {
            "entity_type": "company",
            "entity_id": ticker,
            "name": info.get("company_name") or ticker,
            "ticker": ticker,
            "trial_count": trial_count,
            "filing_count": filing_count,
            "cik": info.get("cik"),
        }
    finally:
        conn.close()


@app.get("/api/entity/drug/{drug_name}")
def entity_drug(drug_name: str):
    """Get drug entity data for dashboard pinning."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT intervention FROM trials
            WHERE intervention ILIKE %s LIMIT 1
        """, (f"%{drug_name}%",))
        drug_row = cur.fetchone()

        if not drug_row:
            cur.close()
            return JSONResponse({"error": f"Drug {drug_name} not found"}, status_code=404)

        cur.execute("""
            SELECT COUNT(*) as cnt FROM trials
            WHERE intervention ILIKE %s
        """, (f"%{drug_name}%",))
        trial_count = cur.fetchone()["cnt"]

        cur.close()
        return {
            "entity_type": "drug",
            "entity_id": drug_name,
            "name": drug_name,
            "trial_count": trial_count,
        }
    finally:
        conn.close()


@app.get("/api/entity/trial/{nct_id}")
def entity_trial(nct_id: str):
    """Get trial entity data for dashboard pinning."""
    nct_id = nct_id.upper()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT nct_id, title, sponsor, phase, status, condition,
                   intervention, enrollment, primary_completion_date,
                   results_posted FROM trials WHERE nct_id = %s
        """, (nct_id,))
        trial = cur.fetchone()

        if not trial:
            cur.close()
            return JSONResponse({"error": f"Trial {nct_id} not found"}, status_code=404)

        trial = dict(trial)
        cur.close()
        return {
            "entity_type": "trial",
            "entity_id": nct_id,
            "name": trial.get("title") or nct_id,
            "nct_id": nct_id,
            "sponsor": trial.get("sponsor"),
            "phase": trial.get("phase"),
            "status": trial.get("status"),
            "condition": trial.get("condition"),
            "intervention": trial.get("intervention"),
            "trial_data": trial,
        }
    finally:
        conn.close()


# ── Earnings ───────────────────────────────────────────────────────────

@app.get("/api/earnings")
def get_earnings_calendar(days: int = 90, ticker: str = None):
    from services.earnings import fetch_earnings
    data, statuses = fetch_earnings(days=days, ticker=ticker)
    return data


# ── Company detail ─────────────────────────────────────────────────────

@app.get("/api/company/{ticker}/quick")
def company_quick(ticker: str):
    """Fast lightweight endpoint: just trials + drugs from DB (no yfinance). For sidebar."""
    ticker = ticker.upper()
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Get sponsor name
        cur.execute("SELECT company_name, ctgov_sponsor_name FROM ticker_map WHERE ticker = %s", (ticker,))
        info = cur.fetchone()
        if not info:
            return {"trials": [], "drugs": []}
        sponsor = info.get("ctgov_sponsor_name") or info.get("company_name") or ticker

        # Trials (fast DB query)
        cur.execute("""
            SELECT nct_id, title, phase, status, condition, intervention
            FROM trials WHERE sponsor ILIKE %s
            ORDER BY phase DESC NULLS LAST LIMIT 25
        """, (f"%{sponsor}%",))
        trials = [dict(r) for r in cur.fetchall()]
        if not trials:
            first_word = sponsor.split()[0]
            cur.execute("""
                SELECT nct_id, title, phase, status, condition, intervention
                FROM trials WHERE sponsor ILIKE %s
                ORDER BY phase DESC NULLS LAST LIMIT 25
            """, (f"%{first_word}%",))
            trials = [dict(r) for r in cur.fetchall()]

        # Extract unique drugs from trials
        drugs = []
        seen = set()
        for t in trials:
            for d in str(t.get("intervention", "") or "").split(";"):
                d = d.strip()
                if d and len(d) > 2 and d.lower() not in seen:
                    seen.add(d.lower())
                    drugs.append({"name": d})
        cur.close()
        return {"trials": trials, "drugs": drugs[:15]}
    except Exception as e:
        logger.warning("company_quick failed: %s", e)
        return {"trials": [], "drugs": []}
    finally:
        try:
            from models.ticker_map import release_connection
            release_connection(conn)
        except Exception:
            conn.close()


@app.get("/api/company/{ticker}")
def company_detail(ticker: str):
    """Get comprehensive company data including insider, holdings, competitors, drugs."""
    from services.company import get_company_detail
    data, statuses = get_company_detail(ticker)
    if "error" in data:
        code = 404 if "not found" in str(data.get("error", "")) else 500
        return JSONResponse(data, status_code=code)
    return wrap_response_flat(data, statuses)


@app.get("/api/company/{ticker}/intel")
def company_intel(ticker: str):
    """Structured intel blocks with gap-fill analysis per section."""
    from services.company import get_company_detail
    from services.intel import build_intel_blocks
    data, statuses = get_company_detail(ticker)
    if "error" in data:
        code = 404 if "not found" in str(data.get("error", "")) else 500
        return JSONResponse(data, status_code=code)
    blocks = build_intel_blocks(data, statuses, ticker)
    return {"ticker": ticker, "sections": blocks}


# ── Summary ────────────────────────────────────────────────────────────

@app.get("/api/summary")
def summary():
    conn = get_connection()
    try:
        cur = conn.cursor()
        tables = ["trials", "catalysts", "filings", "publications", "ticker_map", "news"]
        counts = {}
        for t in tables:
            cur.execute(f"SELECT COUNT(*) AS n FROM {t}")
            counts[t] = cur.fetchone()["n"]

        cur.execute("""
            SELECT source, ticker, rows_upserted, status, finished_at
            FROM ingestion_log ORDER BY finished_at DESC LIMIT 20
        """)
        log = [dict(r) for r in cur.fetchall()]
        for row in log:
            if row.get("finished_at"):
                row["finished_at"] = str(row["finished_at"])

        cur.close()
        return {"counts": counts, "log": log}
    finally:
        conn.close()


# ── Drug detail ────────────────────────────────────────────────────────

def _fetch_ctgov_drug_trials(drug_name: str) -> list[dict]:
    """Fetch trials for a drug from ClinicalTrials.gov via curl."""
    try:
        import urllib.parse
        encoded = urllib.parse.quote(drug_name)
        result = subprocess.run(
            ["curl", "-s", "-f",
             f"https://clinicaltrials.gov/api/v2/studies?query.intr={encoded}&format=json&pageSize=50"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        import json as _j
        data = _j.loads(result.stdout)
        trials = []
        for study in data.get("studies", []):
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})

            nct_id = ident.get("nctId", "")
            if not nct_id:
                continue

            phases = "; ".join(design_mod.get("phases", []))
            interventions = arms_mod.get("interventions", [])
            int_names = "; ".join(i.get("name", "") for i in interventions)
            conditions = "; ".join(cond_mod.get("conditions", [])[:3])

            trials.append({
                "nct_id": nct_id,
                "title": ident.get("briefTitle", ""),
                "phase": phases,
                "status": status_mod.get("overallStatus", ""),
                "condition": conditions,
                "intervention": int_names,
                "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                "results_posted": 1 if status_mod.get("resultsFirstPostDateStruct") else 0,
                "_source": "clinicaltrials.gov",
            })
        return trials
    except Exception as e:
        logger.warning("CT.gov drug search failed for %s: %s", drug_name, e)
        return []


@app.get("/api/drug/{drug_name}")
def drug_detail(drug_name: str):
    """Get drug details and associated trials — local DB + ClinicalTrials.gov."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Step 1: Local DB — no limit, get ALL matching trials
        cur.execute("""
            SELECT nct_id, title, phase, status, condition, intervention,
                   primary_completion_date, start_date, last_updated, sponsor, results_posted
            FROM trials WHERE intervention ILIKE %s
            ORDER BY phase DESC NULLS LAST, primary_completion_date DESC NULLS LAST
        """, (f"%{drug_name}%",))
        db_trials = [dict(r) for r in cur.fetchall()]
        cur.close()
    finally:
        conn.close()

    # Step 2: Fetch from ClinicalTrials.gov
    ctgov_trials = _fetch_ctgov_drug_trials(drug_name)

    # Step 3: Merge + deduplicate (local DB takes priority)
    seen_nct = set()
    all_trials = []
    for t in db_trials:
        if t['nct_id'] not in seen_nct:
            seen_nct.add(t['nct_id'])
            all_trials.append(t)
    for t in ctgov_trials:
        if t['nct_id'] not in seen_nct:
            seen_nct.add(t['nct_id'])
            all_trials.append(t)

    # Step 4: Sort by phase DESC
    phase_order = {
        'PHASE4': 4, 'PHASE3': 3, 'PHASE2': 2, 'PHASE1': 1,
        'Phase 4': 4, 'Phase 3': 3, 'Phase 2': 2, 'Phase 1': 1,
        'Phase I': 1, 'Phase II': 2, 'Phase III': 3, 'Phase IV': 4,
        'Phase 1/2': 1.5, 'Phase 2/3': 2.5,
        'PHASE2; PHASE3': 2.5, 'PHASE1; PHASE2': 1.5,
    }
    all_trials.sort(key=lambda t: phase_order.get(t.get('phase', ''), 0), reverse=True)

    sponsors = list(set(t['sponsor'] for t in all_trials if t.get('sponsor')))[:5]

    max_phase = 'Phase 1'
    for t in all_trials:
        ph = t.get('phase', '')
        if phase_order.get(ph, 0) > phase_order.get(max_phase, 0):
            max_phase = ph

    conditions = list(set(t['condition'] for t in all_trials if t.get('condition')))[:5]

    return {
        "drug_name": drug_name,
        "phase": max_phase,
        "sponsors": sponsors,
        "conditions": conditions,
        "trials": all_trials,
        "trial_count": len(all_trials),
        "_sources": {"local_db": len(db_trials), "clinicaltrials_gov": len(ctgov_trials)},
    }


# ── Trial detail ───────────────────────────────────────────────────────

def _get_trial_outcome(nct_id: str) -> dict | None:
    """Look up the trial's own outcome from v2.outcome_signal + v5 training data."""
    from services.orchestrator import _known_outcomes, _init_known_outcomes
    _init_known_outcomes()

    # Check validated training data first
    if nct_id in _known_outcomes:
        ko = _known_outcomes[nct_id]
        label_map = {"success": "endpoint met", "failure": "endpoint not met"}
        return {
            "outcome": label_map.get(ko.get("outcome", ""), ko.get("outcome", "")),
            "reason": ko.get("reasons", ""),
            "source": "validated_training_data",
            "confidence": "high",
        }

    # Check v2.outcome_signal
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""SELECT coarse_label, coarse_source, coarse_confidence, evidence_snippet
            FROM v2.outcome_signal WHERE nct_id = %s
            AND coarse_label NOT IN ('not_evaluable', 'no_result')""", (nct_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            label = row["coarse_label"]
            display_map = {
                "positive": "endpoint met", "negative": "endpoint not met",
                "mixed": "mixed results", "inconclusive": "inconclusive",
                "safety_failure": "safety failure",
            }
            reason_map = {
                "positive": "primary endpoint met (p<0.05)",
                "negative": "primary endpoint not met (p>=0.05)",
                "mixed": "mixed results across endpoints",
                "safety_failure": "terminated due to safety concerns",
            }
            # Check for terminated-specific reasons from evidence_snippet
            snippet = row.get("evidence_snippet") or ""
            source = row.get("coarse_source") or ""
            if source in ("terminated_why_stopped", "status_terminated"):
                why_lower = snippet.lower()
                if "crl" in why_lower or "complete response letter" in why_lower:
                    display = "CRL received"
                elif any(w in why_lower for w in ["efficacy", "futility", "did not meet"]):
                    display = "lack of efficacy"
                elif any(w in why_lower for w in ["safety", "adverse", "toxicity"]):
                    display = "safety failure"
                elif any(w in why_lower for w in ["enrollment", "recruitment", "accrual"]):
                    display = "enrollment failure"
                elif any(w in why_lower for w in ["business", "funding", "financial", "strategic"]):
                    display = "business decision"
                elif any(w in why_lower for w in ["regulatory", "fda hold", "clinical hold"]):
                    display = "regulatory hold"
                else:
                    display = display_map.get(label, label)
            else:
                display = display_map.get(label, label)

            return {
                "outcome": display,
                "reason": reason_map.get(label, snippet[:200] if snippet else ""),
                "source": source,
                "confidence": str(row.get("coarse_confidence") or "medium"),
            }
    except Exception:
        pass
    return None


def _fetch_ctgov_trial(nct_id: str) -> dict | None:
    """Fetch full trial data from ClinicalTrials.gov via curl (bypasses User-Agent blocking)."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-f", f"https://clinicaltrials.gov/api/v2/studies/{nct_id.upper()}?format=json"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        import json as _json
        study = _json.loads(result.stdout)
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        desc_mod = proto.get("descriptionModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})
        design_info = design_mod.get("designInfo", {})
        masking = design_info.get("maskingInfo", {})
        design_str = " ".join(filter(None, [
            design_info.get("allocation", ""),
            design_info.get("interventionModel", ""),
            masking.get("masking", "") if isinstance(masking, dict) else "",
        ])).strip()
        locations = [
            {"facility": l.get("facility",""), "city": l.get("city",""),
             "state": l.get("state",""), "country": l.get("country",""), "status": l.get("status","")}
            for l in (contacts_mod.get("locations") or [])
        ]
        return {
            "nct_id": ident.get("nctId", nct_id.upper()),
            "title": ident.get("briefTitle", ""),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "phase": "; ".join(design_mod.get("phases", [])),
            "status": status_mod.get("overallStatus", ""),
            "condition": "; ".join(cond_mod.get("conditions", [])),
            "intervention": "; ".join(i.get("name","") for i in arms_mod.get("interventions",[])),
            "enrollment": design_mod.get("enrollmentInfo", {}).get("count"),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
            "results_posted": bool(status_mod.get("resultsFirstPostDateStruct")),
            "brief_summary": desc_mod.get("briefSummary", ""),
            "detailed_description": desc_mod.get("detailedDescription", ""),
            "study_type": design_mod.get("studyType", ""),
            "study_design": design_str or None,
            "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
            "why_stopped": status_mod.get("whyStopped", ""),
            "principal_investigator": "",
            "locations": locations,
            "_source": "clinicaltrials.gov",
        }
    except Exception as e:
        logger.warning("CT.gov fetch failed for %s: %s", nct_id, e)
        return None


@app.get("/api/trial/{nct_id}")
def trial_detail(nct_id: str):
    from services.trials import get_trial_detail
    data, src_status = get_trial_detail(nct_id)

    # If DB has data but it's incomplete, try to fill gaps from CT.gov
    if data is not None:
        missing_key_fields = not data.get("brief_summary") and not data.get("eligibility_criteria") and not data.get("locations")
        if missing_key_fields:
            ctgov = _fetch_ctgov_trial(nct_id)
            if ctgov:
                for key in ["brief_summary", "detailed_description", "eligibility_criteria",
                            "locations", "study_type", "study_design", "principal_investigator", "why_stopped"]:
                    if not data.get(key) and ctgov.get(key):
                        data[key] = ctgov[key]
                data["_enriched_from"] = "clinicaltrials.gov"

        # Attach trial outcome from v2.outcome_signal
        data["trial_outcome"] = _get_trial_outcome(nct_id)
        return data

    # Not in DB at all — fetch entirely from CT.gov
    ctgov = _fetch_ctgov_trial(nct_id)
    if ctgov:
        return ctgov
    try:
        import subprocess, json as _json
        result = subprocess.run(
            ["curl", "-s", "-f", f"https://clinicaltrials.gov/api/v2/studies/{nct_id.upper()}?format=json"],
            capture_output=True, text=True, timeout=20,
        )
        if result.returncode == 0 and result.stdout.strip():
            study = _json.loads(result.stdout)
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod = proto.get("conditionsModule", {})
            arms_mod = proto.get("armsInterventionsModule", {})
            desc_mod = proto.get("descriptionModule", {})
            elig_mod = proto.get("eligibilityModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            interventions = arms_mod.get("interventions", [])
            int_names = "; ".join(i.get("name", "") for i in interventions)
            conditions = "; ".join(cond_mod.get("conditions", []))
            phases = "; ".join(design_mod.get("phases", []))
            design_info = design_mod.get("designInfo", {})
            masking = design_info.get("maskingInfo", {})
            design_str = " ".join(filter(None, [
                design_info.get("allocation", ""),
                design_info.get("interventionModel", ""),
                masking.get("masking", "") if isinstance(masking, dict) else "",
            ]))
            enroll = design_mod.get("enrollmentInfo", {}).get("count")

            locations = []
            for loc in (contacts_mod.get("locations") or []):
                locations.append({
                    "facility": loc.get("facility", ""),
                    "city": loc.get("city", ""),
                    "state": loc.get("state", ""),
                    "country": loc.get("country", ""),
                    "status": loc.get("status", ""),
                })

            return {
                "nct_id": ident.get("nctId", nct_id.upper()),
                "title": ident.get("briefTitle", ""),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                "phase": phases,
                "status": status_mod.get("overallStatus", ""),
                "condition": conditions,
                "intervention": int_names,
                "enrollment": enroll,
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "primary_completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
                "results_posted": bool(status_mod.get("resultsFirstPostDateStruct")),
                "brief_summary": desc_mod.get("briefSummary", ""),
                "detailed_description": desc_mod.get("detailedDescription", ""),
                "study_type": design_mod.get("studyType", ""),
                "study_design": design_str,
                "eligibility_criteria": elig_mod.get("eligibilityCriteria", ""),
                "why_stopped": status_mod.get("whyStopped", ""),
                "principal_investigator": "",
                "locations": locations,
                "_source": "clinicaltrials.gov",
            }
    except Exception as e:
        logger.warning("ClinicalTrials.gov fallback failed for %s: %s", nct_id, e)

    return JSONResponse({"error": f"Trial {nct_id} not found"}, status_code=404)


# ── Backfill ───────────────────────────────────────────────────────────

_backfill_process = None


@app.get("/api/backfill/status")
def backfill_status():
    """Get current backfill progress status."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM trials WHERE brief_summary IS NOT NULL")
        updated = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM trials")
        total = cur.fetchone()["count"]

        cur.execute("""
            SELECT source, ticker as nct_id, status, error_message, finished_at as time
            FROM ingestion_log
            WHERE source = 'clinicaltrials_backfill'
            ORDER BY finished_at DESC
            LIMIT 20
        """)
        logs = cur.fetchall()
        cur.close()

        is_running = False
        if logs:
            last_activity = logs[0]["time"]
            if last_activity:
                from datetime import timezone
                last_time = datetime.fromisoformat(str(last_activity).replace('Z', '+00:00'))
                is_running = (datetime.now(last_time.tzinfo) - last_time) < timedelta(minutes=2)

        global _backfill_process
        if _backfill_process is not None and _backfill_process.poll() is None:
            is_running = True

        recent_logs = []
        for log in logs[:10]:
            recent_logs.append({
                "nct_id": log["nct_id"] or "Unknown",
                "status": "success" if log["status"] == "ok" else "error",
                "time": str(log["time"]) if log["time"] else None,
                "message": log["error_message"] if log["error_message"] else None,
            })

        TARGET_TRIALS = 200000
        return {
            "updated": updated,
            "total": total,
            "target": TARGET_TRIALS,
            "percent_complete": (updated / total * 100) if total > 0 else 0,
            "percent_target": (total / TARGET_TRIALS * 100) if TARGET_TRIALS > 0 else 0,
            "is_running": is_running,
            "recent_logs": recent_logs,
            "message": f"{total:,} trials in database (Target: {TARGET_TRIALS:,})",
        }
    finally:
        conn.close()


@app.post("/api/backfill/start")
def start_backfill(background_tasks: BackgroundTasks):
    """Start the bulk import backfill process."""
    global _backfill_process
    if _backfill_process is not None and _backfill_process.poll() is None:
        return JSONResponse({"success": False, "error": "Backfill is already running"})

    try:
        env = os.environ.copy()
        env["DATABASE_URL"] = DATABASE_URL
        _backfill_process = subprocess.Popen(
            [sys.executable, "bulk_import_trials.py", "--resume"],
            stdout=open("bulk_import.log", "a"),
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        return {
            "success": True,
            "message": "Backfill started successfully.",
            "pid": _backfill_process.pid,
        }
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/api/backfill/stop")
def stop_backfill():
    """Stop the running backfill process."""
    global _backfill_process
    if _backfill_process is None or _backfill_process.poll() is not None:
        return JSONResponse({"success": False, "error": "No backfill is currently running"})

    try:
        _backfill_process.terminate()
        _backfill_process.wait(timeout=5)
        _backfill_process = None
        return {"success": True, "message": "Backfill stopped successfully"}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── Agent (LLM data-gathering) ─────────────────────────────────────────

from pydantic import BaseModel

class AgentRequest(BaseModel):
    query: str

@app.post("/api/agent/gather")
def agent_gather(req: AgentRequest):
    """LLM-powered data-gathering agent (qwen3:8b via Ollama)."""
    from agent.loop import run_agent
    return run_agent(req.query)


@app.get("/api/screening/upcoming")
def screening_upcoming(phase: str = None, area: str = None, limit: int = 20):
    """Screen ClinicalTrials.gov for trials likely to report results soon."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
    from trial_screener import screen_trials
    return screen_trials(phase=phase, area=area, limit=limit)


# ── AI Intelligence Layer ──────────────────────────────────────────────

@app.get("/api/ticker-intel/{ticker}")
def ticker_intel(ticker: str):
    """Full AI orchestration: predict + RAG + analysis for a ticker."""
    from services.company import get_company_detail
    from services.orchestrator import run_ticker_intel

    data, statuses = get_company_detail(ticker)
    if "error" in data:
        return JSONResponse(data, status_code=404)

    trials = data.get("trials") or []
    drugs = data.get("drugs") or []
    result = run_ticker_intel(ticker.upper(), trials, drugs)
    result["verified_data"] = {
        "trials_count": len(trials),
        "drugs_count": len(drugs),
    }
    return result


@app.get("/api/trial-narrative/{nct_id}")
def trial_narrative(nct_id: str):
    """Generate Claude AI narrative analysis for a trial. Cached 24h."""
    from services.orchestrator import predict_trial_for_ticker, _load_rag_evidence, _load_drug_science

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT nct_id, title, phase, status, condition, intervention,
                       sponsor, enrollment, study_design, brief_summary
                       FROM trials WHERE nct_id = %s""", (nct_id.upper(),))
        row = cur.fetchone()
        cur.close()
    finally:
        try:
            from models.ticker_map import release_connection
            release_connection(conn)
        except Exception:
            conn.close()

    if not row:
        return {"narrative": None}

    trial = dict(row)

    # Get prediction (cached)
    _, prediction = predict_trial_for_ticker(trial)

    # Get evidence
    rag_evidence = _load_rag_evidence(trial)
    drug_science = _load_drug_science(trial.get("intervention", ""))

    # Generate narrative
    from services.claude_analysis import generate_narrative
    result = generate_narrative(trial, prediction, rag_evidence, drug_science)
    return {"narrative": result.get("narrative") if result else None,
            "model": result.get("model") if result else None}


@app.get("/api/trial-intel/{nct_id}")
def trial_intel(nct_id: str):
    """AI prediction + analysis for a single trial by NCT ID."""
    from services.orchestrator import predict_trial_for_ticker, retrieve_supporting, generate_analysis

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT nct_id, title, phase, status, condition, intervention,
                   primary_completion_date, sponsor, results_posted,
                   enrollment, study_design, brief_summary
            FROM trials WHERE nct_id = %s
        """, (nct_id.upper(),))
        row = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    if not row:
        ctgov = _fetch_ctgov_trial(nct_id)
        if ctgov:
            row = ctgov

    if not row:
        return JSONResponse({"error": f"Trial {nct_id} not found"}, status_code=404)

    trial = dict(row) if not isinstance(row, dict) else row
    from services.orchestrator import classify_domain
    domain = classify_domain(trial)
    from services.orchestrator import predict_trial_for_ticker
    domain, prediction = predict_trial_for_ticker(trial)
    supporting = retrieve_supporting(trial, domain)
    analysis = generate_analysis(trial, prediction, supporting, domain)

    # Ensure trial is enriched with AACT data (may not happen if prediction was cached)
    try:
        from services.ensemble_signals import _enrich_trial_from_aact
        _enrich_trial_from_aact(trial)
    except Exception:
        pass

    from services.orchestrator import compute_trial_analytics
    analytics = compute_trial_analytics(trial)

    # Build focal trial with enriched AACT fields
    masking_labels = {0: "Open Label", 1: "Single Blind", 2: "Double Blind",
                      3: "Triple Blind", 4: "Quadruple Blind"}
    masking_level = 0
    masking_str = str(trial.get("masking", "") or "").upper()
    if "QUADRUPLE" in masking_str: masking_level = 4
    elif "TRIPLE" in masking_str: masking_level = 3
    elif "DOUBLE" in masking_str: masking_level = 2
    elif "SINGLE" in masking_str: masking_level = 1

    focal_trial = {
        "nct_id": trial.get("nct_id", ""),
        "title": trial.get("title", ""),
        "phase": trial.get("phase", ""),
        "condition": trial.get("condition", ""),
        "enrollment": trial.get("enrollment"),
        "intervention": trial.get("intervention", ""),
        "status": trial.get("status", ""),
        "sponsor": trial.get("sponsor", ""),
    }
    # Add enriched AACT fields if available
    if trial.get("allocation"):
        focal_trial["allocation"] = str(trial["allocation"]).title()
    if masking_level or trial.get("masking"):
        focal_trial["masking"] = masking_labels.get(masking_level, str(trial.get("masking", "")))
    if trial.get("intervention_model"):
        focal_trial["intervention_model"] = str(trial["intervention_model"]).title()
    if trial.get("primary_purpose"):
        focal_trial["primary_purpose"] = str(trial["primary_purpose"]).title()
    if trial.get("number_of_arms") or trial.get("num_arms"):
        focal_trial["num_arms"] = int(trial.get("number_of_arms") or trial.get("num_arms") or 0)
    if trial.get("minimum_age") or trial.get("min_age"):
        focal_trial["age_range"] = str(trial.get("minimum_age", trial.get("min_age", "")))
        if trial.get("maximum_age") or trial.get("max_age"):
            focal_trial["age_range"] += f" – {trial.get('maximum_age', trial.get('max_age', ''))}"
    if trial.get("study_duration_months"):
        focal_trial["duration_months"] = round(float(trial["study_duration_months"]), 1)
    if trial.get("num_countries"):
        focal_trial["num_countries"] = int(trial["num_countries"])

    return {
        "focal_trial": focal_trial,
        "domain": domain,
        "domain_uncertain": False,
        "prediction": prediction,
        "reason_flags": prediction.get("reasons", {}),
        "analysis": analysis,
        "supporting_trials": supporting,
        "analytics": analytics,
    }


def _get_prediction_v2(trial):
    """Run Predictor V2 (Gradient Boosting) if available."""
    try:
        from services.predictor_v2 import predict_v2
        return predict_v2(trial)
    except Exception as e:
        logger.warning("Predictor V2 failed: %s", e)
        return None


@app.get("/api/drug-intel/{drug_name}")
def drug_intel(drug_name: str):
    """AI prediction + analysis for a specific drug's trials."""
    from services.orchestrator import run_ticker_intel

    # Get drug trials
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT nct_id, title, phase, status, condition, intervention,
                   primary_completion_date, sponsor, results_posted,
                   enrollment, study_design, brief_summary
            FROM trials WHERE intervention ILIKE %s
            ORDER BY phase DESC NULLS LAST, primary_completion_date DESC NULLS LAST
            LIMIT 20
        """, (f"%{drug_name}%",))
        trials = [dict(r) for r in cur.fetchall()]
        cur.close()
    finally:
        conn.close()

    # Deduplicate
    seen = set()
    unique_trials = []
    for t in trials:
        if t["nct_id"] not in seen:
            seen.add(t["nct_id"])
            unique_trials.append(t)

    return run_ticker_intel(drug_name, unique_trials, [])


@app.post("/api/simulation/run")
def run_simulation(req: dict):
    """Run a drug demand simulation using MiroFish or local model."""
    from services.mirofish_sim import run_demand_simulation
    drug_name = req.get("drug_name", "")
    condition = req.get("condition", "")
    disease_group = req.get("disease_group", "other")
    population_size = min(int(req.get("population_size", 1000)), 5000)
    if not drug_name or not condition:
        return JSONResponse({"error": "drug_name and condition required"}, status_code=400)
    result = run_demand_simulation(drug_name, condition, disease_group, population_size)
    return result


@app.get("/api/simulation/cached")
def get_cached_sim(drug_name: str, condition: str, disease_group: str = "other"):
    """Get cached simulation result if available."""
    from services.mirofish_sim import get_cached_simulation
    result = get_cached_simulation(drug_name, condition, disease_group)
    if result is None:
        return JSONResponse({"error": "No cached simulation"}, status_code=404)
    return result


class ChatRequest(BaseModel):
    message: str
    context: dict = {}

@app.post("/api/chat/{ticker}")
def ticker_chat(ticker: str, req: ChatRequest):
    """Handle chatbox messages scoped to a ticker page."""
    from services.orchestrator import handle_chat
    return handle_chat(ticker.upper(), req.message, req.context)


class WorkspaceRequest(BaseModel):
    message: str
    history: list = []
    workspace_state: dict = {}

@app.post("/api/workspace/chat")
def workspace_chat(req: WorkspaceRequest):
    """AI Workspace chat — agent + prediction with structured card actions."""
    from services.workspace_handler import handle_workspace_message
    return handle_workspace_message(req.message, req.history, req.workspace_state)


# ── Training Monitor ───────────────────────────────────────────────────

@app.get("/api/training-status")
def training_status():
    """Live training metrics parsed from log output."""
    import re as _re, json as _json
    LOG_FILE = r"C:\Users\kbysn\AppData\Local\Temp\claude\C--Users-kbysn-Desktop-training1\10a67a93-b6e0-468c-8696-169427756c32\tasks\bdyy0fnv8.output"
    RAG_FILE = r"C:\Users\kbysn\Desktop\training1\predictor_v2\data\rag_v2.jsonl"
    TRAIN_FILE = r"C:\Users\kbysn\Desktop\training1\predictor_v2\data\train_9b.jsonl"

    result = {"step": 0, "epoch": 0, "loss": None, "grad_norm": None, "lr": None,
              "speed": None, "done": False, "loss_history": [], "rag_samples": [], "samples": []}
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        # Check if done
        result["done"] = "EXPERIMENT 1 COMPLETE" in text or "Done! Final loss" in text

        # Extract all loss entries
        losses = _re.findall(r"'loss': '([0-9.]+)'", text)
        epochs = _re.findall(r"'epoch': '([0-9.]+)'", text)
        grads = _re.findall(r"'grad_norm': '([0-9.]+)'", text)
        lrs = _re.findall(r"'learning_rate': '([0-9.e-]+)'", text)

        if losses:
            result["loss_history"] = [float(l) for l in losses]
            result["loss"] = float(losses[-1])
            result["step"] = len(losses) * 10  # logged every 10 steps
        if epochs:
            result["epoch"] = float(epochs[-1])
        if grads:
            result["grad_norm"] = float(grads[-1])
        if lrs:
            result["lr"] = float(lrs[-1])

        # Speed: parse from progress bar
        speeds = _re.findall(r'(\d+\.\d+)s/it\]', text)
        if speeds:
            result["speed"] = float(speeds[-1])

        # Final eval results if done
        if result["done"]:
            acc_m = _re.search(r'Accuracy:\s+([0-9.]+)', text)
            f1m_m = _re.search(r'F1 \(macro\):\s+([0-9.]+)', text)
            f1f_m = _re.search(r'F1 \(failure\):\s+([0-9.]+)', text)
            mcc_m = _re.search(r'MCC:\s+([0-9.]+)', text)
            if acc_m: result["val_accuracy"] = float(acc_m.group(1))
            if f1m_m: result["val_f1_macro"] = float(f1m_m.group(1))
            if f1f_m: result["val_f1_failure"] = float(f1f_m.group(1))
            if mcc_m: result["val_mcc"] = float(mcc_m.group(1))

    except FileNotFoundError:
        pass

    # RAG — first 200 facts only (full set too large for browser)
    try:
        with open(RAG_FILE, "r", encoding="utf-8") as f:
            rag_lines = [_json.loads(l) for l in f.readlines()]
        result["rag_all"] = [{"type": r.get("type",""), "text": r["text"][:300],
                              "drug": r.get("drug",""), "tags": r.get("tags",[])} for r in rag_lines[:200]]
        result["rag_count"] = len(rag_lines)
    except:
        result["rag_all"] = []

    # Training samples
    try:
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            sample_lines = [_json.loads(l) for l in [f.readline() for _ in range(5)] if l.strip()]
        result["samples"] = [{"input": s["messages"][1]["content"][:600], "output": s["messages"][2]["content"]} for s in sample_lines]
    except:
        pass

    # Training data records (full CSV)
    try:
        import pandas as _pd
        TRAIN_CSV = r"C:\Users\kbysn\Desktop\training1\predictor_v2\data\train.csv"
        df = _pd.read_csv(TRAIN_CSV)
        records = []
        for _, row in df.head(100).iterrows():
            records.append({
                "nct_id": row.get("nct_id",""),
                "label": row.get("label",""),
                "title": str(row.get("title",""))[:80],
                "phase": row.get("phase",""),
                "enrollment": row.get("enrollment",""),
                "condition": str(row.get("condition",""))[:60],
                "intervention": str(row.get("intervention",""))[:60],
                "sponsor": str(row.get("sponsor",""))[:30],
                "modality": str(row.get("modality",""))[:30],
                "has_drug_science": len(str(row.get("drug_science",""))) > 50,
                "text_length": len(str(row.get("full_text",""))),
            })
        result["training_data"] = records
        result["training_data_total"] = len(df)
    except:
        result["training_data"] = []

    return result


@app.get("/api/training-trial/{nct_id}")
def training_trial_detail(nct_id: str):
    """Get full detail of a training trial — all text, drug science, everything."""
    import pandas as _pd, json as _json
    TRAIN_CSV = r"C:\Users\kbysn\Desktop\training1\predictor_v2\data\train.csv"
    TRAIN_JSONL = r"C:\Users\kbysn\Desktop\training1\predictor_v2\data\train_formatted.jsonl"
    try:
        df = _pd.read_csv(TRAIN_CSV)
        row = df[df["nct_id"] == nct_id.upper()]
        if row.empty:
            return JSONResponse({"error": "Trial not found in training data"}, status_code=404)
        r = row.iloc[0]
        result = {}
        for col in df.columns:
            val = r.get(col)
            if _pd.isna(val): result[col] = None
            else: result[col] = str(val) if not isinstance(val, (int, float, bool)) else val

        # Also get the formatted training input/output
        try:
            with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    rec = _json.loads(line)
                    if nct_id.upper() in rec["messages"][1]["content"]:
                        result["training_input"] = rec["messages"][1]["content"]
                        result["training_output"] = rec["messages"][2]["content"]
                        result["system_prompt"] = rec["messages"][0]["content"]
                        break
        except: pass

        return result
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/training-monitor", response_class=HTMLResponse)
def training_monitor_page():
    monitor_path = r"C:\Users\kbysn\Desktop\training1\predictor_v2\training_monitor.html"
    with open(monitor_path, encoding="utf-8") as f:
        return f.read()


# ── Model Visualization ───────────────────────────────────────────────

@app.get("/api/model/trees")
def model_trees(count: int = 3):
    """Extract tree structures from XGBoost model for visualization."""
    import pickle, json as _json
    ENSEMBLE_DIR = os.path.join(os.path.dirname(__file__), "model_data", "ensemble_v10")

    try:
        with open(os.path.join(ENSEMBLE_DIR, "xgb_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "feat_cols.json")) as f:
            feat_cols = _json.load(f)

        # Get XGBClassifier from CalibratedClassifierCV
        xgb = model.estimator if hasattr(model, "estimator") else model
        booster = xgb.get_booster()
        trees_json = booster.get_dump(dump_format="json", with_stats=True)

        # Feature name mapping: f0 → feat_cols[0]
        DISPLAY_NAMES = {
            "feat_phase_num": "Phase", "feat_is_phase2": "Is Phase 2", "feat_is_phase3": "Is Phase 3",
            "feat_is_early_phase": "Early Phase", "feat_is_interventional": "Interventional",
            "feat_therapeutic_area": "Therapeutic Area", "feat_indication_category": "Indication",
            "feat_is_oncology": "Oncology", "feat_is_rare_disease": "Rare Disease",
            "feat_modality": "Drug Modality", "feat_is_combination": "Combination Therapy",
            "feat_has_biomarker_selection": "Biomarker Selected", "target_class": "Drug Target",
            "feat_enrollment_size": "Enrollment", "enrollment_log": "Enrollment (log)",
            "feat_has_randomization": "Randomized", "feat_has_control_group": "Control Group",
            "feat_has_placebo": "Placebo", "feat_is_double_blind": "Double Blind",
            "feat_has_active_comparator": "Active Comparator", "feat_is_orphan_indication": "Orphan Drug",
            "design_quality_score": "Design Quality", "feat_primary_endpoint_type": "Endpoint Type",
            "feat_has_efficacy_endpoint": "Efficacy Endpoint", "feat_is_safety_trial": "Safety Trial",
            "feat_num_conditions": "# Conditions", "feat_num_interventions": "# Interventions",
            "sponsor_completion_rate": "Sponsor Completion Rate", "sponsor_failure_rate": "Sponsor Failure Rate",
            "sponsor_total_log": "Sponsor Trial Count", "phase_x_oncology": "Phase × Oncology",
            "phase_x_rare": "Phase × Rare Disease", "combination_x_phase": "Combination × Phase",
            "allocation": "Allocation", "intervention_model": "Intervention Model",
            "primary_purpose": "Primary Purpose", "masking_level": "Blinding Level",
            "gender_restriction": "Gender Restricted", "min_age": "Min Age", "max_age": "Max Age",
            "accepts_healthy": "Healthy Volunteers", "age_range": "Age Range",
            "is_pediatric": "Pediatric", "num_arms": "# Arms",
            "num_secondary_endpoints": "Secondary Endpoints", "num_secondary_log": "Endpoint Complexity",
            "num_countries": "Countries", "is_multi_sponsor": "Multi-Sponsor",
            "eligibility_complexity": "Eligibility Complexity", "eligibility_log": "Eligibility (log)",
            "study_duration_months": "Study Duration",
        }

        def map_feature(f_idx):
            """Convert f0, f1, ... to readable name."""
            try:
                idx = int(f_idx.replace("f", ""))
                col = feat_cols[idx] if idx < len(feat_cols) else f_idx
                return DISPLAY_NAMES.get(col, col.replace("feat_", "").replace("_", " ").title())
            except:
                return f_idx

        def process_node(node):
            """Recursively process tree node, mapping feature names."""
            result = {"nodeid": node.get("nodeid", 0), "depth": node.get("depth", 0)}
            if node.get("cover"):
                result["cover"] = round(node["cover"], 1)
            if "leaf" in node:
                result["leaf"] = round(node["leaf"], 6)
                result["type"] = "leaf"
            else:
                result["type"] = "split"
                result["feature"] = map_feature(node.get("split", "?"))
                result["feature_raw"] = node.get("split", "?")
                result["threshold"] = round(node.get("split_condition", 0), 4)
                result["yes"] = node.get("yes")
                result["no"] = node.get("no")
                children = node.get("children", [])
                if children:
                    result["children"] = [process_node(c) for c in children]
            return result

        # Extract first N trees
        extracted = []
        for i in range(min(count, len(trees_json))):
            tree = _json.loads(trees_json[i])
            extracted.append({"tree_id": i, "root": process_node(tree)})

        # Model summary
        importances = xgb.feature_importances_
        top_features = sorted(zip(feat_cols, importances), key=lambda x: -x[1])[:10]

        # Count splits across all trees
        split_counts = {}
        for t_json in trees_json[:50]:  # sample 50 trees
            t = _json.loads(t_json)
            def count_splits(node):
                if "split" in node:
                    fname = map_feature(node["split"])
                    split_counts[fname] = split_counts.get(fname, 0) + 1
                for c in node.get("children", []):
                    count_splits(c)
            count_splits(t)

        return {
            "trees": extracted,
            "summary": {
                "total_trees": len(trees_json),
                "features_used": len(feat_cols),
                "top_features": [{"name": map_feature(f"f{feat_cols.index(c)}") if c in feat_cols else c, "importance": round(float(imp)*100, 1)} for c, imp in top_features],
                "top_split_features": sorted(split_counts.items(), key=lambda x: -x[1])[:10],
                "learning_rate": xgb.get_params().get("learning_rate", "?"),
                "max_depth": xgb.get_params().get("max_depth", "?"),
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── HTML pages ─────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "company.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/news", response_class=HTMLResponse)
def news_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "news.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/company/{ticker}", response_class=HTMLResponse)
def company_page(ticker: str):
    with open(os.path.join(os.path.dirname(__file__), "static", "company.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/drug/{drug_name}", response_class=HTMLResponse)
def drug_page(drug_name: str):
    with open(os.path.join(os.path.dirname(__file__), "static", "drug.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/trial/{nct_id}", response_class=HTMLResponse)
def trial_page(nct_id: str):
    with open(os.path.join(os.path.dirname(__file__), "static", "trial.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/recent", response_class=HTMLResponse)
def recent_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "recent.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/api/trials/recent")
def recent_trials(limit: int = 50):
    """Get recently updated trials from ClinicalTrials.gov."""
    from services.trial_updater import get_recently_updated
    return get_recently_updated(limit=limit)


@app.post("/api/trials/sync")
def sync_trials(days: int = 3):
    """Manually trigger a trial sync from ClinicalTrials.gov."""
    from services.trial_updater import run_daily_update
    return run_daily_update(days_back=days)


@app.get("/workspace", response_class=HTMLResponse)
def workspace_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "workspace.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/backfill-monitor", response_class=HTMLResponse)
def backfill_monitor_page():
    with open(os.path.join(os.path.dirname(__file__), "static", "backfill-monitor.html"), encoding="utf-8") as f:
        return f.read()


# ── Static files (must be last) ───────────────────────────────────────

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
