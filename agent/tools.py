"""
Data-gathering tools for the agent.
Each tool queries a data source and returns a list of dicts.
"""
import logging
import httpx
from models.ticker_map import get_connection
from config import CTGOV_BASE

logger = logging.getLogger(__name__)

# ── Tool registry ────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict] = {}


def _register(name, description, parameters):
    """Decorator to register a tool function."""
    def decorator(fn):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "execute": fn,
        }
        return fn
    return decorator


def get_tool_descriptions() -> str:
    """Format all tools for the system prompt."""
    lines = []
    for i, (name, tool) in enumerate(TOOL_REGISTRY.items(), 1):
        lines.append(f"{i}. {name} — {tool['description']}")
        lines.append(f"   Args: {tool['parameters']}")
    return "\n".join(lines)


# ── Tool 1: Search local trials DB ──────────────────────────────

@_register(
    "search_trials_db",
    "Search the local clinical trials database by keyword, phase, or status",
    '{"query": "optional search text", "phase": "optional e.g. Phase 2", "status": "optional e.g. TERMINATED", "limit": "optional int, default 10"}',
)
def search_trials_db(query=None, phase=None, status=None, limit=10, **_):
    limit = min(int(limit) if limit else 10, 20)
    conn = get_connection()
    cur = conn.cursor()
    try:
        where, params = [], []
        if query:
            where.append("(title ILIKE %s OR condition ILIKE %s OR sponsor ILIKE %s)")
            params.extend([f"%{query}%"] * 3)
        if phase:
            where.append("phase ILIKE %s")
            params.append(f"%{phase}%")
        if status:
            where.append("status ILIKE %s")
            params.append(f"%{status}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""SELECT nct_id, title, sponsor, phase, status, condition, enrollment
                FROM trials {clause}
                ORDER BY nct_id DESC LIMIT %s""",
            params + [limit],
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Truncate long titles
        for r in rows:
            if r.get("title") and len(r["title"]) > 120:
                r["title"] = r["title"][:117] + "..."
        return rows
    except Exception as e:
        logger.error("search_trials_db failed: %s", e)
        return [{"error": str(e)}]
    finally:
        cur.close()
        conn.close()


# ── Tool 2: Get trial detail ────────────────────────────────────

@_register(
    "get_trial_detail",
    "Get full details for a specific clinical trial by NCT ID",
    '{"nct_id": "required, e.g. NCT01234567"}',
)
def get_trial_detail(nct_id=None, **_):
    if not nct_id:
        return [{"error": "nct_id is required"}]
    nct_id = nct_id.strip().upper()
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT nct_id, title, sponsor, phase, status, condition,
                      intervention, enrollment, brief_summary, study_type,
                      study_design, why_stopped, results_posted,
                      start_date, primary_completion_date
               FROM trials WHERE nct_id = %s""",
            [nct_id],
        )
        row = cur.fetchone()
        if row:
            return [dict(row)]
        return [{"error": f"Trial {nct_id} not found in database"}]
    except Exception as e:
        logger.error("get_trial_detail failed: %s", e)
        return [{"error": str(e)}]
    finally:
        cur.close()
        conn.close()


# ── Tool 3: Live search ClinicalTrials.gov ──────────────────────

@_register(
    "search_clinicaltrials_gov",
    "Search ClinicalTrials.gov live API for trials not in the local database",
    '{"query": "required search text", "phase": "optional e.g. PHASE2", "status": "optional e.g. TERMINATED"}',
)
def search_clinicaltrials_gov(query=None, phase=None, status=None, **_):
    if not query:
        return [{"error": "query is required"}]
    params = {
        "query.term": query,
        "pageSize": 10,
        "format": "json",
    }
    filters = []
    if phase:
        # Normalize: PHASE2 -> Phase 2, PHASE3 -> Phase 3
        p = phase.upper().replace("PHASE", "").replace(" ", "").strip()
        filters.append(f"AREA[Phase]Phase {p}")
    if status:
        filters.append(f"AREA[OverallStatus]{status}")
    if filters:
        params["filter.advanced"] = " AND ".join(filters)
    try:
        import urllib.request
        import urllib.parse
        url = CTGOV_BASE + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json as _json
            data = _json.loads(resp.read())
        studies = data.get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            cond_mod = proto.get("conditionsModule", {})
            design_mod = proto.get("designModule", {})
            enroll_mod = design_mod.get("enrollmentInfo", {})
            results.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", "")[:120],
                "status": status_mod.get("overallStatus", ""),
                "phase": ",".join(design_mod.get("phases", [])),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                "condition": ", ".join(cond_mod.get("conditions", [])[:3]),
                "enrollment": enroll_mod.get("count"),
            })
        return results
    except Exception as e:
        logger.error("search_clinicaltrials_gov failed: %s", e)
        return [{"error": str(e)}]


# ── Tool 4: Search news ─────────────────────────────────────────

@_register(
    "search_news",
    "Search biotech news articles by keyword or ticker symbol",
    '{"query": "optional search text", "ticker": "optional ticker e.g. MRNA", "limit": "optional int, default 5"}',
)
def search_news(query=None, ticker=None, limit=5, **_):
    if not query and not ticker:
        return [{"error": "at least query or ticker is required"}]
    limit = min(int(limit) if limit else 5, 10)
    conn = get_connection()
    cur = conn.cursor()
    try:
        where, params = [], []
        if query:
            where.append("(title ILIKE %s OR summary ILIKE %s)")
            params.extend([f"%{query}%"] * 2)
        if ticker:
            where.append("tickers ILIKE %s")
            params.append(f"%{ticker.upper()}%")
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""SELECT title, source, published_at, url, sentiment, catalyst_type, tickers
                FROM news {clause}
                ORDER BY published_at DESC NULLS LAST LIMIT %s""",
            params + [limit],
        )
        rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            if r.get("published_at"):
                r["published_at"] = str(r["published_at"])
        return rows
    except Exception as e:
        logger.error("search_news failed: %s", e)
        return [{"error": str(e)}]
    finally:
        cur.close()
        conn.close()


# ── Tool 5: Get company info ────────────────────────────────────

@_register(
    "get_company_info",
    "Get company information and trial/filing counts for a ticker symbol",
    '{"ticker": "required, e.g. MRNA"}',
)
def get_company_info(ticker=None, **_):
    if not ticker:
        return [{"error": "ticker is required"}]
    ticker = ticker.strip().upper()
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT t.ticker, t.company_name, t.cik,
                      (SELECT COUNT(*) FROM trials tr WHERE tr.sponsor ILIKE '%%' || t.company_name || '%%') AS trial_count,
                      (SELECT COUNT(*) FROM filings f WHERE f.ticker = t.ticker) AS filing_count
               FROM ticker_map t WHERE t.ticker = %s""",
            [ticker],
        )
        row = cur.fetchone()
        if row:
            return [dict(row)]
        return [{"error": f"Ticker {ticker} not found"}]
    except Exception as e:
        logger.error("get_company_info failed: %s", e)
        return [{"error": str(e)}]
    finally:
        cur.close()
        conn.close()


# ── Tool 6: Search executive biography via DuckDuckGo ────────

@_register(
    "search_executive_bio",
    "Search the web for executive/officer biographical info (education, career history, past roles). Use for career timeline queries.",
    '{"name": "required full name", "company": "optional current company"}',
)
def search_executive_bio(name=None, company=None, **_):
    if not name:
        return [{"error": "name is required"}]
    query = f"{name} {company or ''} CEO biography career education"
    try:
        # Use DuckDuckGo HTML search (more reliable than instant answer API)
        import re
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (compatible; BiotechTerminal/1.0)"},
            timeout=15,
            follow_redirects=True,
        )
        html = r.text
        results = []

        # Extract search result snippets
        # DuckDuckGo HTML results are in <a class="result__snippet"> or similar
        snippets = re.findall(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|span|div)',
            html, re.DOTALL
        )
        # Also try result__a for titles
        titles = re.findall(
            r'class="result__a"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )
        urls = re.findall(
            r'class="result__url"[^>]*>(.*?)</(?:a|span)',
            html, re.DOTALL
        )

        # Clean HTML tags from snippets
        def clean(text):
            return re.sub(r'<[^>]+>', '', text).strip()

        for i, snippet in enumerate(snippets[:8]):
            text = clean(snippet)
            if text and len(text) > 20:
                results.append({
                    "source": "web",
                    "title": clean(titles[i]) if i < len(titles) else "",
                    "text": text[:500],
                    "url": clean(urls[i]) if i < len(urls) else "",
                })

        if not results:
            # Fallback: try Wikipedia API directly
            try:
                w = httpx.get(
                    "https://en.wikipedia.org/api/rest_v1/page/summary/" + name.replace(" ", "_"),
                    timeout=10,
                )
                if w.status_code == 200:
                    wd = w.json()
                    if wd.get("extract"):
                        results.append({
                            "source": "wikipedia",
                            "text": wd["extract"][:800],
                            "url": wd.get("content_urls", {}).get("desktop", {}).get("page", ""),
                        })
            except Exception:
                pass

        if not results:
            return [{"info": f"No web results found for '{name}'", "query": query}]
        return results

    except Exception as e:
        logger.error("search_executive_bio failed: %s", e)
        return [{"error": str(e)}]
