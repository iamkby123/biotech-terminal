"""
AI Workspace Handler — self-contained research tool.
Gathers ALL data from the internet (ClinicalTrials.gov API, DuckDuckGo web search).
Does NOT use the local database. Fully independent.
"""
import re
import json
import logging
import time
import urllib.request
import urllib.parse

logger = logging.getLogger(__name__)

CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"


def _classify_intent(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["predict", "probability", "outcome", "chance", "succeed", "fail", "likelihood", "will it"]):
        return "predict"
    if any(w in msg for w in ["show me", "open", "navigate", "go to", "browse", "website"]):
        return "browse"
    if any(w in msg for w in ["find", "search", "look up", "trials for", "pipeline", "what trials", "list", "phase"]):
        return "search"
    if any(w in msg for w in ["news", "latest", "recent", "article", "press release"]):
        return "news"
    if any(w in msg for w in ["what is", "tell me about", "who is", "explain", "how does"]):
        return "research"
    return "general"


# ═══════════════════════════════════════════════════════════════
# WEB DATA GATHERING — all online, no local DB
# ═══════════════════════════════════════════════════════════════

def _search_ctgov(query: str, phase: str = None, status: str = None, max_results: int = 8) -> list:
    """Search ClinicalTrials.gov API directly."""
    params = {"query.term": query, "pageSize": max_results, "format": "json"}
    filters = []
    if phase:
        p = phase.upper().replace("PHASE", "").replace(" ", "").strip()
        if p.isdigit():
            filters.append(f"AREA[Phase]Phase {p}")
    if status:
        filters.append(f"AREA[OverallStatus]{status}")
    if filters:
        params["filter.advanced"] = " AND ".join(filters)
    try:
        url = CTGOV_API + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        results = []
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
            results.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "status": status_mod.get("overallStatus", ""),
                "phase": ", ".join(design_mod.get("phases", [])),
                "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
                "condition": ", ".join(cond_mod.get("conditions", [])[:3]),
                "enrollment": enroll_mod.get("count"),
                "intervention": ", ".join(interventions[:3]),
                "brief_summary": (desc_mod.get("briefSummary") or "")[:200],
                "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
                "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
                "why_stopped": status_mod.get("whyStoppedText", ""),
                "has_results": s.get("hasResults", False),
            })
        return results
    except Exception as e:
        logger.warning("ClinicalTrials.gov API search failed: %s", e)
        return []


def _get_ctgov_trial(nct_id: str) -> dict:
    """Fetch a single trial from ClinicalTrials.gov API."""
    try:
        url = f"{CTGOV_API}/{nct_id}?format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            s = json.loads(resp.read())
        proto = s.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        design_mod = proto.get("designModule", {})
        desc_mod = proto.get("descriptionModule", {})
        enroll_mod = design_mod.get("enrollmentInfo", {})
        arms_mod = proto.get("armsInterventionsModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        interventions = [i.get("name", "") for i in arms_mod.get("interventions", [])]
        study_design = []
        if design_mod.get("designInfo", {}).get("allocation"):
            study_design.append(design_mod["designInfo"]["allocation"])
        if design_mod.get("designInfo", {}).get("masking", {}).get("masking"):
            study_design.append(design_mod["designInfo"]["masking"]["masking"])
        return {
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "official_title": ident.get("officialTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": ", ".join(design_mod.get("phases", [])),
            "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "condition": ", ".join(cond_mod.get("conditions", [])[:5]),
            "enrollment": enroll_mod.get("count"),
            "intervention": ", ".join(interventions[:5]),
            "brief_summary": desc_mod.get("briefSummary", ""),
            "study_design": ", ".join(study_design),
            "study_type": design_mod.get("studyType", ""),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("completionDateStruct", {}).get("date", ""),
            "why_stopped": status_mod.get("whyStoppedText", ""),
            "has_results": s.get("hasResults", False),
            "source": "clinicaltrials.gov",
        }
    except Exception as e:
        logger.warning("ClinicalTrials.gov fetch failed for %s: %s", nct_id, e)
        return {}


def _web_search(query: str, max_results: int = 5) -> list:
    """Search DuckDuckGo for general web info."""
    try:
        import httpx
        r = httpx.get("https://html.duckduckgo.com/html/", params={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=15, follow_redirects=True)
        if r.status_code != 200:
            return []
        results = []
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            r.text, re.DOTALL)
        for url, title, snippet in blocks[:max_results]:
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if "uddg=" in url:
                m = re.search(r'uddg=([^&]+)', url)
                if m:
                    url = urllib.parse.unquote(m.group(1))
            if title and snippet:
                results.append({"title": title, "url": url, "snippet": snippet})
        return results
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════
# ACTION BUILDERS
# ═══════════════════════════════════════════════════════════════

def _trials_to_cards(trials: list) -> list:
    actions = []
    for t in trials:
        if not t.get("nct_id"):
            continue
        nct = t["nct_id"]
        actions.append({
            "type": "create_card", "id": f"trial-{nct}",
            "title": (t.get("title") or f"Trial {nct}")[:55],
            "card_type": "trial_data",
            "content": t.get("title", ""),
            "data": t,
        })
    return actions


def _web_results_to_cards(results: list, prefix: str = "web") -> list:
    actions = []
    for i, r in enumerate(results):
        actions.append({
            "type": "create_card", "id": f"{prefix}-{i}",
            "title": (r.get("title") or "Web Result")[:55],
            "card_type": "research",
            "content": r.get("snippet", ""),
            "data": r,
        })
    return actions


def _build_browse_action(message: str) -> dict:
    url_match = re.search(r'https?://\S+', message)
    if url_match:
        return {"type": "navigate_browser", "url": url_match.group()}
    nct_match = re.search(r'NCT\d{7,8}', message.upper())
    if nct_match:
        return {"type": "navigate_browser", "url": f"https://clinicaltrials.gov/study/{nct_match.group()}"}
    msg_lower = message.lower()
    if "clinicaltrials" in msg_lower or "clinical trials" in msg_lower:
        return {"type": "navigate_browser", "url": "https://clinicaltrials.gov/"}
    if "fda" in msg_lower:
        return {"type": "navigate_browser", "url": "https://www.fda.gov/drugs"}
    if "pubmed" in msg_lower:
        return {"type": "navigate_browser", "url": "https://pubmed.ncbi.nlm.nih.gov/"}
    words = message.replace("show me", "").replace("open", "").replace("browse", "").replace("go to", "").strip()
    if words:
        return {"type": "navigate_browser", "url": f"https://clinicaltrials.gov/search?term={words.replace(' ', '+')}"}
    return {"type": "navigate_browser", "url": "https://clinicaltrials.gov/"}


def _run_prediction(nct_id: str, trial_data: dict) -> dict:
    """Run the fine-tuned 0.8B model prediction."""
    try:
        from services.orchestrator import predict_trial_for_ticker, _enrich_trial_features
        enriched = _enrich_trial_features(trial_data)
        domain, prediction = predict_trial_for_ticker(enriched)
        return {
            "type": "create_card", "id": f"pred-{nct_id}",
            "title": f"Prediction: {nct_id}",
            "card_type": "prediction",
            "content": f"{prediction['outcome'].upper()} ({prediction.get('confidence', 'medium')} confidence)",
            "data": {
                "nct_id": nct_id,
                "outcome": prediction.get("outcome", "unknown"),
                "confidence": prediction.get("confidence", "medium"),
                "probability": prediction.get("probability", 0.5),
                "source": prediction.get("source", ""),
                "drivers": prediction.get("drivers", [])[:6],
                "explanation": prediction.get("explanation", ""),
                "reasons": prediction.get("reasons", {}),
                "presumption_overridden": prediction.get("presumption_overridden"),
                "reason_weights": prediction.get("reason_weights", {}),
            }
        }
    except Exception as e:
        logger.warning("Prediction failed for %s: %s", nct_id, e)
        return {
            "type": "create_card", "id": f"pred-{nct_id}",
            "title": f"Prediction: {nct_id}", "card_type": "prediction",
            "content": f"Prediction failed: {e}", "data": {"error": str(e)},
        }


# ═══════════════════════════════════════════════════════════════
# MAIN HANDLER
# ═══════════════════════════════════════════════════════════════

def handle_workspace_message(message: str, history: list, workspace_state: dict) -> dict:
    """Main entry — gathers all data from the web, not local DB."""
    t0 = time.time()
    intent = _classify_intent(message)
    actions = []
    sources = []
    summary = ""
    existing_cards = set(workspace_state.get("cards", []))

    if intent == "predict":
        nct_match = re.search(r'NCT\d{7,8}', message.upper())
        if nct_match:
            nct_id = nct_match.group()
            trial = _get_ctgov_trial(nct_id)
            if trial and trial.get("nct_id"):
                # Trial data card
                cid = f"trial-{nct_id}"
                actions.append({
                    "type": "update_card" if cid in existing_cards else "create_card",
                    "id": cid, "title": (trial.get("title") or nct_id)[:55],
                    "card_type": "trial_data", "content": trial.get("title", ""),
                    "data": trial,
                })
                # Prediction card
                pred = _run_prediction(nct_id, trial)
                pred["type"] = "update_card" if pred["id"] in existing_cards else "create_card"
                actions.append(pred)
                # Browser
                actions.append({"type": "navigate_browser", "url": f"https://clinicaltrials.gov/study/{nct_id}"})
                outcome = pred.get("data", {}).get("outcome", "unknown")
                prob = pred.get("data", {}).get("probability", 0)
                summary = f"Prediction for {nct_id}: {outcome.upper()} with {round(prob*100,1)}% failure probability. Data fetched from ClinicalTrials.gov."
                sources = ["clinicaltrials.gov", "qwen_0.8b_model"]
            else:
                summary = f"Could not find trial {nct_id} on ClinicalTrials.gov."
        else:
            # Search ClinicalTrials.gov, then predict the top result
            query = message.lower().replace("predict", "").replace("outcome of", "").strip()
            trials = _search_ctgov(query or message, max_results=5)
            actions.extend(_trials_to_cards(trials))
            if trials and trials[0].get("nct_id"):
                first = trials[0]
                full = _get_ctgov_trial(first["nct_id"])
                if full:
                    pred = _run_prediction(first["nct_id"], full)
                    actions.append(pred)
            summary = f"Found {len(trials)} trials on ClinicalTrials.gov. Predicted the top result."
            sources = ["clinicaltrials.gov", "qwen_0.8b_model"]

    elif intent == "browse":
        actions.append(_build_browse_action(message))
        summary = f"Opening {actions[0]['url']}"
        sources = []

    elif intent == "search":
        # Extract phase from message
        phase = None
        phase_match = re.search(r'phase\s*(\d)', message, re.IGNORECASE)
        if phase_match:
            phase = phase_match.group(1)
        # Extract status
        status = None
        for s in ["RECRUITING", "COMPLETED", "TERMINATED", "ACTIVE"]:
            if s.lower() in message.lower():
                status = s
                break
        # Search ClinicalTrials.gov
        query = re.sub(r'phase\s*\d', '', message, flags=re.IGNORECASE)
        query = re.sub(r'(find|search|look up|list|trials for)', '', query, flags=re.IGNORECASE).strip()
        trials = _search_ctgov(query or message, phase=phase, status=status, max_results=8)
        actions.extend(_trials_to_cards(trials))
        # Navigate to the search on ClinicalTrials.gov
        search_url = f"https://clinicaltrials.gov/search?term={urllib.parse.quote(query or message)}"
        if phase:
            search_url += f"&phase={phase}"
        actions.append({"type": "navigate_browser", "url": search_url})
        summary = f"Found {len(trials)} trials on ClinicalTrials.gov for '{query.strip() or message}'."
        sources = ["clinicaltrials.gov"]

    elif intent == "news":
        query = re.sub(r'(news|latest|recent|articles?|press releases?)\s*(about|for|on)?', '', message, flags=re.IGNORECASE).strip()
        results = _web_search(f"{query} biotech clinical trial news 2026", max_results=6)
        actions.extend(_web_results_to_cards(results, "news"))
        summary = f"Found {len(results)} news articles about '{query}'."
        sources = ["duckduckgo"]

    elif intent == "research":
        query = re.sub(r'(what is|tell me about|who is|explain|how does)', '', message, flags=re.IGNORECASE).strip()
        # Web search for general info
        results = _web_search(f"{query} biotech pharmaceutical", max_results=5)
        actions.extend(_web_results_to_cards(results, "research"))
        # Also check ClinicalTrials.gov if it looks trial-related
        if any(w in message.lower() for w in ["trial", "drug", "treatment", "therapy", "vaccine"]):
            trials = _search_ctgov(query, max_results=4)
            actions.extend(_trials_to_cards(trials))
            sources.append("clinicaltrials.gov")
        summary = f"Researched '{query}' — found {len(results)} web sources" + (f" and {len([a for a in actions if a.get('card_type')=='trial_data'])} trials." if "clinicaltrials.gov" in sources else ".")
        sources.append("duckduckgo")

    else:  # general
        # Try ClinicalTrials.gov first
        trials = _search_ctgov(message, max_results=5)
        if trials:
            actions.extend(_trials_to_cards(trials))
            sources.append("clinicaltrials.gov")
        # Also web search
        results = _web_search(f"{message} biotech", max_results=4)
        if results:
            actions.extend(_web_results_to_cards(results))
            sources.append("duckduckgo")
        n = len(actions)
        summary = f"Found {n} results from online sources." if n else "No results found online."

    elapsed = round(time.time() - t0, 2)
    return {
        "message": summary,
        "actions": actions,
        "sources": sources,
        "elapsed": elapsed,
    }
