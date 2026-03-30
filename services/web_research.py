"""
Web Research Service — DuckDuckGo-based research for trial analysis.
Gathers real information about companies, drugs, trials, and regulatory actions.
"""
from __future__ import annotations

import re
import time
import logging
from typing import List, Optional

import httpx

from cache import TTLCache

logger = logging.getLogger(__name__)

research_cache = TTLCache(max_size=200, default_ttl=1800)  # 30 min

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def _ddg_search(query: str, max_results: int = 5) -> List[dict]:
    """Search DuckDuckGo HTML and extract results."""
    try:
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=15,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return []

        html = r.text
        results = []

        # Extract result blocks
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL
        )

        for url, title, snippet in blocks[:max_results]:
            # Clean HTML tags
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            # Decode DuckDuckGo redirect URLs
            if "uddg=" in url:
                url_match = re.search(r'uddg=([^&]+)', url)
                if url_match:
                    from urllib.parse import unquote
                    url = unquote(url_match.group(1))
            if title and snippet:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results

    except Exception as e:
        logger.warning(f"DuckDuckGo search failed for '{query}': {e}")
        return []


def research_trial(
    nct_id: str,
    sponsor: str = "",
    drug_name: str = "",
    condition: str = "",
    title: str = "",
) -> dict:
    """
    Research a clinical trial by searching the web for relevant information.
    Returns structured research data with sources.
    """
    cache_key = f"research_{nct_id}"
    cached = research_cache.get(cache_key)
    if cached is not None:
        return cached[0]

    t0 = time.time()
    all_sources = []
    research = {}

    # Clean inputs
    sponsor = (sponsor or "").strip()
    drug_name = (drug_name or "").strip()
    condition = (condition or "").strip()
    title = (title or "").strip()

    # Extract drug name from intervention if it looks like a drug
    if not drug_name and title:
        # Try to find drug name in title
        words = title.split()
        for w in words:
            if len(w) > 4 and w[0].isupper() and not w.isupper():
                drug_name = w
                break

    # 1. Company research
    if sponsor:
        query = f"{sponsor} biotech pharmaceutical company pipeline clinical trials"
        results = _ddg_search(query, 4)
        research["company"] = results
        all_sources.extend(results)
        time.sleep(0.5)  # Rate limit

    # 2. Drug/intervention research
    if drug_name:
        query = f"{drug_name} mechanism of action clinical trial results efficacy"
        results = _ddg_search(query, 4)
        research["drug"] = results
        all_sources.extend(results)
        time.sleep(0.5)

    # 3. Trial-specific research
    if nct_id:
        query = f"{nct_id} clinical trial results {drug_name}"
        results = _ddg_search(query, 3)
        research["trial"] = results
        all_sources.extend(results)
        time.sleep(0.5)

    # 4. Regulatory/FDA research
    if drug_name:
        query = f"{drug_name} FDA approval CRL regulatory status {sponsor}"
        results = _ddg_search(query, 3)
        research["regulatory"] = results
        all_sources.extend(results)
        time.sleep(0.5)

    # 5. Competition/landscape
    if condition:
        query = f"{condition} approved treatments drugs competitive landscape"
        results = _ddg_search(query, 3)
        research["competition"] = results
        all_sources.extend(results)

    # Deduplicate sources by URL
    seen_urls = set()
    unique_sources = []
    for s in all_sources:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(s)

    elapsed = round(time.time() - t0, 2)

    result = {
        "research": research,
        "sources": unique_sources[:15],  # Top 15 unique sources
        "elapsed": elapsed,
        "queries_run": sum(1 for k in research if research[k]),
    }

    research_cache.set(cache_key, result)
    return result


def format_research_for_prompt(research_data: dict) -> str:
    """Format research results into a text block for the Qwen analysis prompt."""
    sections = []

    for category, results in research_data.get("research", {}).items():
        if not results:
            continue
        label = {
            "company": "COMPANY INFORMATION",
            "drug": "DRUG/TREATMENT INFORMATION",
            "trial": "TRIAL-SPECIFIC DATA",
            "regulatory": "REGULATORY STATUS",
            "competition": "COMPETITIVE LANDSCAPE",
        }.get(category, category.upper())

        snippets = []
        for r in results[:3]:
            snippets.append(f"- [{r['title']}]: {r['snippet']}")
        sections.append(f"=== {label} ===\n" + "\n".join(snippets))

    return "\n\n".join(sections) if sections else "No web research data available."
