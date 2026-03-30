"""
Intel gap-fill service.
Inspects company data, computes per-section completeness,
returns factual summaries for data-rich sections,
calls LLM agent ONLY for gaps.
"""
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def build_intel_blocks(data: dict, statuses: list, ticker: str) -> list[dict]:
    """Build structured intel blocks with source_type tags."""
    company_name = data.get("info", {}).get("company_name", ticker)
    trials = data.get("trials") or []
    drugs = data.get("drugs") or []
    blocks = []

    # ── 1. Trial Overview ──────────────────────────────────────────
    if len(trials) >= 3:
        phases = Counter(t.get("phase", "Unknown") for t in trials)
        statuses_c = Counter(t.get("status", "Unknown") for t in trials)
        phase_str = ", ".join(f"{v} {k}" for k, v in sorted(phases.items()))
        status_str = ", ".join(f"{v} {k}" for k, v in sorted(statuses_c.items()))
        blocks.append({
            "section": "trial_overview",
            "title": "Trial Overview",
            "status": "complete",
            "content": f"{len(trials)} clinical trials registered: {phase_str}. Status breakdown: {status_str}.",
            "source_type": "source_data",
            "source_coverage": 1.0,
            "details": {"total": len(trials), "phases": dict(phases)},
            "ai_note": None,
        })
    elif len(trials) > 0:
        blocks.append({
            "section": "trial_overview",
            "title": "Trial Overview",
            "status": "partial",
            "content": f"Only {len(trials)} trial(s) found in database for {company_name}.",
            "source_type": "hybrid",
            "source_coverage": min(len(trials) / 5.0, 1.0),
            "details": None,
            "ai_note": "Limited trial data. Additional trials may exist on ClinicalTrials.gov.",
        })
    else:
        blocks.append({
            "section": "trial_overview",
            "title": "Trial Overview",
            "status": "missing",
            "content": "No clinical trial data available in database.",
            "source_type": "llm_generated",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": "No trials found. Company may not have public clinical programs.",
        })

    # ── 2. Recent Trials ───────────────────────────────────────────
    recent = sorted(
        [t for t in trials if t.get("primary_completion_date")],
        key=lambda t: t["primary_completion_date"],
        reverse=True,
    )[:3]
    if recent:
        items = "; ".join(
            f"{t['title'][:60]} ({t.get('phase','?')}, {t.get('status','?')})"
            for t in recent
        )
        blocks.append({
            "section": "recent_trials",
            "title": "Recent Trial Activity",
            "status": "complete" if len(recent) >= 2 else "partial",
            "content": f"Most recent trials: {items}.",
            "source_type": "source_data",
            "source_coverage": min(len(recent) / 3.0, 1.0),
            "details": None,
            "ai_note": None,
        })
    else:
        blocks.append({
            "section": "recent_trials",
            "title": "Recent Trial Activity",
            "status": "missing",
            "content": "No recent trial activity data available.",
            "source_type": "llm_generated",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": None,
        })

    # ── 3. Pipeline Summary ────────────────────────────────────────
    if len(drugs) >= 2:
        drug_items = ", ".join(
            f"{d['name']} ({d.get('phase','?')})" for d in drugs[:6]
        )
        blocks.append({
            "section": "pipeline_summary",
            "title": "Drug Pipeline",
            "status": "complete",
            "content": f"{len(drugs)} drug candidate(s) in development: {drug_items}.",
            "source_type": "source_data",
            "source_coverage": 1.0,
            "details": None,
            "ai_note": None,
        })
    elif len(drugs) == 1:
        blocks.append({
            "section": "pipeline_summary",
            "title": "Drug Pipeline",
            "status": "partial",
            "content": f"1 known drug candidate: {drugs[0]['name']} ({drugs[0].get('phase','?')}).",
            "source_type": "hybrid",
            "source_coverage": 0.5,
            "details": None,
            "ai_note": "Limited pipeline visibility. Company may have additional undisclosed programs.",
        })
    else:
        blocks.append({
            "section": "pipeline_summary",
            "title": "Drug Pipeline",
            "status": "missing",
            "content": "No drug pipeline data available.",
            "source_type": "llm_generated",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": None,
        })

    # ── 4. Key Outcomes ────────────────────────────────────────────
    with_results = [t for t in trials if t.get("results_posted")]
    if with_results:
        items = "; ".join(
            f"{t['title'][:50]} ({t.get('phase','?')}, {t.get('condition','?')})"
            for t in with_results[:4]
        )
        blocks.append({
            "section": "key_outcomes",
            "title": "Key Trial Outcomes",
            "status": "complete",
            "content": f"{len(with_results)} trial(s) with posted results: {items}.",
            "source_type": "source_data",
            "source_coverage": 1.0,
            "details": None,
            "ai_note": None,
        })
    else:
        blocks.append({
            "section": "key_outcomes",
            "title": "Key Trial Outcomes",
            "status": "missing",
            "content": "No trials with posted results found.",
            "source_type": "source_data",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": "Results may be pending or not yet publicly posted.",
        })

    # ── 5. Development Stage Summary ───────────────────────────────
    if len(trials) >= 3:
        phases = Counter(t.get("phase", "Unknown") for t in trials)
        phase_lines = ", ".join(f"{k}: {v}" for k, v in sorted(phases.items()))
        blocks.append({
            "section": "development_stage_summary",
            "title": "Development Stage",
            "status": "complete",
            "content": f"Phase distribution across {len(trials)} trials — {phase_lines}.",
            "source_type": "source_data",
            "source_coverage": 1.0,
            "details": dict(phases),
            "ai_note": None,
        })
    elif len(trials) > 0:
        blocks.append({
            "section": "development_stage_summary",
            "title": "Development Stage",
            "status": "partial",
            "content": f"Limited data: {len(trials)} trial(s) found.",
            "source_type": "hybrid",
            "source_coverage": min(len(trials) / 3.0, 1.0),
            "details": None,
            "ai_note": "Insufficient trials for reliable stage distribution.",
        })
    else:
        blocks.append({
            "section": "development_stage_summary",
            "title": "Development Stage",
            "status": "missing",
            "content": "No trial data to determine development stage.",
            "source_type": "llm_generated",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": None,
        })

    # ── 6. Risk Factors (always AI) ────────────────────────────────
    # Build risk assessment from available data — no LLM needed for common patterns
    risks = []
    if len(trials) > 0:
        phase_counts = Counter(t.get("phase", "") for t in trials)
        if not any(k for k in phase_counts if "3" in str(k) or "4" in str(k)):
            risks.append("No late-stage (Phase 3+) trials — approval timeline uncertain")
        terminated = sum(1 for t in trials if "terminated" in str(t.get("status", "")).lower())
        if terminated > 0:
            risks.append(f"{terminated} terminated trial(s) in pipeline")
    if len(drugs) <= 1:
        risks.append("Narrow pipeline — limited diversification")
    if not with_results:
        risks.append("No posted trial results — outcome data pending")

    if risks:
        blocks.append({
            "section": "risk_factors",
            "title": "Risk Assessment",
            "status": "partial",
            "content": " • ".join(risks) + ".",
            "source_type": "hybrid",
            "source_coverage": 0.3,
            "details": None,
            "ai_note": "Risk factors inferred from available trial and pipeline data. Not financial advice.",
        })
    else:
        blocks.append({
            "section": "risk_factors",
            "title": "Risk Assessment",
            "status": "missing",
            "content": "Insufficient data to assess clinical development risks.",
            "source_type": "llm_generated",
            "source_coverage": 0.0,
            "details": None,
            "ai_note": "No trial data available for risk assessment.",
        })

    return blocks
