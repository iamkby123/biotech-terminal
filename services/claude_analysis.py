"""
Claude API narrative analysis for clinical trials.
Generates natural language explanations of trial predictions.
Cached for 24h per trial+probability to minimize API costs.
"""
import os
import logging
import time

logger = logging.getLogger(__name__)

# In-memory cache: {key: (timestamp, narrative)}
_narrative_cache = {}
_CACHE_TTL = 86400  # 24 hours


def generate_narrative(trial: dict, prediction: dict, rag_evidence: list, drug_science: dict) -> dict | None:
    """Generate a Claude-powered narrative analysis of a trial prediction.

    Returns {"narrative": str, "model": str} or None if disabled/failed.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    nct_id = trial.get("nct_id", "")
    prob = prediction.get("probability", 0.5)
    cache_key = f"narrative:{nct_id}:{prob:.3f}"

    # Check cache
    if cache_key in _narrative_cache:
        ts, cached = _narrative_cache[cache_key]
        if time.time() - ts < _CACHE_TTL:
            return cached

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        # Build context for Claude
        outcome = prediction.get("outcome", "unknown")
        confidence = prediction.get("confidence", "medium")
        fail_pct = round(prob * 100, 1)
        succ_pct = round(100 - fail_pct, 1)

        phase = trial.get("phase", "")
        condition = trial.get("condition", "")
        sponsor = trial.get("sponsor", "")
        enrollment = trial.get("enrollment", "")
        intervention = trial.get("intervention", "")

        # Feature impacts (SHAP)
        impacts = prediction.get("feature_impacts", [])
        impact_text = ""
        if impacts:
            impact_lines = []
            for imp in impacts[:6]:
                direction = "toward failure" if imp["direction"] == "risk" else "toward success"
                impact_lines.append(f"- {imp['feature']}: {imp['impact']:+.1f}% ({direction})")
            impact_text = "Key feature contributions:\n" + "\n".join(impact_lines)

        # Drug science
        mech_text = ""
        if drug_science.get("mechanism"):
            mech_text = drug_science["mechanism"][0].get("abstract", "")[:300]

        # RAG evidence
        evidence_text = ""
        if rag_evidence:
            ev_lines = []
            for ev in rag_evidence[:5]:
                text = ev.get("text", ev) if isinstance(ev, dict) else str(ev)
                ev_lines.append(f"- {text[:200]}")
            evidence_text = "Historical evidence:\n" + "\n".join(ev_lines)

        # Dual model
        dual = prediction.get("dual", {})
        dual_text = ""
        if dual:
            opt = dual.get("optimist", {})
            pes = dual.get("pessimist", {})
            dual_text = f"Optimist model: {opt.get('success_pct', '?')}% success. Pessimist model: {pes.get('failure_pct', '?')}% failure. Models {dual.get('agreement', 'DISAGREE')}."

        prompt = f"""Trial: {nct_id} | {phase} | {condition}
Sponsor: {sponsor} | Enrollment: {enrollment} patients
Drug: {intervention}
Prediction: {outcome.upper()} ({fail_pct}% failure probability, {confidence} confidence)
{dual_text}

{impact_text}

Drug mechanism: {mech_text[:300] if mech_text else 'Not available'}

{evidence_text}

Write a 2-3 paragraph analysis explaining this prediction. Be specific: cite the numbers, name the condition and drug, reference the feature impacts. First paragraph: explain the prediction and key drivers. Second paragraph: historical context and competitor results. Third paragraph: risk factors and what to watch for. Be direct and analytical, not hedging."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system="You are a biotech analyst writing concise trial prediction analyses for investors. Be specific with numbers and data. Write in plain language — avoid ML jargon, do not reference model names, algorithms, or technical metrics like MCC or SHAP. No disclaimers or caveats about limitations.",
            messages=[{"role": "user", "content": prompt}],
        )

        narrative = response.content[0].text
        result = {
            "narrative": narrative,
            "model": "claude-haiku-4-5",
            "tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        # Cache
        _narrative_cache[cache_key] = (time.time(), result)

        # Prune old cache entries (keep max 500)
        if len(_narrative_cache) > 500:
            oldest_keys = sorted(_narrative_cache, key=lambda k: _narrative_cache[k][0])[:100]
            for k in oldest_keys:
                del _narrative_cache[k]

        return result

    except Exception as e:
        logger.warning("Claude narrative failed: %s", e)
        return None
