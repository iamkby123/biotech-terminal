"""
Predictor V2 — Ensemble prediction combining 6 signals.
Primary: ensemble_signals.compute_ensemble() (XGBoost + condition + target + sponsor + design + similar)
Fallback: old GB models if ensemble artifacts missing.
"""
import logging

logger = logging.getLogger(__name__)


def predict_v2(trial: dict) -> dict:
    """Run ensemble prediction on a trial. Returns prediction dict with signal breakdown."""
    try:
        from services.ensemble_signals import compute_ensemble
        result = compute_ensemble(trial)
        if result and result.get("outcome"):
            return result
    except Exception as e:
        logger.warning("Ensemble prediction failed, trying fallback: %s", e)

    # Fallback: simple heuristic if ensemble fails
    return {
        "outcome": "unknown",
        "probability": 0.50,
        "confidence": "low",
        "source": "fallback",
        "signals": [],
        "drivers": [],
    }
