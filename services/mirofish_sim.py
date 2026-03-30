"""
MiroFish Simulation Runner — Drug Demand Modeling.
Generates patient populations and runs demand simulations using MiroFish's
multi-agent framework. Results cached per (drug_name, condition).
"""
from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from typing import Optional
from pathlib import Path

import httpx

from cache import TTLCache
from services.disease_mapping import (
    generate_civilian_population,
    get_population_summary,
    DISEASE_PROFILES,
)

logger = logging.getLogger(__name__)

# Cache simulation results for 30 min
sim_cache = TTLCache(max_size=50, default_ttl=1800)

MIROFISH_BASE = "http://localhost:5001/api"
MIROFISH_AVAILABLE = False

# Directory for simulation profiles and results
SIM_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "simulations")
os.makedirs(SIM_DIR, exist_ok=True)


def check_mirofish_available() -> bool:
    """Check if MiroFish backend is running."""
    global MIROFISH_AVAILABLE
    try:
        r = httpx.get(f"{MIROFISH_BASE}/simulation/list", timeout=3)
        MIROFISH_AVAILABLE = r.status_code == 200
    except Exception:
        MIROFISH_AVAILABLE = False
    return MIROFISH_AVAILABLE


def _cache_key(drug_name: str, condition: str, disease_group: str) -> str:
    return hashlib.md5(f"{drug_name}:{condition}:{disease_group}".lower().encode()).hexdigest()


def run_demand_simulation(
    drug_name: str,
    condition_name: str,
    disease_group: str,
    population_size: int = 1000,
    use_mirofish: bool = True,
) -> dict:
    """
    Run a drug demand simulation.

    If MiroFish is available and use_mirofish=True, runs a full multi-agent sim.
    Otherwise, runs a local statistical demand model using the population profiles.
    """
    key = _cache_key(drug_name, condition_name, disease_group)
    cached = sim_cache.get(key)
    if cached is not None:
        value, meta = cached
        return value

    t0 = time.time()

    # Generate patient population
    civilians = generate_civilian_population(
        disease_group=disease_group,
        condition_name=condition_name,
        drug_name=drug_name,
        population_size=population_size,
    )
    population_summary = get_population_summary(civilians)

    # Save profiles for reference
    sim_dir = os.path.join(SIM_DIR, key)
    os.makedirs(sim_dir, exist_ok=True)
    profiles_path = os.path.join(sim_dir, "profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in civilians], f, indent=2, default=str)

    # Try MiroFish full simulation
    mirofish_result = None
    if use_mirofish and check_mirofish_available():
        mirofish_result = _run_mirofish_sim(civilians, drug_name, condition_name, sim_dir)

    # If MiroFish unavailable or failed, run local demand model
    if mirofish_result is None:
        demand_result = _run_local_demand_model(civilians, drug_name, condition_name, disease_group)
    else:
        demand_result = mirofish_result

    elapsed = round(time.time() - t0, 2)

    result = {
        "drug_name": drug_name,
        "condition": condition_name,
        "disease_group": disease_group,
        "population": population_summary,
        "demand": demand_result,
        "simulation_type": "mirofish" if mirofish_result else "local_model",
        "elapsed_seconds": elapsed,
    }

    sim_cache.set(key, result)

    # Save results
    with open(os.path.join(sim_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _run_mirofish_sim(civilians, drug_name: str, condition_name: str, sim_dir: str) -> Optional[dict]:
    """Run simulation through MiroFish REST API."""
    try:
        # Save Reddit-format profiles
        reddit_profiles = [c.to_mirofish_reddit_profile() for c in civilians]
        profiles_path = os.path.join(sim_dir, "reddit_profiles.json")
        with open(profiles_path, "w", encoding="utf-8") as f:
            json.dump(reddit_profiles, f, indent=2)

        # Create simulation
        r = httpx.post(f"{MIROFISH_BASE}/simulation/create", json={
            "project_id": f"biotech_{drug_name.lower().replace(' ', '_')}",
            "enable_twitter": False,
            "enable_reddit": True,
        }, timeout=30)

        if r.status_code != 200:
            logger.warning(f"MiroFish create failed: {r.status_code}")
            return None

        sim_id = r.json().get("data", {}).get("simulation_id")
        if not sim_id:
            return None

        # Start simulation with pre-built profiles
        r = httpx.post(f"{MIROFISH_BASE}/simulation/start", json={
            "simulation_id": sim_id,
            "platform": "reddit",
            "max_rounds": 30,
            "enable_graph_memory_update": False,
        }, timeout=30)

        if r.status_code != 200:
            logger.warning(f"MiroFish start failed: {r.status_code}")
            return None

        # Poll for completion (max 60 seconds)
        for _ in range(12):
            time.sleep(5)
            r = httpx.get(f"{MIROFISH_BASE}/simulation/{sim_id}/run-status", timeout=10)
            if r.status_code == 200:
                status = r.json().get("data", {})
                if status.get("runner_status") == "completed":
                    break

        # Get results
        r = httpx.get(f"{MIROFISH_BASE}/simulation/{sim_id}/agent-stats", timeout=10)
        agent_stats = r.json().get("data", []) if r.status_code == 200 else []

        r = httpx.get(f"{MIROFISH_BASE}/simulation/{sim_id}/actions?limit=200", timeout=10)
        actions = r.json().get("data", []) if r.status_code == 200 else []

        # Analyze sentiment from actions
        positive_posts = sum(1 for a in actions if "positive" in str(a.get("content", "")).lower()
                            or "hopeful" in str(a.get("content", "")).lower()
                            or "effective" in str(a.get("content", "")).lower())
        negative_posts = sum(1 for a in actions if "concern" in str(a.get("content", "")).lower()
                            or "side effect" in str(a.get("content", "")).lower()
                            or "expensive" in str(a.get("content", "")).lower())

        total_posts = max(len(actions), 1)

        return {
            "demand_score": round(positive_posts / total_posts * 100, 1),
            "concern_score": round(negative_posts / total_posts * 100, 1),
            "total_interactions": len(actions),
            "positive_sentiment_pct": round(positive_posts / total_posts * 100, 1),
            "negative_sentiment_pct": round(negative_posts / total_posts * 100, 1),
            "agent_engagement": len(agent_stats),
            "source": "mirofish_simulation",
        }

    except Exception as e:
        logger.warning(f"MiroFish simulation failed: {e}")
        return None


def _run_local_demand_model(civilians, drug_name: str, condition_name: str, disease_group: str) -> dict:
    """
    Local statistical demand model when MiroFish is unavailable.
    Models demand based on patient population characteristics.
    """
    n = len(civilians)
    if n == 0:
        return {"demand_score": 0, "source": "local_model"}

    profile = DISEASE_PROFILES.get(disease_group, DISEASE_PROFILES.get("other", {}))

    # Base demand = treatment-seeking * insured
    seeking = sum(1 for c in civilians if c.treatment_seeking)
    insured = sum(1 for c in civilians if c.has_insurance)
    adherent = sum(1 for c in civilians if c.adherent)

    # Effective demand pool
    demand_pool = sum(1 for c in civilians if c.treatment_seeking and c.has_insurance)
    demand_rate = demand_pool / n

    # Adjust for competition (more alternatives = less demand for new drug)
    alt_therapies = profile.get("alternative_therapies", 5)
    competition_factor = 1.0 / (1.0 + alt_therapies * 0.1)

    # Adjust for urgency (higher urgency = more likely to try new drug)
    avg_urgency = sum(c.treatment_urgency for c in civilians) / n
    urgency_factor = 0.5 + avg_urgency * 0.5

    # Final demand score (0-100)
    demand_score = demand_rate * competition_factor * urgency_factor * 100

    # Adherence-adjusted demand (long-term value)
    adherence_rate = adherent / n
    sustained_demand = demand_score * adherence_rate

    # Comorbidity impact (patients with comorbidities may have drug interactions)
    comorb_patients = sum(1 for c in civilians if len(c.comorbidities) > 0)
    comorb_conflict_rate = comorb_patients / n * 0.15  # 15% of comorbid patients may have conflicts

    # Age distribution segments
    age_segments = {
        "18-35": sum(1 for c in civilians if 18 <= c.age < 35),
        "35-50": sum(1 for c in civilians if 35 <= c.age < 50),
        "50-65": sum(1 for c in civilians if 50 <= c.age < 65),
        "65+": sum(1 for c in civilians if c.age >= 65),
    }

    # Gender split
    gender_split = {
        "female": sum(1 for c in civilians if c.gender == "female"),
        "male": sum(1 for c in civilians if c.gender == "male"),
    }

    # Market penetration estimate (first year)
    year1_penetration = demand_score * 0.15
    year3_penetration = demand_score * 0.45

    # Epidemiology-based market sizing
    from services.disease_mapping import EPIDEMIOLOGY, DEFAULT_EPIDEMIOLOGY
    epi = EPIDEMIOLOGY.get(disease_group, DEFAULT_EPIDEMIOLOGY)
    global_pop = 8_000_000_000
    prevalence = epi["prevalence_per_100k"]
    total_patients = int(global_pop * prevalence / 100_000)
    addressable_market = int(total_patients * demand_rate * competition_factor)
    avg_cost = epi["avg_cost_usd"]
    total_market_bn = epi["market_bn"]
    growth_rate = epi["growth_rate"]
    unmet_need = epi["unmet_need"]

    # Revenue projections
    year1_patients = int(addressable_market * 0.01 * urgency_factor)  # 1% capture * urgency
    year3_patients = int(addressable_market * 0.05 * urgency_factor)
    year5_patients = int(addressable_market * 0.12 * urgency_factor)
    year1_revenue_m = round(year1_patients * avg_cost / 1_000_000, 1)
    year3_revenue_m = round(year3_patients * avg_cost / 1_000_000, 1)
    year5_revenue_m = round(year5_patients * avg_cost / 1_000_000, 1)

    # Adoption curve (quarterly for 5 years = 20 points)
    adoption_curve = []
    for q in range(20):
        year = q / 4
        # S-curve: slow start, rapid growth, plateau
        pct = 100 / (1 + 50 * (2.718 ** (-0.8 * year)))  # logistic growth
        adoption_curve.append({"quarter": f"Q{(q%4)+1}Y{int(year)+1}", "adoption_pct": round(pct, 1)})

    # Price sensitivity (what happens at different price points)
    price_sensitivity = []
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        price = int(avg_cost * mult)
        # Higher price = fewer patients, but inversely proportional (not linear)
        patients = int(year3_patients * (1.0 / mult) ** 0.5)
        revenue = round(patients * price / 1_000_000, 1)
        price_sensitivity.append({"price_usd": price, "patients": patients, "revenue_m": revenue})

    return {
        "demand_score": round(demand_score, 1),
        "sustained_demand_score": round(sustained_demand, 1),
        "demand_pool_pct": round(demand_rate * 100, 1),
        "competition_factor": round(competition_factor, 3),
        "urgency_factor": round(urgency_factor, 3),
        "comorbidity_conflict_rate": round(comorb_conflict_rate * 100, 1),
        "year1_penetration_pct": round(year1_penetration, 1),
        "year3_penetration_pct": round(year3_penetration, 1),
        "age_segments": age_segments,
        "gender_split": gender_split,
        "alternative_therapies": alt_therapies,
        "avg_treatment_duration_months": profile.get("typical_duration_months", 24),
        # NEW: Epidemiology & market data
        "epidemiology": {
            "prevalence_per_100k": prevalence,
            "total_patients_global": total_patients,
            "addressable_market": addressable_market,
            "mortality_rate": epi["mortality_rate"],
            "unmet_need_score": unmet_need,
        },
        "market": {
            "total_market_bn": total_market_bn,
            "growth_rate_pct": round(growth_rate * 100, 1),
            "avg_treatment_cost_usd": avg_cost,
        },
        "revenue_projections": {
            "year1": {"patients": year1_patients, "revenue_m": year1_revenue_m},
            "year3": {"patients": year3_patients, "revenue_m": year3_revenue_m},
            "year5": {"patients": year5_patients, "revenue_m": year5_revenue_m},
        },
        "adoption_curve": adoption_curve,
        "price_sensitivity": price_sensitivity,
        "source": "local_demand_model",
    }


def get_cached_simulation(drug_name: str, condition_name: str, disease_group: str) -> Optional[dict]:
    """Return cached simulation if available."""
    key = _cache_key(drug_name, condition_name, disease_group)
    cached = sim_cache.get(key)
    if cached is not None:
        return cached[0]

    # Check disk cache
    sim_dir = os.path.join(SIM_DIR, key)
    results_path = os.path.join(sim_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)

    return None
