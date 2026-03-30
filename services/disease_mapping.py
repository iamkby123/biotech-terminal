"""
Disease-to-Civilian Mapping for MiroFish Integration.
Maps v2.condition_hierarchy disease groups to MiroFish civilian (agent) profiles
for drug demand modeling simulations.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Disease group → patient population characteristics + real epidemiology
DISEASE_PROFILES = {
    "oncology_solid": {
        "age_range": (45, 80),
        "age_mean": 62,
        "gender_ratio": 0.5,
        "treatment_urgency": 0.95,
        "comorbidities": ["cardiovascular", "diabetes", "depression", "chronic_pain"],
        "comorbidity_rate": 0.65,
        "alternative_therapies": 3,
        "treatment_seeking_rate": 0.90,
        "adherence_rate": 0.85,
        "insurance_barrier": 0.15,
        "typical_duration_months": 12,
        # Epidemiology (global estimates)
        "prevalence_per_100k": 440,
        "incidence_per_100k": 190,
        "mortality_rate": 0.35,
        "avg_treatment_cost_usd": 150000,
        "global_market_size_bn": 180,
        "market_growth_rate": 0.08,
        "unmet_need_score": 7,  # 1-10
    },
    "oncology_hematologic": {
        "age_range": (30, 75),
        "age_mean": 55,
        "gender_ratio": 0.45,
        "treatment_urgency": 0.95,
        "comorbidities": ["infection_susceptibility", "fatigue", "anemia"],
        "comorbidity_rate": 0.70,
        "alternative_therapies": 4,
        "treatment_seeking_rate": 0.92,
        "adherence_rate": 0.88,
        "insurance_barrier": 0.12,
        "typical_duration_months": 18,
    },
    "cardiovascular": {
        "age_range": (40, 85),
        "age_mean": 65,
        "gender_ratio": 0.40,
        "treatment_urgency": 0.70,
        "comorbidities": ["diabetes", "obesity", "hypertension", "kidney_disease"],
        "comorbidity_rate": 0.75,
        "alternative_therapies": 8,
        "treatment_seeking_rate": 0.70,
        "adherence_rate": 0.60,
        "insurance_barrier": 0.10,
        "typical_duration_months": 36,
    },
    "neurology": {
        "age_range": (35, 80),
        "age_mean": 58,
        "gender_ratio": 0.52,
        "treatment_urgency": 0.60,
        "comorbidities": ["depression", "anxiety", "sleep_disorder", "cognitive_decline"],
        "comorbidity_rate": 0.60,
        "alternative_therapies": 5,
        "treatment_seeking_rate": 0.55,
        "adherence_rate": 0.50,
        "insurance_barrier": 0.20,
        "typical_duration_months": 48,
    },
    "autoimmune": {
        "age_range": (20, 65),
        "age_mean": 42,
        "gender_ratio": 0.70,  # more female
        "treatment_urgency": 0.55,
        "comorbidities": ["fatigue", "depression", "joint_pain", "skin_conditions"],
        "comorbidity_rate": 0.55,
        "alternative_therapies": 6,
        "treatment_seeking_rate": 0.65,
        "adherence_rate": 0.55,
        "insurance_barrier": 0.18,
        "typical_duration_months": 60,
    },
    "infectious_disease": {
        "age_range": (18, 70),
        "age_mean": 40,
        "gender_ratio": 0.50,
        "treatment_urgency": 0.85,
        "comorbidities": ["immunocompromised", "diabetes", "respiratory"],
        "comorbidity_rate": 0.30,
        "alternative_therapies": 4,
        "treatment_seeking_rate": 0.80,
        "adherence_rate": 0.75,
        "insurance_barrier": 0.08,
        "typical_duration_months": 3,
    },
    "rare_disease": {
        "age_range": (0, 50),
        "age_mean": 25,
        "gender_ratio": 0.50,
        "treatment_urgency": 0.80,
        "comorbidities": ["developmental_delay", "organ_damage", "metabolic"],
        "comorbidity_rate": 0.70,
        "alternative_therapies": 1,
        "treatment_seeking_rate": 0.95,
        "adherence_rate": 0.90,
        "insurance_barrier": 0.30,
        "typical_duration_months": 120,
    },
    "metabolic": {
        "age_range": (30, 75),
        "age_mean": 55,
        "gender_ratio": 0.48,
        "treatment_urgency": 0.45,
        "comorbidities": ["cardiovascular", "obesity", "kidney_disease", "neuropathy"],
        "comorbidity_rate": 0.70,
        "alternative_therapies": 10,
        "treatment_seeking_rate": 0.60,
        "adherence_rate": 0.45,
        "insurance_barrier": 0.08,
        "typical_duration_months": 120,
    },
    "respiratory": {
        "age_range": (25, 80),
        "age_mean": 55,
        "gender_ratio": 0.48,
        "treatment_urgency": 0.60,
        "comorbidities": ["cardiovascular", "anxiety", "sleep_disorder", "obesity"],
        "comorbidity_rate": 0.55,
        "alternative_therapies": 6,
        "treatment_seeking_rate": 0.65,
        "adherence_rate": 0.55,
        "insurance_barrier": 0.10,
        "typical_duration_months": 36,
    },
    "dermatology": {
        "age_range": (15, 65),
        "age_mean": 38,
        "gender_ratio": 0.55,
        "treatment_urgency": 0.30,
        "comorbidities": ["depression", "anxiety", "autoimmune"],
        "comorbidity_rate": 0.35,
        "alternative_therapies": 7,
        "treatment_seeking_rate": 0.50,
        "adherence_rate": 0.45,
        "insurance_barrier": 0.25,
        "typical_duration_months": 24,
    },
    "ophthalmology": {
        "age_range": (40, 85),
        "age_mean": 65,
        "gender_ratio": 0.52,
        "treatment_urgency": 0.65,
        "comorbidities": ["diabetes", "hypertension", "cardiovascular"],
        "comorbidity_rate": 0.60,
        "alternative_therapies": 4,
        "treatment_seeking_rate": 0.75,
        "adherence_rate": 0.70,
        "insurance_barrier": 0.15,
        "typical_duration_months": 24,
    },
    "renal": {
        "age_range": (35, 80),
        "age_mean": 60,
        "gender_ratio": 0.45,
        "treatment_urgency": 0.75,
        "comorbidities": ["cardiovascular", "diabetes", "anemia", "bone_disease"],
        "comorbidity_rate": 0.80,
        "alternative_therapies": 4,
        "treatment_seeking_rate": 0.85,
        "adherence_rate": 0.75,
        "insurance_barrier": 0.12,
        "typical_duration_months": 60,
    },
}

# Default for unknown disease groups
DEFAULT_PROFILE = {
    "age_range": (25, 70),
    "age_mean": 50,
    "gender_ratio": 0.50,
    "treatment_urgency": 0.50,
    "comorbidities": ["other_chronic"],
    "comorbidity_rate": 0.40,
    "alternative_therapies": 5,
    "treatment_seeking_rate": 0.60,
    "adherence_rate": 0.55,
    "insurance_barrier": 0.15,
    "typical_duration_months": 24,
}


# Real-world epidemiology and market data per disease group
EPIDEMIOLOGY = {
    "oncology_solid":       {"prevalence_per_100k": 440, "incidence_per_100k": 190, "mortality_rate": 0.35, "avg_cost_usd": 150000, "market_bn": 180, "growth_rate": 0.08, "unmet_need": 7},
    "oncology_hematologic": {"prevalence_per_100k": 55,  "incidence_per_100k": 25,  "mortality_rate": 0.40, "avg_cost_usd": 200000, "market_bn": 45,  "growth_rate": 0.10, "unmet_need": 8},
    "cardiovascular":       {"prevalence_per_100k": 6000,"incidence_per_100k": 800, "mortality_rate": 0.15, "avg_cost_usd": 25000,  "market_bn": 95,  "growth_rate": 0.04, "unmet_need": 5},
    "neurology":            {"prevalence_per_100k": 1200,"incidence_per_100k": 150, "mortality_rate": 0.08, "avg_cost_usd": 50000,  "market_bn": 40,  "growth_rate": 0.07, "unmet_need": 9},
    "autoimmune":           {"prevalence_per_100k": 2400,"incidence_per_100k": 80,  "mortality_rate": 0.02, "avg_cost_usd": 35000,  "market_bn": 65,  "growth_rate": 0.06, "unmet_need": 6},
    "infectious_disease":   {"prevalence_per_100k": 3000,"incidence_per_100k": 1500,"mortality_rate": 0.05, "avg_cost_usd": 5000,   "market_bn": 55,  "growth_rate": 0.05, "unmet_need": 4},
    "rare_disease":         {"prevalence_per_100k": 10,  "incidence_per_100k": 2,   "mortality_rate": 0.25, "avg_cost_usd": 300000, "market_bn": 25,  "growth_rate": 0.12, "unmet_need": 10},
    "metabolic":            {"prevalence_per_100k": 8500,"incidence_per_100k": 600, "mortality_rate": 0.03, "avg_cost_usd": 15000,  "market_bn": 120, "growth_rate": 0.05, "unmet_need": 5},
    "respiratory":          {"prevalence_per_100k": 4000,"incidence_per_100k": 400, "mortality_rate": 0.06, "avg_cost_usd": 20000,  "market_bn": 50,  "growth_rate": 0.04, "unmet_need": 5},
    "dermatology":          {"prevalence_per_100k": 3500,"incidence_per_100k": 500, "mortality_rate": 0.01, "avg_cost_usd": 10000,  "market_bn": 30,  "growth_rate": 0.06, "unmet_need": 4},
    "ophthalmology":        {"prevalence_per_100k": 2000,"incidence_per_100k": 200, "mortality_rate": 0.01, "avg_cost_usd": 15000,  "market_bn": 20,  "growth_rate": 0.05, "unmet_need": 6},
    "renal":                {"prevalence_per_100k": 1500,"incidence_per_100k": 100, "mortality_rate": 0.10, "avg_cost_usd": 80000,  "market_bn": 15,  "growth_rate": 0.07, "unmet_need": 7},
}
DEFAULT_EPIDEMIOLOGY = {"prevalence_per_100k": 1000, "incidence_per_100k": 100, "mortality_rate": 0.05, "avg_cost_usd": 30000, "market_bn": 20, "growth_rate": 0.05, "unmet_need": 5}


@dataclass
class CivilianProfile:
    """A single simulated patient/civilian for MiroFish."""
    user_id: int
    username: str
    name: str
    age: int
    gender: str
    bio: str
    persona: str
    disease_group: str
    primary_condition: str
    comorbidities: List[str] = field(default_factory=list)
    treatment_urgency: float = 0.5
    treatment_seeking: bool = True
    has_insurance: bool = True
    adherent: bool = True
    interested_topics: List[str] = field(default_factory=list)
    activity_level: float = 0.5
    sentiment_bias: float = 0.0

    def to_mirofish_reddit_profile(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": int(500 + self.activity_level * 2000),
            "age": self.age,
            "gender": self.gender,
            "interested_topics": self.interested_topics,
        }

    def to_dict(self) -> dict:
        return asdict(self)


# Name pools for generation
FIRST_NAMES_M = ["James","Robert","Michael","David","John","William","Thomas","Daniel","Mark","Steven",
                 "Charles","Joseph","Andrew","Kenneth","Paul","Brian","George","Edward","Richard","Kevin"]
FIRST_NAMES_F = ["Mary","Patricia","Jennifer","Linda","Barbara","Susan","Jessica","Sarah","Karen","Lisa",
                 "Nancy","Betty","Helen","Sandra","Donna","Carol","Ruth","Sharon","Michelle","Laura"]
LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
              "Anderson","Taylor","Thomas","Moore","Jackson","Martin","Lee","Thompson","White","Harris"]


def generate_civilian_population(
    disease_group: str,
    condition_name: str,
    drug_name: str = "",
    population_size: int = 100,
    seed: int = 42,
) -> List[CivilianProfile]:
    """Generate a population of civilians for MiroFish simulation."""
    rng = random.Random(seed)
    profile = DISEASE_PROFILES.get(disease_group, DEFAULT_PROFILE)

    civilians = []
    for i in range(population_size):
        age_lo, age_hi = profile["age_range"]
        age = rng.randint(age_lo, age_hi)
        is_female = rng.random() < profile["gender_ratio"]
        gender = "female" if is_female else "male"

        first = rng.choice(FIRST_NAMES_F if is_female else FIRST_NAMES_M)
        last = rng.choice(LAST_NAMES)
        name = f"{first} {last}"
        username = f"{first.lower()}_{last.lower()}_{rng.randint(100,999)}"

        # Comorbidities
        comorbs = []
        for c in profile["comorbidities"]:
            if rng.random() < profile["comorbidity_rate"] / len(profile["comorbidities"]):
                comorbs.append(c)

        # Treatment seeking
        seeking = rng.random() < profile["treatment_seeking_rate"]
        has_insurance = rng.random() > profile["insurance_barrier"]
        adherent = rng.random() < profile["adherence_rate"]
        urgency = profile["treatment_urgency"] + rng.uniform(-0.1, 0.1)

        # Activity level based on urgency and age
        activity = 0.3 + urgency * 0.4 + rng.uniform(-0.1, 0.2)
        activity = max(0.1, min(1.0, activity))

        # Sentiment: patients with high urgency and few alternatives are more positive toward new drugs
        alt_factor = profile["alternative_therapies"]
        if alt_factor <= 2:
            sentiment = rng.uniform(0.2, 0.7)  # hopeful for new treatment
        elif alt_factor >= 7:
            sentiment = rng.uniform(-0.3, 0.3)  # more skeptical, many options
        else:
            sentiment = rng.uniform(-0.1, 0.5)

        # Bio
        comorb_str = f" Also managing {', '.join(comorbs)}." if comorbs else ""
        insurance_str = " Uninsured." if not has_insurance else ""
        bio = (f"{age}-year-old {gender} diagnosed with {condition_name}.{comorb_str}{insurance_str} "
               f"{'Actively seeking treatment options.' if seeking else 'Monitoring condition.'}")

        # Persona
        if urgency > 0.8:
            persona_trait = "Urgently seeking effective treatment, willing to try new therapies"
        elif urgency > 0.5:
            persona_trait = "Moderately concerned about condition, researching treatment options"
        else:
            persona_trait = "Managing chronic condition, cautious about new treatments"

        persona = (f"Patient with {condition_name}. {persona_trait}. "
                  f"{'Has insurance coverage' if has_insurance else 'Concerned about treatment costs'}. "
                  f"{'Takes medications as prescribed' if adherent else 'Sometimes misses doses or stops early'}.")

        topics = [condition_name.lower(), disease_group, "health", "medicine"]
        if drug_name:
            topics.append(drug_name.lower())
        if comorbs:
            topics.extend(comorbs[:2])

        civilian = CivilianProfile(
            user_id=i + 1,
            username=username,
            name=name,
            age=age,
            gender=gender,
            bio=bio,
            persona=persona,
            disease_group=disease_group,
            primary_condition=condition_name,
            comorbidities=comorbs,
            treatment_urgency=round(urgency, 2),
            treatment_seeking=seeking,
            has_insurance=has_insurance,
            adherent=adherent,
            interested_topics=topics,
            activity_level=round(activity, 2),
            sentiment_bias=round(sentiment, 2),
        )
        civilians.append(civilian)

    return civilians


def get_population_summary(civilians: List[CivilianProfile]) -> dict:
    """Compute summary statistics for a civilian population."""
    n = len(civilians)
    if n == 0:
        return {"population_size": 0}

    ages = [c.age for c in civilians]
    seeking = sum(1 for c in civilians if c.treatment_seeking)
    insured = sum(1 for c in civilians if c.has_insurance)
    adherent = sum(1 for c in civilians if c.adherent)
    female = sum(1 for c in civilians if c.gender == "female")

    # Comorbidity counts
    comorb_counts = {}
    for c in civilians:
        for cm in c.comorbidities:
            comorb_counts[cm] = comorb_counts.get(cm, 0) + 1

    return {
        "population_size": n,
        "avg_age": round(sum(ages) / n, 1),
        "age_range": [min(ages), max(ages)],
        "female_pct": round(female / n * 100, 1),
        "treatment_seeking_pct": round(seeking / n * 100, 1),
        "insured_pct": round(insured / n * 100, 1),
        "adherent_pct": round(adherent / n * 100, 1),
        "avg_urgency": round(sum(c.treatment_urgency for c in civilians) / n, 2),
        "avg_sentiment": round(sum(c.sentiment_bias for c in civilians) / n, 2),
        "top_comorbidities": sorted(comorb_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        "demand_estimate": round(seeking / n * insured / n * 100, 1),  # % who would actually seek + afford
    }
