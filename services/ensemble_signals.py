"""
Ensemble Signals — 6 independent prediction signals combined into a final score.
Each signal outputs a failure probability (0-1). Missing signals redistribute weight.
"""
import os
import json
import math
import pickle
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _release(conn):
    """Return connection to pool or close it."""
    try:
        from models.ticker_map import release_connection
        release_connection(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_THIS_DIR)
ENSEMBLE_DIR = os.path.join(_APP_DIR, "model_data", "ensemble_v10")
DUAL_DIR = os.path.join(_APP_DIR, "model_data", "dual_models")

# Lazy-loaded artifacts
_xgb_model = None
_xgb_encoders = None
_xgb_feat_cols = None
_condition_rates = None
_target_rates = None
_target_families = None
_loaded = False

# Dual models
_optimist = None
_pessimist = None
_opt_encoders = None
_pes_encoders = None
_opt_feats = None
_pes_feats = None
_dual_loaded = False

_aact_cache = {}

def _enrich_trial_from_aact(trial: dict) -> dict:
    """Enrich trial dict with AACT design/eligibility fields if not already present."""
    nct_id = trial.get("nct_id", "")
    if not nct_id:
        return trial

    # Check cache
    if nct_id in _aact_cache:
        trial.update(_aact_cache[nct_id])
        return trial

    # Only query if key fields are missing
    if trial.get("allocation") and trial.get("minimum_age"):
        return trial

    try:
        import psycopg2
        import psycopg2.extras
        AACT_URL = os.environ.get("AACT_DATABASE_URL", "postgresql://kby123:Kevinby452423!@aact-db.ctti-clinicaltrials.org:5432/aact")
        conn = psycopg2.connect(AACT_URL, cursor_factory=psycopg2.extras.RealDictCursor,
                                connect_timeout=5)
        cur = conn.cursor()

        enriched = {}

        # Single JOIN query instead of 4 separate queries
        cur.execute("""
            SELECT d.allocation, d.intervention_model, d.primary_purpose, d.masking,
                   d.subject_masked, d.caregiver_masked, d.investigator_masked, d.outcomes_assessor_masked,
                   e.gender, e.minimum_age, e.maximum_age, e.healthy_volunteers, e.criteria,
                   s.number_of_arms, s.start_date, s.primary_completion_date, s.completion_date,
                   (SELECT COUNT(DISTINCT country) FROM ctgov.facilities WHERE nct_id = %s) as num_countries
            FROM ctgov.studies s
            LEFT JOIN ctgov.designs d ON d.nct_id = s.nct_id
            LEFT JOIN ctgov.eligibilities e ON e.nct_id = s.nct_id
            WHERE s.nct_id = %s
        """, (nct_id, nct_id))
        row = cur.fetchone()

        if row:
            # Design fields
            enriched["allocation"] = row.get("allocation", "")
            enriched["intervention_model"] = row.get("intervention_model", "")
            enriched["primary_purpose"] = row.get("primary_purpose", "")
            enriched["masking"] = row.get("masking", "")
            enriched["subject_masked"] = row.get("subject_masked")
            enriched["caregiver_masked"] = row.get("caregiver_masked")
            enriched["investigator_masked"] = row.get("investigator_masked")
            enriched["outcomes_assessor_masked"] = row.get("outcomes_assessor_masked")
            # Eligibility fields
            enriched["gender"] = row.get("gender", "")
            enriched["minimum_age"] = row.get("minimum_age", "")
            enriched["maximum_age"] = row.get("maximum_age", "")
            enriched["healthy_volunteers"] = row.get("healthy_volunteers", "")
            enriched["eligibility_criteria"] = row.get("criteria", "")
            # Study fields
            enriched["number_of_arms"] = row.get("number_of_arms")
            enriched["num_countries"] = row.get("num_countries", 0)
            start = row.get("start_date")
            end = row.get("primary_completion_date") or row.get("completion_date")
            if start and end:
                try:
                    delta = (end - start).days
                    if delta > 0:
                        enriched["study_duration_months"] = round(delta / 30.44, 1)
                except:
                    pass

        conn.close()

        # v2 schema data — use connection pool
        try:
            from models.ticker_map import get_connection, release_connection
            neon_conn = get_connection()
            neon_cur = neon_conn.cursor()

            neon_cur.execute("""
                SELECT
                    (SELECT COUNT(*) FROM v2.endpoint WHERE nct_id = %s AND endpoint_type = 'secondary') as sec_ep,
                    (SELECT COUNT(*) FROM v2.trial_sponsor WHERE nct_id = %s AND role = 'collaborator') as collabs
            """, (nct_id, nct_id))
            r = neon_cur.fetchone()
            if r:
                enriched["num_secondary_endpoints"] = r["sec_ep"]
                enriched["is_multi_sponsor"] = 1 if r["collabs"] > 0 else 0

            neon_cur.close()
            release_connection(neon_conn)
        except Exception as e2:
            logger.debug("Neon enrichment failed: %s", e2)

        _aact_cache[nct_id] = enriched
        trial.update(enriched)
    except Exception as e:
        logger.debug("AACT enrichment failed for %s: %s", nct_id, e)
        _aact_cache[nct_id] = {}  # cache failure to avoid repeated attempts

    return trial


SIGNAL_WEIGHTS = {
    "xgb": 0.25,
    "condition": 0.15,
    "target": 0.10,
    "sponsor": 0.08,
    "design": 0.08,
    "similar": 0.08,
    "endpoint": 0.10,
    "drug_history": 0.08,
    "regulatory": 0.08,
}


def _load_artifacts():
    global _xgb_model, _xgb_encoders, _xgb_feat_cols, _condition_rates, _target_rates, _target_families, _loaded
    if _loaded:
        return
    _loaded = True
    try:
        with open(os.path.join(ENSEMBLE_DIR, "xgb_model.pkl"), "rb") as f:
            _xgb_model = pickle.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "encoders.pkl"), "rb") as f:
            _xgb_encoders = pickle.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "feat_cols.json")) as f:
            _xgb_feat_cols = json.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "condition_rates.json")) as f:
            _condition_rates = json.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "drug_target_rates.json")) as f:
            _target_rates = json.load(f)
        with open(os.path.join(ENSEMBLE_DIR, "target_families.json")) as f:
            _target_families = json.load(f)
        logger.info("Ensemble V10 artifacts loaded (%d features)", len(_xgb_feat_cols))
    except Exception as e:
        logger.warning("Failed to load ensemble artifacts: %s", e)


# ═══════════════════════════════════════════════════════════════
# SIGNAL 1: XGBoost Model
# ═══════════════════════════════════════════════════════════════
def _parse_age_years(age_str):
    """Parse '18 Years' -> 18, '6 Months' -> 0.5."""
    if not age_str or str(age_str) in ("None", "nan", ""):
        return float("nan")
    try:
        parts = str(age_str).strip().split()
        val = float(parts[0])
        if len(parts) > 1 and "month" in parts[1].lower(): val /= 12
        elif len(parts) > 1 and "day" in parts[1].lower(): val /= 365
        return val
    except:
        return float("nan")


def _compute_masking_level(trial):
    """Compute masking level 0-4 from trial data."""
    masking = str(trial.get("masking", "") or "").upper()
    if "QUADRUPLE" in masking: return 4
    if "TRIPLE" in masking: return 3
    if "DOUBLE" in masking: return 2
    if "SINGLE" in masking: return 1
    # Check individual masks
    count = sum([
        bool(trial.get("subject_masked")),
        bool(trial.get("caregiver_masked")),
        bool(trial.get("investigator_masked")),
        bool(trial.get("outcomes_assessor_masked")),
    ])
    if count > 0: return min(count, 4)
    # Fallback: check study_design text
    design = str(trial.get("study_design", "") or "").lower()
    if "quadruple" in design: return 4
    if "triple" in design: return 3
    if "double" in design: return 2
    if "single" in design and "blind" in design: return 1
    return 0


def compute_xgb_signal(trial: dict) -> dict:
    _load_artifacts()
    if _xgb_model is None or _xgb_feat_cols is None:
        return None

    # Feature engineering (must match V10 training)
    phase_num = 0
    phase = str(trial.get("phase", "") or "")
    if "4" in phase: phase_num = 4
    elif "3" in phase: phase_num = 3
    elif "2" in phase: phase_num = 2
    elif "1" in phase: phase_num = 1

    enrollment = 0
    try:
        enrollment = float(trial.get("enrollment") or trial.get("feat_enrollment_size") or 0)
    except:
        pass

    design = str(trial.get("study_design", "") or "").lower()
    condition = str(trial.get("condition", "") or "")
    intervention = str(trial.get("intervention", "") or "")
    summary = str(trial.get("brief_summary", "") or "")

    # Classify drug target
    target_class = "unknown"
    if _target_families:
        int_lower = intervention.lower()
        for family, keywords in _target_families.items():
            if any(kw in int_lower for kw in keywords):
                target_class = family
                break

    is_oncology = 1 if any(w in condition.lower() for w in ["oncol", "cancer", "tumor", "carcinoma", "lymphoma", "leukemia"]) else 0
    is_combination = 1 if ";" in intervention else 0

    # New V10 features
    allocation = str(trial.get("allocation", "") or "UNKNOWN").upper()
    intervention_model = str(trial.get("intervention_model", "") or "UNKNOWN").upper()
    primary_purpose = str(trial.get("primary_purpose", "") or "UNKNOWN").upper()
    masking_level = _compute_masking_level(trial)

    gender = str(trial.get("gender", trial.get("sex", "")) or "ALL").upper()
    gender_restriction = 1 if gender in ("MALE", "FEMALE") else 0
    min_age_raw = trial.get("minimum_age", trial.get("min_age"))
    max_age_raw = trial.get("maximum_age", trial.get("max_age"))
    min_age = _parse_age_years(min_age_raw) if not isinstance(min_age_raw, (int, float)) else float(min_age_raw)
    max_age = _parse_age_years(max_age_raw) if not isinstance(max_age_raw, (int, float)) else float(max_age_raw)
    if math.isnan(min_age): min_age = 18.0
    if math.isnan(max_age): max_age = 99.0
    accepts_healthy = 1 if str(trial.get("healthy_volunteers", "") or "").lower() in ("true", "yes", "t") else 0

    num_arms = int(trial.get("number_of_arms", trial.get("num_arms", 2)) or 2)
    num_sec = int(trial.get("num_secondary_endpoints", 0) or 0)
    num_countries = int(trial.get("num_countries", 1) or 1)
    is_multi = int(trial.get("is_multi_sponsor", 0) or 0)

    elig_text = str(trial.get("eligibility_criteria", "") or "")
    elig_complexity = len(elig_text.split())
    duration = float(trial.get("study_duration_months", 24) or 24)

    computed = {
        # V9 features
        "feat_phase_num": phase_num,
        "feat_is_phase2": 1 if phase_num == 2 else 0,
        "feat_is_phase3": 1 if phase_num == 3 else 0,
        "feat_is_early_phase": 1 if phase_num <= 1 else 0,
        "feat_is_interventional": 1,
        "feat_is_oncology": is_oncology,
        "feat_is_rare_disease": 0,
        "feat_is_combination": is_combination,
        "feat_has_biomarker_selection": 0,
        "feat_enrollment_size": enrollment,
        "feat_has_randomization": 1 if "random" in design else 0,
        "feat_has_control_group": 1 if "control" in design or "random" in design else 0,
        "feat_has_placebo": 1 if "placebo" in design else 0,
        "feat_is_double_blind": 1 if "double" in design else 0,
        "feat_has_active_comparator": 0,
        "feat_is_orphan_indication": 0,
        "feat_has_efficacy_endpoint": 1,
        "feat_is_safety_trial": 0,
        "feat_num_conditions": len(condition.split(";")) if condition else 1,
        "feat_num_interventions": len(intervention.split(";")) if intervention else 1,
        "sponsor_completion_rate": float(trial.get("sponsor_completion_rate", 0.5) or 0.5),
        "sponsor_failure_rate": float(trial.get("sponsor_failure_rate", 0.2) or 0.2),
        "sponsor_total_log": math.log1p(float(trial.get("sponsor_total_trials", 0) or 0)),
        "design_quality_score": float(trial.get("design_quality_score", 3) or 3),
        "phase_x_oncology": phase_num * is_oncology,
        "phase_x_rare": 0,
        "combination_x_phase": is_combination * phase_num,
        "enrollment_log": math.log1p(enrollment),
        "target_class": target_class,
        # New V10 features
        "allocation": allocation,
        "intervention_model": intervention_model,
        "primary_purpose": primary_purpose,
        "masking_level": masking_level,
        "gender_restriction": gender_restriction,
        "min_age": min_age,
        "max_age": max_age,
        "accepts_healthy": accepts_healthy,
        "age_range": max_age - min_age,
        "is_pediatric": 1 if min_age < 18 else 0,
        "num_arms": num_arms,
        "num_secondary_endpoints": num_sec,
        "num_secondary_log": math.log1p(num_sec),
        "num_countries": num_countries,
        "is_multi_sponsor": is_multi,
        "eligibility_complexity": elig_complexity,
        "eligibility_log": math.log1p(elig_complexity),
        "study_duration_months": duration,
    }

    # Build feature vector
    feat_vals = []
    CATEGORICAL = {"feat_therapeutic_area", "feat_indication_category", "feat_modality",
                   "feat_primary_endpoint_type", "target_class",
                   "allocation", "intervention_model", "primary_purpose"}
    for col in _xgb_feat_cols:
        if col in CATEGORICAL:
            raw = str(trial.get(col, computed.get(col, "unknown")) or "unknown")
            le = _xgb_encoders.get(col)
            if le:
                try:
                    feat_vals.append(float(le.transform([raw])[0]))
                except:
                    feat_vals.append(0.0)
            else:
                feat_vals.append(0.0)
        else:
            feat_vals.append(float(computed.get(col, trial.get(col, 0)) or 0))

    X = np.array([feat_vals])
    prob = _xgb_model.predict_proba(X)[0]
    fail_prob = float(prob[1])

    # Get feature importances for drivers
    try:
        base_model = _xgb_model.estimator if hasattr(_xgb_model, "estimator") else _xgb_model
        if hasattr(base_model, "estimators_"):
            base_model = base_model.estimators_[0]
        importances = base_model.feature_importances_ if hasattr(base_model, "feature_importances_") else []
    except:
        importances = []

    drivers = []
    if len(importances) == len(_xgb_feat_cols):
        for col, imp in sorted(zip(_xgb_feat_cols, importances), key=lambda x: -x[1])[:8]:
            drivers.append({"feature": col, "importance": round(float(imp) * 100, 1)})

    # Per-trial SHAP-style feature contributions (not global importances)
    feature_impacts = []
    try:
        import xgboost as xgb
        booster = _xgb_model.estimator.get_booster() if hasattr(_xgb_model, "estimator") else None
        if booster:
            dmat = xgb.DMatrix(X, feature_names=_xgb_feat_cols)
            contribs = booster.predict(dmat, pred_contribs=True)[0]
            for col, val, contrib in zip(_xgb_feat_cols, feat_vals, contribs[:-1]):
                if abs(float(contrib)) > 0.005:
                    feature_impacts.append({
                        "feature": col,
                        "value": round(float(val), 2),
                        "impact": round(float(contrib) * 100, 1),
                        "direction": "risk" if contrib > 0 else "protective",
                    })
            feature_impacts.sort(key=lambda x: abs(x["impact"]), reverse=True)
            feature_impacts = feature_impacts[:10]
    except Exception as e:
        logger.warning("SHAP contrib failed: %s", e)
    return {
        "prob": round(fail_prob, 4),
        "drivers": drivers,
        "feature_impacts": feature_impacts,
        "detail": f"ML model analyzing {len(_xgb_feat_cols)} trial characteristics, trained on 785 completed trials",
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL 2: Condition Base Rate
# ═══════════════════════════════════════════════════════════════
def compute_condition_signal(trial: dict) -> dict:
    _load_artifacts()
    if not _condition_rates:
        return None

    condition = str(trial.get("condition", "") or "").lower()
    indication = str(trial.get("feat_indication_category", "") or "").lower()

    best_match = None
    best_rate = None
    best_n = 0

    # Check indication category first
    if indication and indication in _condition_rates:
        r = _condition_rates[indication]
        best_match = indication
        best_rate = r["rate"]
        best_n = r["n"]

    # Check condition text for more specific matches
    for cond_key, r in sorted(_condition_rates.items(), key=lambda x: -len(x[0])):
        if cond_key in condition:
            if best_rate is None or (r.get("source") == "literature" and best_rate < r["rate"]):
                best_match = cond_key
                best_rate = r["rate"]
                best_n = r["n"]
            break

    if best_rate is None:
        return None

    n_str = f"n={best_n}" if best_n > 0 else "literature estimate"
    return {"prob": round(best_rate, 4), "detail": f"{best_match}: {round(best_rate*100,1)}% failure ({n_str})"}


# ═══════════════════════════════════════════════════════════════
# SIGNAL 3: Drug Target Success Rate
# ═══════════════════════════════════════════════════════════════
def compute_target_signal(trial: dict) -> dict:
    _load_artifacts()
    intervention = str(trial.get("intervention", "") or "").lower()

    if _target_families:
        for family, keywords in _target_families.items():
            if any(kw in intervention for kw in keywords):
                if _target_rates and family in _target_rates:
                    r = _target_rates[family]
                    return {"prob": round(r["rate"], 4), "detail": f"{family}: {round(r['rate']*100,1)}% failure (n={r['n']})"}
                # Even without specific rate, the target family provides context
                return {"prob": 0.50, "detail": f"Target: {family} (no specific rate data)"}

    return None


# ═══════════════════════════════════════════════════════════════
# SIGNAL 4: Sponsor Track Record
# ═══════════════════════════════════════════════════════════════
def compute_sponsor_signal(trial: dict) -> dict:
    try:
        from models.ticker_map import get_connection
        sponsor = str(trial.get("sponsor", "") or "")
        if not sponsor or len(sponsor) < 3:
            return None

        conn = get_connection()
        cur = conn.cursor()
        sp_kw = sponsor.split(",")[0].split()[0] if sponsor else ""
        if len(sp_kw) < 3:
            cur.close(); _release(conn)
            return None

        cur.execute("SELECT status, COUNT(*) as n FROM trials WHERE sponsor ILIKE %s GROUP BY status", (f"%{sp_kw}%",))
        counts = {r["status"]: r["n"] for r in cur.fetchall()}
        cur.close(); _release(conn)

        completed = counts.get("COMPLETED", 0) + counts.get("Completed", 0)
        terminated = counts.get("TERMINATED", 0) + counts.get("Terminated", 0)
        total = sum(counts.values())

        if total < 3:
            return None

        fail_rate = terminated / max(completed + terminated, 1)
        # Regress toward mean for small samples
        if total < 20:
            fail_rate = 0.5 * fail_rate + 0.5 * 0.30  # prior: 30% failure
        return {"prob": round(fail_rate, 4), "detail": f"{sponsor[:25]}: {round((1-fail_rate)*100)}% completion ({total} trials)"}
    except Exception as e:
        logger.debug("Sponsor signal failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# SIGNAL 5: Design Quality
# ═══════════════════════════════════════════════════════════════
def compute_design_signal(trial: dict) -> dict:
    """Assess design quality using AACT structural data + text fallback."""
    design = str(trial.get("study_design", "") or "").lower()
    enrollment = 0
    try:
        enrollment = float(trial.get("enrollment") or 0)
    except:
        pass

    score = 0
    checks = []

    # Use actual AACT allocation field (much more reliable than text matching)
    allocation = str(trial.get("allocation", "") or "").upper()
    if allocation == "RANDOMIZED":
        score += 2; checks.append("randomized (confirmed)")
    elif "random" in design:
        score += 1; checks.append("randomized (text)")

    # Use actual masking level from AACT
    masking_level = _compute_masking_level(trial)
    if masking_level >= 2:
        score += 2; checks.append(f"{'double' if masking_level==2 else 'triple' if masking_level==3 else 'quadruple'}-blind")
    elif masking_level == 1:
        score += 1; checks.append("single-blind")
    elif "double" in design or "blind" in design:
        score += 1; checks.append("blinded (text)")

    # Placebo control
    if "placebo" in design or "placebo" in str(trial.get("intervention", "") or "").lower():
        score += 1; checks.append("placebo-controlled")

    # Number of arms (from AACT)
    num_arms = int(trial.get("number_of_arms", trial.get("num_arms", 0)) or 0)
    if num_arms >= 2:
        score += 1; checks.append(f"{num_arms} arms")

    # Enrollment power
    if enrollment > 300:
        score += 2; checks.append(f"n={int(enrollment)} (well-powered)")
    elif enrollment > 100:
        score += 1; checks.append(f"n={int(enrollment)}")

    # Intervention model
    int_model = str(trial.get("intervention_model", "") or "").upper()
    if int_model == "PARALLEL":
        score += 1; checks.append("parallel design")

    # Primary purpose clarity
    purpose = str(trial.get("primary_purpose", "") or "").upper()
    if purpose == "TREATMENT":
        score += 0.5; checks.append("treatment purpose")

    # Map score (0-10.5) to failure probability
    max_score = 10.5
    fail_prob = 0.75 - (min(score, max_score) / max_score) * 0.50  # 10.5 → 0.25, 0 → 0.75
    detail = f"Design strength: {score:.0f}/10 ({', '.join(checks[:4])})" if checks else "No design info available"
    return {"prob": round(max(0.15, min(0.80, fail_prob)), 4), "detail": detail}


# ═══════════════════════════════════════════════════════════════
# SIGNAL 6: Similar Trial Outcome Ratio
# ═══════════════════════════════════════════════════════════════
def compute_similar_signal(trial: dict) -> dict:
    """Find similar trials by condition + phase, weighted by design similarity."""
    try:
        from models.ticker_map import get_connection
        condition = str(trial.get("condition", "") or "")
        phase = str(trial.get("phase", "") or "")
        nct_id = str(trial.get("nct_id", ""))

        if not condition or len(condition) < 3:
            return None

        cond_words = [w.strip() for w in condition.replace(";", " ").replace(",", " ").split() if len(w.strip()) > 3]
        if not cond_words:
            return None

        conn = get_connection()
        cur = conn.cursor()

        cond_like = " OR ".join(["condition ILIKE %s"] * min(len(cond_words), 3))
        params = [f"%{w}%" for w in cond_words[:3]]

        cur.execute(f"""
            SELECT nct_id, status, study_design FROM trials
            WHERE ({cond_like}) AND nct_id != %s
            AND status IN ('COMPLETED','Completed','TERMINATED','Terminated')
            LIMIT 50
        """, params + [nct_id])

        rows = cur.fetchall()
        cur.close(); _release(conn)

        if len(rows) < 3:
            return None

        # Weight by design similarity — trials with matching allocation get 1.5× weight
        focal_allocation = str(trial.get("allocation", "") or "").upper()
        focal_is_randomized = focal_allocation == "RANDOMIZED" or "random" in str(trial.get("study_design", "") or "").lower()

        weighted_completed = 0
        weighted_terminated = 0
        total = 0

        for r in rows:
            status = r["status"]
            sim_design = str(r.get("study_design", "") or "").lower()
            sim_is_randomized = "random" in sim_design

            # 1.5× weight for design-matched trials
            weight = 1.5 if (focal_is_randomized == sim_is_randomized) else 1.0

            if status in ("COMPLETED", "Completed"):
                weighted_completed += weight
            elif status in ("TERMINATED", "Terminated"):
                weighted_terminated += weight
            total += 1

        w_total = weighted_completed + weighted_terminated
        if w_total == 0:
            return None

        fail_rate = weighted_terminated / w_total
        raw_completed = sum(1 for r in rows if r["status"] in ("COMPLETED", "Completed"))
        raw_terminated = total - raw_completed
        detail = f"{raw_completed} completed, {raw_terminated} terminated out of {total} similar"
        if focal_is_randomized:
            detail += " (design-weighted)"
        return {"prob": round(fail_rate, 4), "detail": detail}
    except Exception as e:
        logger.debug("Similar signal failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# ENSEMBLE COMBINER
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# SIGNAL 7: Clinical Endpoint Factors
# ═══════════════════════════════════════════════════════════════
def compute_endpoint_signal(trial: dict) -> dict:
    """Assess risk based on primary endpoint type and design."""
    summary = str(trial.get("brief_summary", "") or "").lower()
    condition = str(trial.get("condition", "") or "").lower()
    endpoint_type = str(trial.get("feat_primary_endpoint_type", "") or str(trial.get("primary_endpoint", ""))).lower()

    # Endpoint type risk scores (from industry data)
    ENDPOINT_RISK = {
        "overall survival": 0.55, "os": 0.55,  # hard endpoint, high bar
        "progression-free survival": 0.45, "pfs": 0.45,
        "response rate": 0.35, "orr": 0.35, "objective response": 0.35,
        "biomarker": 0.40, "hba1c": 0.30, "ldl": 0.25,
        "patient reported": 0.55, "quality of life": 0.55, "pro": 0.55,
        "safety": 0.60, "tolerability": 0.55,
        "pharmacokinetic": 0.25, "pk": 0.25, "auc": 0.25,
        "efficacy": 0.45,
        "composite": 0.55,  # composite endpoints harder to hit
    }

    fail_prob = 0.45  # default
    matched_endpoint = None
    for ep_key, risk in sorted(ENDPOINT_RISK.items(), key=lambda x: -len(x[0])):
        if ep_key in endpoint_type or ep_key in summary:
            fail_prob = risk
            matched_endpoint = ep_key
            break

    # Adjust for surrogate vs clinical endpoints
    is_surrogate = any(w in summary for w in ["biomarker", "surrogate", "hba1c", "ldl", "blood pressure", "tumor size"])
    if is_surrogate:
        fail_prob -= 0.10  # surrogate endpoints easier to hit
        matched_endpoint = (matched_endpoint or "surrogate") + " (surrogate)"

    # Adjust for composite endpoints
    if "composite" in summary or "co-primary" in summary:
        fail_prob += 0.08  # harder to hit all components
        matched_endpoint = (matched_endpoint or "") + " (composite)"

    # Adjust for endpoint complexity (num_secondary_endpoints is a top-3 V10 feature)
    num_sec = int(trial.get("num_secondary_endpoints", 0) or 0)
    sec_detail = ""
    if num_sec > 40:
        fail_prob += 0.10  # highly complex trial
        sec_detail = f" | {num_sec} secondary endpoints (very complex)"
    elif num_sec > 20:
        fail_prob += 0.05  # moderately complex
        sec_detail = f" | {num_sec} secondary endpoints"
    elif num_sec > 0:
        sec_detail = f" | {num_sec} secondary endpoints"

    detail = f"Endpoint: {matched_endpoint or 'unknown'} — {round(fail_prob*100)}% failure{sec_detail}"
    return {"prob": round(max(0.1, min(0.9, fail_prob)), 4), "detail": detail}


# ═══════════════════════════════════════════════════════════════
# SIGNAL 8: Historical Drug/Sponsor Factors
# ═══════════════════════════════════════════════════════════════
def compute_drug_history_signal(trial: dict) -> dict:
    """Assess risk based on drug's prior trial history and sponsor's condition-specific track record."""
    try:
        from models.ticker_map import get_connection
        drug_name = str(trial.get("intervention", "") or "")
        sponsor = str(trial.get("sponsor", "") or "")
        condition = str(trial.get("condition", "") or "")
        phase = str(trial.get("phase", "") or "")

        if not drug_name or len(drug_name) < 3:
            return None

        conn = get_connection()
        cur = conn.cursor()

        factors = []
        fail_adjustments = []

        # 1. Drug's prior trial results
        drug_kw = drug_name.split(";")[0].split(",")[0].strip()
        if len(drug_kw) > 3:
            cur.execute("""SELECT status, COUNT(*) as n FROM trials
                WHERE intervention ILIKE %s GROUP BY status""", (f"%{drug_kw[:20]}%",))
            drug_counts = {r["status"]: r["n"] for r in cur.fetchall()}
            drug_total = sum(drug_counts.values())
            drug_completed = drug_counts.get("COMPLETED", 0) + drug_counts.get("Completed", 0)
            drug_terminated = drug_counts.get("TERMINATED", 0) + drug_counts.get("Terminated", 0)

            if drug_total >= 2:
                drug_fail_rate = drug_terminated / max(drug_completed + drug_terminated, 1)
                factors.append(f"Drug has {drug_total} prior trials ({drug_completed} completed, {drug_terminated} terminated)")
                fail_adjustments.append(drug_fail_rate)

        # 2. Sponsor's track record in THIS condition
        if sponsor and condition:
            sp_kw = sponsor.split(",")[0].split()[0]
            cond_kw = [w for w in condition.split() if len(w) > 3][:1]
            if sp_kw and cond_kw:
                cur.execute("""SELECT status, COUNT(*) as n FROM trials
                    WHERE sponsor ILIKE %s AND condition ILIKE %s GROUP BY status""",
                    (f"%{sp_kw}%", f"%{cond_kw[0]}%"))
                sp_cond = {r["status"]: r["n"] for r in cur.fetchall()}
                sp_cond_total = sum(sp_cond.values())
                if sp_cond_total >= 2:
                    sp_completed = sp_cond.get("COMPLETED", 0) + sp_cond.get("Completed", 0)
                    sp_terminated = sp_cond.get("TERMINATED", 0) + sp_cond.get("Terminated", 0)
                    sp_fail_rate = sp_terminated / max(sp_completed + sp_terminated, 1)
                    factors.append(f"Sponsor has {sp_cond_total} trials in this condition ({sp_completed} completed)")
                    fail_adjustments.append(sp_fail_rate)

        # 3. Phase progression check (did prior phase succeed?)
        phase_num = 4 if "4" in phase else 3 if "3" in phase else 2 if "2" in phase else 1
        if phase_num >= 2 and drug_kw:
            prior_phase = f"Phase {phase_num - 1}"
            cur.execute("""SELECT status FROM trials
                WHERE intervention ILIKE %s AND phase ILIKE %s
                AND status IN ('COMPLETED','Completed') LIMIT 1""",
                (f"%{drug_kw[:20]}%", f"%{prior_phase}%"))
            has_prior = cur.fetchone()
            if has_prior:
                factors.append(f"Prior {prior_phase} completed — positive progression signal")
                fail_adjustments.append(0.30)  # prior success lowers risk
            else:
                factors.append(f"No completed {prior_phase} found — may lack prior validation")

        cur.close()
        _release(conn)

        if not fail_adjustments:
            return None

        avg_fail = sum(fail_adjustments) / len(fail_adjustments)
        detail = " | ".join(factors[:2]) if factors else "Drug/sponsor history"
        return {"prob": round(max(0.1, min(0.9, avg_fail)), 4), "detail": detail}
    except Exception as e:
        logger.debug("Drug history signal failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# SIGNAL 9: Market & Regulatory Factors
# ═══════════════════════════════════════════════════════════════
def compute_regulatory_signal(trial: dict) -> dict:
    """Assess risk based on regulatory pathway and market factors."""
    summary = str(trial.get("brief_summary", "") or "").lower()
    condition = str(trial.get("condition", "") or "").lower()
    phase = str(trial.get("phase", "") or "")

    factors = []
    fail_prob = 0.45  # default

    # Fast track / breakthrough / accelerated approval signals
    regulatory_boosts = {
        "breakthrough therapy": -0.15,
        "fast track": -0.10,
        "accelerated approval": -0.12,
        "priority review": -0.08,
        "orphan drug": -0.10,
        "rare pediatric": -0.12,
    }
    for designation, adjustment in regulatory_boosts.items():
        if designation in summary:
            fail_prob += adjustment
            factors.append(f"{designation.title()} designation ({adjustment*100:+.0f}%)")

    # Unmet need (conditions with few approved treatments)
    HIGH_UNMET_NEED = ["alzheimer", "als", "glioblastoma", "pancreatic cancer", "nash",
                       "huntington", "scleroderma", "pulmonary fibrosis", "sickle cell"]
    LOW_UNMET_NEED = ["hypertension", "diabetes type 2", "depression", "asthma", "pain",
                      "cholesterol", "acid reflux", "erectile dysfunction"]

    for cond in HIGH_UNMET_NEED:
        if cond in condition:
            fail_prob -= 0.05  # high unmet need = regulatory support, but also hard biology
            factors.append(f"High unmet need condition ({cond})")
            break

    for cond in LOW_UNMET_NEED:
        if cond in condition:
            fail_prob += 0.05  # crowded market, high bar for approval
            factors.append(f"Established treatment area ({cond}) — high approval bar")
            break

    # Phase-specific regulatory risk
    phase_risk = {"Phase 1": 0.65, "Phase 2": 0.50, "Phase 3": 0.35, "Phase 4": 0.20}
    for p, risk in phase_risk.items():
        if p.replace("Phase ", "") in phase:
            fail_prob = (fail_prob + risk) / 2  # blend with phase base rate
            factors.append(f"{p}: {round(risk*100)}% historical phase failure rate")
            break

    # ── New V10 adjustments from AACT eligibility/duration data ──
    # Healthy volunteer studies (typically Phase 1 PK/safety) have lower risk
    accepts_healthy = trial.get("accepts_healthy") or (
        str(trial.get("healthy_volunteers", "") or "").lower() in ("true", "yes", "t"))
    if accepts_healthy:
        fail_prob -= 0.08
        factors.append("Healthy volunteers accepted (-8%)")

    # Gender-restricted trials are more targeted
    gender = str(trial.get("gender", trial.get("sex", "")) or "").upper()
    if gender in ("MALE", "FEMALE"):
        fail_prob -= 0.03
        factors.append(f"{gender.title()}-only trial — targeted (-3%)")

    # Long studies have higher dropout risk
    duration = float(trial.get("study_duration_months", 0) or 0)
    if duration > 48:
        fail_prob += 0.05
        factors.append(f"Long study ({duration:.0f} months) — higher dropout risk (+5%)")
    elif 0 < duration < 12:
        fail_prob -= 0.03
        factors.append(f"Short study ({duration:.0f} months) — lower variability (-3%)")

    # Very restrictive eligibility may cause enrollment issues
    elig_text = str(trial.get("eligibility_criteria", "") or "")
    elig_words = len(elig_text.split())
    if elig_words > 500:
        fail_prob += 0.03
        factors.append(f"Complex eligibility ({elig_words} words) — enrollment risk (+3%)")

    # Multi-country trials show regulatory commitment
    num_countries = int(trial.get("num_countries", 0) or 0)
    if num_countries > 10:
        fail_prob -= 0.03
        factors.append(f"Global trial ({num_countries} countries) — strong commitment (-3%)")

    if not factors:
        factors.append("Standard regulatory pathway")

    detail = " | ".join(factors[:3])
    return {"prob": round(max(0.1, min(0.9, fail_prob)), 4), "detail": detail}


def compute_ensemble(trial: dict) -> dict:
    """Compute all 9 signals and combine with weighted average."""
    # Enrich trial with AACT design/eligibility data for V10 features
    trial = _enrich_trial_from_aact(trial)

    signals = []
    total_weight = 0

    signal_funcs = [
        ("XGBoost Model", "xgb", compute_xgb_signal),
        ("Condition Risk", "condition", compute_condition_signal),
        ("Clinical Endpoint", "endpoint", compute_endpoint_signal),
        ("Drug Target", "target", compute_target_signal),
        ("Sponsor Record", "sponsor", compute_sponsor_signal),
        ("Drug History", "drug_history", compute_drug_history_signal),
        ("Design Quality", "design", compute_design_signal),
        ("Regulatory Path", "regulatory", compute_regulatory_signal),
        ("Similar Trials", "similar", compute_similar_signal),
    ]

    for name, key, func in signal_funcs:
        try:
            result = func(trial)
            weight = SIGNAL_WEIGHTS[key]
            if result is not None:
                signals.append({
                    "name": name,
                    "prob": result["prob"],
                    "weight": weight,
                    "detail": result.get("detail", ""),
                    "available": True,
                    "drivers": result.get("drivers", []),
                    "feature_impacts": result.get("feature_impacts", []),
                })
                total_weight += weight
            else:
                signals.append({"name": name, "prob": None, "weight": weight, "detail": "Not available", "available": False})
        except Exception as e:
            logger.debug("Signal %s failed: %s", name, e)
            signals.append({"name": name, "prob": None, "weight": SIGNAL_WEIGHTS[key], "detail": f"Error: {e}", "available": False})

    # Compute weighted average (only available signals)
    if total_weight > 0:
        weighted_sum = sum(s["prob"] * s["weight"] for s in signals if s["available"])
        final_prob = weighted_sum / total_weight
    else:
        final_prob = 0.50  # default

    # Normalize contributions
    for s in signals:
        if s["available"]:
            s["contribution"] = round(s["prob"] * s["weight"] / max(total_weight, 0.01) * 100, 1)
        else:
            s["contribution"] = 0

    outcome = "failure" if final_prob >= 0.5 else "success"
    confidence = "high" if abs(final_prob - 0.5) > 0.25 else ("medium" if abs(final_prob - 0.5) > 0.10 else "low")

    # Get drivers and per-trial feature impacts from XGBoost signal
    xgb_drivers = next((s.get("drivers", []) for s in signals if s["name"] == "XGBoost Model" and s["available"]), [])
    xgb_impacts = next((s.get("feature_impacts", []) for s in signals if s["name"] == "XGBoost Model" and s["available"]), [])

    # ═══ DUAL MODEL PREDICTIONS (Optimist vs Pessimist) ═══
    dual = _compute_dual_prediction(trial)

    return {
        "outcome": outcome,
        "probability": round(final_prob, 4),
        "confidence": confidence,
        "source": "ensemble_v10",
        "signals": signals,
        "drivers": xgb_drivers,
        "feature_impacts": xgb_impacts,
        "dual": dual,
    }


def _load_dual():
    global _optimist, _pessimist, _opt_encoders, _pes_encoders, _opt_feats, _pes_feats, _dual_loaded
    if _dual_loaded:
        return
    _dual_loaded = True
    try:
        with open(os.path.join(DUAL_DIR, "optimist_model.pkl"), "rb") as f:
            d = pickle.load(f)
            _optimist = d["model"]
            _opt_encoders = d["encoders"]
            _opt_feats = d["feat_cols"]
        with open(os.path.join(DUAL_DIR, "pessimist_model.pkl"), "rb") as f:
            d = pickle.load(f)
            _pessimist = d["model"]
            _pes_encoders = d["encoders"]
            _pes_feats = d["feat_cols"]
        logger.info("Dual models (optimist + pessimist) loaded")
    except Exception as e:
        logger.warning("Failed to load dual models: %s", e)


def _encode_for_dual(trial, feat_cols, encoders):
    """Encode a trial for one of the dual models (V2 expanded features)."""
    summary = str(trial.get("brief_summary", "") or "").lower()
    condition = str(trial.get("condition", "") or "")
    phase = str(trial.get("phase", "") or "")
    enrollment = 0
    try: enrollment = float(trial.get("enrollment") or 0)
    except: pass

    phase_num = 4 if "4" in phase else 3 if "3" in phase else 2 if "2" in phase else 1 if "1" in phase else 0

    # Parse new AACT-sourced features from trial dict
    allocation = str(trial.get("allocation", "") or "UNKNOWN").upper()
    intervention_model = str(trial.get("intervention_model", "") or "UNKNOWN").upper()
    primary_purpose = str(trial.get("primary_purpose", "") or "UNKNOWN").upper()
    masking_level = _compute_masking_level(trial)

    gender = str(trial.get("gender", trial.get("sex", "")) or "ALL").upper()
    min_age_raw = trial.get("minimum_age", trial.get("min_age"))
    max_age_raw = trial.get("maximum_age", trial.get("max_age"))
    min_age = _parse_age_years(min_age_raw) if not isinstance(min_age_raw, (int, float)) else float(min_age_raw or 18)
    max_age = _parse_age_years(max_age_raw) if not isinstance(max_age_raw, (int, float)) else float(max_age_raw or 99)
    if math.isnan(min_age): min_age = 18.0
    if math.isnan(max_age): max_age = 99.0

    num_arms = int(trial.get("number_of_arms", trial.get("num_arms", 2)) or 2)
    num_sec = int(trial.get("num_secondary_endpoints", 0) or 0)
    accepts_healthy = 1 if str(trial.get("healthy_volunteers", "") or "").lower() in ("true", "yes", "t") else 0
    is_multi = int(trial.get("is_multi_sponsor", 0) or 0)
    elig_text = str(trial.get("eligibility_criteria", "") or "")
    elig_complexity = len(elig_text.split())
    duration = float(trial.get("study_duration_months", 24) or 24)

    computed = {
        "phase_num": phase_num,
        "enrollment": enrollment,
        "enrollment_log": math.log1p(enrollment),
        "has_randomization": 1 if "random" in summary else 0,
        "has_blinding": 1 if "blind" in summary or "mask" in summary else 0,
        "has_placebo": 1 if "placebo" in summary else 0,
        "has_control": 1 if "control" in summary else 0,
        "design_quality": sum([1 for kw in ["random","blind","placebo","control"] if kw in summary]),
        "sponsor_type": str(trial.get("sponsor_type", "") or ""),
        "sponsor_name": str(trial.get("sponsor", "") or ""),
        "allocation": allocation,
        "intervention_model_cat": intervention_model,
        "masking_level": masking_level,
        "num_arms": num_arms,
        "is_multi_sponsor": is_multi,
        "accepts_healthy": accepts_healthy,
        "disease_group": str(trial.get("disease_group", "") or "other"),
        "condition_name": condition,
        "modality": str(trial.get("modality", trial.get("intervention_type", "")) or ""),
        "intervention_type": str(trial.get("intervention_type", "") or ""),
        "primary_purpose": primary_purpose,
        "is_oncology": 1 if any(w in condition.lower() for w in ["oncol", "cancer", "tumor"]) else 0,
        "is_rare": 1 if "rare" in condition.lower() else 0,
        "condition_words": len(condition.split()),
        "summary_length": len(str(trial.get("brief_summary", "") or "")),
        "gender_restriction": 1 if gender in ("MALE", "FEMALE") else 0,
        "min_age": min_age,
        "max_age": max_age,
        "num_secondary_endpoints": num_sec,
        "eligibility_complexity": elig_complexity,
        "study_duration_months": duration,
    }

    CATEGORICAL = {"sponsor_type", "sponsor_name", "disease_group", "condition_name",
                   "modality", "intervention_type", "allocation", "intervention_model_cat",
                   "primary_purpose"}
    vals = []
    for col in feat_cols:
        if col in CATEGORICAL:
            le = encoders.get(col)
            raw = str(computed.get(col, "unknown"))
            if le:
                try: vals.append(float(le.transform([raw])[0]))
                except: vals.append(0.0)
            else:
                vals.append(0.0)
        else:
            vals.append(float(computed.get(col, 0)))
    return np.array([vals])


def _compute_dual_prediction(trial):
    """Run both optimist and pessimist models independently."""
    _load_dual()
    if _optimist is None or _pessimist is None:
        return None

    try:
        X_opt = _encode_for_dual(trial, _opt_feats, _opt_encoders)
        X_pes = _encode_for_dual(trial, _pes_feats, _pes_encoders)

        opt_prob = float(_optimist.predict_proba(X_opt)[0][1])  # P(failure)
        pes_prob = float(_pessimist.predict_proba(X_pes)[0][1])  # P(failure)

        # Optimist: P(success) = 1 - P(failure), with success-biased interpretation
        opt_success = round((1 - opt_prob) * 100, 1)
        opt_failure = round(opt_prob * 100, 1)

        # Pessimist: P(failure) with failure-biased interpretation
        pes_success = round((1 - pes_prob) * 100, 1)
        pes_failure = round(pes_prob * 100, 1)

        # Agreement analysis
        opt_says = "success" if opt_prob < 0.5 else "failure"
        pes_says = "success" if pes_prob < 0.5 else "failure"
        agree = opt_says == pes_says
        agreement_label = "AGREE" if agree else "DISAGREE"

        # When they disagree, the trial is genuinely uncertain
        if agree:
            combined_conf = "high" if abs(opt_prob - 0.5) > 0.2 and abs(pes_prob - 0.5) > 0.2 else "medium"
        else:
            combined_conf = "low"  # disagreement = uncertainty

        return {
            "optimist": {
                "success_pct": opt_success,
                "failure_pct": opt_failure,
                "prediction": opt_says,
                "confidence": "high" if abs(opt_prob - 0.5) > 0.25 else "medium" if abs(opt_prob - 0.5) > 0.1 else "low",
                "bias": "Starts optimistic and looks for red flags in trial design and sponsor track record",
                "features": "Evaluates: randomization, blinding, enrollment size, sponsor experience, trial design",
            },
            "pessimist": {
                "success_pct": pes_success,
                "failure_pct": pes_failure,
                "prediction": pes_says,
                "confidence": "high" if abs(pes_prob - 0.5) > 0.25 else "medium" if abs(pes_prob - 0.5) > 0.1 else "low",
                "bias": "Starts skeptical and looks for strengths in the condition, drug class, and historical data",
                "features": "Evaluates: disease difficulty, drug mechanism, competitor outcomes, regulatory pathway",
            },
            "agreement": agreement_label,
            "combined_confidence": combined_conf,
        }
    except Exception as e:
        logger.warning("Dual prediction failed: %s", e)
        return None
