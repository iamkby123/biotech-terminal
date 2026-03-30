"""
Production orchestrator: routes prediction, RAG, and analysis.
Does NOT retrain models. Uses existing artifacts only.
"""
import os, sys, json, time, logging, re
from pathlib import Path

logger = logging.getLogger(__name__)

# Add scripts dir to path for hybrid_production and query_rag
_SCRIPTS = str(Path(__file__).resolve().parent.parent.parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

_initialized = False
_rf_model = None
_rf_feat_cols = None
_rf_enc_feat_cols = None
_rf_label_encs = None

# Fine-tuned Qwen V5 state
_qwen_model = None
_qwen_tokenizer = None
_qwen_loaded = False

# v5 known outcomes — loaded once at startup, never re-read from disk
_known_outcomes: dict = {}
_known_outcomes_loaded = False

# Per-NCT prediction cache — avoids re-running RF + Qwen for the same trial
_prediction_cache: dict = {}

REASON_KEYS = ["lack_of_efficacy", "safety_issue", "trial_design_issue",
               "funding_or_business", "regulatory", "manufacturing_issue",
               "sponsor_track_record", "insider_signal", "unknown"]

QWEN_PREDICT_SYSTEM = (
    "You are a clinical trial outcome predictor. "
    "Given pre-outcome trial design features, predict whether the trial will succeed or fail. "
    "Your PRIMARY task is accurate outcome prediction. "
    "For failures, also provide failure reasons and a brief explanation. "
    "Output ONLY valid JSON."
)

QWEN_ANALYSIS_SYSTEM = (
    "You are a clinical trial analyst. A statistical model has predicted an outcome based purely "
    "on pre-trial DESIGN features (phase, enrollment, randomization, endpoint type, therapeutic area). "
    "The model does NOT have access to actual trial results. "
    "Your job: explain in 2-3 sentences why the listed design features support the predicted outcome. "
    "Focus only on the trial design characteristics provided. Do NOT describe or infer what actually "
    "happened to the trial. Do NOT reference efficacy results, termination reasons, or real-world outcomes. "
    "Do NOT contradict the predicted outcome. Do NOT start with 'The trial was predicted to'. "
    "Output ONLY valid JSON: {\"explanation\": \"your explanation here\"}"
)


def _init_rf():
    """Load RF V6 model (lightweight, no GPU)."""
    global _rf_model, _rf_feat_cols, _rf_enc_feat_cols, _rf_label_encs, _initialized
    if _initialized:
        return

    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    v6_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "v6" / "dataset_v6_trainable.csv"
    if not v6_path.exists():
        logger.warning("V6 dataset not found at %s, RF unavailable", v6_path)
        _initialized = True
        return

    df = pd.read_csv(v6_path)
    feat_cols = [c for c in df.columns if c.startswith("feat_") and c not in
                 ["feat_sponsor_name", "feat_has_real_outcome", "feat_study_type", "feat_is_interventional"]]

    cat_cols = [c for c in feat_cols if df[c].dtype == "object"]
    label_encs = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encs[col] = le

    enc_cols = [c + "_enc" if c in cat_cols else c for c in feat_cols]
    for c in enc_cols:
        if df[c].dtype == "bool":
            df[c] = df[c].astype(int)
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[enc_cols] = df[enc_cols].fillna(0)

    X = df[enc_cols].values.astype(np.float32)
    y = (df["final_label"] == "failure").astype(int).values

    rf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X, y)

    _rf_model = rf
    _rf_feat_cols = feat_cols
    _rf_enc_feat_cols = enc_cols
    _rf_label_encs = label_encs
    _initialized = True
    logger.info("RF V6 model loaded (%d features, %d training rows)", len(enc_cols), len(X))


def _init_qwen():
    """DISABLED — prediction now uses ensemble (GB+signals), not LLM.
    LLM was too slow to load and didn't improve predictions."""
    global _qwen_model, _qwen_tokenizer, _qwen_loaded
    _qwen_loaded = True
    logger.info("Qwen LLM loading SKIPPED — using ensemble prediction instead")
    return
    # Original loading code below (disabled):
    if _qwen_loaded:
        return
    _qwen_loaded = True
    try:
        from unsloth import FastLanguageModel
        base = Path(__file__).resolve().parent.parent.parent.parent / "outputs"
        # Prefer V8 (0.8B, fast) over V5 (9B, slow)
        v8_path = base / "qwen35_v8" / "final_adapter"
        v5_path = base / "qwen35_lora_v5" / "final_adapter"
        if v8_path.exists():
            adapter_path = str(v8_path)
            version = "V8 (0.8B)"
            seq_len = 1536
        elif v5_path.exists():
            adapter_path = str(v5_path)
            version = "V5 (9B)"
            seq_len = 1280
        else:
            logger.error("No Qwen adapter found at %s or %s", v8_path, v5_path)
            return
        logger.info("Loading fine-tuned Qwen %s from %s ...", version, adapter_path)
        t0 = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path, max_seq_length=seq_len, dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        _qwen_model = model
        _qwen_tokenizer = tokenizer
        logger.info("Qwen %s loaded in %.1fs", version, time.time() - t0)
    except Exception as e:
        logger.error("Failed to load Qwen: %s", e)


def _qwen_generate(system_prompt, user_text):
    """Generate from fine-tuned Qwen V5. Returns parsed JSON dict.
    Thinking mode disabled — outputs JSON directly without <think> chain-of-thought."""
    import torch
    if _qwen_model is None:
        return {"error": "Qwen V5 not loaded"}

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]

    # enable_thinking=False: skips the <think>...</think> block entirely.
    # Without this, Qwen3.5 spends ~200-400 tokens on chain-of-thought before JSON,
    # consuming the token budget and adding 10-30s of unnecessary GPU time.
    try:
        input_text = _qwen_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        # Older tokenizer versions don't support enable_thinking — fall back gracefully
        input_text = _qwen_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    inputs = _qwen_tokenizer(text=input_text, return_tensors="pt", padding=True).to(_qwen_model.device)

    with torch.no_grad():
        outputs = _qwen_model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=False)

    response = _qwen_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Strip any residual think blocks and extract JSON
    try:
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        return json.loads(match.group()) if match else {"raw": clean}
    except (json.JSONDecodeError, AttributeError):
        return {"raw": clean if clean else response[:500]}


def _qwen_generate_long(system_prompt, user_text, max_tokens=1024):
    """Generate longer text from Qwen (for analysis reports). Returns raw text."""
    import torch
    if _qwen_model is None:
        return ""

    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
    try:
        input_text = _qwen_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        input_text = _qwen_tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    inputs = _qwen_tokenizer(text=input_text, return_tensors="pt", padding=True).to(_qwen_model.device)
    with torch.no_grad():
        outputs = _qwen_model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3, do_sample=True)

    response = _qwen_tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
    return clean


def _format_trial_features(trial):
    """Format trial into the text prompt Qwen V5 was trained on."""
    def _get(key, default=""):
        return trial.get(key) or trial.get(f"feat_{key}") or trial.get("filled_fields", {}).get(key, default)

    phase = _get("phase", "")
    phase_num = 0
    if "4" in str(phase): phase_num = 4
    elif "3" in str(phase): phase_num = 3
    elif "2" in str(phase): phase_num = 2
    elif "1" in str(phase): phase_num = 1

    condition = _get("condition", "unknown")
    intervention = _get("intervention", "unknown")
    enrollment = trial.get("enrollment") or trial.get("feat_enrollment_size") or 0
    try:
        enrollment = int(enrollment)
    except (ValueError, TypeError):
        enrollment = 0

    bucket = "tiny" if enrollment < 50 else ("small" if enrollment < 200 else ("medium" if enrollment < 500 else "large"))
    design_text = (_get("study_design", "") or "").lower()
    area = trial.get("filled_fields", {}).get("condition_category", "other")

    lines = ["Predict the outcome of this clinical trial:\n"]
    lines.append(f"Phase: {phase_num}")
    lines.append(f"Therapeutic Area: {area}")
    lines.append(f"Indication: {area}")
    lines.append(f"Modality: {trial.get('filled_fields', {}).get('intervention_type', 'drug')}")
    lines.append(f"Enrollment: {enrollment} ({bucket})")
    lines.append(f"Primary Endpoint: {trial.get('primary_endpoint', 'efficacy') or 'efficacy'}")

    design = []
    if "random" in design_text: design.append("randomized")
    if "control" in design_text or "placebo" in str(intervention).lower(): design.append("controlled")
    if "placebo" in str(intervention).lower(): design.append("placebo-controlled")
    if "double" in design_text: design.append("double-blind")
    if design:
        lines.append(f"Design: {', '.join(design)}")

    chars = []
    from services.enrich import _kw, _ONCOLOGY, _RARE
    cond_text = str(condition)
    if _kw(cond_text, _ONCOLOGY): chars.append("oncology")
    if _kw(cond_text, _RARE): chars.append("rare disease")
    if chars:
        lines.append(f"Characteristics: {', '.join(chars)}")

    cond_count = len([c for c in str(condition).split(";") if c.strip()])
    int_count = len([i for i in str(intervention).split(";") if i.strip()])
    lines.append(f"Conditions: {cond_count}")
    lines.append(f"Interventions: {int_count}")

    return "\n".join(lines)


def classify_domain(trial):
    """Classify trial as oncology or non_oncology."""
    onc_kw = {"cancer","tumor","tumour","carcinoma","sarcoma","leukemia","lymphoma",
              "melanoma","glioma","myeloma","neoplasm","metasta","oncolog","blastoma"}
    text = " ".join([
        str(trial.get("condition", "")),
        str(trial.get("feat_therapeutic_area", "")),
        str(trial.get("feat_indication_category", "")),
    ]).lower()
    is_onc = any(k in text for k in onc_kw) or trial.get("feat_is_oncology") in [True, "True", 1]
    return "oncology" if is_onc else "non_oncology"


def _enrich_trial_features(trial):
    """Map raw DB trial fields to feat_* columns for the RF model."""
    from services.enrich import classify_condition, classify_intervention_type, _kw, _ONCOLOGY, _RARE

    condition = str(trial.get("condition", "") or "")
    intervention = str(trial.get("intervention", "") or "")
    phase = str(trial.get("phase", "") or "")
    design = str(trial.get("study_design", "") or "").lower()
    enrollment = trial.get("enrollment") or 0
    try:
        enrollment = int(enrollment)
    except (ValueError, TypeError):
        enrollment = 0

    phase_num = 0
    if "4" in phase: phase_num = 4
    elif "3" in phase: phase_num = 3
    elif "2" in phase: phase_num = 2
    elif "1" in phase: phase_num = 1

    area = classify_condition(condition)
    is_onc = _kw(condition, _ONCOLOGY)

    enriched = dict(trial)
    enriched.setdefault("feat_therapeutic_area", area)
    enriched.setdefault("feat_phase_num", phase_num)
    enriched.setdefault("feat_is_phase2", phase_num == 2)
    enriched.setdefault("feat_is_phase3", phase_num == 3)
    enriched.setdefault("feat_study_type", "INTERVENTIONAL")
    enriched.setdefault("feat_is_interventional", True)
    enriched.setdefault("feat_modality", classify_intervention_type(intervention) if intervention else "drug")
    enriched.setdefault("feat_is_combination", ";" in intervention and "placebo" not in intervention.lower())
    enriched.setdefault("feat_has_biomarker_selection", _kw(condition + " " + str(trial.get("title", "")), {"biomarker", "mutation", "her2", "egfr", "braf"}))
    enriched.setdefault("feat_indication_category", area)
    enriched.setdefault("feat_is_oncology", is_onc)
    enriched.setdefault("feat_is_rare_disease", _kw(condition, _RARE))
    enriched.setdefault("feat_enrollment_size", enrollment)
    enriched.setdefault("feat_enrollment_bucket", "tiny" if enrollment < 50 else ("small" if enrollment < 200 else ("medium" if enrollment < 500 else "large")))
    enriched.setdefault("feat_has_randomization", "random" in design)
    enriched.setdefault("feat_has_control_group", "random" in design or "placebo" in intervention.lower())
    enriched.setdefault("feat_has_placebo", "placebo" in intervention.lower())
    enriched.setdefault("feat_is_double_blind", "double" in design)
    enriched.setdefault("feat_has_active_comparator", "active" in design)
    enriched.setdefault("feat_is_orphan_indication", _kw(condition, _RARE))
    enriched.setdefault("feat_primary_endpoint_type", "efficacy")
    enriched.setdefault("feat_has_efficacy_endpoint", True)
    enriched.setdefault("feat_is_safety_trial", False)
    enriched.setdefault("feat_num_conditions", len([c for c in condition.split(";") if c.strip()]))
    enriched.setdefault("feat_num_interventions", len([i for i in intervention.split(";") if i.strip()]))
    enriched.setdefault("feat_is_early_phase", phase_num <= 1)
    enriched.setdefault("feat_sponsor_name", trial.get("sponsor", ""))
    enriched.setdefault("feat_has_real_outcome", 0)
    return enriched


def compute_trial_analytics(trial):
    """Compute DB-based analytics for a trial. No LLM, pure stats."""
    from models.ticker_map import get_connection

    condition = str(trial.get("condition", "") or "")
    phase = str(trial.get("phase", "") or "")
    sponsor = str(trial.get("sponsor", "") or "")
    design = str(trial.get("study_design", "") or "").lower()
    intervention = str(trial.get("intervention", "") or "").lower()
    enrollment = 0
    try:
        enrollment = int(trial.get("enrollment") or 0)
    except (ValueError, TypeError):
        pass

    # Extract first meaningful condition keyword
    cond_keyword = ""
    for w in condition.replace(";", " ").replace(",", " ").split():
        if len(w) > 4 and w.lower() not in ("with", "patients", "study", "phase", "trial", "disease"):
            cond_keyword = w
            break

    # Phase number
    phase_num = ""
    if "3" in phase: phase_num = "Phase 3"
    elif "2" in phase: phase_num = "Phase 2"
    elif "4" in phase: phase_num = "Phase 4"
    elif "1" in phase: phase_num = "Phase 1"

    analytics = {}

    try:
        conn = get_connection()
        cur = conn.cursor()

        # 1. Historical base rates for condition + phase
        if cond_keyword and phase_num:
            cur.execute("""SELECT status, COUNT(*) as n FROM trials
                WHERE condition ILIKE %s AND phase ILIKE %s
                GROUP BY status""", (f"%{cond_keyword}%", f"%{phase_num}%"))
            status_counts = {r["status"]: r["n"] for r in cur.fetchall()}
            completed = status_counts.get("COMPLETED", 0)
            terminated = status_counts.get("TERMINATED", 0)
            total = completed + terminated
            rate = round(completed / total * 100) if total > 0 else 0
            label = "HIGH" if rate > 60 else ("MEDIUM" if rate > 40 else "LOW")
            analytics["base_rates"] = {
                "condition": cond_keyword,
                "phase": phase_num,
                "completed": completed,
                "terminated": terminated,
                "total": total,
                "rate": rate,
                "label": label,
            }

        # 2. Design quality — use AACT structural data when available
        allocation = str(trial.get("allocation", "") or "").upper()
        masking_level = int(trial.get("masking_level", 0) or 0)
        masking_str = str(trial.get("masking", "") or "").upper()
        if not masking_level and masking_str:
            if "QUADRUPLE" in masking_str: masking_level = 4
            elif "TRIPLE" in masking_str: masking_level = 3
            elif "DOUBLE" in masking_str: masking_level = 2
            elif "SINGLE" in masking_str: masking_level = 1

        checks = [
            {"name": "Randomized", "pass": allocation == "RANDOMIZED" or "random" in design},
            {"name": "Double-blind", "pass": masking_level >= 2 or "double" in design},
            {"name": "Placebo-controlled", "pass": "placebo" in intervention or "placebo" in design},
            {"name": "Control group", "pass": allocation == "RANDOMIZED" or "random" in design or "placebo" in intervention},
            {"name": "Enrollment > 50", "pass": enrollment > 50},
            {"name": "Efficacy endpoint", "pass": True},
        ]
        score = sum(1 for c in checks if c["pass"])

        # Build trial characteristics from AACT enrichment
        int_model = str(trial.get("intervention_model", "") or "")
        purpose = str(trial.get("primary_purpose", "") or "")
        num_arms = int(trial.get("number_of_arms", trial.get("num_arms", 0)) or 0)
        gender = str(trial.get("gender", trial.get("sex", "")) or "")
        min_age = str(trial.get("minimum_age", trial.get("min_age", "")) or "")
        max_age = str(trial.get("maximum_age", trial.get("max_age", "")) or "")
        duration = trial.get("study_duration_months")
        num_countries = int(trial.get("num_countries", 0) or 0)
        num_sec_ep = int(trial.get("num_secondary_endpoints", 0) or 0)

        masking_labels = {0: "Open Label", 1: "Single Blind", 2: "Double Blind",
                          3: "Triple Blind", 4: "Quadruple Blind"}

        analytics["design_quality"] = {
            "score": score,
            "total": len(checks),
            "checks": checks,
        }

        # Trial characteristics from AACT
        characteristics = {}
        if allocation: characteristics["Allocation"] = allocation.title()
        if masking_level >= 0: characteristics["Masking"] = masking_labels.get(masking_level, f"Level {masking_level}")
        if int_model: characteristics["Intervention Model"] = int_model.title()
        if purpose: characteristics["Primary Purpose"] = purpose.title()
        if num_arms: characteristics["Number of Arms"] = num_arms
        if gender and gender.upper() not in ("", "ALL"): characteristics["Gender"] = gender.title()
        if min_age: characteristics["Age Range"] = f"{min_age}" + (f" – {max_age}" if max_age else "+")
        if duration: characteristics["Planned Duration"] = f"{duration:.0f} months" if isinstance(duration, float) else f"{duration} months"
        if num_countries > 0: characteristics["Countries"] = num_countries
        if num_sec_ep > 0: characteristics["Secondary Endpoints"] = num_sec_ep
        analytics["trial_characteristics"] = characteristics

        # 3. Sponsor track record
        if sponsor:
            sponsor_first = sponsor.split(",")[0].strip()
            # Use first significant word for matching
            sponsor_kw = sponsor_first.split()[0] if sponsor_first else ""
            if len(sponsor_kw) > 2:
                cur.execute("""SELECT status, COUNT(*) as n FROM trials
                    WHERE sponsor ILIKE %s GROUP BY status""", (f"%{sponsor_kw}%",))
                sp_counts = {r["status"]: r["n"] for r in cur.fetchall()}
                sp_completed = sp_counts.get("COMPLETED", 0)
                sp_terminated = sp_counts.get("TERMINATED", 0)
                sp_total = sum(sp_counts.values())
                sp_rate = round(sp_completed / (sp_completed + sp_terminated) * 100) if (sp_completed + sp_terminated) > 0 else 0
                label = "EXPERIENCED" if sp_total > 10 else ("MODERATE" if sp_total > 3 else "NEW")
                analytics["sponsor_record"] = {
                    "name": sponsor_first[:30],
                    "total": sp_total,
                    "completed": sp_completed,
                    "terminated": sp_terminated,
                    "rate": sp_rate,
                    "label": label,
                }

        # 4. Competitive landscape
        if cond_keyword:
            cur.execute("""SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'COMPLETED') as completed,
                COUNT(*) FILTER (WHERE status = 'TERMINATED') as terminated,
                COUNT(*) FILTER (WHERE status IN ('RECRUITING','ACTIVE_NOT_RECRUITING')) as active
                FROM trials WHERE condition ILIKE %s""", (f"%{cond_keyword}%",))
            r = cur.fetchone()
            total = r["total"]
            label = "HIGH" if total > 100 else ("MEDIUM" if total > 20 else "LOW")
            analytics["competition"] = {
                "condition": cond_keyword,
                "total": total,
                "completed": r["completed"],
                "terminated": r["terminated"],
                "active": r["active"],
                "label": label,
            }

        # 5. Drug class / modality success rates
        if intervention:
            # Detect modality from intervention text
            modality = "drug"
            int_lower = intervention.lower()
            if any(w in int_lower for w in ["antibod", "mab", "-mab", "biologic"]):
                modality = "biologic/antibody"
            elif any(w in int_lower for w in ["mrna", "rna", "vaccine"]):
                modality = "mRNA/vaccine"
            elif any(w in int_lower for w in ["gene therap", "aav", "crispr", "editing"]):
                modality = "gene therapy"
            elif any(w in int_lower for w in ["cell therap", "car-t", "car t"]):
                modality = "cell therapy"
            elif any(w in int_lower for w in ["device", "implant", "stent"]):
                modality = "device"

            if cond_keyword:
                cur.execute("""SELECT status, COUNT(*) as n FROM trials
                    WHERE condition ILIKE %s AND intervention ILIKE %s
                    GROUP BY status""",
                    (f"%{cond_keyword}%", f"%{intervention.split()[0] if intervention.split() else intervention}%"))
                mc = {r["status"]: r["n"] for r in cur.fetchall()}
                mc_comp = mc.get("COMPLETED", 0) + mc.get("Completed", 0)
                mc_term = mc.get("TERMINATED", 0) + mc.get("Terminated", 0)
                mc_total = mc_comp + mc_term
                analytics["drug_class"] = {
                    "modality": modality,
                    "intervention": intervention.split(",")[0][:40] if intervention else "",
                    "completed": mc_comp,
                    "terminated": mc_term,
                    "total": mc_total,
                    "success_rate": round(mc_comp / mc_total * 100) if mc_total > 0 else None,
                }

        # 6. Enrollment comparison (percentile vs similar trials)
        if enrollment > 0 and cond_keyword and phase_num:
            cur.execute("""SELECT enrollment FROM trials
                WHERE condition ILIKE %s AND phase ILIKE %s
                AND enrollment IS NOT NULL AND enrollment > 0
                ORDER BY enrollment""",
                (f"%{cond_keyword}%", f"%{phase_num}%"))
            all_enrollments = [r["enrollment"] for r in cur.fetchall()]
            if all_enrollments:
                import statistics
                median_e = int(statistics.median(all_enrollments))
                below = sum(1 for e in all_enrollments if e <= enrollment)
                percentile = round(below / len(all_enrollments) * 100)
                analytics["enrollment_comparison"] = {
                    "trial_enrollment": enrollment,
                    "median_enrollment": median_e,
                    "min_enrollment": min(all_enrollments),
                    "max_enrollment": max(all_enrollments),
                    "percentile": percentile,
                    "sample_size": len(all_enrollments),
                }

        # 7. Phase transition rates
        if cond_keyword and phase_num:
            next_phase = {"Phase 1": "Phase 2", "Phase 2": "Phase 3", "Phase 3": "Phase 4"}.get(phase_num)
            if next_phase:
                cur.execute("""SELECT COUNT(DISTINCT nct_id) as n FROM trials
                    WHERE condition ILIKE %s AND phase ILIKE %s""",
                    (f"%{cond_keyword}%", f"%{next_phase}%"))
                next_count = cur.fetchone()["n"]
                cur.execute("""SELECT COUNT(DISTINCT nct_id) as n FROM trials
                    WHERE condition ILIKE %s AND phase ILIKE %s""",
                    (f"%{cond_keyword}%", f"%{phase_num}%"))
                this_count = cur.fetchone()["n"]
                if this_count > 0:
                    transition_rate = round(next_count / this_count * 100)
                    analytics["phase_transition"] = {
                        "current_phase": phase_num,
                        "next_phase": next_phase,
                        "current_count": this_count,
                        "next_count": next_count,
                        "transition_rate": min(transition_rate, 100),
                    }

        # 8. Trial timeline
        start = trial.get("start_date") or trial.get("start_date_struct")
        completion = trial.get("primary_completion_date") or trial.get("completion_date")
        if start and completion:
            try:
                from datetime import datetime
                def _parse_date(d):
                    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%Y-%m-%dT%H:%M:%S"):
                        try: return datetime.strptime(str(d)[:10], fmt[:len(str(d)[:10])])
                        except: continue
                    return None
                sd = _parse_date(start)
                cd = _parse_date(completion)
                if sd and cd and cd > sd:
                    duration_months = round((cd - sd).days / 30.44)
                    # Typical durations by phase
                    typical = {"Phase 1": 18, "Phase 2": 30, "Phase 3": 36, "Phase 4": 24}.get(phase_num, 30)
                    analytics["timeline"] = {
                        "start_date": str(start)[:10],
                        "completion_date": str(completion)[:10],
                        "duration_months": duration_months,
                        "typical_months": typical,
                        "vs_typical": "faster" if duration_months < typical * 0.8 else ("slower" if duration_months > typical * 1.2 else "typical"),
                    }
            except Exception:
                pass

        cur.close()
        conn.close()
    except Exception as e:
        logger.warning("Trial analytics failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # RISK BREAKDOWN SCORES (1-10, deeply computed)
    # ═══════════════════════════════════════════════════════════════
    try:
        # Known high-failure conditions
        CONDITION_FAIL_RATES = {
            "alzheimer": 97, "nash": 85, "als": 95, "sepsis": 90, "huntington": 95,
            "parkinson": 85, "glioblastoma": 95, "pancreatic": 88, "scleroderma": 88,
            "lupus": 82, "schizophrenia": 75, "obesity": 70, "depression": 50,
        }
        cond_lower = condition.lower()
        known_fail_rate = None
        for ck, fr in CONDITION_FAIL_RATES.items():
            if ck in cond_lower:
                known_fail_rate = fr
                break

        # ── EFFICACY RISK ──
        base_rate = analytics.get("base_rates", {}).get("rate", 50)
        dc = analytics.get("drug_class", {})
        dc_rate = dc.get("success_rate") if dc.get("total", 0) > 2 else None

        efficacy_sub = []
        efficacy_risk = max(1, min(10, 10 - round(base_rate / 10)))

        if known_fail_rate:
            efficacy_risk = max(efficacy_risk, min(10, round(known_fail_rate / 10)))
            efficacy_sub.append(f"Known high-failure condition: {known_fail_rate}% historical failure rate")
        efficacy_sub.append(f"Phase+condition base rate: {base_rate}% success (n={analytics.get('base_rates',{}).get('total',0)})")
        if dc_rate is not None:
            efficacy_sub.append(f"Drug class success rate: {dc_rate}% (n={dc.get('total',0)})")
            if dc_rate < 30:
                efficacy_risk = max(efficacy_risk, 7)
        pt = analytics.get("phase_transition", {})
        if pt.get("transition_rate"):
            efficacy_sub.append(f"Phase advancement rate: {pt['transition_rate']}% ({pt.get('current_phase','')} → {pt.get('next_phase','')})")

        # ── SAFETY RISK ──
        phase_risk_map = {"Phase 1": 8, "Phase 2": 6, "Phase 3": 4, "Phase 4": 2}
        safety_risk = phase_risk_map.get(phase_num, 5)
        safety_sub = []
        safety_sub.append(f"{phase_num}: {'early phase — limited safety data, dose-finding' if safety_risk > 5 else 'later phase — larger safety database'}")
        if enrollment > 0:
            if enrollment < 50:
                safety_sub.append(f"Very small trial (n={enrollment}) — safety signals may be missed")
                safety_risk = max(safety_risk, 7)
            elif enrollment > 1000:
                safety_sub.append(f"Large trial (n={enrollment}) — good statistical power for safety")
                safety_risk = min(safety_risk, safety_risk - 1)
        if "random" in design:
            safety_sub.append("Randomized design helps isolate drug-related adverse events")
        else:
            safety_sub.append("Non-randomized — harder to attribute adverse events to drug")
            safety_risk = min(10, safety_risk + 1)

        # ── DESIGN RISK ──
        dq_data = analytics.get("design_quality", {})
        dq = dq_data.get("score", 3)
        design_risk = max(1, min(10, 10 - dq))
        design_sub = []
        checks = dq_data.get("checks", [])
        for c in checks:
            if c.get("pass"):
                design_sub.append(f"✓ {c['name']}")
            else:
                design_sub.append(f"✗ {c['name']} — increases bias risk")
        ec = analytics.get("enrollment_comparison", {})
        if ec.get("percentile") is not None:
            pctile = ec["percentile"]
            if pctile < 25:
                design_sub.append(f"Enrollment at {pctile}th percentile — below typical for this phase+condition")
                design_risk = min(10, design_risk + 1)
            elif pctile > 75:
                design_sub.append(f"Enrollment at {pctile}th percentile — well-powered study")

        # ── COMPETITION RISK ──
        comp = analytics.get("competition", {})
        comp_total = comp.get("total", 0)
        comp_active = comp.get("active", 0)
        comp_completed = comp.get("completed", 0)
        competition_risk = 8 if comp_total > 200 else 6 if comp_total > 50 else 4 if comp_total > 10 else 2
        competition_sub = []
        competition_sub.append(f"{comp_total} total trials in {cond_keyword}")
        if comp_active:
            competition_sub.append(f"{comp_active} currently active — crowded pipeline")
        if comp_completed:
            competition_sub.append(f"{comp_completed} completed — established treatment landscape")
        if competition_risk >= 7:
            competition_sub.append("Highly competitive: new entrant must show clear superiority")
        elif competition_risk <= 3:
            competition_sub.append("Low competition: potential first-mover advantage")

        # ── REGULATORY RISK ──
        sponsor_rec = analytics.get("sponsor_record", {})
        sp_label = sponsor_rec.get("label", "NEW")
        sp_total = sponsor_rec.get("total", 0)
        sp_completed = sponsor_rec.get("completed", 0)
        sp_terminated = sponsor_rec.get("terminated", 0)
        sp_rate = sponsor_rec.get("rate", 0)
        regulatory_risk = 3 if sp_label == "EXPERIENCED" else 5 if sp_label == "MODERATE" else 7
        regulatory_sub = []
        regulatory_sub.append(f"Sponsor: {sponsor[:25]} ({sp_label.lower()})")
        if sp_total > 0:
            regulatory_sub.append(f"Track record: {sp_total} trials, {sp_completed} completed, {sp_terminated} terminated ({sp_rate}% completion)")
        if sp_label == "NEW":
            regulatory_sub.append("New/small sponsor — limited regulatory experience, higher execution risk")
            regulatory_risk = max(regulatory_risk, 7)
        elif sp_label == "EXPERIENCED" and sp_rate > 80:
            regulatory_sub.append("Strong regulatory track record with high completion rate")

        # Phase-specific regulatory context
        if "1" in str(phase_num) and "2" not in str(phase_num):
            regulatory_sub.append("Phase 1: IND must be active, FDA safety monitoring")
        elif "3" in str(phase_num):
            regulatory_sub.append("Phase 3: pivotal trial, NDA/BLA submission path, potential for accelerated approval")

        # ── OVERALL RISK ──
        avg_risk = round((efficacy_risk + safety_risk + design_risk + competition_risk + regulatory_risk) / 5, 1)

        analytics["risk_scores"] = [
            {"label": "Efficacy", "score": efficacy_risk, "sub_factors": efficacy_sub,
             "explanation": efficacy_sub[0] if efficacy_sub else ""},
            {"label": "Safety", "score": safety_risk, "sub_factors": safety_sub,
             "explanation": safety_sub[0] if safety_sub else ""},
            {"label": "Design", "score": design_risk, "sub_factors": design_sub,
             "explanation": f"Design quality {dq}/6"},
            {"label": "Competition", "score": competition_risk, "sub_factors": competition_sub,
             "explanation": competition_sub[0] if competition_sub else ""},
            {"label": "Regulatory", "score": regulatory_risk, "sub_factors": regulatory_sub,
             "explanation": regulatory_sub[0] if regulatory_sub else ""},
        ]
        analytics["overall_risk"] = avg_risk
    except Exception as e:
        logger.warning("Risk score computation failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # DRUG MECHANISM ANALYSIS (from local PubMed files)
    # ═══════════════════════════════════════════════════════════════
    try:
        drug_name = str(trial.get("intervention", "") or "")
        drug_science = _load_drug_science(drug_name)
        if drug_science:
            mech_summary = {}
            if drug_science.get("mechanism"):
                mech_summary["mechanism"] = drug_science["mechanism"][0].get("abstract", "")[:500]
            if drug_science.get("targets"):
                mech_summary["targets"] = drug_science["targets"][0].get("abstract", "")[:400]
            if drug_science.get("pharmacokinetics"):
                mech_summary["pharmacokinetics"] = drug_science["pharmacokinetics"][0].get("abstract", "")[:400]
            if drug_science.get("safety"):
                mech_summary["safety_profile"] = drug_science["safety"][0].get("abstract", "")[:400]
            if drug_science.get("prior_results"):
                mech_summary["prior_results"] = [e.get("abstract", "")[:300] for e in drug_science["prior_results"][:2]]
            if mech_summary:
                analytics["drug_mechanism"] = {
                    "drug_name": drug_name.split(";")[0].strip(),
                    "data": mech_summary,
                    "source": "pubmed_local",
                }
    except Exception as e:
        logger.warning("Drug mechanism load failed: %s", e)

    # ═══════════════════════════════════════════════════════════════
    # CONFIDENCE CALIBRATION
    # ═══════════════════════════════════════════════════════════════
    try:
        # Compare this trial's profile to historical base rates
        base = analytics.get("base_rates", {})
        dc = analytics.get("drug_class", {})
        pt = analytics.get("phase_transition", {})
        ec = analytics.get("enrollment_comparison", {})

        calibration = {
            "phase_condition_rate": base.get("rate", None),
            "phase_condition_n": base.get("total", 0),
            "drug_class_rate": dc.get("success_rate", None),
            "drug_class_n": dc.get("total", 0),
            "enrollment_percentile": ec.get("percentile", None),
            "phase_advancement_rate": pt.get("transition_rate", None),
        }

        # Compute composite confidence
        rates = [v for k, v in calibration.items() if v is not None and k.endswith("_rate")]
        if rates:
            avg_rate = sum(rates) / len(rates)
            calibration["composite_historical_success"] = round(avg_rate, 1)
            calibration["confidence_level"] = "high" if len(rates) >= 3 else "medium" if len(rates) >= 2 else "low"
            calibration["data_points"] = len(rates)
        else:
            calibration["composite_historical_success"] = None
            calibration["confidence_level"] = "insufficient_data"
            calibration["data_points"] = 0

        analytics["calibration"] = calibration
    except Exception as e:
        logger.warning("Calibration computation failed: %s", e)

    return analytics


def predict_rf(trial):
    """Run RF V6 prediction on a trial."""
    import numpy as np
    _init_rf()
    if _rf_model is None:
        return {"outcome": "unknown", "confidence": "low", "probability": None, "source": "rf_v6", "error": "RF not loaded"}

    # Enrich raw trial with feat_* columns if missing
    trial_enriched = _enrich_trial_features(trial)

    # Build feature vector
    row = {}
    for col in _rf_feat_cols:
        val = trial_enriched.get(col, trial_enriched.get("filled_fields", {}).get(col))
        row[col] = val

    # Encode
    import pandas as pd
    df_row = pd.DataFrame([row])
    for col in [c for c in _rf_feat_cols if c in _rf_label_encs]:
        le = _rf_label_encs[col]
        val = str(df_row[col].iloc[0]) if col in df_row else "unknown"
        try:
            df_row[col + "_enc"] = le.transform([val])[0]
        except ValueError:
            df_row[col + "_enc"] = 0

    enc_cols = _rf_enc_feat_cols
    for c in enc_cols:
        if c not in df_row.columns:
            df_row[c] = 0
        if df_row[c].dtype == "bool":
            df_row[c] = df_row[c].astype(int)
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")
    df_row[enc_cols] = df_row[enc_cols].fillna(0)

    X = df_row[enc_cols].values.astype(np.float32)
    proba = _rf_model.predict_proba(X)[0]
    pred = "failure" if proba[1] > 0.5 else "success"
    conf = "high" if max(proba) > 0.7 else "medium"

    # Compute per-feature contributions using flip-perturbation:
    # For binary features: flip 0→1 or 1→0 to see true counterfactual impact.
    # For continuous features: perturb to 0 (or flip sign if already 0).
    drivers = []
    base_fail_prob = float(proba[1])
    feature_names_readable = {
        "feat_enrollment_size":           "Enrollment size",
        "feat_phase_num":                 "Trial phase",
        "feat_is_phase1":                 "Phase 1",
        "feat_is_phase2":                 "Phase 2",
        "feat_is_phase3":                 "Phase 3",
        "feat_is_phase4":                 "Phase 4",
        "feat_is_oncology":               "Oncology indication",
        "feat_is_rare_disease":           "Rare disease",
        "feat_is_infectious":             "Infectious disease",
        "feat_is_cardiovascular":         "Cardiovascular",
        "feat_is_metabolic":              "Metabolic disease",
        "feat_is_neurological":           "Neurological",
        "feat_has_randomization":         "Randomized design",
        "feat_has_control_group":         "Control group",
        "feat_has_placebo":               "Placebo controlled",
        "feat_is_double_blind":           "Double blind",
        "feat_has_active_comparator":     "Active comparator",
        "feat_is_combination":            "Combination therapy",
        "feat_has_biomarker_selection":   "Biomarker patient selection",
        "feat_is_orphan_indication":      "Orphan drug designation",
        "feat_has_efficacy_endpoint":     "Efficacy endpoint",
        "feat_is_safety_trial":           "Safety/tolerability trial",
        "feat_is_early_phase":            "Early phase (1/1-2)",
        "feat_has_surrogate_endpoint":    "Surrogate endpoint",
        "feat_has_os_endpoint":           "Overall survival endpoint",
        "feat_has_pfs_endpoint":          "PFS endpoint",
        "feat_has_response_endpoint":     "Response rate endpoint",
        "feat_has_qol_endpoint":          "Quality of life endpoint",
        "feat_is_multicenter":            "Multi-center trial",
        "feat_is_international":          "International trial",
        "feat_sponsor_is_industry":       "Industry sponsor",
        "feat_sponsor_is_academic":       "Academic sponsor",
        "feat_enrollment_bucket_enc":     "Enrollment tier",
        "feat_therapeutic_area_enc":      "Therapeutic area",
        "feat_modality_enc":              "Drug modality",
        "feat_indication_category_enc":   "Indication category",
        "feat_primary_endpoint_type_enc": "Primary endpoint type",
        "feat_num_conditions":            "Number of conditions",
        "feat_num_interventions":         "Number of interventions",
        "feat_num_arms":                  "Number of trial arms",
        "feat_has_results":               "Results already posted",
        "feat_duration_months":           "Trial duration (months)",
    }
    try:
        for i, col in enumerate(enc_cols):
            val = float(X[0, i])
            X_pert = X.copy()

            # Flip-perturbation: binary → flip; continuous → zero out
            is_binary = col not in ["feat_enrollment_size", "feat_phase_num",
                                    "feat_enrollment_bucket_enc", "feat_therapeutic_area_enc",
                                    "feat_modality_enc", "feat_indication_category_enc",
                                    "feat_primary_endpoint_type_enc",
                                    "feat_num_conditions", "feat_num_interventions",
                                    "feat_num_arms", "feat_duration_months"]
            if is_binary:
                X_pert[0, i] = 1.0 - val  # flip 0↔1
            else:
                X_pert[0, i] = 0.0

            pert_prob = _rf_model.predict_proba(X_pert)[0][1]
            # delta: positive = current value increases failure risk vs the counterfactual
            # For binary flip: delta>0 means having val=1 (vs 0) increases failure
            delta = base_fail_prob - pert_prob
            if is_binary:
                # If we flipped 0→1 and failure went up, it means val=1 is risky
                # Flip sign so delta still means "current value's contribution to failure"
                if val == 0.0:
                    delta = -delta

            if abs(delta) > 0.003:  # lower threshold — catch all meaningful contributors
                readable = feature_names_readable.get(
                    col, col.replace("feat_", "").replace("_enc", "").replace("_", " ").title()
                )
                # Format display value
                if col == "feat_enrollment_size":
                    display_val = f"{int(val)} pts"
                elif col == "feat_phase_num":
                    display_val = f"Phase {int(val)}" if val > 0 else ""
                elif col == "feat_duration_months":
                    display_val = f"{int(val)}mo" if val > 0 else ""
                elif col in ("feat_num_conditions", "feat_num_interventions", "feat_num_arms"):
                    display_val = str(int(val)) if val > 0 else ""
                elif is_binary:
                    display_val = "Yes" if val == 1.0 else "No"
                else:
                    display_val = ""
                drivers.append({
                    "feature": readable,
                    "value": display_val,
                    "impact": round(delta * 100, 1),
                    "direction": "risk" if delta > 0 else "safe",
                })
        # Sort by absolute impact, show top 12
        drivers.sort(key=lambda d: abs(d["impact"]), reverse=True)
        drivers = drivers[:12]
    except Exception as e:
        logger.warning("Feature contribution failed: %s", e)

    return {
        "outcome": pred,
        "confidence": conf,
        "probability": round(float(proba[1]), 4),
        "source": "rf_v6",
        "drivers": drivers,
    }


def _heuristic_reasons(prediction, trial):
    """Fallback heuristic reason flags when Qwen is unavailable."""
    reasons = {k: False for k in REASON_KEYS}
    if prediction.get("outcome") == "failure":
        reasons["lack_of_efficacy"] = True
        enrollment = trial.get("enrollment") or trial.get("feat_enrollment_size") or 0
        try:
            if int(enrollment) < 50:
                reasons["trial_design_issue"] = True
        except (ValueError, TypeError):
            pass
    return reasons


QWEN_PREDICT_FULL_SYSTEM = (
    "You are a clinical trial outcome analyst. Your default position is SUCCESS — "
    "every trial is presumed successful unless you find compelling evidence of failure.\n\n"
    "Examine the trial characteristics and actively search for reasons it might fail. "
    "If you find sufficient evidence, override the presumption to FAILURE.\n\n"
    "For each active failure reason, assign a weight (0.0-1.0, active weights sum to 1.0) "
    "and explain WHY that reason carries more or less weight than others.\n\n"
    "Failure reasons to evaluate:\n"
    "- lack_of_efficacy: weak mechanism, poor endpoint choice, underpowered\n"
    "- safety_issue: known toxicity, SAE history, drug class risks\n"
    "- trial_design_issue: enrollment too small, no randomization, poor controls\n"
    "- funding_or_business: small company, no revenue, high burn rate\n"
    "- regulatory: prior FDA holds, failed submissions in class\n"
    "- manufacturing_issue: CMC issues, supply chain, formulation problems\n"
    "- sponsor_track_record: sponsor's history of trial failures/terminations\n"
    "- insider_signal: unusual options activity, insider selling\n\n"
    "Output ONLY valid JSON:\n"
    '{"outcome":"success"|"failure",'
    '"confidence":"high"|"medium"|"low",'
    '"failure_probability":0.0-1.0,'
    '"presumption_overridden":true|false,'
    '"reasons":{"reason_name":true|false,...},'
    '"reason_weights":{"active_reason":0.0-1.0,...},'
    '"weight_explanation":"why each reason is weighted as it is",'
    '"drivers":[{"feature":"name","value":"val","impact":float,"direction":"risk"|"safe"},...up to 10],'
    '"explanation":"2-3 sentences"}'
    "\nFor SUCCESS: presumption_overridden=false, all reasons false, reason_weights={}. "
    "Output ONLY valid JSON, no other text."
)


    # _enrich_with_tools removed — replaced by web research in generate_analysis()


def predict_trial_for_ticker(trial):
    """Primary: GB V2 (verified p-value labels). Fallback: RF V6.
    LLM is used ONLY for generate_analysis(), not for prediction."""
    nct_id = trial.get("nct_id", "")

    # Return cached result if available
    if nct_id and nct_id in _prediction_cache:
        cached = _prediction_cache[nct_id]
        return cached["domain"], cached["prediction"]

    domain = classify_domain(trial)

    # ── Primary path: Gradient Boosting V2 (best accuracy) ──────────────────
    try:
        from services.predictor_v2 import predict_v2
        gb_result = predict_v2(trial)
        if gb_result and gb_result.get("outcome"):
            prediction = {
                "outcome": gb_result["outcome"],
                "confidence": gb_result["confidence"],
                "probability": gb_result["probability"],
                "source": gb_result.get("source", "ensemble_v10"),
                "domain": domain,
                "reasons": {k: False for k in REASON_KEYS},
                "reason_source": "ensemble",
                "drivers": gb_result.get("drivers", []),
                "signals": gb_result.get("signals", []),
                "dual": gb_result.get("dual"),
                "explanation": f"Ensemble predicts {gb_result['outcome']} with {gb_result['probability']*100:.1f}% failure probability.",
            }
            if nct_id:
                _prediction_cache[nct_id] = {"domain": domain, "prediction": prediction}
            return domain, prediction
    except Exception as e:
        logger.warning("GB V2 prediction failed: %s", e)

    # ── Secondary: Qwen LLM prediction (if GB fails) ────────────────────────
    if _qwen_model is not None:
        try:
            features_text = _format_trial_features(trial)
            parsed = _qwen_generate(QWEN_PREDICT_FULL_SYSTEM, features_text)

            if "error" not in parsed and "raw" not in parsed and "outcome" in parsed:
                outcome = str(parsed.get("outcome", "unknown")).lower()
                if outcome not in ("success", "failure"):
                    outcome = "failure"

                confidence = str(parsed.get("confidence", "medium")).lower()
                fail_prob = float(parsed.get("failure_probability", 0.5))
                fail_prob = max(0.0, min(1.0, fail_prob))

                # Reason flags — only meaningful for failures
                qwen_reasons = parsed.get("reasons", {})
                if outcome == "success":
                    reasons = {k: False for k in REASON_KEYS}
                else:
                    reasons = {k: bool(qwen_reasons.get(k, False)) for k in REASON_KEYS}

                # Drivers from Qwen — validate and normalise
                raw_drivers = parsed.get("drivers", [])
                drivers = []
                for d in raw_drivers[:12]:
                    if not isinstance(d, dict):
                        continue
                    impact = float(d.get("impact", 0))
                    direction = str(d.get("direction", "risk")).lower()
                    if direction not in ("risk", "safe"):
                        direction = "risk" if impact > 0 else "safe"
                    drivers.append({
                        "feature": str(d.get("feature", "Unknown")),
                        "value": str(d.get("value", "")),
                        "impact": round(impact, 1),
                        "direction": direction,
                    })
                # Sort by absolute impact descending
                drivers.sort(key=lambda x: abs(x["impact"]), reverse=True)

                # V8 fields: presumption_overridden, reason_weights, weight_explanation
                presumption_overridden = parsed.get("presumption_overridden", outcome == "failure")
                reason_weights = parsed.get("reason_weights", {})
                weight_explanation = parsed.get("weight_explanation", "")

                prediction = {
                    "outcome": outcome,
                    "confidence": confidence,
                    "probability": round(fail_prob, 4),
                    "source": "qwen_v8",
                    "domain": domain,
                    "reasons": reasons,
                    "reason_source": "qwen_v8_innocent_until_guilty",
                    "drivers": drivers,
                    "explanation": parsed.get("explanation", ""),
                    "presumption_overridden": presumption_overridden,
                    "reason_weights": reason_weights,
                    "weight_explanation": weight_explanation,
                }

                if nct_id:
                    _prediction_cache[nct_id] = {"domain": domain, "prediction": prediction}
                return domain, prediction

            logger.warning("Qwen returned unexpected structure: %s", list(parsed.keys()))

        except Exception as e:
            logger.warning("Qwen prediction failed: %s", e)

    # ── Fallback: RF V6 ──────────────────────────────────────────────────────
    logger.info("Falling back to RF V6 for %s", nct_id)
    prediction = predict_rf(trial)
    prediction["domain"] = domain
    prediction["source"] = "rf_v6_fallback"
    prediction["reasons"] = _heuristic_reasons(prediction, trial)
    prediction["reason_source"] = "heuristic"

    if nct_id:
        _prediction_cache[nct_id] = {"domain": domain, "prediction": prediction}
    return domain, prediction


def _batch_db_trial_details(nct_ids):
    """Fetch trial details from DB for a list of NCT IDs. Single query, fast."""
    if not nct_ids:
        return {}
    try:
        from models.ticker_map import get_connection
        conn = get_connection()
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(nct_ids))
        cur.execute(f"""
            SELECT nct_id, title, condition, intervention, enrollment, status,
                   phase, brief_summary, study_design, primary_endpoint, sponsor
            FROM trials WHERE nct_id IN ({placeholders})
        """, nct_ids)
        rows = {r["nct_id"]: dict(r) for r in cur.fetchall()}
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        logger.warning("Batch DB lookup failed: %s", e)
        return {}


def _compute_tag_match(focal, candidate):
    """Compute tag-based match % between focal trial and a candidate."""
    tags_matched = 0
    tags_total = 0

    # 1. Condition match
    tags_total += 1
    fc = (focal.get("condition") or "").lower()
    cc = (candidate.get("condition") or "").lower()
    if fc and cc:
        # Check if any condition word overlaps
        fc_words = set(w for w in fc.replace(";", " ").replace(",", " ").split() if len(w) > 3)
        cc_words = set(w for w in cc.replace(";", " ").replace(",", " ").split() if len(w) > 3)
        if fc_words & cc_words:
            tags_matched += 1

    # 2. Phase match
    tags_total += 1
    fp = str(focal.get("phase") or "")
    cp = str(candidate.get("phase") or "")
    if fp and cp and any(d in fp for d in "1234") and any(d in cp for d in "1234"):
        fp_num = max(int(d) for d in fp if d.isdigit())
        cp_num = max(int(d) for d in cp if d.isdigit())
        if fp_num == cp_num:
            tags_matched += 1

    # 3. Enrollment range match (within 2x)
    tags_total += 1
    fe = focal.get("enrollment") or 0
    ce = candidate.get("enrollment") or 0
    if fe and ce and fe > 0 and ce > 0:
        ratio = max(fe, ce) / max(min(fe, ce), 1)
        if ratio <= 2.0:
            tags_matched += 1

    # 4. Sponsor match
    tags_total += 1
    fs = (focal.get("sponsor") or "").lower().split()[0] if focal.get("sponsor") else ""
    cs = (candidate.get("sponsor") or "").lower().split()[0] if candidate.get("sponsor") else ""
    if fs and cs and (fs in cs or cs in fs):
        tags_matched += 1

    # 5. Design match (randomized, blinded, placebo)
    fd = (focal.get("study_design") or "").lower()
    cd = (candidate.get("study_design") or "").lower()
    for feature in ["random", "double", "placebo", "open"]:
        tags_total += 1
        if (feature in fd) == (feature in cd):
            tags_matched += 1

    # 6. Endpoint match
    tags_total += 1
    fep = (focal.get("primary_endpoint") or "").lower()
    cep = (candidate.get("primary_endpoint") or "").lower()
    if fep and cep:
        fep_words = set(w for w in fep.split() if len(w) > 3)
        cep_words = set(w for w in cep.split() if len(w) > 3)
        if fep_words & cep_words:
            tags_matched += 1

    return round(tags_matched / max(tags_total, 1), 3)


def _build_match_tags(focal, candidate):
    """Build list of matching tag names between focal and candidate trial."""
    tags = []
    fc = (focal.get("condition") or "").lower()
    cc = (candidate.get("condition") or "").lower()
    if fc and cc and any(w in cc for w in fc.split(";")[0].replace(",", " ").split() if len(w) > 3):
        tags.append("condition")
    fp = str(focal.get("phase") or "")
    cp = str(candidate.get("phase") or "")
    if fp and cp and any(d in fp for d in "1234") and any(d in cp for d in "1234"):
        if max((int(d) for d in fp if d.isdigit()), default=0) == max((int(d) for d in cp if d.isdigit()), default=0):
            tags.append("phase")
    fe = focal.get("enrollment") or 0
    ce = candidate.get("enrollment") or 0
    if fe and ce and fe > 0 and ce > 0 and max(fe, ce) / max(min(fe, ce), 1) <= 2.0:
        tags.append("enrollment")
    fs = (focal.get("sponsor") or "").lower().split()[0] if focal.get("sponsor") else ""
    cs = (candidate.get("sponsor") or "").lower().split()[0] if candidate.get("sponsor") else ""
    if fs and cs and len(fs) > 2 and (fs in cs or cs in fs):
        tags.append("sponsor")
    fd = (focal.get("study_design") or "").lower()
    cd = (candidate.get("study_design") or "").lower()
    if "random" in fd and "random" in cd:
        tags.append("randomized")
    if "double" in fd and "double" in cd:
        tags.append("double-blind")
    if "placebo" in fd and "placebo" in cd:
        tags.append("placebo")
    return tags


def retrieve_supporting(trial, domain, k=5):
    """Retrieve similar COMPLETED trials with known outcomes from DB."""
    try:
        from models.ticker_map import get_connection

        # Use in-memory cache — loaded once at startup, never re-read from disk
        if not _known_outcomes_loaded:
            _init_known_outcomes()
        known_outcomes = _known_outcomes

        conn = get_connection()
        cur = conn.cursor()

        focal_nct = trial.get("nct_id", "")
        condition = trial.get("condition", "") or ""
        phase = trial.get("phase", "") or ""

        cond_words = [w.strip() for w in condition.replace(";", " ").replace(",", " ").split() if len(w.strip()) > 3]

        candidates = []

        # Strategy 1: Match by condition keywords — prefer trials with results_posted
        # Include TERMINATED trials too — they have useful outcome/failure info
        if cond_words:
            cond_like = " OR ".join(["condition ILIKE %s"] * min(len(cond_words), 3))
            params = [f"%{w}%" for w in cond_words[:3]]
            cur.execute(f"""
                SELECT nct_id, title, condition, intervention, enrollment, status,
                       phase, brief_summary, study_design, primary_endpoint, sponsor,
                       results_posted
                FROM trials
                WHERE status IN ('COMPLETED', 'Completed', 'TERMINATED', 'Terminated')
                  AND nct_id != %s
                  AND ({cond_like})
                ORDER BY results_posted DESC, RANDOM()
                LIMIT 50
            """, [focal_nct] + params)
            candidates.extend([dict(r) for r in cur.fetchall()])

        # Strategy 2: Broaden to same phase if needed
        if len(candidates) < k and phase:
            phase_like = f"%{phase.replace('PHASE','').replace(' ','').strip()}%"
            existing_ids = [c["nct_id"] for c in candidates]
            exclude_placeholders = ",".join(["%s"] * (len(existing_ids) + 1))
            cur.execute(f"""
                SELECT nct_id, title, condition, intervention, enrollment, status,
                       phase, brief_summary, study_design, primary_endpoint, sponsor,
                       results_posted
                FROM trials
                WHERE status IN ('COMPLETED', 'Completed', 'TERMINATED', 'Terminated')
                  AND nct_id NOT IN ({exclude_placeholders})
                  AND phase ILIKE %s
                ORDER BY results_posted DESC, RANDOM()
                LIMIT 30
            """, [focal_nct] + existing_ids + [phase_like])
            candidates.extend([dict(r) for r in cur.fetchall()])

        # Fetch outcome signals from v2 for candidates that have results
        candidate_ncts = [c["nct_id"] for c in candidates]
        v2_outcomes = {}
        v2_why_stopped = {}
        if candidate_ncts:
            try:
                ph = ",".join(["%s"] * len(candidate_ncts))
                cur.execute(f"""
                    SELECT nct_id, coarse_label, coarse_confidence
                    FROM v2.outcome_signal
                    WHERE nct_id IN ({ph})
                      AND coarse_label NOT IN ('no_result', 'not_evaluable')
                """, candidate_ncts)
                for r in cur.fetchall():
                    v2_outcomes[r["nct_id"]] = {
                        "outcome": r["coarse_label"],
                        "confidence": r["coarse_confidence"],
                    }
            except Exception:
                pass
            # Fetch why_stopped for terminated trials
            try:
                cur.execute(f"""
                    SELECT nct_id, why_stopped
                    FROM v2.trial
                    WHERE nct_id IN ({ph})
                      AND why_stopped IS NOT NULL AND why_stopped != ''
                """, candidate_ncts)
                for r in cur.fetchall():
                    v2_why_stopped[r["nct_id"]] = r["why_stopped"]
            except Exception:
                pass

        cur.close()
        conn.close()

        # Score candidates and attach outcomes
        scored = []
        for c in candidates:
            score = _compute_tag_match(trial, c)
            tags = _build_match_tags(trial, c)
            c["similarity_score"] = score
            c["match_tags"] = tags
            c["trial_id"] = c.get("nct_id", "")

            # Attach known outcome: training data > v2.outcome_signal > ctgov flag > status
            nct = c.get("nct_id", "")
            if nct in known_outcomes:
                c["outcome"] = known_outcomes[nct]["outcome"]
                c["reason_summary"] = known_outcomes[nct]["reasons"]
                c["outcome_source"] = "validated"
            elif nct in v2_outcomes:
                raw_label = v2_outcomes[nct]["outcome"]
                label_map = {"positive": "endpoint met", "negative": "endpoint not met",
                             "mixed": "mixed results", "inconclusive": "inconclusive",
                             "safety_failure": "safety failure"}
                c["outcome"] = label_map.get(raw_label, raw_label)
                reason_map = {"positive": "primary endpoint met (p<0.05)",
                              "negative": "primary endpoint not met (p>=0.05)",
                              "mixed": "mixed results across endpoints",
                              "safety_failure": "terminated due to safety concerns"}
                c["reason_summary"] = reason_map.get(raw_label, "extracted from posted results")
                c["outcome_source"] = "v2_extracted"
            elif c.get("results_posted"):
                c["outcome"] = "results posted"
                c["reason_summary"] = "results available on ClinicalTrials.gov"
                c["outcome_source"] = "ctgov"
            elif c.get("status", "").upper() in ("TERMINATED",):
                why = v2_why_stopped.get(nct, "")
                if why:
                    why_lower = why.lower()
                    if any(w in why_lower for w in ["complete response letter", "crl received", "crl issued", "fda issued a crl", "refused to file"]):
                        c["outcome"] = "CRL received"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["efficacy", "futility", "endpoint", "did not meet", "insufficient efficacy", "failed to demonstrate"]):
                        c["outcome"] = "lack of efficacy"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["safety", "adverse event", "toxicity", "death", "serious adverse", "hepatotoxicity", "dose-limiting"]):
                        c["outcome"] = "safety failure"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["enrollment", "recruitment", "accrual", "slow enrollment", "unable to enroll", "difficulty enrolling"]):
                        c["outcome"] = "enrollment failure"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["business", "funding", "financial", "strategic", "commercial", "sponsor decision", "budget", "resource"]):
                        c["outcome"] = "business decision"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["regulatory", "fda hold", "clinical hold", "partial clinical hold"]):
                        c["outcome"] = "regulatory hold"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["covid", "pandemic", "coronavirus"]):
                        c["outcome"] = "COVID impact"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    elif any(w in why_lower for w in ["superseded", "replaced", "another study", "successor"]):
                        c["outcome"] = "superseded"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                    else:
                        c["outcome"] = "terminated"
                        c["reason_summary"] = why[:200]
                        c["outcome_source"] = "terminated_reason"
                else:
                    c["outcome"] = "terminated"
                    c["reason_summary"] = "trial terminated, reason not specified"
                    c["outcome_source"] = "status_only"
            else:
                c["outcome"] = "completed"
                c["reason_summary"] = "no results posted yet"
                c["outcome_source"] = "status_only"

            if c.get("brief_summary"):
                sentences = c["brief_summary"].split(". ")
                c["brief_summary"] = ". ".join(sentences[:2]) + ("." if sentences else "")
            scored.append(c)

        # Sort: prioritize known outcomes, then by score
        _source_rank = {"validated": 0, "v2_extracted": 1, "terminated_reason": 2, "ctgov": 3, "status_only": 4}
        scored.sort(key=lambda x: (
            _source_rank.get(x.get("outcome_source"), 3),
            -x["similarity_score"]
        ))
        return scored[:k]

    except Exception as e:
        logger.warning("DB-based evidence retrieval failed: %s", e)
        return []


QWEN_ANALYSIS_SYSTEM = (
    "You are a biotech analyst writing a detailed research report on a clinical trial. "
    "You have access to web research data about the company, drug, and regulatory landscape.\n\n"
    "Write a comprehensive analysis covering these sections:\n"
    "## Company Overview\nWho is the sponsor, their track record, pipeline strength.\n"
    "## Drug & Mechanism\nWhat the drug does, how it works, drug class history, prior clinical data.\n"
    "## Trial Design Assessment\nIs the design strong enough? Enrollment, endpoints, controls.\n"
    "## Competitive Landscape\nWhat else is approved or in development for this condition.\n"
    "## Risk Factors\nSpecific reasons this could fail, with evidence.\n"
    "## Bull Case\nWhy it could succeed, with evidence.\n"
    "## Conclusion\nOverall assessment.\n\n"
    "Reference the web research snippets provided. Cite sources by name when possible. "
    "Be specific — use real data, names, and numbers. No generic statements. "
    "Write 4-6 paragraphs total across all sections. Output plain text with ## headers."
)


def _load_drug_science(drug_name):
    """Load PubMed drug science from local files (no web search needed)."""
    import re as _re
    _app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    drug_science_dir = os.path.join(_app_dir, "model_data", "drug_science")
    if not os.path.exists(drug_science_dir):
        return {}

    # Try exact match first, then fuzzy
    drug_clean = drug_name.split(";")[0].split(",")[0].strip()

    # Build candidate names: full name, generic name in parens, first word
    candidates = [drug_clean]
    paren_match = _re.search(r'\(([^)]+)\)', drug_clean)
    if paren_match:
        candidates.append(paren_match.group(1).strip())
    candidates.append(drug_clean.split("(")[0].strip())
    candidates.append(drug_clean.split()[0])

    fpath = None
    for candidate in candidates:
        fname = _re.sub(r'[^a-zA-Z0-9]', '_', candidate) + ".json"
        test_path = os.path.join(drug_science_dir, fname)
        if os.path.exists(test_path):
            fpath = test_path
            break

    if fpath is None:
        # Fuzzy search across all files
        for candidate in candidates:
            drug_lower = candidate.lower().replace(" ", "_").replace("-", "_")
            if len(drug_lower) < 3:
                continue
            for f in os.listdir(drug_science_dir):
                f_lower = f.lower().replace("-", "_")
                if drug_lower in f_lower or f_lower.startswith(drug_lower[:min(len(drug_lower), 10)]):
                    fpath = os.path.join(drug_science_dir, f)
                    break
            if fpath:
                break

    if fpath is None:
        return {}

    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("science", {})
    except:
        return {}


def _load_rag_evidence(trial):
    """Load RAG V2 facts relevant to this trial."""
    _app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rag_path = os.path.join(_app_dir, "model_data", "rag_v2.jsonl")
    if not os.path.exists(rag_path):
        return []

    try:
        with open(rag_path, "r", encoding="utf-8") as f:
            facts = [json.loads(l) for l in f.readlines()]
    except:
        return []

    phase = str(trial.get("phase", "")).lower()
    condition = str(trial.get("condition", "")).lower()
    intervention = str(trial.get("intervention", "")).lower()
    sponsor = str(trial.get("sponsor", "")).lower().split()[0] if trial.get("sponsor") else ""
    drug_words = set(w.strip().lower().split()[0] for w in intervention.split(";") if w.strip() and len(w.strip()) > 2)

    scored = []
    for fact in facts:
        tags = [t.lower() for t in fact.get("tags", [])]
        drug = fact.get("drug", "").lower()
        score = 0
        if drug and any(dw in drug or drug.startswith(dw) for dw in drug_words): score += 10
        if any(phase and t in phase for t in tags): score += 3
        if any(t in condition for t in tags if len(t) > 3): score += 2
        if any(sponsor and t == sponsor for t in tags): score += 2
        if score > 0: scored.append((score, fact["text"]))

    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:7]]


def generate_analysis(trial, prediction, supporting, domain):
    """Generate analysis using local drug science + RAG (no web search)."""
    t0 = time.time()

    nct_id = trial.get("nct_id", "")
    sponsor = trial.get("sponsor", "")
    drug_name = trial.get("intervention", "") or trial.get("drug", "")
    condition = trial.get("condition", "")
    title = trial.get("title", "")

    # 1. Load local drug science (PubMed files) — FAST, no web search
    drug_science = _load_drug_science(drug_name)
    rag_evidence = _load_rag_evidence(trial)
    sources = []  # No web sources, all local

    # 2. Build analysis from local data — no LLM needed, instant
    outcome = prediction.get("outcome", "unknown")
    prob = prediction.get("probability", 0.5)
    phase = trial.get("phase", "")
    enrollment = trial.get("enrollment", "")
    drivers = prediction.get("drivers", [])

    # 3. Build structured analysis from local drug science + RAG — FAST, no LLM/web needed
    parts = []

    # Company Overview
    parts.append(f"## Company Overview")
    parts.append(f"{sponsor or 'Unknown sponsor'} is conducting this {phase} trial for {condition}.")
    if enrollment:
        parts.append(f"The trial enrolls {enrollment} patients.")

    # Drug Mechanism (from PubMed files)
    parts.append(f"\n## Drug & Mechanism")
    if drug_science.get("mechanism"):
        mech = drug_science["mechanism"][0].get("abstract", "")[:400]
        parts.append(f"**{drug_name}**: {mech}")
    else:
        parts.append(f"{drug_name or 'The intervention'} is being evaluated in a {phase} trial.")

    # Drug Targets
    if drug_science.get("targets"):
        target_text = drug_science["targets"][0].get("abstract", "")[:300]
        parts.append(f"\n**Molecular targets**: {target_text}")

    # Safety Profile
    if drug_science.get("safety"):
        safety_text = drug_science["safety"][0].get("abstract", "")[:300]
        parts.append(f"\n## Safety Profile")
        parts.append(safety_text)

    # Prior Clinical Data
    if drug_science.get("prior_results"):
        parts.append(f"\n## Prior Clinical Data")
        for entry in drug_science["prior_results"][:2]:
            parts.append(f"- {entry.get('abstract', '')[:200]}")

    # RAG Evidence
    if rag_evidence:
        parts.append(f"\n## Historical Evidence")
        for ev in rag_evidence[:5]:
            parts.append(f"- {ev[:200]}")

    # Similar Trials Summary
    if supporting:
        succ_count = sum(1 for s in supporting if 'success' in str(s.get('outcome','')).lower() or 'met' in str(s.get('outcome','')).lower())
        fail_count = sum(1 for s in supporting if 'fail' in str(s.get('outcome','')).lower() or 'not met' in str(s.get('outcome','')).lower())
        parts.append(f"\n## Similar Trials")
        parts.append(f"Of {len(supporting)} similar completed trials: {succ_count} succeeded, {fail_count} failed.")

    # Prediction Summary
    parts.append(f"\n## Prediction Summary")
    fail_pct = round(float(prob) * 100, 1)
    succ_pct = round(100 - fail_pct, 1)
    parts.append(f"The Gradient Boosting model predicts **{outcome.upper()}** with {succ_pct}% confidence ({fail_pct}% failure probability).")
    if drivers:
        top_drivers = [f"{d.get('feature','?')}: {d.get('value','?')} ({d.get('importance', d.get('impact', 0))}%)" for d in drivers[:3]]
        parts.append(f"Key factors: {', '.join(top_drivers)}.")

    elapsed = round(time.time() - t0, 2)

    # Build source list from drug science files
    if drug_science:
        for section in ["mechanism", "targets", "safety", "prior_results"]:
            for entry in drug_science.get(section, [])[:1]:
                title = entry.get("title", "")
                if title:
                    sources.append({"title": title[:100], "url": "", "snippet": entry.get("abstract", "")[:100]})

    return {
        "text": "\n".join(parts),
        "sources": sources[:10],
        "source_type": "local_drug_science",
        "elapsed": elapsed,
    }


def run_ticker_intel(ticker, trials, drugs):
    """Full orchestration: predict + retrieve + analyze for a ticker's focal trial."""
    if not trials:
        return {
            "ticker": ticker,
            "domain": "unknown",
            "prediction": None,
            "analysis": {"text": "No trials available for prediction.", "source_type": "llm_generated"},
            "supporting_trials": [],
        }

    # Pick focal trial: find the best UPCOMING trial (no results yet, nearest completion)
    # Priority: highest phase → active/recruiting status → nearest completion date → largest enrollment
    UPCOMING_STATUSES = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ACTIVE, NOT RECRUITING",
                         "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "COMPLETED"}

    def trial_score(t):
        """Score trials for focal selection. Higher = better candidate for prediction."""
        score = 0
        # Skip trials with posted results — those are past, not upcoming
        if t.get("results_posted") in (1, True, "1"):
            return -1000

        status = str(t.get("status", "")).upper().strip()
        phase = str(t.get("phase", ""))

        # Phase score (Phase 3 > 2 > 1)
        if "3" in phase: score += 300
        elif "2" in phase: score += 200
        elif "4" in phase: score += 150
        elif "1" in phase: score += 100

        # Status score (active near completion > recruiting > completed)
        if "ACTIVE" in status and "NOT" in status: score += 50  # active, not recruiting = near completion
        elif "RECRUITING" in status: score += 40
        elif "NOT_YET" in status: score += 30
        elif "COMPLETED" in status: score += 20  # completed but no results = awaiting readout

        # Enrollment bonus (larger = more important)
        enrollment = t.get("enrollment") or 0
        if enrollment and int(enrollment) > 100: score += 10

        return score

    scored = [(trial_score(t), i, t) for i, t in enumerate(trials)]
    scored.sort(key=lambda x: (-x[0], x[1]))  # highest score first, preserve order on ties

    focal = scored[0][2] if scored and scored[0][0] > -1000 else trials[0]

    # Log the selection
    logger.info("Focal trial selected: %s (phase=%s, status=%s, results_posted=%s, score=%d)",
                focal.get("nct_id", "?"), focal.get("phase", "?"),
                focal.get("status", "?"), focal.get("results_posted", "?"),
                scored[0][0] if scored else 0)

    # 1. Domain + Prediction
    domain, prediction = predict_trial_for_ticker(focal)

    # 2. RAG retrieval
    supporting = retrieve_supporting(focal, domain)

    # 3. Analysis (async-safe, but we run sync for simplicity)
    analysis = generate_analysis(focal, prediction, supporting, domain)

    # Build selection reason
    f_status = str(focal.get("status", "")).upper()
    f_results = focal.get("results_posted") in (1, True, "1")
    if "ACTIVE" in f_status and "NOT" in f_status:
        selection_reason = "Active, near completion — no results yet"
    elif "RECRUITING" in f_status:
        selection_reason = "Currently recruiting — upcoming readout"
    elif "COMPLETED" in f_status and not f_results:
        selection_reason = "Completed — awaiting results publication"
    elif "COMPLETED" in f_status:
        selection_reason = "Completed with results"
    else:
        selection_reason = "Most advanced trial in pipeline"

    return {
        "ticker": ticker,
        "focal_trial": {
            "nct_id": focal.get("nct_id", ""),
            "title": focal.get("title", ""),
            "phase": focal.get("phase", ""),
            "condition": focal.get("condition", ""),
            "enrollment": focal.get("enrollment"),
            "intervention": focal.get("intervention", ""),
            "selection_reason": selection_reason,
            "status": focal.get("status", ""),
        },
        "domain": domain,
        "domain_uncertain": False,
        "prediction": prediction,
        "reason_flags": prediction.get("reasons", {}),
        "analysis": analysis,
        "supporting_trials": supporting,
    }


def handle_chat(ticker, message, context):
    """Handle a chatbox message using the agent with page context."""
    try:
        from agent.loop import run_agent

        # Build context string from page data
        ctx_parts = [f"Company: {ticker}"]
        if context.get("focal_trial"):
            ft = context["focal_trial"]
            ctx_parts.append(f"Focal trial: {ft.get('nct_id','')} — {ft.get('title','')}")
        if context.get("prediction"):
            p = context["prediction"]
            ctx_parts.append(f"Prediction: {p.get('outcome','')} ({p.get('confidence','')}, source: {p.get('source','')})")
        if context.get("domain"):
            ctx_parts.append(f"Domain: {context['domain']}")

        ctx = "\n".join(ctx_parts)
        query = f"Context for {ticker}:\n{ctx}\n\nUser question: {message}\n\nAnswer concisely using available data. If you need to search, use tools. Do not fabricate facts."

        result = run_agent(query)
        return {
            "response": result.get("summary", "I couldn't find an answer."),
            "sources": result.get("sources", []),
            "tool_calls": result.get("tool_calls_made", 0),
        }
    except Exception as e:
        logger.warning("Chat failed: %s", e)
        return {"response": "Chat unavailable.", "error": str(e)}


# ── Auto-initialize models on import ─────────────────────────────
def _init_known_outcomes():
    """Load v5 training outcomes into memory once. Never re-reads from disk."""
    global _known_outcomes, _known_outcomes_loaded
    if _known_outcomes_loaded:
        return
    try:
        import pandas as pd
        v5_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "v5" / "dataset_v5_trainable.csv"
        v5 = pd.read_csv(v5_path)
        for _, row in v5.iterrows():
            reasons = []
            for r in ["lack_of_efficacy", "safety_issue", "funding_or_business", "regulatory", "trial_design_issue"]:
                if row.get(f"label_reason_{r}"):
                    reasons.append(r.replace("_", " "))
            _known_outcomes[row["nct_id"]] = {
                "outcome": row["final_label"],
                "reasons": ", ".join(reasons) if reasons else "unknown",
            }
        logger.info("Loaded %d known outcomes from v5 dataset", len(_known_outcomes))
    except Exception as e:
        logger.warning("Could not load v5 outcomes: %s", e)
    _known_outcomes_loaded = True


def init_all_models():
    """Initialize RF + Qwen V5 models + v5 outcomes cache. Call at server startup."""
    logger.info("Initializing prediction models...")
    _init_rf()
    _init_qwen()
    _init_known_outcomes()
    logger.info("All models initialized. Qwen loaded: %s", _qwen_model is not None)
