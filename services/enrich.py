"""
Field-level gap-filling for trials and drugs.
All logic is deterministic — NO LLM calls, NO external APIs.
Only enriches existing records; never adds or removes trials.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Placeholder drug names to filter out ─────────────────────────

PLACEHOLDER_DRUGS = {
    "placebo", "standard of care", "soc", "control", "comparator",
    "active comparator", "best supportive care", "bsc", "observation",
    "no intervention", "usual care", "investigator's choice",
    "physician's choice", "watchful waiting", "sham", "vehicle",
    "normal saline", "saline", "matching placebo", "dummy",
}

# ── Condition classification keywords ────────────────────────────

_ONCOLOGY = {"cancer","tumor","tumour","carcinoma","sarcoma","leukemia","lymphoma",
             "melanoma","glioma","myeloma","neoplasm","metasta","oncolog","blastoma"}
_CARDIO = {"cardiovascular","heart","cardiac","coronary","atrial","hypertension",
           "cholesterol","lipid","atherosclerosis","stroke","thrombosis","arrhythmia"}
_NEURO = {"alzheimer","parkinson","multiple sclerosis","epilepsy","migraine",
          "neuropathy","dementia","huntington","als","amyotrophic","schizophrenia",
          "depression","bipolar","seizure","neurology","cns","brain"}
_INFECT = {"hiv","hepatitis","infection","bacterial","viral","fungal","malaria",
           "tuberculosis","influenza","covid","sars","sepsis","pneumonia"}
_AUTOIMMUNE = {"lupus","rheumatoid","psoriasis","crohn","colitis","inflammatory bowel",
               "ankylosing","autoimmune","atopic dermatitis"}
_RARE = {"orphan","rare","fabry","gaucher","pompe","hunter","hurler",
         "cystic fibrosis","hemophilia","sickle cell","duchenne","rett"}
_RESPIRATORY = {"asthma","copd","pulmonary","respiratory","lung fibrosis","bronchitis"}
_METABOLIC = {"diabetes","obesity","metabolic","fatty liver","nash","nafld","gout"}


def _kw(text, keywords):
    t = text.lower()
    return any(k in t for k in keywords)


# ── Classification functions ─────────────────────────────────────

def classify_condition(text):
    """Classify condition text into a therapeutic area."""
    if _kw(text, _ONCOLOGY): return "oncology"
    if _kw(text, _CARDIO): return "cardiovascular"
    if _kw(text, _NEURO): return "neurology"
    if _kw(text, _INFECT): return "infectious_disease"
    if _kw(text, _AUTOIMMUNE): return "autoimmune"
    if _kw(text, _RARE): return "rare_disease"
    if _kw(text, _RESPIRATORY): return "respiratory"
    if _kw(text, _METABOLIC): return "metabolic"
    return "other"


def classify_intervention_type(intervention_text):
    """Classify intervention into drug/biologic/device/other."""
    t = intervention_text.lower()
    # Biologics
    if any(k in t for k in ["antibody","mab ","umab","izumab","ximab","zumab",
                             "vaccine","cell therapy","car-t","car t","gene therapy",
                             "mrna","sirna","antisense","oligonucleotide"]):
        return "biologic"
    # Devices
    if any(k in t for k in ["device","implant","stent","catheter","pump","stimulat"]):
        return "device"
    # Radiation
    if any(k in t for k in ["radiation","radiotherapy","proton","brachytherapy"]):
        return "radiation"
    # Small molecule (default for named drugs)
    if any(k in t for k in ["nib","mib","tinib","ciclib","rafenib","parin",
                             "statin","sartan","pril","olol","azole","mycin",
                             "cillin","cycline","vir ","navir","previr","tablet",
                             "capsule","oral","mg","injection"]):
        return "small_molecule"
    # If it looks like a proper drug name (capitalized word)
    if re.search(r'\b[A-Z][a-z]{3,}(?:mab|nib|lib|vir|tid)\b', intervention_text):
        return "small_molecule"
    return "drug"  # generic fallback


def classify_mechanism(drug_name):
    """Classify drug mechanism from name suffix/prefix patterns."""
    name = drug_name.lower().strip()

    # mRNA
    if "mrna" in name or name.startswith("mrna-"):
        return "mRNA therapeutic"

    # Monoclonal antibodies (-mab suffix)
    if re.search(r'[a-z]mab$', name) or re.search(r'[a-z]mab\b', name):
        # Sub-classify by target
        if "tu" in name[-6:] or "xi" in name[-6:]: return "chimeric monoclonal antibody"
        if "zu" in name[-6:] or "nu" in name[-6:]: return "humanized monoclonal antibody"
        if "mu" in name[-6:]: return "murine monoclonal antibody"
        return "monoclonal antibody"

    # Kinase inhibitors (-nib/-tinib)
    if re.search(r'(?:tinib|nib)$', name):
        return "kinase inhibitor"

    # CDK inhibitors (-ciclib)
    if name.endswith("ciclib"):
        return "CDK inhibitor"

    # Proteasome inhibitors (-zomib)
    if name.endswith("zomib"):
        return "proteasome inhibitor"

    # ADCs (antibody-drug conjugates)
    if "adc" in name or "conjugate" in name or re.search(r'vedotin|emtansine|deruxtecan|govitecan', name):
        return "antibody-drug conjugate"

    # PD-1/PD-L1 checkpoint inhibitors
    if any(k in name for k in ["pembrolizumab","nivolumab","atezolizumab","durvalumab","avelumab","cemiplimab"]):
        return "PD-1/PD-L1 inhibitor"

    # PARP inhibitors
    if name.endswith("parib"):
        return "PARP inhibitor"

    # Antivirals (-vir)
    if re.search(r'[a-z]vir$', name):
        return "antiviral"

    # Peptides (-tide)
    if name.endswith("tide"):
        return "peptide"

    # Vaccines
    if "vaccine" in name or "vax" in name:
        return "vaccine"

    # CAR-T / cell therapy
    if "car-t" in name or "car t" in name or "cel " in name:
        return "cell therapy"

    # Gene therapy
    if "gene" in name or re.search(r'[a-z]gene$', name):
        return "gene therapy"

    # siRNA
    if "sirna" in name or name.endswith("siran"):
        return "siRNA"

    # ASO (antisense)
    if "antisense" in name or name.endswith("ersen"):
        return "antisense oligonucleotide"

    return "unknown"


def bucket_enrollment(n):
    """Categorize enrollment size."""
    if not n or n <= 0: return None
    if n < 50: return "tiny"
    if n < 200: return "small"
    if n < 500: return "medium"
    return "large"


def build_trial_description(trial):
    """Build a short description from existing trial fields."""
    parts = []
    phase = trial.get("phase", "")
    if phase:
        parts.append(phase)

    condition = trial.get("condition", "")
    if condition:
        # Take first condition, truncate
        first_cond = condition.split(";")[0].split(",")[0].strip()
        if len(first_cond) > 50:
            first_cond = first_cond[:47] + "..."
        parts.append(first_cond)

    parts.append("trial")

    intervention = trial.get("intervention", "")
    if intervention:
        first_int = intervention.split(";")[0].strip()
        if first_int.lower() not in PLACEHOLDER_DRUGS:
            if len(first_int) > 40:
                first_int = first_int[:37] + "..."
            parts.append(f"evaluating {first_int}")

    enrollment = trial.get("enrollment")
    if enrollment and enrollment > 0:
        parts.append(f"in {enrollment} patients")

    return " ".join(parts) + "."


def summarize_design(design_text):
    """Extract key design features from study_design text."""
    if not design_text:
        return None
    t = design_text.lower()
    features = []
    if "randomiz" in t: features.append("randomized")
    if "double" in t and "blind" in t: features.append("double-blind")
    elif "single" in t and "blind" in t: features.append("single-blind")
    elif "open" in t and "label" in t: features.append("open-label")
    if "placebo" in t: features.append("placebo-controlled")
    if "parallel" in t: features.append("parallel-group")
    if "crossover" in t: features.append("crossover")
    if not features:
        return None
    return ", ".join(features).capitalize()


def is_placeholder_drug(name):
    """Check if a drug name is actually a placeholder/control."""
    if not name:
        return True
    clean = name.strip().lower()
    # Exact matches
    if clean in PLACEHOLDER_DRUGS:
        return True
    # Partial matches
    if clean.startswith("placebo") or clean.startswith("sham"):
        return True
    # Very short generic names
    if len(clean) <= 2:
        return True
    # Dosage strings (e.g., "625 mg", "10mg", "100 ml")
    if re.match(r'^\d+\s*(mg|ml|mcg|iu|g|%|cc)\b', clean):
        return True
    # Pure numbers
    if re.match(r'^\d+$', clean):
        return True
    return False


# ── Main enrichment function ─────────────────────────────────────

def enrich_company_data(data: dict) -> dict:
    """
    Fill missing fields in trials and drugs using deterministic rules only.
    Never adds/removes records. Never calls LLM. Never fetches external data.
    """
    trials = data.get("trials") or []
    drugs = data.get("drugs") or []

    enriched_trials = 0
    enriched_drugs = 0
    removed_placebo = 0

    for trial in trials:
        filled = {}

        # 1. Short description (if no brief_summary)
        if not trial.get("brief_summary") and trial.get("title"):
            desc = build_trial_description(trial)
            if desc:
                filled["description"] = desc

        # 2. Intervention type
        intervention = trial.get("intervention", "") or ""
        if intervention:
            itype = classify_intervention_type(intervention)
            filled["intervention_type"] = itype

        # 3. Condition category
        condition = trial.get("condition", "") or ""
        if condition:
            filled["condition_category"] = classify_condition(condition)

        # 4. Enrollment bucket
        enrollment = trial.get("enrollment")
        if enrollment:
            bucket = bucket_enrollment(enrollment)
            if bucket:
                filled["enrollment_bucket"] = bucket

        # 5. Design summary
        design = trial.get("study_design", "") or ""
        if design:
            summary = summarize_design(design)
            if summary:
                filled["design_summary"] = summary

        if filled:
            trial["filled_fields"] = filled
            trial["source_type"] = "hybrid"
            enriched_trials += 1
        else:
            trial["source_type"] = "source_data"

    # Enrich drugs
    for drug in drugs:
        filled = {}
        name = drug.get("name", "") or ""

        # 1. Filter placeholders
        if is_placeholder_drug(name):
            drug["_exclude"] = True
            removed_placebo += 1
            continue

        # 2. Mechanism class
        mech = classify_mechanism(name)
        if mech != "unknown":
            filled["mechanism_class"] = mech

        # 3. Drug type from name
        itype = classify_intervention_type(name)
        filled["drug_type"] = itype

        # 4. Description
        condition = drug.get("condition", "") or ""
        phase = drug.get("phase", "") or ""
        if condition:
            filled["description"] = f"{name} — {phase} for {condition}" if phase else f"{name} for {condition}"

        if filled:
            drug["filled_fields"] = filled
            drug["source_type"] = "hybrid"
            enriched_drugs += 1
        else:
            drug["source_type"] = "source_data"

    # Remove placeholder drugs
    before = len(drugs)
    data["drugs"] = [d for d in drugs if not d.get("_exclude")]
    after = len(data["drugs"])

    logger.info(
        "Enrichment: %d/%d trials filled, %d/%d drugs filled, %d placeholders removed",
        enriched_trials, len(trials), enriched_drugs, after, before - after,
    )

    return data
