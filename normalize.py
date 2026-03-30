"""
Centralized normalization for the biotech terminal.

All parsing, formatting, and canonicalization logic lives here.
Replaces scattered normalization in ticker_map.py, news.py, fda_calendar.py, api.py.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Optional


# ── Phase normalization ──────────────────────────────────────────────────

_PHASE_MAP = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE2": "Phase 2",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NA": "N/A",
    # Handle already-normalized inputs gracefully
    "EARLY PHASE 1": "Early Phase 1",
    "PHASE 1": "Phase 1",
    "PHASE 2": "Phase 2",
    "PHASE 3": "Phase 3",
    "PHASE 4": "Phase 4",
}

_PHASE_RANK = {
    "Early Phase 1": 1,
    "Phase 1": 2,
    "Phase 1/2": 3,
    "Phase 2": 4,
    "Phase 2/3": 5,
    "Phase 3": 6,
    "Phase 3/4": 7,
    "Phase 4": 8,
    "N/A": 0,
}


def normalize_phase(raw: str | None) -> str | None:
    """
    Normalize ClinicalTrials.gov phase strings.

    Examples:
        "PHASE3"           -> "Phase 3"
        "PHASE1; PHASE2"   -> "Phase 1/2"
        "PHASE2; PHASE3"   -> "Phase 2/3"
        "EARLY_PHASE1"     -> "Early Phase 1"
        None               -> None
    """
    if not raw:
        return None

    raw = raw.strip()
    if not raw:
        return None

    # Split on semicolons (ClinicalTrials.gov uses "; " for multi-phase)
    parts = [p.strip().upper().replace(" ", "") for p in raw.split(";")]
    # Also handle slash separator
    if len(parts) == 1 and "/" in parts[0]:
        parts = [p.strip().upper().replace(" ", "") for p in raw.split("/")]

    normalized_parts = []
    for part in parts:
        # Try direct lookup
        mapped = _PHASE_MAP.get(part)
        if mapped:
            normalized_parts.append(mapped)
        elif part.startswith("PHASE"):
            # Extract number
            num = part.replace("PHASE", "").strip()
            if num:
                normalized_parts.append(f"Phase {num}")
        else:
            normalized_parts.append(part)

    if not normalized_parts:
        return raw  # Return original if we can't normalize

    # Deduplicate and sort by rank
    unique = list(dict.fromkeys(normalized_parts))
    unique.sort(key=lambda p: _PHASE_RANK.get(p, 99))

    if len(unique) == 1:
        return unique[0]

    # Combine adjacent phases: ["Phase 1", "Phase 2"] -> "Phase 1/2"
    nums = []
    for p in unique:
        m = re.match(r"Phase (\d)", p)
        if m:
            nums.append(m.group(1))
    if len(nums) == len(unique) and len(nums) > 1:
        return f"Phase {'/'.join(nums)}"

    return "; ".join(unique)


def phase_rank(normalized_phase: str | None) -> int:
    """Return integer rank for sorting. Higher = later phase."""
    if not normalized_phase:
        return -1
    return _PHASE_RANK.get(normalized_phase, 0)


# ── Status normalization ─────────────────────────────────────────────────

_STATUS_MAP = {
    "RECRUITING": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active, Not Recruiting",
    "ACTIVE, NOT RECRUITING": "Active, Not Recruiting",
    "NOT_YET_RECRUITING": "Not Yet Recruiting",
    "NOT YET RECRUITING": "Not Yet Recruiting",
    "COMPLETED": "Completed",
    "TERMINATED": "Terminated",
    "WITHDRAWN": "Withdrawn",
    "SUSPENDED": "Suspended",
    "ENROLLING_BY_INVITATION": "Enrolling by Invitation",
    "ENROLLING BY INVITATION": "Enrolling by Invitation",
    "UNKNOWN_STATUS": "Unknown",
    "UNKNOWN STATUS": "Unknown",
    "APPROVED_FOR_MARKETING": "Approved for Marketing",
    "AVAILABLE": "Available",
    "NO_LONGER_AVAILABLE": "No Longer Available",
    "TEMPORARILY_NOT_AVAILABLE": "Temporarily Not Available",
    "WITHHELD": "Withheld",
}


def normalize_status(raw: str | None) -> str | None:
    """Normalize ClinicalTrials.gov status strings."""
    if not raw:
        return None
    key = raw.strip().upper().replace(" ", "_")
    mapped = _STATUS_MAP.get(key)
    if mapped:
        return mapped
    # Also try with spaces
    mapped = _STATUS_MAP.get(raw.strip().upper())
    if mapped:
        return mapped
    # Return title case as fallback
    return raw.strip().replace("_", " ").title()


# ── Sponsor/company normalization ────────────────────────────────────────

_SUFFIXES = re.compile(
    r',?\s*('
    r'Inc\.?|Incorporated|Corp\.?|Corporation|Ltd\.?|Limited|'
    r'LLC|L\.L\.C\.|LLP|L\.L\.P\.|LP|L\.P\.|'
    r'PLC|plc|P\.L\.C\.|SE|AG|GmbH|S\.A\.|S\.A|SA|N\.V\.|NV|'
    r'Co\.?|Company|& Co\.?|'
    r'Pharmaceuticals?|Therapeutics?|Biosciences?|BioTherapeutics?|'
    r'Sciences?|Oncology|Biopharmaceuticals?|Biopharma|'
    r'Genomics?|Biologics?|Biotech|Biotechnology|'
    r'Medical|Healthcare|Health|Diagnostics?|'
    r'Laboratories?|Labs?|Research|'
    r'International|Global|Group|Holdings?|'
    r'Pty|N\.A\.|NA'
    r')\s*$',
    re.IGNORECASE
)


def normalize_sponsor(name: str | None) -> str:
    """
    Normalize company/sponsor name for deterministic matching.

    Examples:
        "Moderna, Inc."     -> "moderna"
        "ModernaTX, Inc."   -> "modernatx"
        "Pfizer Inc."       -> "pfizer"
        "Eli Lilly and Company" -> "eli lilly"
    """
    if not name:
        return ""
    result = name.strip()
    # Iteratively strip suffixes (handles "Corp. Inc." etc.)
    for _ in range(3):
        new = _SUFFIXES.sub("", result).strip()
        if new == result:
            break
        result = new
    # Strip trailing commas and periods
    result = result.rstrip(",. ")
    return result.lower().strip()


def normalize_sponsor_first_word(name: str | None, min_length: int = 5) -> str | None:
    """Extract first word of sponsor name if long enough for matching."""
    normalized = normalize_sponsor(name)
    if not normalized:
        return None
    first = normalized.split()[0].rstrip(",.")
    return first if len(first) >= min_length else None


# ── Date normalization ───────────────────────────────────────────────────

_DATE_FORMATS = [
    "%a, %d %b %Y %H:%M:%S %z",     # "Thu, 20 Mar 2026 08:00:00 +0000"
    "%a, %d %b %Y %H:%M:%S %Z",     # "Thu, 20 Mar 2026 08:00:00 EST"
    "%Y-%m-%dT%H:%M:%S%z",           # "2026-03-20T08:00:00+00:00"
    "%Y-%m-%dT%H:%M:%SZ",            # "2026-03-20T08:00:00Z"
    "%Y-%m-%d %H:%M:%S",             # "2026-03-20 08:00:00"
    "%a, %d %b %Y %H:%M:%S",         # "Thu, 20 Mar 2026 08:00:00"
    "%d %b %Y %H:%M:%S %z",          # "20 Mar 2026 08:00:00 +0000"
    "%Y-%m-%dT%H:%M:%S.%f%z",        # "2026-03-20T08:00:00.000+00:00"
    "%Y-%m-%d",                       # "2026-03-20"
    "%B %d, %Y",                      # "March 20, 2026"
    "%B %d %Y",                       # "March 20 2026"
    "%m/%d/%Y",                       # "03/20/2026"
    "%d/%m/%Y",                       # "20/03/2026"
]


def normalize_date(raw: str | None) -> datetime | None:
    """
    Parse any date string into a datetime object.
    Returns None on failure — never raises.
    """
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None

    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    # Fallback: try fromisoformat
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass

    return None


def normalize_date_to_iso(raw: str | None) -> str | None:
    """Parse any date string and return ISO-sortable 'YYYY-MM-DD HH:MM:SS' string."""
    dt = normalize_date(raw)
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalize_date_to_isoformat(raw: str | None) -> str | None:
    """Parse any date string and return full ISO 8601 string."""
    dt = normalize_date(raw)
    if dt is None:
        return None
    return dt.isoformat()


# ── Ticker normalization ────────────────────────────────────────────────

def normalize_ticker_list(raw: str | None) -> list[str]:
    """
    Parse comma-separated tickers into a sorted uppercase list.

    Examples:
        "MRNA,PFE"   -> ["MRNA", "PFE"]
        "mrna"       -> ["MRNA"]
        None         -> []
        ""           -> []
    """
    if not raw:
        return []
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return sorted(set(tickers))


# ── Numeric safety ──────────────────────────────────────────────────────

def safe_float(val, default: float = 0.0) -> float:
    """
    Convert value to a JSON-safe float, replacing NaN/Inf with default.
    Handles pandas NaN, numpy inf, None, and string values.
    """
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def safe_int(val, default: int = 0) -> int:
    """Convert value to int safely."""
    if val is None:
        return default
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


# ── HTML cleaning ───────────────────────────────────────────────────────

def clean_html(text: str | None, max_length: int = 800) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_length]
