# domains/isac/config.py
from __future__ import annotations
from typing import Any, Dict, List

DOMAIN_NAME = "ISAC"

# -------------------------
# ISAC keyword lists
# -------------------------
KEYWORDS = [
    # core terms
    "isac", "integrated sensing and communication", "integrated sensing & communication",
    "jcas", "joint communication and sensing", "joint sensing and communication",
    "dual-function radar communication", "df rc", "df-radar", "dfrc",
    "radar-communication", "radar communication",

    # sensing tasks/metrics
    "range", "doppler", "range-doppler", "micro-doppler", "angle of arrival", "aoa", "angle of departure", "aod",
    "clutter", "cfar", "tracking", "localization", "positioning", "sensing", "radar sensing",

    # signals/waveforms
    "ofdm sensing", "ofdm radar", "otfs", "fmcw", "pmcw", "chirp", "preamble sensing",
    "pilot", "reference signal", "prs", "srs", "csi-rs", "ssb", "prach",

    # joint design
    "joint waveform", "resource allocation", "joint beamforming", "dual use", "co-design", "trade-off",
    "precoding", "beam management", "mimo radar", "massive mimo",

    # cellular context
    "3gpp", "5g-advanced", "5g advanced", "nr", "6g", "ran", "gnb", "ue",
    "tdd sensing", "uplink sensing", "downlink sensing",
]

WIKI_KEYWORDS = [
    "integrated sensing", "joint communication", "radar", "ofdm", "otfs",
    "mimo", "beamforming", "range doppler", "3gpp", "5g", "6g"
]

def _has_any_kw(text: str, kws: List[str]) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in kws)

def cheap_prefilter(sample: Dict[str, Any], parse_metadata_fn) -> bool:
    """
    Fast, *high-recall* filter to avoid wasting LLM calls.
    ISAC-focused implementation.
    """
    cat = str(sample.get("Category", sample.get("category", "")) or "").strip().lower()
    meta = parse_metadata_fn(sample.get("Metadata", sample.get("metadata")))
    content = str(sample.get("Content", sample.get("content", "")) or "")

    # Standard docs: allow likely 3GPP series where ISAC/positioning/sensing may appear
    if cat == "standard":
        series = str(meta.get("series", meta.get("Series", ""))).strip()
        # ISAC can show up in TRs/TSs across series 22, 36, 37, 38
        if series in {"22", "36", "37", "38"}:
            return True
        # fallback: keyword scan
        return _has_any_kw(content[:6000], KEYWORDS)

    if cat == "wiki":
        return _has_any_kw(content[:6000], WIKI_KEYWORDS) or _has_any_kw(str(meta.get("title", "")), WIKI_KEYWORDS)

    if cat == "arxiv":
        title = str(meta.get("title", "")).lower()
        abstract = str(meta.get("abstract", "")).lower()
        return _has_any_kw(title + "\n" + abstract, KEYWORDS)

    if cat == "web":
        url = str(meta.get("url", "")).lower()
        # keep if url hints or content hints
        return _has_any_kw(url, ["isac", "jcas", "joint-communication", "radar-communication", "dfrc"]) or _has_any_kw(content[:6000], KEYWORDS)

    # Unknown category: just keyword scan the beginning
    return _has_any_kw(content[:6000], KEYWORDS)
