#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset


# ----------------------------
# Logging
# ----------------------------

def setup_logging(log_file: Optional[str] = None, verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stderr)]

    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


# ----------------------------
# Built-in domain profiles
# ----------------------------

DOMAIN_PROFILES = {
    "isac": {
        "display_name": "Integrated Sensing and Communication",
        "include": (
            "ISAC, JCAS, dual-function radar communication, sensing-assisted communication, "
            "communication-assisted sensing, OFDM/OTFS/FMCW/PMCW sensing, range/Doppler/angle estimation, "
            "radar sensing, joint waveform design, joint beamforming, sensing-communication tradeoffs, "
            "PRS/SRS/CSI-RS/SSB reuse for sensing, 5G-Advanced/6G sensing, and 3GPP cellular sensing."
        ),
        "exclude": (
            "pure radar with no communication linkage, pure communication with no sensing component, "
            "generic localization without communication-sensing coupling, marketing, business news, or unrelated IoT/security topics."
        ),
        "keywords": [
            "isac", "integrated sensing and communication", "integrated sensing & communication",
            "jcas", "joint communication and sensing", "joint sensing and communication",
            "dual-function radar communication", "dual function radar communication", "dfrc",
            "radar-communication", "radar communication", "communication and sensing",
            "sensing and communication", "range-doppler", "range doppler", "doppler",
            "micro-doppler", "angle of arrival", "aoa", "angle of departure", "aod",
            "target detection", "target tracking", "tracking", "localization", "positioning",
            "sensing", "radar sensing", "clutter", "cfar", "radar cross section", "rcs",
            "range resolution", "velocity resolution", "parameter estimation", "ofdm sensing",
            "ofdm radar", "otfs", "fmcw", "pmcw", "chirp", "radar waveform",
            "sensing waveform", "preamble sensing", "pilot reuse", "reference signal",
            "prs", "srs", "csi-rs", "ssb", "prach", "joint waveform",
            "joint beamforming", "joint precoding", "dual use", "co-design", "trade-off",
            "sensing-communication tradeoff", "communication-sensing tradeoff",
            "resource allocation", "beam management", "mimo radar", "massive mimo sensing",
            "5g-advanced", "5g advanced", "6g", "nr sensing", "cellular sensing",
            "ran sensing", "gnb sensing", "ue sensing", "uplink sensing", "downlink sensing",
            "tdd sensing", "3gpp sensing",
        ],
        "strong_terms": [
            "isac", "integrated sensing and communication", "jcas",
            "joint communication and sensing", "joint sensing and communication",
            "dual-function radar communication", "dfrc", "ofdm sensing", "ofdm radar",
            "radar communication", "sensing and communication",
        ],
        "anchors": [
            "wireless", "radio", "cellular", "telecom", "5g", "5g-advanced", "6g",
            "nr", "ran", "gnb", "ue", "uplink", "downlink", "beam", "mimo",
            "ofdm", "pilot", "reference signal", "3gpp",
        ],
    },

    "sagin": {
        "display_name": "Space-Air-Ground Integrated Networks / Non-Terrestrial Networks",
        "include": (
            "SAGIN, NTN, satellite-terrestrial integration, NR-NTN, IoT-NTN, NB-IoT over NTN, "
            "LEO/MEO/GEO satellites, HAPS, UAV/aerial base stations, satellite gateways, NTN terminals, "
            "bent-pipe and regenerative payloads, feeder/service/user links, Doppler compensation, timing advance, "
            "long propagation delay, link budget, spot beams, satellite/beam/gateway handover, inter-satellite links, "
            "constellation routing, DTN, gateway diversity, satellite resource allocation, and 3GPP Rel-17/18/19 NTN."
        ),
        "exclude": (
            "generic astronomy, astrophysics, launch vehicles, spacecraft design, satellite imagery, remote sensing, "
            "GNSS/GPS/navigation, UAV flight control, or generic satellite business/news unless telecom network operation is central."
        ),
        "keywords": [
            "sagin", "space-air-ground integrated network", "space air ground integrated network",
            "space-air-ground integrated networks", "space air ground integrated networks",
            "space-air-ground-sea integrated network", "space air ground sea integrated network",
            "ntn", "non-terrestrial network", "non terrestrial network",
            "non-terrestrial networks", "non terrestrial networks",
            "integrated satellite-terrestrial network", "integrated satellite terrestrial network",
            "satellite-terrestrial network", "satellite terrestrial network",
            "aerial-terrestrial network", "aerial terrestrial network",
            "space-ground integrated network", "space ground integrated network",

            "leo satellite", "meo satellite", "geo satellite", "ngso satellite", "gso satellite",
            "low earth orbit", "medium earth orbit", "geostationary orbit",
            "non-geostationary orbit", "non geostationary orbit",
            "satellite constellation", "leo constellation", "mega-constellation", "mega constellation",
            "satellite swarm", "multi-orbit network", "multi orbit network",
            "inter-satellite link", "inter satellite link", "isl",
            "feeder link", "service link", "gateway link", "user link",
            "satellite backhaul", "satellite access", "satellite payload",
            "bent-pipe payload", "bent pipe payload", "transparent payload",
            "regenerative payload", "on-board processing", "on board processing",

            "haps", "high altitude platform", "high-altitude platform",
            "hap station", "high altitude platform station",
            "uav communication", "uav communications", "unmanned aerial vehicle communication",
            "aerial base station", "aerial access point", "drone base station",
            "uav base station", "balloon platform", "airborne platform",

            "terrestrial network integration", "satellite-terrestrial integration",
            "ntn integration", "5g ntn", "nr ntn", "iot ntn",
            "nb-iot ntn", "iot over ntn", "direct-to-device", "direct to device",
            "direct-to-cell", "direct to cell", "satellite direct-to-device",
            "satellite direct to device", "mobile satellite service", "mss",
            "fixed satellite service", "fss", "satellite access network",
            "ntn terminal", "ntn gateway", "satellite gateway", "ground gateway",
            "gateway selection", "earth station", "user terminal",

            "doppler shift", "doppler compensation", "doppler pre-compensation",
            "doppler pre compensation", "doppler shift compensation",
            "frequency pre-compensation", "timing advance", "large propagation delay",
            "long propagation delay", "round trip time", "rtt", "delay compensation",
            "propagation delay compensation", "link budget", "free-space path loss",
            "free space path loss", "fspl", "atmospheric attenuation", "rain attenuation",
            "shadowing", "elevation angle", "slant range", "beam footprint", "spot beam",
            "beam hopping", "beam steering", "satellite beam", "earth-fixed beam",
            "earth fixed beam", "earth-moving beam", "earth moving beam", "ephemeris",
            "orbit prediction",

            "satellite handover", "ntn handover", "beam handover", "gateway handover",
            "inter-satellite handover", "inter satellite handover", "cell reselection in ntn",
            "mobility management in ntn", "orbital mobility", "high mobility",
            "moving cell", "moving beam", "service continuity", "handover due to satellite movement",

            "orbital routing", "constellation routing", "satellite routing",
            "delay-tolerant networking", "delay tolerant networking", "dtn",
            "store-and-forward", "store and forward", "contact plan", "contact graph routing",
            "multi-hop satellite", "multi hop satellite", "multi-layer satellite network",
            "multi layer satellite network", "space segment routing",
            "traffic steering", "load balancing across satellites", "gateway diversity",

            "satellite resource allocation", "ntn resource allocation", "beam resource allocation",
            "spectrum sharing with satellite", "satellite spectrum sharing",
            "power allocation for ntn", "beam scheduling", "satellite scheduling",
            "constellation management", "coverage enhancement", "global coverage",
            "rural coverage", "remote area coverage", "disaster recovery communication",
            "integrated access and backhaul over satellite", "satellite iab",

            "3gpp", "release 17", "rel-17", "rel 17", "release 18", "rel-18", "rel 18",
            "release 19", "rel-19", "rel 19", "5g-advanced", "5g advanced", "5g", "6g",
            "nr", "ran", "gnb", "ng-ran", "ue", "user equipment", "core network",
            "satellite ran", "ntn ran", "ntn nr", "ntn iot", "ntn nb-iot",
        ],
        "strong_terms": [
            "sagin", "space-air-ground integrated network", "space air ground integrated network",
            "ntn", "non-terrestrial network", "non terrestrial network",
            "nr ntn", "5g ntn", "iot ntn", "nb-iot ntn",
            "integrated satellite-terrestrial network", "satellite-terrestrial network",
            "direct-to-device", "direct to device", "direct-to-cell", "direct to cell",
            "satellite handover", "doppler shift compensation", "inter-satellite link",
            "high altitude platform station", "uav communication", "satellite backhaul",
        ],
        "broad_terms": [
            "satellite", "leo", "meo", "geo", "orbit", "constellation", "uav", "drone",
            "haps", "high altitude platform", "earth station", "gateway", "aerospace",
            "orbital", "spacecraft", "payload",
        ],
        "anchors": [
            "wireless", "radio", "cellular", "telecom", "mobile network", "5g", "5g-advanced",
            "6g", "nr", "ran", "gnb", "ng-ran", "ue", "user equipment", "core network",
            "uplink", "downlink", "access network", "backhaul", "spectrum", "bandwidth",
            "beam", "handover", "latency", "coverage", "link budget", "doppler",
            "propagation delay", "gateway", "satellite access", "3gpp",
        ],
        "negative_terms": [
            "astronomy", "astrophysics", "remote sensing", "earth observation",
            "weather satellite", "navigation satellite", "gnss", "gps",
            "launch vehicle", "rocket", "space mission", "spacecraft design",
            "aerodynamics", "flight control", "drone photography", "aerial mapping",
            "satellite imagery", "image classification",
        ],
    },

    "jcc": {
        "display_name": "Joint Communication and Computation / MEC",
        "include": (
            "JCC, joint communication and computation, communication-computation co-design, radio-compute co-design, "
            "mobile/multi-access edge computing, task/computation offloading, MEC servers, edge service placement, "
            "service migration, edge orchestration, traffic steering, LADN, uplink classifier, compute-aware scheduling, "
            "joint radio and compute resource allocation, CPU cycle allocation, edge server selection, workload scheduling, "
            "latency-energy tradeoffs, transmission/computation/queueing delay, UE energy, wireless edge intelligence, "
            "split inference, collaborative inference, AI inference offloading, and federated learning over wireless."
        ),
        "exclude": (
            "generic cloud computing, data centers, Kubernetes, serverless, microservices, generic edge computing, "
            "generic AI/ML, generic caching/load balancing, or pure communication topics without computation/offloading/MEC coupling."
        ),
        "keywords": [
            "joint communication and computation", "joint communications and computation", "jcc",
            "communication-computation co-design", "communication and computation co-design",
            "radio-compute co-design", "radio and compute resource allocation",
            "joint radio and computing resource allocation", "joint communication and computing",
            "computing force network", "cfn", "compute-aware networking", "computing-aware networking",

            "mobile edge computing", "multi-access edge computing", "mec", "mec server",
            "edge application server", "edge enabler server", "edge configuration server",
            "edge service continuity", "edge service migration", "edge service placement",
            "mec orchestration", "edge orchestration", "traffic steering", "local breakout",
            "local area data network", "ladn", "uplink classifier", "ul classifier", "branching point",

            "task offloading", "computation offloading", "compute offloading", "offloading decision",
            "offloading policy", "offloading ratio", "binary offloading", "partial offloading",
            "local execution", "edge execution", "remote execution", "uplink offloading",
            "downlink result delivery", "task partitioning", "task dependency",

            "compute resource allocation", "computing resource allocation", "cpu cycle allocation",
            "cpu frequency allocation", "server resource allocation", "joint scheduling",
            "radio resource scheduling", "compute-aware scheduling", "bandwidth allocation",
            "power allocation", "transmit power allocation", "time-frequency resource allocation",
            "workload scheduling", "server selection", "edge server selection", "load balancing",

            "energy-delay tradeoff", "latency-energy tradeoff", "latency constrained computation",
            "latency-constrained computation", "delay constrained computation", "energy efficient offloading",
            "computation rate", "task completion latency", "end-to-end latency", "queueing delay",
            "transmission delay", "computation delay", "execution delay", "processing delay",
            "ue energy consumption", "deadline constrained", "deadline-aware", "low latency computation",

            "wireless edge intelligence", "edge intelligence", "split inference", "split computing",
            "collaborative inference", "edge inference", "on-device inference", "model partitioning",
            "ai inference offloading", "federated learning over wireless", "distributed learning over wireless",

            "3gpp", "5g-advanced", "5g advanced", "6g", "5g", "nr", "ran", "gnb",
            "base station", "ue", "user equipment", "core network", "application function", "af",
            "network exposure function", "nef", "network data analytics function", "nwdaf",
            "network slice", "network slicing", "urllc", "uplink", "downlink",
        ],
        "strong_terms": [
            "joint communication and computation", "communication-computation co-design",
            "radio-compute co-design", "radio and compute resource allocation",
            "joint radio and computing resource allocation", "mobile edge computing",
            "multi-access edge computing", "computing force network", "task offloading",
            "computation offloading", "compute offloading",
        ],
        "broad_terms": [
            "edge computing", "edge cloud", "edge server", "edge platform", "edge service",
            "edge orchestration", "edge intelligence", "federated learning", "split inference",
            "model partitioning", "service placement", "load balancing",
        ],
        "anchors": [
            "wireless", "radio", "cellular", "telecom", "mobile network", "5g", "5g-advanced",
            "6g", "nr", "ran", "gnb", "base station", "ue", "user equipment", "uplink",
            "downlink", "bandwidth", "spectrum", "transmission power", "latency", "urllc",
            "network slice", "traffic steering", "3gpp", "core network",
        ],
        "negative_terms": [
            "cloud computing", "data center", "datacenter", "kubernetes", "container orchestration",
            "serverless", "virtual machine", "microservice", "distributed system", "devops",
        ],
    },
    "ris": {
        "display_name": "Reconfigurable Intelligent Surfaces",
        "include": (
            "RIS, intelligent reflecting surface (IRS), large intelligent surface (LIS), reconfigurable metasurface, "
            "programmable metasurface, holographic MIMO/radio, metasurface unit cells, reflecting elements, "
            "passive beamforming, reflective beamforming, cascaded channel estimation, BS-RIS-UE link, "
            "star-ris, intelligent omni-surface (IOS), and radio propagation control."
        ),
        "exclude": (
            "optical, optics, photonic, photovoltaic, solar, acoustic, acoustics, seismic, thermal, "
            "mechanical, or pure material synthesis without wireless communication context."
        ),
        "keywords": [
            "ris", "reconfigurable intelligent surface", "reconfigurable intelligent surfaces",
            "intelligent reflecting surface", "intelligent reflecting surfaces", "irs",
            "large intelligent surface", "large intelligent surfaces", "lis",
            "smart radio environment", "smart electromagnetic environment",
            "reconfigurable metasurface", "programmable metasurface",
            "holographic mimo", "holographic radio", "holographic surface",
            "metasurface", "metamaterial", "meta-atom", "unit cell",
            "reflecting element", "passive element", "nearly passive surface",
            "passive beamforming", "reflective beamforming", "joint active and passive beamforming",
            "bs-ris-ue", "cascaded channel", "cascaded csi", "ris channel",
            "star-ris", "intelligent omni-surface", "ios",
            "ris phase optimization", "passive precoding", "generalized snell"
        ],
        "strong_terms": [
            "reconfigurable intelligent surface", "intelligent reflecting surface",
            "large intelligent surface", "ris-assisted", "irs-assisted",
            "passive beamforming", "cascaded channel", "ris channel estimation"
        ],
        "anchors": [
            "wireless", "radio", "cellular", "telecom", "5g", "5g-advanced", "6g",
            "nr", "ran", "gnb", "ue", "uplink", "downlink", "mimo", "beamforming",
            "propagation", "3gpp"
        ],
    },
    "semcom": {
        "display_name": "Semantic Communications",
        "include": (
            "Semantic communication, SemCom, joint source-channel coding (JSCC), goal-oriented communication, "
            "task-oriented communication, semantic encoding/decoding, semantic fidelity, task accuracy, "
            "semantic metrics, knowledge base integration, context-aware communication, and AI-native air interface."
        ),
        "exclude": (
            "generic NLP, chatbot, recommendation system, question answering, search engine, text mining, "
            "document retrieval, or pure AI/ML without telecom communication linkage."
        ),
        "keywords": [
            "semantic communication", "semantic communications", "semcom",
            "semantic-aware", "semantic-native", "goal-oriented communication",
            "task-oriented communication", "intent-oriented communication",
            "meaning transmission", "joint source-channel coding", "jscc",
            "deep jscc", "neural jscc", "semantic jscc",
            "semantic source coding", "semantic channel coding",
            "semantic encoder", "semantic decoder", "semantic codec",
            "semantic similarity", "semantic fidelity", "semantic distortion",
            "semantic capacity", "semantic rate", "semantic spectral efficiency",
            "meaning recovery", "context-aware communication", "pragmatic communication",
            "ai-native air interface", "foundation model for communication"
        ],
        "strong_terms": [
            "semantic communication", "semantic communications", "semcom",
            "goal-oriented communication", "task-oriented communication",
            "semantic encoder", "semantic decoder", "semantic jscc",
            "wireless semantic communication"
        ],
        "anchors": [
            "wireless", "radio", "cellular", "telecom", "5g", "5g-advanced", "6g",
            "nr", "ran", "gnb", "ue", "uplink", "downlink", "channel", "fading",
            "snr", "bandwidth", "spectrum", "3gpp"
        ],
    },
}


# ----------------------------
# Keyword loading/filtering
# ----------------------------

def load_keywords(domain: str, keywords_file: Optional[str]) -> List[str]:
    if keywords_file:
        with open(keywords_file, "r", encoding="utf-8") as f:
            if keywords_file.endswith(".json"):
                data = json.load(f)
                if isinstance(data, dict):
                    kws = data.get(domain) or data.get(domain.lower()) or data.get("keywords")
                    if not isinstance(kws, list):
                        raise ValueError("JSON keywords file must contain a list under domain name or 'keywords'.")
                    return [str(x) for x in kws]
                if isinstance(data, list):
                    return [str(x) for x in data]
                raise ValueError("JSON keywords file must be a list or dict.")
            return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    profile = DOMAIN_PROFILES.get(domain.lower())
    if not profile:
        return []
    return list(profile["keywords"])


def contains_any(text: str, terms: List[str], use_word_boundaries: bool = True) -> bool:
    keep, _ = keyword_filter(text, terms, min_keyword_hits=1, use_word_boundaries=use_word_boundaries)
    return keep


def keyword_filter(
    text: str,
    keywords: List[str],
    min_keyword_hits: int,
    use_word_boundaries: bool,
) -> Tuple[bool, List[str]]:
    hits = []
    lowered = text.lower()

    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue

        kw_lower = kw.lower()

        if use_word_boundaries and re.fullmatch(r"[a-zA-Z0-9_\-]+", kw_lower):
            pattern = r"(?<![a-zA-Z0-9_])" + re.escape(kw_lower) + r"(?![a-zA-Z0-9_])"
            if re.search(pattern, lowered):
                hits.append(kw)
        else:
            if kw_lower in lowered:
                hits.append(kw)

    return len(hits) >= min_keyword_hits, hits


def domain_keyword_filter(
    text: str,
    domain: str,
    keywords: List[str],
    min_keyword_hits: int,
    use_word_boundaries: bool,
) -> Tuple[bool, List[str], str]:
    """
    Domain-aware cheap filter.

    Strong domain terms pass directly.
    Broad terms require telecom anchors.
    Negative terms suppress examples unless strong terms are present.
    """
    profile = DOMAIN_PROFILES.get(domain.lower())
    basic_keep, hits = keyword_filter(text, keywords, min_keyword_hits, use_word_boundaries)

    if profile is None:
        return basic_keep, hits, "generic_keyword_filter"

    strong_terms = profile.get("strong_terms", [])
    broad_terms = profile.get("broad_terms", [])
    anchors = profile.get("anchors", [])
    negative_terms = profile.get("negative_terms", [])

    has_strong = contains_any(text, strong_terms, use_word_boundaries)
    has_broad = contains_any(text, broad_terms, use_word_boundaries)
    has_anchor = contains_any(text, anchors, use_word_boundaries)
    has_negative = contains_any(text, negative_terms, use_word_boundaries)

    if has_strong:
        return True, hits, "strong_domain_term"

    if has_negative and not has_anchor:
        return False, hits, "negative_without_telecom_anchor"

    if has_broad:
        return basic_keep and has_anchor, hits, "broad_term_requires_telecom_anchor"

    return basic_keep, hits, "domain_keyword_filter"


# ----------------------------
# TeleQnA normalization
# ----------------------------

OPTION_RE = re.compile(r"\boption\s*([0-9]+)\b", re.IGNORECASE)


def find_nested_question_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    if "question" in row or "Question" in row:
        return row

    nested_values = [v for v in row.values() if isinstance(v, dict)]
    for v in nested_values:
        if "question" in v or "Question" in v:
            return v

    return row


def normalize_option_key(key: str) -> Optional[int]:
    k = str(key).strip().lower()
    m = re.match(r"^(?:option|choice|answer)[\s_\-]*([0-9]+)$", k)
    if m:
        return int(m.group(1))
    return None


def collect_options(row: Dict[str, Any]) -> List[Tuple[int, str]]:
    options: Dict[int, str] = {}

    for key, value in row.items():
        idx = normalize_option_key(str(key))
        if idx is not None and value not in (None, ""):
            options[idx] = str(value)

    for opt_container_key in ["options", "Options", "choices", "Choices", "answers", "Answers"]:
        if opt_container_key not in row:
            continue

        container = row[opt_container_key]

        if isinstance(container, list):
            for i, value in enumerate(container, start=1):
                if value not in (None, ""):
                    options.setdefault(i, str(value))

        elif isinstance(container, dict):
            has_zero_key = any(str(k).strip() == "0" for k in container.keys())

            for key, value in container.items():
                if value in (None, ""):
                    continue

                idx = None
                try:
                    idx = int(key)
                except Exception:
                    idx = normalize_option_key(str(key))

                if idx is None:
                    continue

                if has_zero_key:
                    idx = idx + 1

                options.setdefault(idx, str(value))

    return sorted(options.items(), key=lambda x: x[0])


def extract_option_id(text: Any, valid_options: Optional[List[int]] = None) -> Optional[int]:
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    m = OPTION_RE.search(s)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options:
            return val

    m = re.search(r"^\s*(?:answer\s*[:\-]?\s*)?([0-9]+)\b", s, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options:
            return val

    m = re.search(r"\b([0-9]+)\b", s)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options:
            return val

    return None


def extract_gold_option(answer: Any, valid_options: List[int]) -> Optional[int]:
    if answer is None:
        return None

    parsed = extract_option_id(answer, valid_options=valid_options)
    if parsed is not None:
        return parsed

    s = str(answer).strip()
    try:
        numeric = int(float(s))
    except Exception:
        return None

    if numeric in valid_options:
        return numeric

    if numeric == 0 and valid_options and min(valid_options) == 1:
        return 1

    shifted = numeric + 1
    if shifted in valid_options:
        return shifted

    return None


def normalize_teleqna_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = find_nested_question_dict(row)

    question = (
        row.get("question")
        or row.get("Question")
        or row.get("prompt")
        or row.get("Prompt")
    )
    if question is None:
        raise ValueError(f"Could not find question in row keys: {list(row.keys())}")

    options = collect_options(row)
    valid_options = [i for i, _ in options]

    answer = (
        row.get("answer")
        if "answer" in row
        else row.get("Answer", row.get("label", row.get("Label", None)))
    )

    gold_option = extract_gold_option(answer, valid_options)

    category = str(row.get("category", row.get("Category", "unknown")))
    explanation = str(row.get("explanation", row.get("Explanation", "")))

    return {
        "question": str(question),
        "options": options,
        "valid_options": valid_options,
        "answer": "" if answer is None else str(answer),
        "gold_option": gold_option,
        "category": category,
        "explanation": explanation,
        "raw": row,
    }


def format_options(options: List[Tuple[int, str]]) -> str:
    if not options:
        return "(no options found)"
    return "\n".join(f"option {i}: {txt}" for i, txt in options)


def item_to_filter_text(item: Dict[str, Any], include_answer: bool = True) -> str:
    parts = [item["question"], format_options(item["options"])]

    if include_answer:
        parts.append(f"answer: {item['answer']}")
        if item.get("explanation"):
            parts.append(f"explanation: {item['explanation']}")

    return "\n".join(parts)


# ----------------------------
# Confidence-only LLM judge
# ----------------------------

CONFIDENCE_JUDGE_SYSTEM = (
    "You are a strict wireless communications domain relevance classifier. "
    "You output only one numeric confidence score."
)

CONFIDENCE_JUDGE_USER = """Decide whether the following TeleQnA multiple-choice Q/A pair is relevant to the domain: {domain_display_name}.

Target domain includes:
{domain_include}

Target domain excludes:
{domain_exclude}

Scoring:
- 1.0 = clearly and substantively relevant to the target domain.
- 0.0 = not relevant to the target domain.
- Use values between 0.0 and 1.0 for partial or uncertain relevance.
- Prefer precision over recall.
- Do not give high scores for generic telecom terms unless there is a clear domain connection.

Return ONLY one floating-point number between 0 and 1.
No explanation. No JSON. No text.

QUESTION:
<<<
{question}
>>>

OPTIONS:
<<<
{options}
>>>

GOLD ANSWER:
<<<
{answer}
>>>

EXPLANATION:
<<<
{explanation}
>>>
"""


def parse_confidence_only(text: str) -> float:
    cleaned = text.strip()

    try:
        score = float(cleaned)
        return max(0.0, min(1.0, score))
    except Exception:
        pass

    for token in cleaned.replace("\n", " ").replace(",", " ").split():
        token = token.strip()
        try:
            score = float(token)
            return max(0.0, min(1.0, score))
        except ValueError:
            continue

    raise ValueError(f"Could not parse confidence score from judge output: {cleaned[:500]}")


def openai_confidence_judge(
    client,
    model: str,
    item: Dict[str, Any],
    domain: str,
    max_chars: int,
    max_tokens: int,
) -> Dict[str, Any]:
    profile = DOMAIN_PROFILES.get(domain.lower(), {})
    domain_display_name = profile.get("display_name", domain)
    domain_include = profile.get("include", f"technical content relevant to {domain}")
    domain_exclude = profile.get("exclude", "unrelated, generic, marketing, or nontechnical content")

    options_text = format_options(item.get("options", []))

    user_msg = CONFIDENCE_JUDGE_USER.format(
        domain_display_name=domain_display_name,
        domain_include=domain_include,
        domain_exclude=domain_exclude,
        question=item.get("question", "")[:max_chars],
        options=options_text[:max_chars],
        answer=str(item.get("answer", ""))[:max_chars],
        explanation=str(item.get("explanation", ""))[:max_chars],
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": CONFIDENCE_JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_output_tokens=max_tokens,
    )

    raw = resp.output_text.strip()
    confidence = parse_confidence_only(raw)

    return {
        "confidence": confidence,
        "raw_output": raw,
    }


# ----------------------------
# Output
# ----------------------------

def make_output_record(
    line_no: int,
    item: Dict[str, Any],
    raw_row: Dict[str, Any],
    domain: str,
    matched_keywords: List[str],
    keyword_reason: str,
    judge_confidence: Optional[float],
    output_format: str,
) -> Dict[str, Any]:
    filter_meta = {
        "domain": domain,
        "matched_keywords": matched_keywords,
        "keyword_reason": keyword_reason,
        "judge_confidence": judge_confidence,
    }

    if output_format == "raw":
        rec = dict(raw_row)
        rec["_domain_filter"] = filter_meta
        return rec

    if output_format == "prompt_completion":
        gold_text = None
        if item["gold_option"] is not None:
            opt_map = dict(item["options"])
            gold_text = opt_map.get(item["gold_option"])

        return {
            "prompt": item["question"],
            "completion": gold_text or item["answer"],
            "options": {f"option {i}": txt for i, txt in item["options"]},
            "gold_option": item["gold_option"],
            "answer": item["answer"],
            "category": item["category"],
            "explanation": item["explanation"],
            "_domain_filter": filter_meta,
        }

    if output_format != "evaluator":
        raise ValueError(f"Unknown output format: {output_format}")

    return {
        "line_no": line_no,
        "question": item["question"],
        "options": {f"option {i}": txt for i, txt in item["options"]},
        "valid_options": item["valid_options"],
        "answer": item["answer"],
        "gold_option": item["gold_option"],
        "category": item["category"],
        "explanation": item["explanation"],
        "_domain_filter": filter_meta,
    }


def log_progress(
    scanned: int,
    normalized: int,
    keyword_kept: int,
    final_kept: int,
    skipped_bad: int,
    skipped_no_options: int,
    skipped_no_gold: int,
    llm_kept: int,
    llm_rejected: int,
    llm_failed: int,
    t0: float,
):
    elapsed = time.perf_counter() - t0
    rate = scanned / elapsed if elapsed > 0 else 0.0

    logging.info(
        "Progress | scanned=%d normalized=%d keyword_kept=%d final_kept=%d "
        "skipped_bad=%d skipped_no_options=%d skipped_no_gold=%d "
        "llm_kept=%d llm_rejected=%d llm_failed=%d rate=%.2f ex/s elapsed=%.1fs",
        scanned,
        normalized,
        keyword_kept,
        final_kept,
        skipped_bad,
        skipped_no_options,
        skipped_no_gold,
        llm_kept,
        llm_rejected,
        llm_failed,
        rate,
        elapsed,
    )


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter netop/TeleQnA into a domain-specific subset. "
            "Supports ISAC, SAGIN, and JCC with built-in keywords plus optional confidence-only LLM judge."
        )
    )

    parser.add_argument("--dataset-name", default="netop/TeleQnA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-path", default=None, help="Optional local TeleQnA-style JSONL instead of HF dataset")

    parser.add_argument("--domain", required=True, choices=["ISAC", "isac", "SAGIN", "sagin", "JCC", "jcc", "RIS", "ris", "SEMCOM", "semcom"])
    parser.add_argument("--keywords-file", default=None, help="Optional .txt or .json keyword file")

    parser.add_argument("--output", required=True)
    parser.add_argument("--output-format", default="evaluator", choices=["evaluator", "raw", "prompt_completion"])

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-kept", type=int, default=None)
    parser.add_argument("--min-keyword-hits", type=int, default=1)
    parser.add_argument("--no-word-boundaries", action="store_true")
    parser.add_argument("--filter-without-answer", action="store_true")
    parser.add_argument("--allow-missing-options", action="store_true")
    parser.add_argument("--allow-missing-gold", action="store_true")

    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--judge-max-chars", type=int, default=5000)
    parser.add_argument("--judge-max-tokens", type=int, default=16)
    parser.add_argument("--min-judge-confidence", type=float, default=0.7)
    parser.add_argument("--sleep-sec", type=float, default=0.0)

    parser.add_argument("--log-file", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-first", type=int, default=5)

    args = parser.parse_args()

    domain = args.domain.lower()
    setup_logging(args.log_file, args.verbose)

    keywords = load_keywords(domain, args.keywords_file)
    if not keywords:
        raise ValueError(f"No keywords found for domain={args.domain}. Pass --keywords-file or use ISAC/SAGIN/JCC.")

    logging.info("Starting TeleQnA domain filtering")
    logging.info("Dataset=%s split=%s data_path=%s", args.dataset_name, args.split, args.data_path)
    logging.info("Domain=%s | keywords=%d | keywords_file=%s", domain.upper(), len(keywords), args.keywords_file)
    logging.info("Output=%s | output_format=%s", args.output, args.output_format)
    logging.info("LLM judge=%s | model=%s | threshold=%.3f", args.use_llm_judge, args.judge_model, args.min_judge_confidence)
    logging.info(
        "Keyword filter | min_hits=%d | word_boundaries=%s | include_answer=%s",
        args.min_keyword_hits,
        not args.no_word_boundaries,
        not args.filter_without_answer,
    )

    client = None
    if args.use_llm_judge:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set, required for --use-llm-judge.")
        from openai import OpenAI
        client = OpenAI()

    if args.data_path:
        rows = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if line:
                    rows.append((line_no, json.loads(line)))
        logging.info("Loaded %d rows from local JSONL", len(rows))
    else:
        ds = load_dataset(args.dataset_name, split=args.split)
        rows = [(i + 1, dict(row)) for i, row in enumerate(ds)]
        logging.info("Loaded %d rows from HF dataset", len(rows))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    scanned = 0
    normalized = 0
    keyword_kept = 0
    final_kept = 0
    skipped_bad = 0
    skipped_no_options = 0
    skipped_no_gold = 0
    llm_kept = 0
    llm_rejected = 0
    llm_failed = 0

    t0 = time.perf_counter()

    with open(args.output, "w", encoding="utf-8") as out_f:
        for line_no, raw_row in rows:
            if args.limit is not None and scanned >= args.limit:
                logging.info("Reached --limit=%d", args.limit)
                break

            if args.max_kept is not None and final_kept >= args.max_kept:
                logging.info("Reached --max-kept=%d", args.max_kept)
                break

            scanned += 1

            try:
                item = normalize_teleqna_row(raw_row)
                normalized += 1
            except Exception as e:
                skipped_bad += 1
                logging.debug("Normalize failed | line=%s | error=%s", line_no, e)
                continue

            if not args.allow_missing_options and not item["options"]:
                skipped_no_options += 1
                logging.debug("Skipping missing options | line=%s", line_no)
                continue

            if not args.allow_missing_gold and item["gold_option"] is None:
                skipped_no_gold += 1
                logging.debug("Skipping missing gold | line=%s | answer=%s", line_no, item["answer"])
                continue

            filter_text = item_to_filter_text(item, include_answer=not args.filter_without_answer)
            kw_keep, matched_keywords, keyword_reason = domain_keyword_filter(
                text=filter_text,
                domain=domain,
                keywords=keywords,
                min_keyword_hits=args.min_keyword_hits,
                use_word_boundaries=not args.no_word_boundaries,
            )

            if args.debug and scanned <= args.debug_first:
                logging.info("==== DEBUG FILTER ====")
                logging.info("line_no=%s", line_no)
                logging.info("question=%s", item["question"])
                logging.info("options=%s", item["options"])
                logging.info("answer=%s gold_option=%s", item["answer"], item["gold_option"])
                logging.info("kw_keep=%s reason=%s matched_keywords=%s", kw_keep, keyword_reason, matched_keywords)
                logging.info("==== END DEBUG FILTER ====")

            if not kw_keep:
                if args.log_every > 0 and scanned % args.log_every == 0:
                    log_progress(
                        scanned, normalized, keyword_kept, final_kept,
                        skipped_bad, skipped_no_options, skipped_no_gold,
                        llm_kept, llm_rejected, llm_failed, t0,
                    )
                continue

            keyword_kept += 1
            judge_confidence = None

            if args.use_llm_judge:
                try:
                    judge_t0 = time.perf_counter()
                    judge_result = openai_confidence_judge(
                        client=client,
                        model=args.judge_model,
                        item=item,
                        domain=domain,
                        max_chars=args.judge_max_chars,
                        max_tokens=args.judge_max_tokens,
                    )
                    judge_time = time.perf_counter() - judge_t0
                    judge_confidence = float(judge_result["confidence"])

                    logging.debug(
                        "LLM judge | line=%s | confidence=%.3f | raw=%s | time=%.2fs",
                        line_no,
                        judge_confidence,
                        judge_result.get("raw_output", ""),
                        judge_time,
                    )

                except Exception as e:
                    llm_failed += 1
                    logging.warning("LLM judge failed | line=%s | error=%s", line_no, e)
                    continue

                if judge_confidence < args.min_judge_confidence:
                    llm_rejected += 1
                    continue

                llm_kept += 1

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            rec = make_output_record(
                line_no=line_no,
                item=item,
                raw_row=raw_row,
                domain=domain.upper(),
                matched_keywords=matched_keywords,
                keyword_reason=keyword_reason,
                judge_confidence=judge_confidence,
                output_format=args.output_format,
            )

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            final_kept += 1

            logging.debug("Kept | line=%s | final_kept=%d", line_no, final_kept)

            if args.log_every > 0 and scanned % args.log_every == 0:
                log_progress(
                    scanned, normalized, keyword_kept, final_kept,
                    skipped_bad, skipped_no_options, skipped_no_gold,
                    llm_kept, llm_rejected, llm_failed, t0,
                )

    summary = {
        "dataset_name": args.dataset_name,
        "split": args.split,
        "data_path": args.data_path,
        "domain": domain.upper(),
        "keywords_file": args.keywords_file,
        "num_keywords": len(keywords),
        "output": args.output,
        "output_format": args.output_format,
        "use_llm_judge": args.use_llm_judge,
        "judge_model": args.judge_model if args.use_llm_judge else None,
        "min_judge_confidence": args.min_judge_confidence if args.use_llm_judge else None,
        "scanned": scanned,
        "normalized": normalized,
        "keyword_kept": keyword_kept,
        "llm_kept": llm_kept if args.use_llm_judge else None,
        "llm_rejected": llm_rejected if args.use_llm_judge else None,
        "llm_failed": llm_failed if args.use_llm_judge else None,
        "final_kept": final_kept,
        "skipped_bad": skipped_bad,
        "skipped_no_options": skipped_no_options,
        "skipped_no_gold": skipped_no_gold,
    }

    summary_path = args.output + ".summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_progress(
        scanned, normalized, keyword_kept, final_kept,
        skipped_bad, skipped_no_options, skipped_no_gold,
        llm_kept, llm_rejected, llm_failed, t0,
    )

    logging.info("Summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("Wrote subset to: %s", args.output)
    logging.info("Wrote summary to: %s", summary_path)


if __name__ == "__main__":
    main()