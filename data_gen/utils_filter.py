# utils_filter.py
from __future__ import annotations

import re
import json
import hashlib
from typing import Any, Dict, List, Optional
import orjson

# -------------------------
# Agnostic Helpers
# -------------------------

LOCAL_REF_PATTERNS = [
    r"\b(in|within) this (paper|section|chapter|document)\b",
    r"\bas shown (above|below)\b",
    r"\bfigure\s*\d+\b",
    r"\btable\s*\d+\b",
    r"\bsection\s*\d+(\.\d+)*\b",
]

def stable_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()

def parse_metadata(meta_field: Any) -> Dict[str, Any]:
    if meta_field is None:
        return {}
    if isinstance(meta_field, dict):
        return meta_field
    if isinstance(meta_field, (bytes, bytearray)):
        try:
            return orjson.loads(meta_field)
        except Exception:
            return {}
    if isinstance(meta_field, str):
        s = meta_field.strip()
        if not s:
            return {}
        try:
            obj = orjson.loads(s)
            return obj if isinstance(obj, dict) else {"_meta": obj}
        except Exception:
            return {"_meta": s}
    return {"_meta": str(meta_field)}

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u0000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []
    if chunk_chars <= 0:
        return [text]
    chunk_overlap = max(0, min(chunk_overlap, chunk_chars - 1))
    step = max(1, chunk_chars - chunk_overlap)
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_chars])
        i += step
    return chunks

def fails_local_ref(prompt: str, response: Any) -> bool:
    resp_str = response if isinstance(response, str) else " ".join(map(str, response)) if response is not None else ""
    blob = (prompt + " " + resp_str).lower()
    return any(re.search(p, blob) for p in LOCAL_REF_PATTERNS)

def extract_json_obj(text: str) -> Optional[str]:
    if not text:
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e + 1]

def parse_json_loose(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    t = t.replace("```json", "```").replace("```", "").strip()
    try:
        obj = orjson.loads(t)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    blob = extract_json_obj(t)
    if not blob:
        return None
    try:
        obj = orjson.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def infer_difficulty(prompt: str, response: str, task_type: str) -> str:
    p = (prompt or "").lower()
    r = (response or "").lower()
    n = len(p) + len(r)
    # Generic length-based/keyword heuristic
    if n > 2200:
        return "advanced"
    if task_type in ["procedure_explanation", "role_responsibility"]:
        return "intermediate"
    return "intermediate"

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default
