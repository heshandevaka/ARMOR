from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Tuple
import importlib

import orjson
import jsonlines
from diskcache import Cache
from datasets import load_dataset

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils_filter import (
    stable_hash, parse_metadata,
    normalize_text, chunk_text, fails_local_ref,
    parse_json_loose, infer_difficulty, safe_float,
)

# --- MOD (1): OpenAI clean-doc post-filter (imports + helpers) ---
import math
import random
import re
import time
from openai import OpenAI
# --- end MOD (1) ---

# DOMAIN default, will be overwritten by domain config
DOMAIN = "UNKNOWN"

def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

class LocalChatLLM:
    def __init__(self, model_name: str):
        tp = env_int("TP_SIZE", 1)
        gpu_mem_util = env_float("GPU_MEMORY_UTILIZATION", 0.90)
        max_model_len = env_int("MAX_MODEL_LEN", 8192)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
        )

        self.params = SamplingParams(
            temperature=env_float("TEMPERATURE", 0.2),
            top_p=env_float("TOP_P", 0.9),
            max_tokens=env_int("MAX_TOKENS", 900),
        )

    def _chat_prompt(self, system: str, user: str) -> str:
        return (
            "<|begin_of_text|>\n"
            "<|start_header_id|>system<|end_header_id|>\n"
            f"{system}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )


    def generate_json_batch(self, system: str, users: List[str]) -> List[Dict[str, Any]]:
        prompts = [self._chat_prompt(system, u) for u in users]
        outs = self.llm.generate(prompts, self.params)
        results = []
        for o in outs:
            t = o.outputs[0].text if o.outputs else ""
            results.append(parse_json_loose(t) or {})
        return results

def iter_teledata(path: str):
    with jsonlines.open(path) as r:
        for obj in r:
            yield obj

def meta_hint(cat: str, meta: Dict[str, Any]) -> str:
    if cat == "standard":
        series = meta.get("series", meta.get("Series", ""))
        rel = meta.get("release", meta.get("Release", ""))
        fn = meta.get("file_name", meta.get("File_name", ""))
        return f"3GPP series={series} release={rel} file={fn}"
    if cat == "arxiv":
        return f"arxiv_id={meta.get('arxiv_id','')} title={meta.get('title','')}"
    if cat == "wiki":
        return f"title={meta.get('title','')} url={meta.get('url','')}"
    if cat == "web":
        return f"url={meta.get('url','')}"
    return str(meta)

def make_example(ex_id: int, doc: Dict[str, Any], chunk_index: int, task_type: str, item: Dict[str, Any], domain_name: str) -> Dict[str, Any]:
    prompt = item["prompt"]
    response = item["response"]
    ex = {
        "id": f"{domain_name.lower()}_{ex_id:08d}",
        "task_type": task_type,
        "domain": domain_name,
        "topic": item.get("topic", ""),
        "prompt": prompt,
        "response": response,
        "difficulty": infer_difficulty(prompt, response, task_type),
        "tags": item.get("tags", []),
        "source_id": doc["doc_id"],
        "source_category": doc["category"],
        "source_metadata": doc.get("metadata", {}),
        "provenance": {"chunk_index": chunk_index},
    }
    if "related_terms" in item:
        ex["related_terms"] = item["related_terms"]
    return ex

def keep_judged(j: Dict[str, Any], min_tech: float, min_clarity: float) -> bool:
    return (
        bool(j.get("keep"))
        and bool(j.get("answerable_without_context"))
        and float(j.get("technical_score", 0.0)) >= min_tech
        and float(j.get("clarity_score", 0.0)) >= min_clarity
    )

# --- MOD (1): OpenAI clean-doc post-filter (prompt + parser + filter) ---
_RELEVANCE_USER_SCORE_ONLY_TEMPLATE = """Decide whether this Tele-Data sample is suitable for a tutorial dataset focused on {domain_name}.
{domain_specific_instructions}

Return ONLY a single number between 0.0 and 1.0 (no JSON, no words):
- 1.0 = clear true positive for a {domain_name} tutorial dataset
- 0.0 = clear false positive

SAMPLE_ID: {sid}
CATEGORY: {cat}
META_HINT: {meta_hint}
CONTENT_SNIPPET:
<<<
{snippet}
>>>
"""

# Note: We'll keep the instructions hardcoded for ISAC in the module or pass them.
# For now, let's keep the ISAC-specific instruction in case it is ISAC.
ISAC_INSTRUCTIONS = """Include explanatory or instructional content about topics such as:
- ISAC / JCAS fundamentals: shared spectrum or hardware, dual-function radar–communication systems
- Joint waveform and signal design for communication and sensing (e.g., OFDM-based sensing, OTFS, pilot/PRS/SRS reuse)
- Sensing tasks in communication systems: range, Doppler, angle estimation, localization, tracking
- Joint beamforming, MIMO, and resource allocation for sensing–communication trade-offs
- Performance trade-offs between sensing accuracy and communication metrics (rate, latency, reliability)
- Standards-oriented or system-level discussion relevant to cellular or wireless ISAC (e.g., 3GPP, 5G-Advanced, 6G)

Include only content that is explanatory, instructional, or tutorial-like.
Exclude marketing material, vendor promotions, business or market analysis, and purely speculative vision papers without technical explanation."""

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _parse_score_only(text: str) -> float:
    t = (text or "").strip()
    m = _FLOAT_RE.search(t)
    if not m:
        raise ValueError(f"Could not parse float from model output: {t!r}")
    v = float(m.group(0))
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"Invalid float from model output: {t!r}")
    return max(0.0, min(1.0, v))

def _shorten(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 20].rstrip() + "\n...[TRUNCATED]..."

def _openai_doc_score(
    client: OpenAI,
    model_name: str,
    domain_name: str,
    sid: str,
    cat: str,
    mh: str,
    snippet: str,
    timeout_s: float,
    max_retries: int,
    base_backoff_s: float,
    relevance_system_prompt: str,
) -> float:
    instructions = ISAC_INSTRUCTIONS if domain_name == "ISAC" else "Include content relevant to the domain."
    prompt = _RELEVANCE_USER_SCORE_ONLY_TEMPLATE.format(
        domain_name=domain_name,
        domain_specific_instructions=instructions,
        sid=sid,
        cat=cat,
        meta_hint=mh,
        snippet=snippet,
    )

    attempt = 0
    while True:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": relevance_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                timeout=timeout_s,
            )
            return _parse_score_only(resp.choices[0].message.content)
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = base_backoff_s * (2 ** (attempt - 1))
            sleep_s *= (0.75 + random.random() * 0.5)
            time.sleep(sleep_s)

def _openai_filter_clean_docs(
    clean_docs: List[Dict[str, Any]],
    cache: Cache,
    openai_model: str,
    threshold: float,
    snippet_chars: int,
    timeout_s: float,
    max_retries: int,
    base_backoff_s: float,
    max_to_judge: int,
    domain_name: str,
    relevance_system_prompt: str,
) -> List[Dict[str, Any]]:
    client = OpenAI()
    kept: List[Dict[str, Any]] = []
    for i, d in enumerate(clean_docs):
        if max_to_judge > 0 and i >= max_to_judge:
            kept.extend(clean_docs[i:])
            break

        sid = str(d.get("doc_id", ""))
        cat = str(d.get("category", ""))
        mh = meta_hint(cat, d.get("metadata", {}) or {})
        snippet = _shorten(d.get("text", "") or "", snippet_chars)

        ck = stable_hash("oa_clean_doc_score", openai_model, sid, cat, mh[:200], snippet[:1200])
        if ck in cache:
            score = float(cache[ck])
        else:
            score = _openai_doc_score(
                client=client,
                model_name=openai_model,
                domain_name=domain_name,
                sid=sid,
                cat=cat,
                mh=mh,
                snippet=snippet,
                timeout_s=timeout_s,
                max_retries=max_retries,
                base_backoff_s=base_backoff_s,
                relevance_system_prompt=relevance_system_prompt,
            )
            cache[ck] = score

        if score >= threshold:
            kept.append(d)

    return kept
# --- end MOD (1) ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Tele-Data dataset path")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--model", required=True, help="HF model name or local path")
    ap.add_argument("--domain", default="isac", help="Domain folder to use (e.g. isac, medical)")

    # --- MOD (2): resume/bypass args ---
    ap.add_argument("--from_clean_docs", default=None)
    ap.add_argument("--from_prejudge_examples", default=None)
    ap.add_argument("--from_openai_filtered_clean_docs", default=None)
    # --- MOD (1): OpenAI args ---
    ap.add_argument("--openai_doc_filter_model", default=None)
    ap.add_argument("--openai_doc_filter_threshold", type=float, default=0.5)
    ap.add_argument("--openai_doc_filter_snippet_chars", type=int, default=2000)
    ap.add_argument("--openai_doc_filter_timeout_s", type=float, default=120.0)
    ap.add_argument("--openai_doc_filter_max_retries", type=int, default=6)
    ap.add_argument("--openai_doc_filter_base_backoff_s", type=float, default=0.75)
    ap.add_argument("--openai_doc_filter_max_to_judge", type=int, default=0)

    args = ap.parse_args()

    # Dynamic Domain Loading
    try:
        domain_cfg = importlib.import_module(f"domains.{args.domain}.config")
        domain_prompts = importlib.import_module(f"domains.{args.domain}.prompts")
    except ImportError as e:
        print(f"Error: Could not load domain '{args.domain}'. Ensure domains/{args.domain}/ exists. ({e})")
        return

    domain_name = domain_cfg.DOMAIN_NAME
    cheap_prefilter = domain_cfg.cheap_prefilter

    out_clean = os.path.join(args.out_dir, f"{args.domain}_clean_docs.jsonl")
    out_clean_openai = os.path.join(args.out_dir, f"{args.domain}_clean_docs_openai_filtered.jsonl")
    out_examples = os.path.join(args.out_dir, "examples.jsonl")
    cache_dir = os.path.join(args.out_dir, f".cache_{args.domain}_builder")

    min_doc_conf = env_float("MIN_DOC_CONF", 0.70)
    chunk_chars = env_int("CHUNK_CHARS", 8000)
    chunk_overlap = env_int("CHUNK_OVERLAP", 800)
    gen_bs = env_int("GEN_BATCH_SIZE", 20)
    judge_bs = env_int("JUDGE_BATCH_SIZE", 20)
    min_tech = env_float("MIN_TECH", 0.70)
    min_clarity = env_float("MIN_CLARITY", 0.70)

    cache = Cache(cache_dir)
    model = LocalChatLLM(args.model)

    clean_docs: List[Dict[str, Any]] = []

    if args.from_prejudge_examples is None:
        if args.from_openai_filtered_clean_docs is not None:
            with jsonlines.open(args.from_openai_filtered_clean_docs) as r:
                for d in r:
                    clean_docs.append(d)
        elif args.from_clean_docs is not None:
            with jsonlines.open(args.from_clean_docs) as r:
                for d in r:
                    clean_docs.append(d)
            if args.openai_doc_filter_model:
                clean_docs = _openai_filter_clean_docs(
                    clean_docs=clean_docs,
                    cache=cache,
                    openai_model=args.openai_doc_filter_model,
                    threshold=args.openai_doc_filter_threshold,
                    snippet_chars=args.openai_doc_filter_snippet_chars,
                    timeout_s=args.openai_doc_filter_timeout_s,
                    max_retries=args.openai_doc_filter_max_retries,
                    base_backoff_s=args.openai_doc_filter_base_backoff_s,
                    max_to_judge=args.openai_doc_filter_max_to_judge,
                    domain_name=domain_name,
                    relevance_system_prompt=domain_prompts.RELEVANCE_SYSTEM,
                )
                with jsonlines.open(out_clean_openai, "w") as w:
                    for d in clean_docs:
                        w.write(d)
        else:
            rel_jobs: List[Tuple[str, Dict[str, Any], str]] = []
            ds = load_dataset(args.input, split="train")

            for sample in ds:
                if not cheap_prefilter(sample, parse_metadata):
                    continue

                sid = sample.get("ID", sample.get("id", ""))
                cat = (sample.get("Category", sample.get("category", "")) or "").lower()
                content = sample.get("Content", sample.get("content", "")) or ""
                meta = parse_metadata(sample.get("Metadata", sample.get("metadata")))
                snippet = content[:2000]

                user = domain_prompts.RELEVANCE_USER.format(
                    sid=sid, cat=cat, meta_hint=meta_hint(cat, meta), snippet=snippet
                )
                key = stable_hash("rel", sid, cat, snippet[:1200])
                rel_jobs.append((key, sample, user))

            for i in range(0, len(rel_jobs), gen_bs):
                batch = rel_jobs[i:i+gen_bs]
                keys = [k for k, _, _ in batch]
                users = [u for _, _, u in batch]

                to_run = [idx for idx, k in enumerate(keys) if k not in cache]
                if to_run:
                    run_users = [users[idx] for idx in to_run]
                    outs = model.generate_json_batch(domain_prompts.RELEVANCE_SYSTEM, run_users)
                    for idx, out in zip(to_run, outs):
                        cache[keys[idx]] = out or {
                            "relevant": False, "confidence": 0.0, "main_topics": [], "notes": "parse_failed"
                        }

                for key, sample, _ in batch:
                    v = cache[key]
                    if not v.get("relevant"):
                        continue
                    conf = safe_float(v.get("confidence"), default=0.0)
                    if conf < min_doc_conf:
                        continue

                    sid = sample.get("ID", sample.get("id", ""))
                    cat = (sample.get("Category", sample.get("category", "")) or "").lower()
                    meta = parse_metadata(sample.get("Metadata", sample.get("metadata")))
                    text = normalize_text(sample.get("Content", sample.get("content", "")) or "")
                    if len(text) < 1200:
                        continue

                    clean_docs.append({
                        "doc_id": sid,
                        "category": cat,
                        "metadata": meta,
                        "text": text,
                        "filter_meta": v,
                    })

            with jsonlines.open(out_clean, "w") as w:
                for d in clean_docs:
                    w.write(d)

            if args.openai_doc_filter_model:
                clean_docs = _openai_filter_clean_docs(
                    clean_docs=clean_docs,
                    cache=cache,
                    openai_model=args.openai_doc_filter_model,
                    threshold=args.openai_doc_filter_threshold,
                    snippet_chars=args.openai_doc_filter_snippet_chars,
                    timeout_s=args.openai_doc_filter_timeout_s,
                    max_retries=args.openai_doc_filter_max_retries,
                    base_backoff_s=args.openai_doc_filter_base_backoff_s,
                    max_to_judge=args.openai_doc_filter_max_to_judge,
                    domain_name=domain_name,
                    relevance_system_prompt=domain_prompts.RELEVANCE_SYSTEM,
                )
                with jsonlines.open(out_clean_openai, "w") as w:
                    for d in clean_docs:
                        w.write(d)

    examples: List[Dict[str, Any]] = []
    ex_id = 0

    if args.from_prejudge_examples is not None:
        with jsonlines.open(args.from_prejudge_examples) as r:
            for ex in r:
                examples.append(ex)
    else:
        gen_users: List[str] = []
        gen_meta: List[Tuple[Dict[str, Any], int, str]] = []

        for doc in clean_docs:
            chunks = chunk_text(doc["text"], chunk_chars, chunk_overlap)
            for ci, ch in enumerate(chunks):
                if len(ch.strip()) < 800:
                    continue
                key = stable_hash("gen", doc["doc_id"], str(ci), ch[:1200])
                if key in cache:
                    gen = cache[key] or {}
                    for task_type, items in gen.items():
                        if not isinstance(items, list): continue
                        for item in items:
                            if not isinstance(item, dict) or "prompt" not in item or "response" not in item: continue
                            if fails_local_ref(item["prompt"], item["response"]): continue
                            ex_id += 1
                            examples.append(make_example(ex_id, doc, ci, task_type, item, domain_name))
                else:
                    gen_users.append(domain_prompts.GEN_USER.format(chunk=ch))
                    gen_meta.append((doc, ci, key))

        for i in range(0, len(gen_users), gen_bs):
            batch_users = gen_users[i:i+gen_bs]
            batch_meta = gen_meta[i:i+gen_bs]
            outs = model.generate_json_batch(domain_prompts.GEN_SYSTEM, batch_users)

            for out, (doc, ci, key) in zip(outs, batch_meta):
                if not out:
                    out = {"concept_qa": [], "procedure_explanation": [], "role_responsibility": [], "common_misconception": [], "definition": []}
                cache[key] = out
                for task_type, items in out.items():
                    if not isinstance(items, list): continue
                    for item in items:
                        if not isinstance(item, dict) or "prompt" not in item or "response" not in item: continue
                        if fails_local_ref(item["prompt"], item["response"]): continue
                        ex_id += 1
                        examples.append(make_example(ex_id, doc, ci, task_type, item, domain_name))

        out_prejudge = os.path.join(args.out_dir, f"{args.domain}_prejudge_examples.jsonl")
        with jsonlines.open(out_prejudge, "w") as w:
            for ex in examples:
                w.write(ex)

    final: List[Dict[str, Any]] = []
    judge_users: List[str] = []
    judge_meta: List[Tuple[Dict[str, Any], str]] = []

    for ex in examples:
        ex_json = orjson.dumps(ex).decode("utf-8")
        key = stable_hash("judge", ex["id"], ex_json)
        if key in cache:
            j = cache[key] or {}
            if keep_judged(j, min_tech, min_clarity):
                ex["quality"] = {"answerable_without_context": bool(j.get("answerable_without_context")), "technical_score": float(j.get("technical_score", 0.0)), "clarity_score": float(j.get("clarity_score", 0.0)), "issues": j.get("issues", [])}
                final.append(ex)
        else:
            judge_users.append(domain_prompts.JUDGE_USER.format(example_json=ex_json))
            judge_meta.append((ex, key))

    for i in range(0, len(judge_users), judge_bs):
        batch_users = judge_users[i:i+judge_bs]
        batch_meta = judge_meta[i:i+judge_bs]
        outs = model.generate_json_batch(domain_prompts.JUDGE_SYSTEM, batch_users)
        for out, (ex, key) in zip(outs, batch_meta):
            if not out:
                out = {"keep": False, "answerable_without_context": False, "technical_score": 0.0, "clarity_score": 0.0, "issues": ["parse_failed"]}
            cache[key] = out
            if keep_judged(out, min_tech, min_clarity):
                ex["quality"] = {"answerable_without_context": bool(out.get("answerable_without_context")), "technical_score": float(out.get("technical_score", 0.0)), "clarity_score": float(out.get("clarity_score", 0.0)), "issues": out.get("issues", [])}
                final.append(ex)

    with jsonlines.open(out_examples, "w") as w:
        for ex in final:
            w.write(ex)

    cache.close()
    print(f"Done. Final examples: {len(final)} -> {out_examples}")

if __name__ == "__main__":
    main()
