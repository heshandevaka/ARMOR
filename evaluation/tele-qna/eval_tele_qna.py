#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

OPTION_RE = re.compile(r"\boption\s*([0-9]+)\b", re.IGNORECASE)

# ----------------------------
# General utilities
# ----------------------------

def parse_dtype(dtype: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def needs_e5_prefix(model_name_or_path: str) -> bool:
    return model_name_or_path is not None and "e5" in model_name_or_path.lower()


def maybe_prefix_query(model_name_or_path: str, question: str) -> str:
    return ("query: " + question) if needs_e5_prefix(model_name_or_path) else question


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return device_arg


def format_prompt(user_content: str, system: Optional[str]) -> str:
    if system: return f"{system}\n\nUser: {user_content}\nAssistant:"
    return user_content


# ----------------------------
# TeleQnA parsing
# ----------------------------

def extract_option_id(text: Any, valid_options: Optional[List[int]] = None) -> Optional[int]:
    if text is None: return None
    s = str(text).strip()
    if not s: return None
    m = OPTION_RE.search(s)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options: return val
    m = re.search(r"^\s*N\s*[\)\:\-]?\s*([0-9]+)\b", s, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options: return val
    m = re.search(r"^\s*(?:answer\s*[:\-]?\s*)?([0-9]+)\b", s, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options: return val
    m = re.search(r"\b([0-9]+)\b", s)
    if m:
        val = int(m.group(1))
        if valid_options is None or val in valid_options: return val
    return None


def extract_gold_option(answer: Any, valid_options: List[int]) -> Optional[int]:
    if answer is None: return None
    s = str(answer).strip()
    parsed = extract_option_id(s, valid_options=valid_options)
    if parsed is not None: return parsed
    try: numeric = int(float(s))
    except Exception: return None
    if numeric in valid_options: return numeric
    if numeric == 0 and valid_options and min(valid_options) == 1: return 1
    shifted = numeric + 1
    if shifted in valid_options: return shifted
    return None


def find_nested_question_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    if "question" in row or "Question" in row: return row
    nested_values = [v for v in row.values() if isinstance(v, dict)]
    for v in nested_values:
        if "question" in v or "Question" in v: return v
    return row


def normalize_option_key(key: str) -> Optional[int]:
    k = str(key).strip().lower()
    m = re.match(r"^(?:option|choice|answer)[\s_\-]*([0-9]+)$", k)
    if m: return int(m.group(1))
    return None


def collect_options(row: Dict[str, Any]) -> List[Tuple[int, str]]:
    options: Dict[int, str] = {}
    for key, value in row.items():
        idx = normalize_option_key(str(key))
        if idx is not None and value not in (None, ""): options[idx] = str(value)
    for opt_container_key in ["options", "Options", "choices", "Choices", "answers", "Answers"]:
        if opt_container_key not in row: continue
        container = row[opt_container_key]
        if isinstance(container, list):
            for i, value in enumerate(container, start=1):
                if value not in (None, ""): options.setdefault(i, str(value))
        elif isinstance(container, dict):
            for key, value in container.items():
                if value in (None, ""): continue
                try: idx = int(key)
                except Exception: idx = normalize_option_key(str(key))
                if idx is None: continue
                if idx == 0: idx = 1
                options.setdefault(idx, str(value))
    return sorted(options.items(), key=lambda x: x[0])


def normalize_teleqna_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = find_nested_question_dict(row)
    question = row.get("question") or row.get("Question") or row.get("prompt") or row.get("Prompt")
    if question is None: raise ValueError(f"Could not find question field in row keys: {list(row.keys())}")
    options = collect_options(row)
    valid_options = [i for i, _ in options]
    answer = row.get("answer") if "answer" in row else row.get("Answer", row.get("label", row.get("Label", None)))
    gold_option = extract_gold_option(answer, valid_options)
    return {
        "question": str(question), "options": options, "valid_options": valid_options,
        "answer": "" if answer is None else str(answer), "gold_option": gold_option,
        "category": str(row.get("category", row.get("Category", "unknown"))),
        "explanation": str(row.get("explanation", row.get("Explanation", ""))), "raw": row,
    }


def format_options(options: List[Tuple[int, str]]) -> str:
    if not options: return "(no options found)"
    return "\n".join(f"option {i}: {text}" for i, text in options)


def valid_option_instruction(options: List[Tuple[int, str]]) -> str:
    valid = [i for i, _ in options]
    if not valid: return "Return exactly one answer in the form: option <number>. Do not explain."
    choices = ", ".join(f"option {i}" for i in valid)
    return f"Return exactly one of: {choices}. Do not explain."


# ----------------------------
# Query encoder & SQLite store
# ----------------------------

class QueryEncoder:
    def __init__(self, model_path: str, device: str = "cuda", trust_remote_code: bool = False):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=trust_remote_code).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str], normalize_embeddings: bool = True, max_length: int = 256) -> np.ndarray:
        batch = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        if normalize_embeddings: emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)


class AutoChunkStore:
    def __init__(self, sqlite_path: str, table: Optional[str] = None, id_col: Optional[str] = None, text_col: Optional[str] = None):
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.table, self.col_vid, self.col_text = self._discover(table, id_col, text_col)
        self._all_cols = self._list_columns(self.table)

    def close(self): self.conn.close()

    def _list_objects(self):
        cur = self.conn.cursor()
        return cur.execute("SELECT type, name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name").fetchall()

    def _list_columns(self, obj_name: str) -> List[str]:
        cur = self.conn.cursor()
        rows = cur.execute(f'PRAGMA table_info("{obj_name}")').fetchall()
        return [r[1] for r in rows]

    def _discover(self, table: Optional[str], id_col: Optional[str], text_col: Optional[str]):
        objects = self._list_objects()
        if not objects: raise RuntimeError("No user tables/views found in sqlite DB.")
        if table is not None:
            cols = self._list_columns(table)
            cvid = id_col or self._guess_id_col(cols)
            ctext = text_col or self._guess_text_col(cols)
            if cvid is None or ctext is None: raise RuntimeError(f"Could not infer id/text columns for sqlite object '{table}'.")
            return table, cvid, ctext
        best, best_score = None, -1
        for _, name in objects:
            cols = self._list_columns(name)
            cvid, ctext = self._guess_id_col(cols), self._guess_text_col(cols)
            score = 0
            if cvid is not None: score += 2
            if ctext is not None: score += 2
            if "vid" in [c.lower() for c in cols]: score += 2
            if "text" in [c.lower() for c in cols]: score += 2
            if score > best_score and cvid is not None and ctext is not None:
                best, best_score = (name, cvid, ctext), score
        if best is None: raise RuntimeError("Could not auto-discover passage store in sqlite DB.")
        return best

    @staticmethod
    def _guess_id_col(cols: List[str]) -> Optional[str]:
        preferred = ["vid", "id", "rowid", "chunk_id", "doc_id"]
        lower_to_orig = {c.lower(): c for c in cols}
        for c in preferred:
            if c in lower_to_orig: return lower_to_orig[c]
        return None

    @staticmethod
    def _guess_text_col(cols: List[str]) -> Optional[str]:
        preferred = ["text", "body", "content", "chunk", "passage"]
        lower_to_orig = {c.lower(): c for c in cols}
        for c in preferred:
            if c in lower_to_orig: return lower_to_orig[c]
        return None

    def fetch_by_vids(self, vids: List[int]) -> List[Dict[str, Any]]:
        if not vids: return []
        wanted_meta = [c for c in ["doc_id", "category", "chunk_id", "metadata_json"] if c in self._all_cols]
        select_cols = [self.col_vid, self.col_text] + wanted_meta
        qcols = ", ".join([f'"{c}"' for c in select_cols])
        ph = ",".join(["?"] * len(vids))
        sql = f'SELECT {qcols} FROM "{self.table}" WHERE "{self.col_vid}" IN ({ph})'
        cur = self.conn.cursor()
        rows = cur.execute(sql, [int(v) for v in vids]).fetchall()
        by_vid = {}
        for row in rows:
            rec = dict(row)
            meta = {}
            if "metadata_json" in rec and rec["metadata_json"]:
                try: meta = json.loads(rec["metadata_json"])
                except Exception: meta = {"metadata_json": rec["metadata_json"]}
            vid = int(rec[self.col_vid])
            by_vid[vid] = {"vid": vid, "text": rec[self.col_text], "doc_id": rec.get("doc_id", str(vid)), "category": rec.get("category", ""), "chunk_id": rec.get("chunk_id", str(vid)), "metadata": meta}
        return [by_vid[v] for v in vids if v in by_vid]


# ----------------------------
# Generation
# ----------------------------

@torch.inference_mode()
def generate_vanilla(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature if do_sample else None, top_p=top_p if do_sample else None, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@torch.inference_mode()
def _next_token_logprobs_batch(model, tokenizer, prompts: List[str], max_len: int) -> torch.Tensor:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
    out = model(**inputs)
    logits = out.logits
    last_pos = inputs["attention_mask"].sum(dim=1) - 1
    next_logits = logits[torch.arange(logits.size(0), device=logits.device), last_pos, :]
    return torch.log_softmax(next_logits, dim=-1)


def _top_p_filtering(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0: return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum > top_p
    mask[..., 0] = False
    sorted_probs[mask] = 0.0
    denom = sorted_probs.sum().clamp(min=1e-12)
    sorted_probs = sorted_probs / denom
    out = torch.zeros_like(probs)
    out[sorted_idx] = sorted_probs
    return out


@torch.inference_mode()
def generate_replug(model, tokenizer, base_user_contents: List[str], system: Optional[str], scores: np.ndarray, max_new_tokens: int, temperature: float, top_p: float, max_len: int) -> str:
    if len(base_user_contents) == 0:
        return generate_vanilla(model, tokenizer, format_prompt("", system), max_new_tokens, temperature, top_p)
    s = torch.tensor(scores[:len(base_user_contents)], dtype=torch.float32, device="cpu")
    lam = torch.softmax(s, dim=0)
    generated, do_sample = "", temperature > 0
    for _ in range(max_new_tokens):
        prompts = [format_prompt(uc + generated, system) for uc in base_user_contents]
        logp_each = _next_token_logprobs_batch(model, tokenizer, prompts, max_len=max_len)
        p_each = logp_each.exp()
        p_ens = (lam.to(p_each.device).unsqueeze(-1) * p_each).sum(dim=0)
        p_ens = p_ens / p_ens.sum().clamp(min=1e-12)
        if do_sample:
            if temperature != 1.0: p_ens = torch.softmax(torch.log(p_ens + 1e-12) / max(temperature, 1e-6), dim=-1)
            p_ens = _top_p_filtering(p_ens, top_p)
            next_id = torch.multinomial(p_ens, num_samples=1).item()
        else: next_id = torch.argmax(p_ens).item()
        generated += tokenizer.decode([next_id], skip_special_tokens=True)
        if next_id == tokenizer.eos_token_id: break
    return generated.strip()


# ----------------------------
# Prompt builders & Model loading
# ----------------------------

def build_single_doc_training_style_prompt(question: str, options: List[Tuple[int, str]], document: str, qa_prompt_prefix: str, doc_prefix: str, question_prefix: str, answer_prefix: str, section_sep: str) -> str:
    return qa_prompt_prefix + doc_prefix + document + section_sep + question_prefix + question + section_sep + "Options:\n" + format_options(options) + section_sep + valid_option_instruction(options) + section_sep + answer_prefix


def build_multi_doc_rag_prompt(question: str, options: List[Tuple[int, str]], contexts: List[Dict[str, Any]], max_context_chars: int, qa_prompt_prefix: str, doc_prefix: str, question_prefix: str, answer_prefix: str, section_sep: str) -> str:
    blocks, total = [], 0
    for i, c in enumerate(contexts, start=1):
        block = f"[{i}] doc_id={c.get('doc_id','')} category={c.get('category','')} chunk_id={c.get('chunk_id','')}\n" + c["text"] + "\n\n"
        if total + len(block) > max_context_chars: break
        blocks.append(block)
        total += len(block)
    return qa_prompt_prefix + doc_prefix + ("".join(blocks).strip() or "(no context)") + section_sep + question_prefix + question + section_sep + "Options:\n" + format_options(options) + section_sep + valid_option_instruction(options) + section_sep + answer_prefix


def build_closed_book_prompt(question: str, options: List[Tuple[int, str]], qa_prompt_prefix: str, question_prefix: str, answer_prefix: str, section_sep: str) -> str:
    return qa_prompt_prefix + question_prefix + question + section_sep + "Options:\n" + format_options(options) + section_sep + valid_option_instruction(options) + section_sep + answer_prefix


def load_generator_model(model_path: str, tokenizer_name: Optional[str], dtype, device: str, trust_remote_code: bool):
    tok_name = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    if os.path.isdir(model_path) and "checkpoints" in model_path:
        file_list = os.listdir(model_path)
        if "pytorch_model.bin" in file_list or "model.safetensors" in file_list:
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype if device != "cpu" else torch.float32, device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code).eval()
            if device in ("cpu", "mps"): model.to(device)
            return tokenizer, model
        if "adapter_model.safetensors" in file_list or "adapter_model.bin" in file_list:
            peft_cfg = PeftConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or peft_cfg.base_model_name_or_path, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(peft_cfg.base_model_name_or_path, torch_dtype=dtype if device != "cpu" else torch.float32, device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code).eval()
            model = PeftModel.from_pretrained(base_model, model_path).eval()
            if device in ("cpu", "mps"): model.to(device)
            return tokenizer, model.merge_and_unload()
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype if device != "cpu" else torch.float32, device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code).eval()
    if device in ("cpu", "mps"): model.to(device)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TeleQnA (MCQ) benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--system", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="netop/TeleQnA")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieval-mode", default="rag", choices=["closed_book", "rag", "replug"])
    parser.add_argument("--faiss-index", default=None)
    parser.add_argument("--sqlite-db", default=None)
    parser.add_argument("--embed-model", default=None)
    parser.add_argument("--query-encoder-path", default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--retrieval-max-len", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--hnsw-efSearch", type=int, default=128)
    parser.add_argument("--sqlite-table", default=None)
    parser.add_argument("--sqlite-id-col", default=None)
    parser.add_argument("--sqlite-text-col", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replug-max-len", type=int, default=4096)
    parser.add_argument("--qa-prompt-prefix", default="")
    parser.add_argument("--doc-prefix", default="Document: ")
    parser.add_argument("--question-prefix", default="Question: ")
    parser.add_argument("--answer-prefix", default="Answer:")
    parser.add_argument("--section-sep", default="\n\n")
    parser.add_argument("--query-with-options", action="store_true")
    parser.add_argument("--save-prompts", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = resolve_device(args.device), parse_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path, summary_path = os.path.join(args.output_dir, "results.jsonl"), os.path.join(args.output_dir, "summary.txt")

    tokenizer, model = load_generator_model(args.model, args.tokenizer, dtype, device, args.trust_remote_code)
    
    query_encoder, index, store = None, None, None
    if args.retrieval_mode != "closed_book":
        if not args.faiss_index or not args.sqlite_db or not args.query_encoder_path:
            raise ValueError(f"{args.retrieval_mode} requires faiss-index, sqlite-db, and query-encoder-path.")
        query_encoder = QueryEncoder(args.query_encoder_path, device=device, trust_remote_code=args.trust_remote_code)
        index = faiss.read_index(args.faiss_index)
        if hasattr(index, "nprobe"): index.nprobe = args.nprobe
        if hasattr(index, "hnsw"): index.hnsw.efSearch = args.hnsw_efSearch
        store = AutoChunkStore(args.sqlite_db, args.sqlite_table, args.sqlite_id_col, args.sqlite_text_col)

    if args.data_path:
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = [(i+1, json.loads(l)) for i, l in enumerate(f) if l.strip()]
    else:
        ds = load_dataset(args.dataset_name, split=args.split)
        data = [(i+1, dict(row)) for i, row in enumerate(ds)]

    total, correct, invalid, by_cat = 0, 0, 0, {}
    total_ret_t, total_gen_t = 0.0, 0.0

    with open(results_path, "w", encoding="utf-8") as out_f:
        for line_no, raw_dp in data:
            if args.limit and total >= args.limit: break
            try: item = normalize_teleqna_row(raw_dp)
            except Exception: continue
            if not item["options"] or item["gold_option"] is None: continue
            
            total += 1
            question, options, val_opts = item["question"], item["options"], item["valid_options"]
            
            ret_t0 = time.perf_counter()
            contexts, retrieved_meta, scores = [], [], np.array([], dtype=np.float32)
            if args.retrieval_mode != "closed_book":
                q_text = question + ("\n" + format_options(options) if args.query_with_options else "")
                qv = query_encoder.encode([maybe_prefix_query(args.embed_model, q_text)], max_length=args.retrieval_max_len)
                scores, ids = index.search(qv, args.retrieval_top_k)
                vids = [int(i) for i in ids[0].tolist() if i >= 0]
                contexts = store.fetch_by_vids(vids)
                retrieved_meta = [{"vid": c["vid"], "doc_id": c.get("doc_id", ""), "category": c.get("category", ""), "chunk_id": c.get("chunk_id", "")} for c in contexts]
            ret_t = time.perf_counter() - ret_t0

            gen_t0 = time.perf_counter()
            if args.retrieval_mode == "closed_book":
                user_content = build_closed_book_prompt(question, options, args.qa_prompt_prefix, args.question_prefix, args.answer_prefix, args.section_sep)
                prompt_text = format_prompt(user_content, args.system)
                candidate = generate_vanilla(model, tokenizer, prompt_text, args.max_new_tokens, args.temperature, args.top_p)
            elif args.retrieval_mode == "rag":
                user_content = build_multi_doc_rag_prompt(question, options, contexts, args.max_context_chars, args.qa_prompt_prefix, args.doc_prefix, args.question_prefix, args.answer_prefix, args.section_sep)
                prompt_text = format_prompt(user_content, args.system)
                candidate = generate_vanilla(model, tokenizer, prompt_text, args.max_new_tokens, args.temperature, args.top_p)
            else:
                per_doc = [build_single_doc_training_style_prompt(question, options, c["text"], args.qa_prompt_prefix, args.doc_prefix, args.question_prefix, args.answer_prefix, args.section_sep) for c in contexts]
                prompt_text = format_prompt(per_doc[0], args.system) if per_doc else ""
                candidate = generate_replug(model, tokenizer, per_doc, args.system, scores[0], args.max_new_tokens, args.temperature, args.top_p, args.replug_max_len)
            gen_t = time.perf_counter() - gen_t0

            pred_opt = extract_option_id(candidate, valid_options=val_opts)
            is_correct = pred_opt == item["gold_option"]
            if pred_opt is None: invalid += 1
            if is_correct: correct += 1
            total_ret_t += ret_t
            total_gen_t += gen_t

            cat = item["category"]
            by_cat.setdefault(cat, {"total": 0, "correct": 0, "invalid": 0})
            by_cat[cat]["total"] += 1
            by_cat[cat]["correct"] += int(is_correct)
            by_cat[cat]["invalid"] += int(pred_opt is None)

            record = {"line_no": line_no, "question": question, "options": {f"option {i}": t for i, t in options}, "gold_option": item["gold_option"], "candidate": candidate, "prediction_option": pred_opt, "correct": is_correct, "category": cat, "retrieval_mode": args.retrieval_mode, "retrieval_time_sec": ret_t, "generation_time_sec": gen_t, "retrieved": retrieved_meta}
            if args.save_prompts: record["prompt_text"] = prompt_text
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{total}] pred={pred_opt} gold={item['gold_option']} correct={is_correct}", file=sys.stderr)

    if store: store.close()
    acc = correct / total if total > 0 else 0.0
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(f"TeleQnA evaluation summary\nAccuracy: {acc:.4f}\nSamples: {total}\nInvalid: {invalid}\n")
        sf.write("\nCategory breakdown\n")
        for cat, s in sorted(by_cat.items()):
            sf.write(f"{cat}: total={s['total']} acc={s['correct']/s['total']:.4f} invalid={s['invalid']}\n")

    print(f"Summary written to {summary_path}", file=sys.stderr)

if __name__ == "__main__":
    main()
