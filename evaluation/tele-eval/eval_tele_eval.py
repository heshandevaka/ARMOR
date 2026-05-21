#!/usr/bin/env python3
import argparse
import json
import os
import random
import sqlite3
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from peft import PeftConfig, PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed

JUDGE_SYSTEM = (
    "You are an automatic grader for question-answer pairs. "
    "You output only one numeric score."
)

JUDGE_USER_TEMPLATE = """You are grading a model answer for a question-answer pair.

Compare the candidate answer to the ground-truth reference.

Score semantics:
- 1.0 = on par with or better than the reference answer (technically correct, covers the essential idea).
- <1.0 = worse than the reference.

Return ONLY one floating-point number between 0 and 1. No explanation, no text, no JSON.

Question:
<<<
{prompt}
>>>

Ground-truth answer:
<<<
{reference}
>>>

Candidate model answer:
<<<
{candidate}
>>>
"""


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "completion" not in obj:
                # Some datasets might use 'question' and 'answer'
                if "question" in obj and "answer" in obj:
                    obj["prompt"] = obj["question"]
                    obj["completion"] = obj["answer"]
                else:
                    raise ValueError(f"Line {line_no} missing 'prompt' or 'completion'. Keys: {list(obj.keys())}")
            yield line_no, obj


def format_prompt(user_content: str, system: Optional[str]) -> str:
    if system:
        return f"{system}\n\nUser: {user_content}\nAssistant:"
    return user_content


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
    return "e5" in model_name_or_path.lower()


class QueryEncoder:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModel.from_pretrained(model_path).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: List[str], normalize_embeddings: bool = True, max_length: int = 256) -> np.ndarray:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        if normalize_embeddings:
            emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype(np.float32)


class AutoChunkStore:
    def __init__(self, sqlite_path: str, table: Optional[str] = None, id_col: Optional[str] = None, text_col: Optional[str] = None):
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.table, self.col_vid, self.col_text = self._discover(table, id_col, text_col)
        self._all_cols = self._list_columns(self.table)

    def close(self):
        self.conn.close()

    def _list_objects(self):
        cur = self.conn.cursor()
        return cur.execute(
            "SELECT type, name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()

    def _list_columns(self, obj_name: str) -> List[str]:
        cur = self.conn.cursor()
        rows = cur.execute(f'PRAGMA table_info("{obj_name}")').fetchall()
        return [r[1] for r in rows]

    def _discover(self, table: Optional[str], id_col: Optional[str], text_col: Optional[str]):
        objects = self._list_objects()
        if not objects:
            raise RuntimeError("No user tables/views found in sqlite DB.")

        if table is not None:
            cols = self._list_columns(table)
            col_vid = id_col or self._guess_id_col(cols)
            col_text = text_col or self._guess_text_col(cols)
            if col_vid is None or col_text is None:
                raise RuntimeError(f"Could not infer id/text columns for sqlite object '{table}'.")
            return table, col_vid, col_text

        best = None
        best_score = -1
        for _, name in objects:
            cols = self._list_columns(name)
            cvid = self._guess_id_col(cols)
            ctext = self._guess_text_col(cols)
            score = 0
            if cvid is not None: score += 2
            if ctext is not None: score += 2
            if "vid" in [c.lower() for c in cols]: score += 2
            if "text" in [c.lower() for c in cols]: score += 2
            if score > best_score and cvid is not None and ctext is not None:
                best = (name, cvid, ctext)
                best_score = score

        if best is None:
            raise RuntimeError("Could not auto-discover passage store in sqlite DB.")
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
            by_vid[int(rec[self.col_vid])] = {
                "vid": int(rec[self.col_vid]),
                "text": rec[self.col_text],
                "doc_id": rec.get("doc_id", str(rec[self.col_vid])),
                "category": rec.get("category", ""),
                "chunk_id": rec.get("chunk_id", str(rec[self.col_vid])),
                "metadata": meta,
            }
        return [by_vid[v] for v in vids if v in by_vid]


@torch.inference_mode()
def generate_vanilla(model, tokenizer, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
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
def generate_replug(model, tokenizer, base_user_contents: List[str], system: Optional[str], scores: np.ndarray,
                    max_new_tokens: int, temperature: float, top_p: float, max_len: int) -> str:
    if len(base_user_contents) == 0:
        prompt_text = format_prompt("", system)
        return generate_vanilla(model, tokenizer, prompt_text, max_new_tokens, temperature, top_p)

    s = torch.tensor(scores[:len(base_user_contents)], dtype=torch.float32, device="cpu")
    lam = torch.softmax(s, dim=0)
    generated = ""
    do_sample = temperature > 0

    for _ in range(max_new_tokens):
        prompts = [format_prompt(uc + generated, system) for uc in base_user_contents]
        logp_each = _next_token_logprobs_batch(model, tokenizer, prompts, max_len=max_len)
        p_each = logp_each.exp()
        p_ens = (lam.to(p_each.device).unsqueeze(-1) * p_each).sum(dim=0)
        p_ens = p_ens / p_ens.sum().clamp(min=1e-12)

        if do_sample:
            if temperature != 1.0:
                p_ens = torch.softmax(torch.log(p_ens + 1e-12) / max(temperature, 1e-6), dim=-1)
            p_ens = _top_p_filtering(p_ens, top_p)
            next_id = torch.multinomial(p_ens, num_samples=1).item()
        else:
            next_id = torch.argmax(p_ens).item()

        generated += tokenizer.decode([next_id], skip_special_tokens=True)
        if next_id == tokenizer.eos_token_id: break

    return generated.strip()


def judge_with_gpt(client: OpenAI, model: str, prompt: str, reference: str, candidate: str, max_tokens: int) -> float:
    user_msg = JUDGE_USER_TEMPLATE.format(prompt=prompt, reference=reference, candidate=candidate)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_completion_tokens=max_tokens,
        )
        cleaned = resp.choices[0].message.content.strip()
        for token in cleaned.replace("\n", " ").split():
            try:
                return max(0.0, min(1.0, float(token)))
            except ValueError:
                continue
        return 0.0
    except Exception as e:
        print(f"Warning: Judge failed for one sample: {e}", file=sys.stderr)
        return 0.0


def maybe_prefix_query(model_name_or_path: str, question: str) -> str:
    return ("query: " + question) if needs_e5_prefix(model_name_or_path) else question


def build_closed_book_prompt(question: str, qa_prompt_prefix: str, question_prefix: str, answer_prefix: str, section_sep: str) -> str:
    return (
        qa_prompt_prefix
        + question_prefix + question
        + section_sep
        + answer_prefix
    )


def build_single_doc_training_style_prompt(question: str, document: str, qa_prompt_prefix: str,
                                           doc_prefix: str, question_prefix: str, answer_prefix: str,
                                           section_sep: str) -> str:
    return (
        qa_prompt_prefix
        + doc_prefix + document
        + section_sep
        + question_prefix + question
        + section_sep
        + answer_prefix
    )


def build_multi_doc_rag_prompt(question: str, contexts: List[Dict[str, Any]], max_context_chars: int,
                               qa_prompt_prefix: str, doc_prefix: str, question_prefix: str,
                               answer_prefix: str, section_sep: str) -> str:
    blocks = []
    total = 0
    for i, c in enumerate(contexts, start=1):
        head = f"[{i}] doc_id={c['doc_id']} category={c.get('category','')} chunk_id={c.get('chunk_id','')}\n"
        block = head + c["text"] + "\n\n"
        if total + len(block) > max_context_chars: break
        blocks.append(block)
        total += len(block)
    doc_text = "".join(blocks).strip() if blocks else "(no retrieved context)"
    return (
        qa_prompt_prefix
        + doc_prefix + doc_text
        + section_sep
        + question_prefix + question
        + section_sep
        + answer_prefix
    )


def load_generator_model(model_path: str, tokenizer_name: Optional[str], dtype, device: str, trust_remote_code: bool):
    tok_name = tokenizer_name or model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token

    if os.path.isdir(model_path) and "checkpoints" in model_path:
        file_list = os.listdir(model_path)
        if "pytorch_model.bin" in file_list or "model.safetensors" in file_list:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=dtype if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code,
            ).eval()
            if device in ("cpu", "mps"): model.to(device)
            return tokenizer, model
        if "adapter_model.safetensors" in file_list or "adapter_model.bin" in file_list:
            peft_cfg = PeftConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or peft_cfg.base_model_name_or_path, trust_remote_code=trust_remote_code)
            if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_cfg.base_model_name_or_path, torch_dtype=dtype if device != "cpu" else torch.float32,
                device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code,
            ).eval()
            model = PeftModel.from_pretrained(base_model, model_path).eval()
            if device in ("cpu", "mps"): model.to(device)
            model = model.merge_and_unload()
            return tokenizer, model
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype if device != "cpu" else torch.float32,
        device_map="auto" if device == "cuda" else None, trust_remote_code=trust_remote_code,
    ).eval()
    if device in ("cpu", "mps"): model.to(device)
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tele-eval (open-ended QnA) benchmark.")

    parser.add_argument("--model", required=True, help="Generator LM path/name")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--system", default=None)

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--retrieval-mode", default="rag", choices=["rag", "replug", "closed_book"])
    parser.add_argument("--faiss-index", default=None)
    parser.add_argument("--sqlite-db", default=None)
    parser.add_argument("--embed-model", default="intfloat/e5-large-v2")
    parser.add_argument("--query-encoder-path", default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=8)
    parser.add_argument("--retrieval-max-len", type=int, default=256)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--hnsw-efSearch", type=int, default=128)

    parser.add_argument("--sqlite-table", default=None)
    parser.add_argument("--sqlite-id-col", default=None)
    parser.add_argument("--sqlite-text-col", default=None)

    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--judge-model", default="gpt-5.2")
    parser.add_argument("--judge-max-tokens", type=int, default=32)
    parser.add_argument("--replug-max-len", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--qa-prompt-prefix", default="")
    parser.add_argument("--doc-prefix", default="Document: ")
    parser.add_argument("--question-prefix", default="Question: ")
    parser.add_argument("--answer-prefix", default="Answer:")
    parser.add_argument("--section-sep", default="\n\n")

    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = args.device

    dtype = parse_dtype(args.dtype)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.txt")

    tokenizer, model = load_generator_model(args.model, args.tokenizer, dtype, device, args.trust_remote_code)

    query_encoder = None
    index = None
    store = None
    if args.retrieval_mode != "closed_book":
        if not args.query_encoder_path or not args.faiss_index or not args.sqlite_db:
            raise ValueError("Retrieval components (query encoder, faiss index, sqlite db) required for RAG/REPLUG modes.")
        query_encoder = QueryEncoder(args.query_encoder_path, device=device)
        index = faiss.read_index(args.faiss_index)
        if hasattr(index, "nprobe"): index.nprobe = args.nprobe
        if hasattr(index, "hnsw"): index.hnsw.efSearch = args.hnsw_efSearch
        store = AutoChunkStore(args.sqlite_db, args.sqlite_table, args.sqlite_id_col, args.sqlite_text_col)

    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Judge will fail.", file=sys.stderr)
    client = OpenAI()

    total = 0
    score_sum = 0.0
    correct = 0
    total_retrieval_time = 0.0
    total_generation_time = 0.0

    with open(results_path, "w", encoding="utf-8") as out_f:
        for line_no, dp in load_jsonl(args.data_path):
            if args.limit and total >= args.limit: break
            total += 1
            question = dp["prompt"]
            reference = dp["completion"]

            retrieval_t0 = time.perf_counter()
            contexts = []
            scores = None
            retrieved_meta = []
            if args.retrieval_mode != "closed_book":
                q = maybe_prefix_query(args.embed_model, question)
                qv = query_encoder.encode([q], normalize_embeddings=True, max_length=args.retrieval_max_len)
                scores, ids = index.search(qv, args.retrieval_top_k)
                vids = [int(i) for i in ids[0].tolist() if i >= 0]
                contexts = store.fetch_by_vids(vids)
                retrieved_meta = [{"vid": c["vid"], "doc_id": c.get("doc_id", ""), "category": c.get("category", ""), "chunk_id": c.get("chunk_id", "")} for c in contexts]
            
            retrieval_time = time.perf_counter() - retrieval_t0

            generation_t0 = time.perf_counter()
            if args.retrieval_mode == "closed_book":
                user_content = build_closed_book_prompt(question, args.qa_prompt_prefix, args.question_prefix, args.answer_prefix, args.section_sep)
                prompt_text = format_prompt(user_content, args.system)
                candidate = generate_vanilla(model, tokenizer, prompt_text, args.max_new_tokens, args.temperature, args.top_p)
            elif args.retrieval_mode == "rag":
                user_content = build_multi_doc_rag_prompt(question, contexts, args.max_context_chars, args.qa_prompt_prefix, args.doc_prefix, args.question_prefix, args.answer_prefix, args.section_sep)
                prompt_text = format_prompt(user_content, args.system)
                candidate = generate_vanilla(model, tokenizer, prompt_text, args.max_new_tokens, args.temperature, args.top_p)
            else: # replug
                per_doc_user_contents = [build_single_doc_training_style_prompt(question, c["text"], args.qa_prompt_prefix, args.doc_prefix, args.question_prefix, args.answer_prefix, args.section_sep) for c in contexts]
                candidate = generate_replug(model, tokenizer, per_doc_user_contents, args.system, scores[0], args.max_new_tokens, args.temperature, args.top_p, args.replug_max_len)
            
            generation_time = time.perf_counter() - generation_t0

            score = judge_with_gpt(client, args.judge_model, question, reference, candidate, args.judge_max_tokens)
            is_correct = score == 1.0
            score_sum += score
            if is_correct: correct += 1

            total_retrieval_time += retrieval_time
            total_generation_time += generation_time

            record = {
                "line_no": line_no, "prompt": question, "reference": reference, "candidate": candidate,
                "score": score, "correct": is_correct, "retrieval_mode": args.retrieval_mode,
                "retrieval_time_sec": retrieval_time, "generation_time_sec": generation_time, "retrieved": retrieved_meta,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{total}] score={score:.3f} correct={is_correct}", file=sys.stderr)

    if store: store.close()

    avg_score = score_sum / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    avg_retrieval_time = total_retrieval_time / total if total > 0 else 0.0
    avg_generation_time = total_generation_time / total if total > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("Model evaluation summary\n=========================\n")
        sf.write(f"Generator model: {args.model}\n")
        sf.write(f"Retrieval mode: {args.retrieval_mode}\n")
        if args.retrieval_mode != "closed_book":
            sf.write(f"Query encoder: {args.query_encoder_path}\n")
            sf.write(f"FAISS: {args.faiss_index}\n")
            sf.write(f"SQLite: {args.sqlite_db}\n")
            sf.write(f"Top-k: {args.retrieval_top_k}\n")
        sf.write(f"Judge: {args.judge_model}\n")
        sf.write(f"Data: {args.data_path}\n")
        sf.write(f"Samples: {total}\n")
        sf.write(f"Average score: {avg_score:.4f}\n")
        sf.write(f"Accuracy (score==1.0): {accuracy:.4f}\n")
        sf.write(f"Average retrieval time (sec): {avg_retrieval_time:.4f}\n")
        sf.write(f"Average generation time (sec): {avg_generation_time:.4f}\n")

    print(f"Summary written to {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
