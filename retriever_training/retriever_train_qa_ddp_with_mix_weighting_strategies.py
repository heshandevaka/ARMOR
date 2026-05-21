#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed


# ============================================================
# Distributed helpers
# ============================================================

def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=torch.distributed.constants.default_pg_timeout)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if is_dist():
        dist.destroy_process_group()


# ============================================================
# Logging
# ============================================================

def build_logger(log_dir: str, name: str = "rag_replug_contriever_retriever_train_qa") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    rank = get_rank()

    logger = logging.getLogger(f"{name}_rank{rank}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(log_dir, f"train_rank{rank}.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ============================================================
# SQLite chunk store
# ============================================================

class AutoChunkStore:
    def __init__(self, sqlite_path: str):
        self.conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.table, self.col_vid, self.col_text, self.col_doc_id = self._discover()

    def _discover(self):
        cur = self.conn.cursor()
        tables = [
            r["name"] for r in cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        ]
        if not tables:
            raise RuntimeError("No user tables found in sqlite DB.")

        best = None
        best_cols = None
        best_score = -1
        for t in tables:
            cols = [r["name"] for r in cur.execute(f"PRAGMA table_info({t})").fetchall()]
            low = set(c.lower() for c in cols)
            score = 0
            if "vid" in low or "id" in low:
                score += 2
            if "text" in low or "content" in low or "chunk" in low:
                score += 2
            if "doc_id" in low:
                score += 2
            if score > best_score:
                best_score = score
                best = t
                best_cols = cols

        low_map = {c.lower(): c for c in best_cols}
        col_vid = low_map.get("vid", low_map.get("id"))
        col_text = low_map.get("text", low_map.get("content", low_map.get("chunk")))
        col_doc_id = low_map.get("doc_id", low_map.get("document_id"))
        if col_vid is None or col_text is None:
            raise RuntimeError(f"Could not infer vid/text columns from table={best}, cols={best_cols}")
        return best, col_vid, col_text, col_doc_id

    def get_rows_by_vids(self, vids: List[int]) -> List[Dict[str, Any]]:
        if not vids:
            return []
        qmarks = ",".join(["?"] * len(vids))
        doc_sel = f", {self.col_doc_id} AS doc_id" if self.col_doc_id else ", NULL AS doc_id"
        sql = f"""
            SELECT {self.col_vid} AS vid, {self.col_text} AS text {doc_sel}
            FROM {self.table}
            WHERE {self.col_vid} IN ({qmarks})
        """
        rows = self.conn.execute(sql, vids).fetchall()
        by_vid = {
            int(r["vid"]): {
                "vid": int(r["vid"]),
                "text": r["text"],
                "doc_id": None if "doc_id" not in r.keys() else r["doc_id"],
            }
            for r in rows
        }
        return [by_vid[v] for v in vids if v in by_vid]

    def get_texts_by_vids(self, vids: List[int]) -> List[str]:
        return [r["text"] for r in self.get_rows_by_vids(vids)]

    def close(self):
        self.conn.close()


# ============================================================
# Encoder helpers
# ============================================================

def needs_e5_prefix(model_name: str) -> bool:
    return "e5" in model_name.lower()


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


class BiEncoderSide(nn.Module):
    def __init__(self, model_name: str, side: str):
        super().__init__()
        assert side in {"query", "doc"}
        self.model_name = model_name
        self.side = side
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def _prefix(self, texts: List[str]) -> List[str]:
        if needs_e5_prefix(self.model_name):
            return [f"{'query' if self.side == 'query' else 'passage'}: {t}" for t in texts]
        return texts

    def encode(self, texts: List[str], max_len: int) -> torch.Tensor:
        texts = self._prefix(texts)
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        batch = {k: v.to(self.encoder.device) for k, v in batch.items()}
        out = self.encoder(**batch)
        emb = mean_pool(out.last_hidden_state, batch["attention_mask"])
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_numpy(self, texts: List[str], max_len: int, batch_size: int = 64) -> np.ndarray:
        self.eval()
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            vec = self.encode(chunk, max_len=max_len)
            all_vecs.append(vec.detach().cpu().numpy().astype(np.float32))
        self.train(self.side == "query")
        return np.concatenate(all_vecs, axis=0)


# ============================================================
# Frozen LM scoring
# ============================================================

@torch.no_grad()
def lm_loglikelihood_of_continuation(
    lm: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    continuations: List[str],
    max_len: int,
) -> torch.Tensor:
    device = next(lm.parameters()).device

    prompt_ids = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    full_texts = [p + c for p, c in zip(prompts, continuations)]
    full_ids = tok(
        full_texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)

    out = lm(**full_ids)
    logprobs = F.log_softmax(out.logits, dim=-1)

    input_ids = full_ids["input_ids"]
    attn = full_ids["attention_mask"]
    prompt_lens = prompt_ids["attention_mask"].sum(dim=1)
    full_lens = attn.sum(dim=1)

    ll = torch.zeros(input_ids.size(0), device=device, dtype=torch.float32)
    for i in range(input_ids.size(0)):
        pl = int(prompt_lens[i].item())
        fl = int(full_lens[i].item())
        start_pred = max(pl - 1, 0)
        end_pred = fl - 1
        targets = input_ids[i, pl:fl]
        pred = logprobs[i, start_pred:end_pred, :]
        ll[i] = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum()
    return ll


# ============================================================
# Data
# ============================================================

QuestionType = Union[str, List[str]]
AnswerType = Union[str, List[str]]


def _pick_first_present(obj: dict, keys: List[str]) -> Optional[object]:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _normalize_question(val: QuestionType) -> Optional[str]:
    if isinstance(val, str):
        q = val.strip()
        return q if q else None
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _normalize_answers(val: AnswerType) -> List[str]:
    if isinstance(val, str):
        val = val.strip()
        return [val] if val else []
    if isinstance(val, list):
        out = []
        for item in val:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out
    return []


def _normalize_int_list(val: object) -> List[int]:
    if isinstance(val, list):
        out = []
        for item in val:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    return []


def _normalize_str_list(val: object) -> List[str]:
    if isinstance(val, list):
        return [str(x) for x in val if isinstance(x, str) and x.strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def load_qa_pairs_from_jsonl(jsonl_path: str, max_examples: int = 0) -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            raw_q = _pick_first_present(obj, ["question", "query", "input", "prompt"])
            raw_a = _pick_first_present(obj, ["completion", "answer", "answers", "output", "target", "gold", "label"])
            q = _normalize_question(raw_q)
            answers = _normalize_answers(raw_a)
            if q is None or not answers:
                continue
            examples.append((q, answers[0]))
            if max_examples > 0 and len(examples) >= max_examples:
                break
    if not examples:
        raise RuntimeError("No QA examples found.")
    return examples


def load_contriever_examples_from_jsonl(jsonl_path: str, max_examples: int = 0) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            question = _normalize_question(_pick_first_present(obj, ["question", "query", "input", "prompt"]))
            if question is None:
                continue
            answers = _normalize_answers(_pick_first_present(obj, ["completion", "answer", "answers", "output", "target", "gold", "label"]))
            positive_text = _pick_first_present(obj, ["top_positive_text", "positive_text", "context"])
            positive_vid = _pick_first_present(obj, ["top_positive_vid"])
            positive_vids = _normalize_int_list(_pick_first_present(obj, ["positive_vids"]))
            positive_texts = _normalize_str_list(_pick_first_present(obj, ["positive_texts"]))
            if positive_text is None and positive_texts:
                positive_text = positive_texts[0]
            if positive_vid is not None:
                try:
                    positive_vid = int(positive_vid)
                except Exception:
                    positive_vid = None
            doc_id = _pick_first_present(obj, ["doc_id", "document_id"])
            reason = _pick_first_present(obj, ["reason", "rationale", "explanation"])
            if not positive_text:
                continue
            examples.append({
                "question": question,
                "answer": answers[0] if answers else "",
                "reason": reason if isinstance(reason, str) else "",
                "doc_id": None if doc_id is None else str(doc_id),
                "top_positive_vid": positive_vid,
                "positive_vids": positive_vids,
                "top_positive_text": str(positive_text),
                "positive_texts": positive_texts,
                "example_id": _pick_first_present(obj, ["example_id", "id"]) or i,
            })
            if max_examples > 0 and len(examples) >= max_examples:
                break
    if not examples:
        raise RuntimeError("No contrastive examples found. Expected aligned JSONL with question + top_positive_text (+ optional positive_vids).")
    return examples


class ShardedDataset:
    def __init__(self, items: List[Any], seed: int):
        self.items = items
        self.seed = seed
        self.rank = get_rank()
        self.world_size = get_world_size()

    def iter_batches(self, batch_size: int, epoch: int):
        rng = random.Random(self.seed + epoch)
        indices = list(range(len(self.items)))
        rng.shuffle(indices)
        indices = indices[self.rank::self.world_size]
        batch = []
        for idx in indices:
            batch.append(self.items[idx])
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ============================================================
# FAISS
# ============================================================

def configure_faiss_index(index, nprobe: int, ef_search: int):
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = ef_search


def load_faiss_index(index_path: str, nprobe: int, ef_search: int):
    index = faiss.read_index(index_path)
    configure_faiss_index(index, nprobe=nprobe, ef_search=ef_search)
    return index


# ============================================================
# Objective helpers
# ============================================================

@dataclass
class TrainConfig:
    objective: str
    top_k: int
    retr_max_len: int
    lm_max_len: int
    teacher_batch_size: int
    retrieval_encode_batch_size: int
    retr_temperature: float
    normalize_by_y_length: bool
    qa_prompt_prefix: str
    doc_prefix: str
    question_prefix: str
    answer_prefix: str
    section_sep: str
    replug_beta: float
    replug_gamma: float
    contrastive_temperature: float
    contrastive_num_negatives: int
    contrastive_exclude_same_doc: bool
    mix_rag_weight: float
    mix_contriever_weight: float
    mix_weighting_strategy: str
    mix_schedule_steps: int
    mix_loss_ema_alpha: float
    mix_entropy_low: float
    mix_entropy_high: float
    mix_min_rag_weight: float
    mix_min_contriever_weight: float


@dataclass
class MixWeightState:
    rag_loss_ema: Optional[float] = None
    contr_loss_ema: Optional[float] = None


def _update_ema(old: Optional[float], new: float, alpha: float) -> float:
    if old is None:
        return float(new)
    return float(alpha * old + (1.0 - alpha) * new)


def compute_mix_weights(
    cfg: TrainConfig,
    global_step: int,
    rag_loss_value: float,
    contr_loss_value: Optional[float],
    target_entropy_value: Optional[float],
    state: MixWeightState,
) -> Tuple[float, float]:
    base_rag = float(cfg.mix_rag_weight)
    base_contr = float(cfg.mix_contriever_weight)

    if contr_loss_value is None:
        return base_rag, 0.0

    strategy = cfg.mix_weighting_strategy
    eps = 1e-8

    if strategy == "static":
        return base_rag, base_contr

    if strategy == "rag_warmup":
        warm = max(1, cfg.mix_schedule_steps)
        frac = min(max((global_step + 1) / warm, 0.0), 1.0)
        return max(cfg.mix_min_rag_weight, base_rag * frac), max(cfg.mix_min_contriever_weight, base_contr)

    if strategy == "contr_warmup":
        warm = max(1, cfg.mix_schedule_steps)
        frac = min(max((global_step + 1) / warm, 0.0), 1.0)
        return max(cfg.mix_min_rag_weight, base_rag), max(cfg.mix_min_contriever_weight, base_contr * frac)

    if strategy == "inverse_loss":
        rag_scale = 1.0 / max(rag_loss_value, eps)
        contr_scale = 1.0 / max(contr_loss_value, eps)
        return max(cfg.mix_min_rag_weight, base_rag * rag_scale), max(cfg.mix_min_contriever_weight, base_contr * contr_scale)

    if strategy == "ema_inverse_loss":
        state.rag_loss_ema = _update_ema(state.rag_loss_ema, rag_loss_value, cfg.mix_loss_ema_alpha)
        state.contr_loss_ema = _update_ema(state.contr_loss_ema, contr_loss_value, cfg.mix_loss_ema_alpha)
        rag_scale = 1.0 / max(state.rag_loss_ema, eps)
        contr_scale = 1.0 / max(state.contr_loss_ema, eps)
        return max(cfg.mix_min_rag_weight, base_rag * rag_scale), max(cfg.mix_min_contriever_weight, base_contr * contr_scale)

    if strategy == "entropy_adaptive":
        if target_entropy_value is None:
            return base_rag, base_contr
        low = cfg.mix_entropy_low
        high = max(cfg.mix_entropy_high, low + 1e-6)
        frac = (target_entropy_value - low) / (high - low)
        frac = min(max(frac, 0.0), 1.0)
        # High entropy => rely more on contrastive retrieval supervision.
        rag_mult = 1.0 - frac
        contr_mult = frac
        return max(cfg.mix_min_rag_weight, base_rag * rag_mult), max(cfg.mix_min_contriever_weight, base_contr * contr_mult)

    raise ValueError(f"Unknown mix_weighting_strategy: {strategy}")


def gather_scalar(x: torch.Tensor) -> float:
    if not is_dist():
        return float(x.item())
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / get_world_size()
    return float(y.item())


def rag_sequence_marginal_nll(retrieval_logits: torch.Tensor, doc_loglik: torch.Tensor) -> torch.Tensor:
    log_p_eta = F.log_softmax(retrieval_logits, dim=0)
    return -torch.logsumexp(log_p_eta + doc_loglik, dim=0)


def replug_kl_loss(retrieval_logits: torch.Tensor, doc_loglik: torch.Tensor, gamma: float, beta: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_q = torch.softmax(doc_loglik / beta, dim=0)
    retr_p = torch.softmax(retrieval_logits / gamma, dim=0)
    kl = (teacher_q * (torch.log(teacher_q + 1e-12) - torch.log(retr_p + 1e-12))).sum()
    return kl, teacher_q, retr_p


def contrastive_infonce_loss(query_emb: torch.Tensor, doc_embs: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = (query_emb @ doc_embs.T).squeeze(0) / temperature
    target = torch.zeros(1, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    return loss, logits


def format_qa_prompt(question: str, document: str, cfg: TrainConfig) -> str:
    return (
        f"{cfg.qa_prompt_prefix}"
        f"{cfg.doc_prefix}{document}"
        f"{cfg.section_sep}{cfg.question_prefix}{question}"
        f"{cfg.section_sep}{cfg.answer_prefix}"
    )


# ============================================================
# Train
# ============================================================

def train(args, logger: logging.Logger):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting process | rank={rank} world_size={world_size} local_rank={local_rank} device={device}")

    set_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    store = AutoChunkStore(args.sqlite_db)
    index = load_faiss_index(args.faiss_index, nprobe=args.nprobe, ef_search=args.hnsw_ef_search)
    logger.info(f"Loaded FAISS index from {args.faiss_index} | ntotal={index.ntotal}")

    if args.objective in {"contriever", "rag_contriever"}:
        logger.info("Loading aligned contrastive examples from JSONL...")
        all_examples = load_contriever_examples_from_jsonl(args.train_jsonl, max_examples=args.max_examples)
    else:
        logger.info("Loading QA examples from JSONL...")
        all_examples = load_qa_pairs_from_jsonl(args.train_jsonl, max_examples=args.max_examples)
    logger.info(f"Loaded {len(all_examples)} total examples before rank sharding")
    dataset = ShardedDataset(all_examples, seed=args.seed)

    query_encoder = BiEncoderSide(args.embed_model, side="query").to(device)
    query_encoder.train()
    ddp_query_encoder = DDP(
        query_encoder,
        device_ids=[local_rank] if device.type == "cuda" else None,
        output_device=local_rank if device.type == "cuda" else None,
        find_unused_parameters=False,
    )

    doc_encoder = BiEncoderSide(args.embed_model, side="doc").to(device)
    doc_encoder.eval()
    for p in doc_encoder.parameters():
        p.requires_grad_(False)

    optimizer = AdamW(ddp_query_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lm_tok = None
    teacher = None
    if args.objective in {"rag", "replug", "rag_contriever"}:
        lm_tok = AutoTokenizer.from_pretrained(args.lm_name, use_fast=True)
        if lm_tok.pad_token_id is None:
            lm_tok.pad_token = lm_tok.eos_token
        logger.info(f"Loading frozen generator LM: {args.lm_name}")
        lm_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.lm_dtype]
        teacher = AutoModelForCausalLM.from_pretrained(
            args.lm_name,
            torch_dtype=lm_dtype if device.type == "cuda" else torch.float32,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    cfg = TrainConfig(
        objective=args.objective,
        top_k=args.top_k,
        retr_max_len=args.retr_max_len,
        lm_max_len=args.lm_max_len,
        teacher_batch_size=args.teacher_batch_size,
        retrieval_encode_batch_size=args.retrieval_encode_batch_size,
        retr_temperature=args.retr_temperature,
        normalize_by_y_length=args.normalize_by_y_length,
        qa_prompt_prefix=args.qa_prompt_prefix,
        doc_prefix=args.doc_prefix,
        question_prefix=args.question_prefix,
        answer_prefix=args.answer_prefix,
        section_sep=args.section_sep,
        replug_beta=args.replug_beta,
        replug_gamma=args.replug_gamma,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_num_negatives=args.contrastive_num_negatives,
        contrastive_exclude_same_doc=args.contrastive_exclude_same_doc,
        mix_rag_weight=args.mix_rag_weight,
        mix_contriever_weight=args.mix_contriever_weight,
        mix_weighting_strategy=args.mix_weighting_strategy,
        mix_schedule_steps=args.mix_schedule_steps,
        mix_loss_ema_alpha=args.mix_loss_ema_alpha,
        mix_entropy_low=args.mix_entropy_low,
        mix_entropy_high=args.mix_entropy_high,
        mix_min_rag_weight=args.mix_min_rag_weight,
        mix_min_contriever_weight=args.mix_min_contriever_weight,
    )

    mix_weight_state = MixWeightState()

    global_step = 0
    epoch = 0
    stop_training = False

    while not stop_training:
        epoch += 1
        logger.info(f"Starting epoch {epoch}")

        for batch in dataset.iter_batches(batch_size=args.batch_size, epoch=epoch):
            t_step0 = time.time()
            loss_terms = []
            avg_retrieved = 0.0
            avg_target_entropy = 0.0
            avg_best_doc_rank = 0.0
            avg_gold_mass = 0.0
            avg_negatives = 0.0
            avg_top_positive_rank = 0.0
            avg_has_top_positive = 0.0
            usable_examples = 0.0
            examined_examples = 0.0
            avg_rag_component = 0.0
            avg_contr_component = 0.0
            avg_weighted_rag_component = 0.0
            avg_weighted_contr_component = 0.0
            avg_applied_rag_weight = 0.0
            avg_applied_contr_weight = 0.0
            mix_examples_with_contr = 0.0

            if cfg.objective in {"rag", "replug"}:
                batch_q = [x for x, _ in batch]
                batch_a = [y for _, y in batch]
            else:
                batch_q = [ex["question"] for ex in batch]
                batch_a = [ex.get("answer", "") for ex in batch]

            # Retrieval using current query encoder + fixed FAISS index
            t0 = time.time()
            with torch.no_grad():
                q_vecs = ddp_query_encoder.module.encode_numpy(
                    batch_q, max_len=cfg.retr_max_len, batch_size=cfg.retrieval_encode_batch_size
                )
            scores_np, vids_np = index.search(q_vecs.astype(np.float32), cfg.top_k)
            retrieval_time = time.time() - t0

            if cfg.objective in {"rag", "replug", "rag_contriever"}:
                docs_per_example: List[List[str]] = []
                rows_per_example: List[List[Dict[str, Any]]] = []
                for i in range(len(batch_q)):
                    vids = [int(v) for v in vids_np[i].tolist() if int(v) >= 0]
                    rows = store.get_rows_by_vids(vids)
                    rows_per_example.append(rows)
                    docs_per_example.append([r["text"] for r in rows if isinstance(r.get("text"), str) and r["text"].strip()])

                # Frozen generator scores p_theta(answer | question, doc)
                t1 = time.time()
                flat_prompts: List[str] = []
                flat_answers: List[str] = []
                per_ex_k: List[int] = []
                for q, a, rows in zip(batch_q, batch_a, rows_per_example):
                    docs = [r["text"] for r in rows if isinstance(r.get("text"), str) and r["text"].strip()]
                    per_ex_k.append(len(docs))
                    for d in docs:
                        flat_prompts.append(format_qa_prompt(q, d, cfg))
                        flat_answers.append(a)

                if len(flat_prompts) == 0:
                    logger.warning("No retrieved docs in this batch; skipping")
                    continue

                ll_chunks = []
                with torch.no_grad():
                    for i in range(0, len(flat_prompts), cfg.teacher_batch_size):
                        ll = lm_loglikelihood_of_continuation(
                            teacher, lm_tok,
                            prompts=flat_prompts[i:i + cfg.teacher_batch_size],
                            continuations=flat_answers[i:i + cfg.teacher_batch_size],
                            max_len=cfg.lm_max_len,
                        )
                        ll_chunks.append(ll)
                ll_all = torch.cat(ll_chunks, dim=0)
                teacher_time = time.time() - t1

                doc_loglik_per_example: List[torch.Tensor] = []
                offset = 0
                for a, k_i in zip(batch_a, per_ex_k):
                    ll_i = ll_all[offset:offset + k_i]
                    if cfg.normalize_by_y_length and k_i > 0:
                        y_len = max(1, len(lm_tok(a, add_special_tokens=False).input_ids))
                        ll_i = ll_i / y_len
                    doc_loglik_per_example.append(ll_i)
                    offset += k_i

                t2 = time.time()
                for ex_or_pair, rows, doc_ll in zip(batch, rows_per_example, doc_loglik_per_example):
                    if cfg.objective in {"rag", "replug"}:
                        q, _a = ex_or_pair
                        ex = None
                    else:
                        ex = ex_or_pair
                        q = ex["question"]

                    docs = [r["text"] for r in rows if isinstance(r.get("text"), str) and r["text"].strip()]
                    if len(docs) == 0:
                        continue
                    avg_retrieved += len(docs)
                    q_emb = ddp_query_encoder.module.encode([q], max_len=cfg.retr_max_len)
                    with torch.no_grad():
                        d_emb = doc_encoder.encode(docs, max_len=cfg.retr_max_len)
                    retr_logits = (q_emb @ d_emb.T).squeeze(0)

                    if cfg.objective == "rag":
                        rag_loss = rag_sequence_marginal_nll(retr_logits / cfg.retr_temperature, doc_ll.to(device))
                        loss_i = rag_loss
                        with torch.no_grad():
                            log_p_eta = F.log_softmax(retr_logits / cfg.retr_temperature, dim=0)
                            log_post = log_p_eta + doc_ll.to(device)
                            log_post = log_post - torch.logsumexp(log_post, dim=0)
                            target_dist = log_post.exp()
                            entropy = -(target_dist * log_post).sum()
                            avg_target_entropy += float(entropy.item())
                            best_idx = int(torch.argmax(doc_ll).item())
                            avg_gold_mass += float(target_dist[best_idx].item())
                    elif cfg.objective == "replug":
                        replug_loss, teacher_q, retr_p = replug_kl_loss(retr_logits, doc_ll.to(device), cfg.replug_gamma, cfg.replug_beta)
                        loss_i = replug_loss
                        with torch.no_grad():
                            entropy = -(teacher_q * torch.log(teacher_q + 1e-12)).sum()
                            avg_target_entropy += float(entropy.item())
                            best_idx = int(torch.argmax(doc_ll).item())
                            avg_gold_mass += float(teacher_q[best_idx].item())
                    else:
                        # Mixed objective: RAG loss plus query-only contrastive loss against aligned positives.
                        rag_loss = rag_sequence_marginal_nll(retr_logits / cfg.retr_temperature, doc_ll.to(device))
                        contr_loss = None
                        contr_logits = None
                        top_positive_rank = None
                        positive_vids = set(ex.get("positive_vids", [])) if ex is not None else set()
                        if ex is not None and ex.get("top_positive_vid") is not None:
                            try:
                                positive_vids.add(int(ex["top_positive_vid"]))
                            except Exception:
                                pass
                        ex_doc_id = ex.get("doc_id") if ex is not None else None
                        top_positive_text = ex.get("top_positive_text") if ex is not None else None

                        if ex is not None:
                            examined_examples += 1.0
                            tp = ex.get("top_positive_vid")
                            if tp is not None:
                                try:
                                    tp = int(tp)
                                    for pos_idx, row in enumerate(rows, start=1):
                                        if int(row["vid"]) == tp:
                                            top_positive_rank = pos_idx
                                            break
                                except Exception:
                                    pass

                        negative_rows = []
                        if top_positive_text:
                            for row in reversed(rows):
                                if int(row["vid"]) in positive_vids:
                                    continue
                                if cfg.contrastive_exclude_same_doc and ex_doc_id is not None and row.get("doc_id") is not None:
                                    if str(row["doc_id"]) == str(ex_doc_id):
                                        continue
                                text = row.get("text")
                                if not isinstance(text, str) or not text.strip():
                                    continue
                                negative_rows.append(row)
                                if len(negative_rows) >= cfg.contrastive_num_negatives:
                                    break

                        if top_positive_text and len(negative_rows) > 0:
                            usable_examples += 1.0
                            avg_negatives += len(negative_rows)
                            if top_positive_rank is not None:
                                avg_top_positive_rank += float(top_positive_rank)
                                avg_has_top_positive += 1.0
                            candidate_texts = [top_positive_text] + [r["text"] for r in negative_rows]
                            with torch.no_grad():
                                contrastive_doc_emb = doc_encoder.encode(candidate_texts, max_len=cfg.retr_max_len)
                            contr_loss, contr_logits = contrastive_infonce_loss(q_emb, contrastive_doc_emb, cfg.contrastive_temperature)
                            mix_examples_with_contr += 1.0

                        with torch.no_grad():
                            log_p_eta = F.log_softmax(retr_logits / cfg.retr_temperature, dim=0)
                            log_post = log_p_eta + doc_ll.to(device)
                            log_post = log_post - torch.logsumexp(log_post, dim=0)
                            target_dist = log_post.exp()
                            entropy = -(target_dist * log_post).sum()
                            entropy_value = float(entropy.item())
                            avg_target_entropy += entropy_value
                            best_idx = int(torch.argmax(doc_ll).item())
                            avg_gold_mass += float(target_dist[best_idx].item())

                        rag_loss_value = float(rag_loss.detach().item())
                        contr_loss_value = float(contr_loss.detach().item()) if contr_loss is not None else None
                        rag_weight_i, contr_weight_i = compute_mix_weights(
                            cfg=cfg,
                            global_step=global_step,
                            rag_loss_value=rag_loss_value,
                            contr_loss_value=contr_loss_value,
                            target_entropy_value=entropy_value,
                            state=mix_weight_state,
                        )
                        loss_i = rag_weight_i * rag_loss
                        avg_rag_component += rag_loss_value
                        avg_weighted_rag_component += rag_weight_i * rag_loss_value
                        avg_applied_rag_weight += rag_weight_i
                        if contr_loss is not None:
                            loss_i = loss_i + contr_weight_i * contr_loss
                            avg_contr_component += contr_loss_value
                            avg_weighted_contr_component += contr_weight_i * contr_loss_value
                            avg_applied_contr_weight += contr_weight_i

                    loss_terms.append(loss_i)
                    with torch.no_grad():
                        if cfg.objective == "rag_contriever" and contr_logits is not None:
                            probs = torch.softmax(contr_logits, dim=0)
                            rank_pos = int((torch.argsort(contr_logits, descending=True) == 0).nonzero(as_tuple=False)[0].item()) + 1
                            avg_best_doc_rank += rank_pos
                            # average gold mass mixes semantics: report contrastive positive mass if available
                            avg_gold_mass += 0.0
                        else:
                            rank_pos = int((torch.argsort(retr_logits, descending=True) == best_idx).nonzero(as_tuple=False)[0].item()) + 1
                            avg_best_doc_rank += rank_pos
                train_time = time.time() - t2
            else:
                teacher_time = 0.0
                t2 = time.time()
                for ex, retrieved_vids in zip(batch, vids_np.tolist()):
                    q = ex["question"]
                    top_positive_text = ex["top_positive_text"]
                    positive_vids = set(ex.get("positive_vids", []))
                    if ex.get("top_positive_vid") is not None:
                        positive_vids.add(int(ex["top_positive_vid"]))
                    ex_doc_id = ex.get("doc_id")

                    retrieved_vids = [int(v) for v in retrieved_vids if int(v) >= 0]
                    retrieved_rows = store.get_rows_by_vids(retrieved_vids)
                    examined_examples += 1.0

                    top_positive_rank = None
                    if ex.get("top_positive_vid") is not None:
                        tp = int(ex["top_positive_vid"])
                        for pos_idx, row in enumerate(retrieved_rows, start=1):
                            if row["vid"] == tp:
                                top_positive_rank = pos_idx
                                break

                    negative_rows = []
                    for row in reversed(retrieved_rows):
                        if row["vid"] in positive_vids:
                            continue
                        if cfg.contrastive_exclude_same_doc and ex_doc_id is not None and row.get("doc_id") is not None:
                            if str(row["doc_id"]) == str(ex_doc_id):
                                continue
                        text = row.get("text")
                        if not isinstance(text, str) or not text.strip():
                            continue
                        negative_rows.append(row)
                        if len(negative_rows) >= cfg.contrastive_num_negatives:
                            break
                    if len(negative_rows) == 0:
                        continue

                    usable_examples += 1.0
                    avg_retrieved += len(retrieved_rows)
                    avg_negatives += len(negative_rows)
                    if top_positive_rank is not None:
                        avg_top_positive_rank += float(top_positive_rank)
                        avg_has_top_positive += 1.0

                    q_emb = ddp_query_encoder.module.encode([q], max_len=cfg.retr_max_len)
                    candidate_texts = [top_positive_text] + [r["text"] for r in negative_rows]
                    with torch.no_grad():
                        d_emb = doc_encoder.encode(candidate_texts, max_len=cfg.retr_max_len)
                    loss_i, logits = contrastive_infonce_loss(q_emb, d_emb, cfg.contrastive_temperature)
                    loss_terms.append(loss_i)
                    with torch.no_grad():
                        probs = torch.softmax(logits, dim=0)
                        avg_gold_mass += float(probs[0].item())
                        rank_pos = int((torch.argsort(logits, descending=True) == 0).nonzero(as_tuple=False)[0].item()) + 1
                        avg_best_doc_rank += rank_pos
                train_time = time.time() - t2
            if not loss_terms:
                logger.warning("No usable examples in batch after retrieval; skipping")
                continue

            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp_query_encoder.parameters(), args.max_grad_norm)
            optimizer.step()

            global_step += 1
            mean_loss = gather_scalar(loss)
            denom = max(1, len(loss_terms))
            if cfg.objective in {"contriever", "rag_contriever"}:
                metric_denom = max(1.0, usable_examples if cfg.objective == "contriever" else denom)
            else:
                metric_denom = float(denom)
            mean_avg_retrieved = avg_retrieved / metric_denom
            mean_target_entropy = avg_target_entropy / metric_denom if cfg.objective in {"rag", "replug", "rag_contriever"} else 0.0
            mean_best_doc_rank = avg_best_doc_rank / metric_denom
            mean_gold_mass = avg_gold_mass / metric_denom
            mean_negatives = avg_negatives / max(1.0, usable_examples) if cfg.objective in {"contriever", "rag_contriever"} else 0.0
            mean_top_positive_rank = (avg_top_positive_rank / max(avg_has_top_positive, 1.0)) if cfg.objective in {"contriever", "rag_contriever"} else 0.0
            hit_top_positive = (avg_has_top_positive / max(1.0, usable_examples)) if cfg.objective in {"contriever", "rag_contriever"} else 0.0
            usable_example_rate = (usable_examples / max(1.0, examined_examples)) if cfg.objective in {"contriever", "rag_contriever"} else 1.0
            mean_rag_component = avg_rag_component / max(1.0, denom) if cfg.objective == "rag_contriever" else 0.0
            mean_contr_component = avg_contr_component / max(1.0, mix_examples_with_contr) if cfg.objective == "rag_contriever" else 0.0
            mean_weighted_rag_component = avg_weighted_rag_component / max(1.0, denom) if cfg.objective == "rag_contriever" else 0.0
            mean_weighted_contr_component = avg_weighted_contr_component / max(1.0, mix_examples_with_contr) if cfg.objective == "rag_contriever" else 0.0
            mean_applied_rag_weight = avg_applied_rag_weight / max(1.0, denom) if cfg.objective == "rag_contriever" else 0.0
            mean_applied_contr_weight = avg_applied_contr_weight / max(1.0, mix_examples_with_contr) if cfg.objective == "rag_contriever" else 0.0
            mix_contr_fraction = (mix_examples_with_contr / max(1.0, denom)) if cfg.objective == "rag_contriever" else 0.0

            if global_step % args.log_every == 0:
                grad_norm_val = float(grad_norm.item()) if grad_norm is not None and torch.is_tensor(grad_norm) else None
                msg = (
                    f"step={global_step} objective={cfg.objective} loss={mean_loss:.6f} batch={len(batch_q)} "
                    f"avg_retrieved={mean_avg_retrieved:.2f} best_doc_rank={mean_best_doc_rank:.2f} "
                    f"best_doc_target_mass={mean_gold_mass:.4f} t_retr={retrieval_time:.2f}s "
                    f"t_teacher={teacher_time:.2f}s t_train={train_time:.2f}s t_total={time.time()-t_step0:.2f}s grad_norm={grad_norm_val}"
                )
                if cfg.objective in {"rag", "replug"}:
                    msg += f" target_entropy={mean_target_entropy:.4f}"
                else:
                    msg += (
                        f" avg_negatives={mean_negatives:.2f} top_positive_hit_rate={hit_top_positive:.4f} "
                        f"avg_top_positive_rank_when_hit={mean_top_positive_rank:.2f} "
                        f"usable_example_rate={usable_example_rate:.4f}"
                    )
                    if cfg.objective == "rag_contriever":
                        msg += (
                            f" mix_weighting={cfg.mix_weighting_strategy} rag_loss={mean_rag_component:.6f} "
                            f"contr_loss={mean_contr_component:.6f} weighted_rag_loss={mean_weighted_rag_component:.6f} "
                            f"weighted_contr_loss={mean_weighted_contr_component:.6f} "
                            f"rag_weight={mean_applied_rag_weight:.6f} contr_weight={mean_applied_contr_weight:.6f} "
                            f"contr_examples_fraction={mix_contr_fraction:.4f} target_entropy={mean_target_entropy:.4f}"
                        )
                logger.info(msg)

            if global_step >= args.max_steps:
                stop_training = True
                break

    barrier()
    if is_main_process():
        ckpt_dir = os.path.join(args.out_dir, "query_encoder_final")
        os.makedirs(ckpt_dir, exist_ok=True)
        ddp_query_encoder.module.encoder.save_pretrained(ckpt_dir)
        ddp_query_encoder.module.tokenizer.save_pretrained(ckpt_dir)
        logger.info(f"Saved final query encoder to {ckpt_dir}")

    store.close()
    cleanup_distributed()


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Distributed retriever training with switchable RAG, RePLUG, Contriever-style contrastive, or mixed RAG+Contriever objectives"
    )

    ap.add_argument("--train_jsonl", required=True, help="QA JSONL for rag/replug, or aligned contrastive JSONL for contriever/rag_contriever")
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--sqlite_db", required=True)

    ap.add_argument("--objective", default="rag", choices=["rag", "replug", "contriever", "rag_contriever"], help="Retriever training objective")

    ap.add_argument("--embed_model", default="intfloat/e5-large-v2")
    ap.add_argument("--lm_name", default=None, help="Required only for rag/replug")
    ap.add_argument("--out_dir", default="./retriever_qa_out")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--retr_temperature", type=float, default=1.0, help="RAG retrieval softmax temperature")
    ap.add_argument("--replug_beta", type=float, default=1.0, help="Teacher distribution temperature for RePLUG")
    ap.add_argument("--replug_gamma", type=float, default=1.0, help="Retriever distribution temperature for RePLUG")

    ap.add_argument("--contrastive_temperature", type=float, default=0.05)
    ap.add_argument("--contrastive_num_negatives", type=int, default=3)
    ap.add_argument("--contrastive_exclude_same_doc", action="store_true", help="Exclude retrieved negatives that share doc_id with the example")
    ap.add_argument("--mix_rag_weight", type=float, default=1.0, help="Weight for the RAG loss term when objective=rag_contriever")
    ap.add_argument("--mix_contriever_weight", type=float, default=1.0, help="Weight for the contrastive loss term when objective=rag_contriever")
    ap.add_argument(
        "--mix_weighting_strategy",
        default="static",
        choices=["static", "rag_warmup", "contr_warmup", "inverse_loss", "ema_inverse_loss", "entropy_adaptive"],
        help="How to weight RAG and contrastive losses when objective=rag_contriever. 'static' reproduces the original behavior exactly.",
    )
    ap.add_argument("--mix_schedule_steps", type=int, default=200, help="Warmup horizon for rag_warmup/contr_warmup weighting strategies")
    ap.add_argument("--mix_loss_ema_alpha", type=float, default=0.9, help="EMA coefficient for ema_inverse_loss weighting")
    ap.add_argument("--mix_entropy_low", type=float, default=1.8, help="Low entropy threshold for entropy_adaptive weighting")
    ap.add_argument("--mix_entropy_high", type=float, default=2.6, help="High entropy threshold for entropy_adaptive weighting")
    ap.add_argument("--mix_min_rag_weight", type=float, default=0.0, help="Minimum applied RAG weight for dynamic mix weighting")
    ap.add_argument("--mix_min_contriever_weight", type=float, default=0.0, help="Minimum applied contrastive weight for dynamic mix weighting")

    ap.add_argument("--retr_max_len", type=int, default=256)
    ap.add_argument("--lm_max_len", type=int, default=4096)
    ap.add_argument("--teacher_batch_size", type=int, default=1)
    ap.add_argument("--retrieval_encode_batch_size", type=int, default=1)
    ap.add_argument("--max_examples", type=int, default=0)

    ap.add_argument("--qa_prompt_prefix", default="")
    ap.add_argument("--doc_prefix", default="Document: ")
    ap.add_argument("--question_prefix", default="Question: ")
    ap.add_argument("--answer_prefix", default="Answer: ")
    ap.add_argument("--section_sep", default="\n\n")

    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--hnsw_ef_search", type=int, default=128)
    ap.add_argument("--lm_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--normalize_by_y_length", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    if args.objective in {"rag", "replug", "rag_contriever"} and not args.lm_name:
        ap.error("--lm_name is required for objective rag/replug/rag_contriever")
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_dir = os.path.join(args.out_dir, "logs")
    logger = build_logger(log_dir)
    try:
        train(args, logger)
    except Exception:
        logger.exception("Training crashed")
        raise
