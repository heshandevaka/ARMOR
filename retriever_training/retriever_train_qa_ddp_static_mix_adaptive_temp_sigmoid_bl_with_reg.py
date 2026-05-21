#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, Dict, Any, Set

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

def build_logger(log_dir: str, name: str = "rag_contriever_static_adaptive_temp_reg") -> logging.Logger:
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
        raise RuntimeError("No aligned mixed examples found. Expected JSONL with question + answer + top_positive_text.")
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
    top_k: int
    retr_max_len: int
    lm_max_len: int
    teacher_batch_size: int
    retrieval_encode_batch_size: int
    normalize_by_y_length: bool
    qa_prompt_prefix: str
    doc_prefix: str
    question_prefix: str
    answer_prefix: str
    section_sep: str
    contrastive_num_negatives: int
    contrastive_exclude_same_doc: bool
    mix_rag_weight: float
    mix_contriever_weight: float
    temperature_min: float
    temperature_max: float
    temperature_l2: float
    regularizers: Set[str]
    weight_anchor_coef: float
    query_distill_coef: float
    rank_distill_coef: float
    query_distill_type: str
    rank_distill_temperature: float


class AdaptiveObjectiveTemperatures(nn.Module):
    """
    Learn temperatures as they appear in the original objectives:
      - RAG retrieval temperature inside p_eta(z|x)
      - InfoNCE temperature inside the contrastive softmax
    """
    def __init__(self, init_retr_temperature: float, init_contrastive_temperature: float,
                 temperature_min: float, temperature_max: float):
        super().__init__()
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        
        if init_retr_temperature <= self.temperature_min or init_retr_temperature >= self.temperature_max:
            init_retr_temperature = min(max(init_retr_temperature, self.temperature_min + 1e-4), self.temperature_max - 1e-4)
        if init_contrastive_temperature <= self.temperature_min or init_contrastive_temperature >= self.temperature_max:
            init_contrastive_temperature = min(max(init_contrastive_temperature, self.temperature_min + 1e-4), self.temperature_max - 1e-4)

        def inv_sigmoid(y):
            sig_x = (y - self.temperature_min) / (self.temperature_max - self.temperature_min)
            return float(np.log(sig_x / (1.0 - sig_x)))

        self.raw_retr_temperature = nn.Parameter(torch.tensor(inv_sigmoid(init_retr_temperature), dtype=torch.float32))
        self.raw_contrastive_temperature = nn.Parameter(torch.tensor(inv_sigmoid(init_contrastive_temperature), dtype=torch.float32))

    def _scaled_sigmoid(self, raw_t: torch.Tensor) -> torch.Tensor:
        return self.temperature_min + (self.temperature_max - self.temperature_min) * torch.sigmoid(raw_t)

    def retr_temperature(self) -> torch.Tensor:
        return self._scaled_sigmoid(self.raw_retr_temperature)

    def contrastive_temperature(self) -> torch.Tensor:
        return self._scaled_sigmoid(self.raw_contrastive_temperature)


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


def contrastive_infonce_loss(query_emb: torch.Tensor, doc_embs: torch.Tensor, temperature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
# Regularization helpers
# ============================================================

def build_base_param_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().float().clone()
        for name, param in model.named_parameters()
    }


def weight_anchor_loss(model: nn.Module, base_snapshot: Dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        base_param = base_snapshot[name].to(param.device, dtype=param.dtype)
        losses.append(F.mse_loss(param, base_param, reduction="mean"))
    if not losses:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def query_distill_loss(q_ft: torch.Tensor, q_base: torch.Tensor, reg_type: str = "cosine") -> torch.Tensor:
    if reg_type == "cosine":
        return (1.0 - F.cosine_similarity(q_ft, q_base, dim=-1)).mean()
    if reg_type == "mse":
        return F.mse_loss(q_ft, q_base, reduction="mean")
    raise ValueError(f"Unsupported query_distill_type: {reg_type}")


def rank_distill_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    t = max(temperature, 1e-6)
    teacher_probs = F.softmax(teacher_logits / t, dim=-1)
    student_log_probs = F.log_softmax(student_logits / t, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (t * t)


def apply_regularizers(
    cfg: TrainConfig,
    query_model: nn.Module,
    base_param_snapshot: Optional[Dict[str, torch.Tensor]],
    q_ft: Optional[torch.Tensor],
    q_base: Optional[torch.Tensor],
    student_logits: Optional[torch.Tensor],
    teacher_logits: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = next(query_model.parameters()).device
    total = torch.zeros((), device=device)
    metrics = {"weight_anchor": 0.0, "query_distill": 0.0, "rank_distill": 0.0}

    if "weight_anchor" in cfg.regularizers:
        if base_param_snapshot is None:
            raise RuntimeError("weight_anchor requested but base_param_snapshot is None")
        l = weight_anchor_loss(query_model, base_param_snapshot)
        total = total + cfg.weight_anchor_coef * l
        metrics["weight_anchor"] = float(l.detach().item())

    if "query_distill" in cfg.regularizers:
        if q_ft is None or q_base is None:
            raise RuntimeError("query_distill requested but q_ft/q_base is None")
        l = query_distill_loss(q_ft, q_base, cfg.query_distill_type)
        total = total + cfg.query_distill_coef * l
        metrics["query_distill"] = float(l.detach().item())

    if "rank_distill" in cfg.regularizers:
        if student_logits is None or teacher_logits is None:
            raise RuntimeError("rank_distill requested but student/teacher logits is None")
        l = rank_distill_loss(student_logits, teacher_logits, cfg.rank_distill_temperature)
        total = total + cfg.rank_distill_coef * l
        metrics["rank_distill"] = float(l.detach().item())

    return total, metrics


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

    logger.info("Loading aligned mixed RAG+InfoNCE examples from JSONL...")
    all_examples = load_contriever_examples_from_jsonl(args.train_jsonl, max_examples=args.max_examples)
    logger.info(f"Loaded {len(all_examples)} total train examples before rank sharding")
    dataset = ShardedDataset(all_examples, seed=args.seed)

    val_dataset = None
    if args.val_jsonl:
        logger.info("Loading val examples from JSONL...")
        val_examples = load_contriever_examples_from_jsonl(args.val_jsonl, max_examples=args.max_examples)
        logger.info(f"Loaded {len(val_examples)} val examples before rank sharding")
        val_dataset = ShardedDataset(val_examples, seed=args.seed + 100)

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

    # Frozen base query encoder used by base-preserving regularizers.
    base_query_teacher = BiEncoderSide(args.embed_model, side="query").to(device)
    base_query_teacher.eval()
    for p in base_query_teacher.parameters():
        p.requires_grad_(False)

    base_param_snapshot = None
    if "weight_anchor" in set(args.regularizers):
        base_param_snapshot = build_base_param_snapshot(base_query_teacher.encoder)

    adaptive_temperatures = AdaptiveObjectiveTemperatures(
        init_retr_temperature=args.retr_temperature,
        init_contrastive_temperature=args.contrastive_temperature,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
    ).to(device)

    encoder_optimizer = AdamW(ddp_query_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    temp_lr = args.temperature_lr if args.temperature_lr > 0 else args.lr
    temp_optimizer = AdamW(adaptive_temperatures.parameters(), lr=temp_lr, weight_decay=0.0)

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
        top_k=args.top_k,
        retr_max_len=args.retr_max_len,
        lm_max_len=args.lm_max_len,
        teacher_batch_size=args.teacher_batch_size,
        retrieval_encode_batch_size=args.retrieval_encode_batch_size,
        normalize_by_y_length=args.normalize_by_y_length,
        qa_prompt_prefix=args.qa_prompt_prefix,
        doc_prefix=args.doc_prefix,
        question_prefix=args.question_prefix,
        answer_prefix=args.answer_prefix,
        section_sep=args.section_sep,
        contrastive_num_negatives=args.contrastive_num_negatives,
        contrastive_exclude_same_doc=args.contrastive_exclude_same_doc,
        mix_rag_weight=args.mix_rag_weight,
        mix_contriever_weight=args.mix_contriever_weight,
        temperature_min=args.temperature_min,
        temperature_max=args.temperature_max,
        temperature_l2=args.temperature_l2,
        regularizers=set(args.regularizers),
        weight_anchor_coef=args.weight_anchor_coef,
        query_distill_coef=args.query_distill_coef,
        rank_distill_coef=args.rank_distill_coef,
        query_distill_type=args.query_distill_type,
        rank_distill_temperature=args.rank_distill_temperature,
    )

    global_step = 0
    epoch = 0
    stop_training = False

    while not stop_training:
        epoch += 1
        logger.info(f"Starting epoch {epoch}")

        def get_val_batch():
            while True:
                for b in val_dataset.iter_batches(batch_size=args.batch_size, epoch=epoch):
                    yield b
        val_iter = iter(get_val_batch()) if val_dataset is not None else None

        train_iter = dataset.iter_batches(batch_size=args.batch_size, epoch=epoch)
        phase_queue = []

        while True:
            if not phase_queue:
                try:
                    train_batch = next(train_iter)
                    has_data = 1.0
                except StopIteration:
                    has_data = 0.0

                # Epoch synchronization: Ensure all ranks finish the epoch together
                has_data_tensor = torch.tensor([has_data], device=device)
                if is_dist():
                    dist.all_reduce(has_data_tensor, op=dist.ReduceOp.MIN)
                if has_data_tensor.item() == 0:
                    break
                phase_queue.append(("train", train_batch))
                if val_iter is not None:
                    phase_queue.append(("val", next(val_iter)))
            
            phase_name, batch = phase_queue.pop(0)

            if phase_name == "train":
                for p in adaptive_temperatures.parameters():
                    p.requires_grad_(False)
                for p in ddp_query_encoder.parameters():
                    p.requires_grad_(True)
            else:
                for p in adaptive_temperatures.parameters():
                    p.requires_grad_(True)
                for p in ddp_query_encoder.parameters():
                    p.requires_grad_(False)

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

            avg_reg_total = 0.0
            avg_reg_weight_anchor = 0.0
            avg_reg_query_distill = 0.0
            avg_reg_rank_distill = 0.0

            avg_retr_temperature = 0.0
            avg_contr_temperature = 0.0
            avg_raw_retr_temperature = 0.0
            avg_raw_contr_temperature = 0.0

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

            rows_per_example: List[List[Dict[str, Any]]] = []
            for i in range(len(batch_q)):
                vids = [int(v) for v in vids_np[i].tolist() if int(v) >= 0]
                rows = store.get_rows_by_vids(vids)
                rows_per_example.append(rows)

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
            for ex, rows, doc_ll in zip(batch, rows_per_example, doc_loglik_per_example):
                q = ex["question"]

                docs = [r["text"] for r in rows if isinstance(r.get("text"), str) and r["text"].strip()]
                if len(docs) == 0:
                    continue

                avg_retrieved += len(docs)
                examined_examples += 1.0

                q_emb = ddp_query_encoder.module.encode([q], max_len=cfg.retr_max_len)
                with torch.no_grad():
                    q_base = base_query_teacher.encode([q], max_len=cfg.retr_max_len)
                    d_emb = doc_encoder.encode(docs, max_len=cfg.retr_max_len)
                retr_logits = (q_emb @ d_emb.T).squeeze(0)
                teacher_rank_logits = (q_base @ d_emb.T).squeeze(0)

                # Adaptive temperatures live inside original objectives
                retr_temperature = adaptive_temperatures.retr_temperature()
                contr_temperature = adaptive_temperatures.contrastive_temperature()

                rag_loss = rag_sequence_marginal_nll(retr_logits / retr_temperature, doc_ll.to(device))

                contr_loss = None
                contr_logits = None
                teacher_contr_logits = None
                top_positive_rank = None
                positive_vids = set(ex.get("positive_vids", []))
                if ex.get("top_positive_vid") is not None:
                    try:
                        positive_vids.add(int(ex["top_positive_vid"]))
                    except Exception:
                        pass
                ex_doc_id = ex.get("doc_id")
                top_positive_text = ex.get("top_positive_text")

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
                        teacher_contr_logits = (q_base @ contrastive_doc_emb.T).squeeze(0)
                    contr_loss, contr_logits = contrastive_infonce_loss(q_emb, contrastive_doc_emb, contr_temperature)

                with torch.no_grad():
                    log_p_eta = F.log_softmax(retr_logits / retr_temperature, dim=0)
                    log_post = log_p_eta + doc_ll.to(device)
                    log_post = log_post - torch.logsumexp(log_post, dim=0)
                    target_dist = log_post.exp()
                    entropy = -(target_dist * log_post).sum()
                    entropy_value = float(entropy.item())
                    avg_target_entropy += entropy_value
                    best_idx = int(torch.argmax(doc_ll).item())
                    avg_gold_mass += float(target_dist[best_idx].item())

                rag_loss_value = float(rag_loss.detach().item())
                contr_loss_value = float(contr_loss.detach().item()) if contr_loss is not None else 0.0

                loss_i = cfg.mix_rag_weight * rag_loss
                avg_rag_component += rag_loss_value
                avg_weighted_rag_component += cfg.mix_rag_weight * rag_loss_value

                if contr_loss is not None:
                    loss_i = loss_i + cfg.mix_contriever_weight * contr_loss
                    avg_contr_component += contr_loss_value
                    avg_weighted_contr_component += cfg.mix_contriever_weight * contr_loss_value

                    loss_i = loss_i + cfg.temperature_l2 * (
                        adaptive_temperatures.raw_retr_temperature.pow(2) +
                        adaptive_temperatures.raw_contrastive_temperature.pow(2)
                    )

                # Base-preserving regularizers shape only the query encoder update.
                if phase_name == "train" and cfg.regularizers:
                    reg_student_logits = contr_logits if contr_logits is not None else retr_logits
                    reg_teacher_logits = teacher_contr_logits if teacher_contr_logits is not None else teacher_rank_logits
                    reg_loss, reg_metrics = apply_regularizers(
                        cfg=cfg,
                        query_model=ddp_query_encoder.module.encoder,
                        base_param_snapshot=base_param_snapshot,
                        q_ft=q_emb,
                        q_base=q_base,
                        student_logits=reg_student_logits,
                        teacher_logits=reg_teacher_logits,
                    )
                    loss_i = loss_i + reg_loss
                    avg_reg_total += float(reg_loss.detach().item())
                    avg_reg_weight_anchor += reg_metrics["weight_anchor"]
                    avg_reg_query_distill += reg_metrics["query_distill"]
                    avg_reg_rank_distill += reg_metrics["rank_distill"]

                loss_terms.append(loss_i)

                with torch.no_grad():
                    avg_retr_temperature += float(retr_temperature.item())
                    avg_contr_temperature += float(contr_temperature.item())
                    avg_raw_retr_temperature += float(adaptive_temperatures.raw_retr_temperature.item())
                    avg_raw_contr_temperature += float(adaptive_temperatures.raw_contrastive_temperature.item())
                    if contr_logits is not None:
                        rank_pos = int((torch.argsort(contr_logits, descending=True) == 0).nonzero(as_tuple=False)[0].item()) + 1
                        avg_best_doc_rank += rank_pos
                    else:
                        rank_pos = int((torch.argsort(retr_logits, descending=True) == best_idx).nonzero(as_tuple=False)[0].item()) + 1
                        avg_best_doc_rank += rank_pos

            train_time = time.time() - t2

            # Batch synchronization: Ensure all ranks skip or process the batch together.
            # Using MIN ensures that if any rank lacks usable data, all ranks skip to stay in sync.
            usable_flag = torch.tensor([1.0 if loss_terms else 0.0], device=device)
            if is_dist():
                dist.all_reduce(usable_flag, op=dist.ReduceOp.MIN)

            if usable_flag.item() == 0:
                logger.warning(f"No usable examples in {phase_name} batch after retrieval; skipping")
                continue

            loss = torch.stack(loss_terms).mean()

            grad_norm = None
            if phase_name == "train":
                encoder_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(ddp_query_encoder.parameters(), args.max_grad_norm)
                encoder_optimizer.step()
                global_step += 1
            else:
                temp_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if is_dist():
                    for p in adaptive_temperatures.parameters():
                        # All ranks must participate in all_reduce even if local grad is None
                        if p.grad is None:
                            p.grad = torch.zeros_like(p.data)
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad /= get_world_size()
                if args.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(adaptive_temperatures.parameters(), args.max_grad_norm)
                temp_optimizer.step()
            mean_loss = gather_scalar(loss)
            denom = max(1, len(loss_terms))
            metric_denom = float(denom)

            mean_avg_retrieved = avg_retrieved / metric_denom
            mean_target_entropy = avg_target_entropy / metric_denom
            mean_best_doc_rank = avg_best_doc_rank / metric_denom
            mean_gold_mass = avg_gold_mass / metric_denom
            mean_negatives = avg_negatives / max(1.0, usable_examples)
            mean_top_positive_rank = avg_top_positive_rank / max(avg_has_top_positive, 1.0)
            hit_top_positive = avg_has_top_positive / max(1.0, usable_examples)
            usable_example_rate = usable_examples / max(1.0, examined_examples)

            mean_rag_component = avg_rag_component / max(1.0, denom)
            mean_contr_component = avg_contr_component / max(1.0, usable_examples)
            mean_weighted_rag_component = avg_weighted_rag_component / max(1.0, denom)
            mean_weighted_contr_component = avg_weighted_contr_component / max(1.0, usable_examples)
            mean_retr_temperature = avg_retr_temperature / max(1.0, denom)
            mean_contr_temperature = avg_contr_temperature / max(1.0, denom)
            mean_raw_retr_temperature = avg_raw_retr_temperature / max(1.0, denom)
            mean_raw_contr_temperature = avg_raw_contr_temperature / max(1.0, denom)
            contr_examples_fraction = usable_examples / max(1.0, denom)

            mean_reg_total = avg_reg_total / max(1.0, denom)
            mean_reg_weight_anchor = avg_reg_weight_anchor / max(1.0, denom)
            mean_reg_query_distill = avg_reg_query_distill / max(1.0, denom)
            mean_reg_rank_distill = avg_reg_rank_distill / max(1.0, denom)

            if global_step % args.log_every == 0:
                grad_norm_val = float(grad_norm.item()) if grad_norm is not None and torch.is_tensor(grad_norm) else None
                msg = (
                    f"phase={phase_name} step={global_step} objective=rag_contriever_static_adaptive_temp_reg loss={mean_loss:.6f} "
                    f"batch={len(batch_q)} avg_retrieved={mean_avg_retrieved:.2f} "
                    f"best_doc_rank={mean_best_doc_rank:.2f} best_doc_target_mass={mean_gold_mass:.4f} "
                    f"t_retr={retrieval_time:.2f}s t_teacher={teacher_time:.2f}s t_train={train_time:.2f}s "
                    f"t_total={time.time()-t_step0:.2f}s grad_norm={grad_norm_val} "
                    f"rag_loss={mean_rag_component:.6f} contr_loss={mean_contr_component:.6f} "
                    f"weighted_rag_loss={mean_weighted_rag_component:.6f} "
                    f"weighted_contr_loss={mean_weighted_contr_component:.6f} "
                    f"rag_weight={cfg.mix_rag_weight:.6f} contr_weight={cfg.mix_contriever_weight:.6f} "
                    f"target_entropy={mean_target_entropy:.4f} avg_negatives={mean_negatives:.2f} "
                    f"top_positive_hit_rate={hit_top_positive:.4f} "
                    f"avg_top_positive_rank_when_hit={mean_top_positive_rank:.2f} "
                    f"usable_example_rate={usable_example_rate:.4f} contr_examples_fraction={contr_examples_fraction:.4f} "
                    f"retr_temperature={mean_retr_temperature:.6f} "
                    f"contrastive_temperature={mean_contr_temperature:.6f} "
                    f"raw_retr_temperature={mean_raw_retr_temperature:.6f} "
                    f"raw_contrastive_temperature={mean_raw_contr_temperature:.6f} "
                    f"reg_total={mean_reg_total:.6f} "
                    f"reg_weight_anchor={mean_reg_weight_anchor:.6f} "
                    f"reg_query_distill={mean_reg_query_distill:.6f} "
                    f"reg_rank_distill={mean_reg_rank_distill:.6f}"
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
        torch.save(
            {
                "raw_retr_temperature": adaptive_temperatures.raw_retr_temperature.detach().cpu(),
                "raw_contrastive_temperature": adaptive_temperatures.raw_contrastive_temperature.detach().cpu(),
                "retr_temperature": adaptive_temperatures.retr_temperature().detach().cpu(),
                "contrastive_temperature": adaptive_temperatures.contrastive_temperature().detach().cpu(),
            },
            os.path.join(args.out_dir, "adaptive_temperatures.pt"),
        )
        logger.info(f"Saved final query encoder to {ckpt_dir}")
        logger.info(f"Saved adaptive temperatures to {os.path.join(args.out_dir, 'adaptive_temperatures.pt')}")

    store.close()
    cleanup_distributed()


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Distributed retriever training with static mixing of RAG and InfoNCE, adaptive objective temperatures, and optional base-preserving regularization."
    )

    ap.add_argument("--train_jsonl", required=True, help="Aligned JSONL with question + answer + top_positive_text (+ optional positive_vids).")
    ap.add_argument("--val_jsonl", default=None, help="Aligned JSONL with validation data.")
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--sqlite_db", required=True)

    ap.add_argument("--embed_model", default="intfloat/e5-large-v2")
    ap.add_argument("--lm_name", required=True)
    ap.add_argument("--out_dir", default="./retriever_qa_out")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature_lr", type=float, default=-1.0, help="LR for adaptive temperatures; if <= 0, uses --lr")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--retr_temperature", type=float, default=1.0, help="Initial RAG retrieval softmax temperature")
    ap.add_argument("--contrastive_temperature", type=float, default=0.05, help="Initial InfoNCE temperature")
    ap.add_argument("--temperature_min", type=float, default=0.05)
    ap.add_argument("--temperature_max", type=float, default=10.0)
    ap.add_argument("--temperature_l2", type=float, default=0.0, help="Optional L2 regularization on log-temperatures")

    ap.add_argument("--contrastive_num_negatives", type=int, default=3)
    ap.add_argument("--contrastive_exclude_same_doc", action="store_true", help="Exclude retrieved negatives that share doc_id with the example")

    ap.add_argument("--mix_rag_weight", type=float, default=1.0, help="Static weight for the RAG loss term")
    ap.add_argument("--mix_contriever_weight", type=float, default=1.0, help="Static weight for the contrastive loss term")

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

    # Base-preserving regularization switches. These are applied only during
    # query-encoder training phases, not during adaptive temperature updates.
    ap.add_argument(
        "--regularizers",
        nargs="*",
        default=[],
        choices=["weight_anchor", "query_distill", "rank_distill"],
        help="Enable any subset of the base-preserving regularizers",
    )
    ap.add_argument("--weight_anchor_coef", type=float, default=1.0)
    ap.add_argument("--query_distill_coef", type=float, default=1.0)
    ap.add_argument("--rank_distill_coef", type=float, default=1.0)
    ap.add_argument("--query_distill_type", default="cosine", choices=["cosine", "mse"])
    ap.add_argument("--rank_distill_temperature", type=float, default=1.0)
    return ap.parse_args()


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
