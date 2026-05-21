#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

def build_logger(log_dir: str, name: str = "rag_retriever_lm_train_qa") -> logging.Logger:
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

    def encode_numpy(self, texts: List[str], max_len: int, batch_size: int = 64) -> np.ndarray:
        was_training = self.training
        self.eval()
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                vec = self.encode(chunk, max_len=max_len)
                all_vecs.append(vec.detach().cpu().numpy().astype(np.float32))
        if was_training:
            self.train()
        return np.concatenate(all_vecs, axis=0)


# ============================================================
# Data
# ============================================================

def _pick_first_present(obj: dict, keys: List[str]) -> Optional[object]:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


def _normalize_question(val: object) -> Optional[str]:
    if isinstance(val, str):
        q = val.strip()
        return q if q else None
    if isinstance(val, list):
        for item in val:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def _normalize_answers(val: object) -> List[str]:
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
    retr_temperature: float
    normalize_by_y_length: bool
    qa_prompt_prefix: str
    doc_prefix: str
    question_prefix: str
    answer_prefix: str
    section_sep: str


def gather_scalar(x: torch.Tensor) -> float:
    if not is_dist():
        return float(x.item())
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y = y / get_world_size()
    return float(y.item())


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad_(requires_grad)


def maybe_ddp(module: nn.Module, should_wrap: bool, local_rank: int, device: torch.device, static_graph: bool = False) -> nn.Module:
    if should_wrap and is_dist():
        return DDP(
            module,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
            static_graph=static_graph,
        )
    return module


def format_qa_prompt(question: str, document: str, cfg: TrainConfig) -> str:
    return (
        f"{cfg.qa_prompt_prefix}"
        f"{cfg.doc_prefix}{document}"
        f"{cfg.section_sep}{cfg.question_prefix}{question}"
        f"{cfg.section_sep}{cfg.answer_prefix}"
    )


def compute_continuation_loglikelihood(
    lm: nn.Module,
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

    ll = torch.zeros(input_ids.size(0), device=device, dtype=logprobs.dtype)
    for i in range(input_ids.size(0)):
        pl = int(prompt_lens[i].item())
        fl = int(full_lens[i].item())
        start_pred = max(pl - 1, 0)
        end_pred = fl - 1
        targets = input_ids[i, pl:fl]
        pred = logprobs[i, start_pred:end_pred, :]
        ll[i] = pred.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum()
    return ll


def rag_sequence_marginal_nll(retrieval_logits: torch.Tensor, doc_loglik: torch.Tensor) -> torch.Tensor:
    log_p_eta = F.log_softmax(retrieval_logits, dim=0)
    return -torch.logsumexp(log_p_eta + doc_loglik, dim=0)


# ============================================================
# LM / LoRA helpers
# ============================================================

def parse_target_modules(s: Optional[str]) -> List[str]:
    if not s:
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return [x.strip() for x in s.split(",") if x.strip()]


def maybe_enable_gradient_checkpointing(model: nn.Module, enabled: bool):
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False


def build_lm_model(args, device: torch.device, optimize_lm: bool, logger: logging.Logger):
    lm_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.lm_dtype]

    if (args.load_in_4bit or args.load_in_8bit) and args.lora_r <= 0:
        raise ValueError("4-bit/8-bit LM loading is only supported here together with LoRA adapters. Set --lora_r > 0.")

    load_kwargs: Dict[str, Any] = {}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = lm_dtype
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    if args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    logger.info(f"Loading generator LM: {args.lm_name}")
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_name, **load_kwargs)
    if device.type == "cuda" and not (args.load_in_4bit or args.load_in_8bit):
        lm_model = lm_model.to(device)

    if hasattr(lm_model, "config"):
        lm_model.config.use_cache = False if args.gradient_checkpointing else lm_model.config.use_cache

    lm_is_peft = False
    if optimize_lm and args.lora_r > 0:
        if args.load_in_4bit or args.load_in_8bit:
            lm_model = prepare_model_for_kbit_training(
                lm_model,
                use_gradient_checkpointing=args.gradient_checkpointing,
            )
        target_modules = parse_target_modules(args.lora_target_modules)
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        lm_model = get_peft_model(lm_model, lora_cfg)
        lm_is_peft = True
        logger.info(
            f"Applied LoRA to LM | r={args.lora_r} alpha={args.lora_alpha} "
            f"dropout={args.lora_dropout} target_modules={target_modules}"
        )
    elif args.lora_r > 0 and not optimize_lm:
        logger.info("LoRA args were provided but LM optimization is disabled, so LoRA adapters were not attached.")

    maybe_enable_gradient_checkpointing(lm_model, args.gradient_checkpointing)

    if not optimize_lm:
        set_requires_grad(lm_model, False)
    elif lm_is_peft:
        # Keep base LM frozen and train only adapters when LoRA is enabled.
        set_requires_grad(lm_model, False)
        for name, p in lm_model.named_parameters():
            if "lora_" in name:
                p.requires_grad_(True)
    else:
        set_requires_grad(lm_model, True)

    lm_model.train() if optimize_lm else lm_model.eval()
    return lm_model, lm_is_peft


def trainable_parameter_summary(model: nn.Module) -> Tuple[int, int]:
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def save_lm_artifacts(lm_model: nn.Module, lm_tok: AutoTokenizer, out_dir: str, merge_lora: bool, logger: logging.Logger):
    os.makedirs(out_dir, exist_ok=True)
    base_model = unwrap_model(lm_model)

    if merge_lora and hasattr(base_model, "merge_and_unload"):
        logger.info("Merging LoRA adapters into base LM weights before save")
        merged = base_model.merge_and_unload()
        merged.save_pretrained(out_dir)
    else:
        base_model.save_pretrained(out_dir)
    lm_tok.save_pretrained(out_dir)


# ============================================================
# Train
# ============================================================

def train(args, logger: logging.Logger):
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting process | rank={rank} world_size={world_size} local_rank={local_rank} device={device}")

    if args.deepspeed_config:
        logger.info(
            "A DeepSpeed config path was provided, but this script still trains with torchrun/DDP only; "
            "the JSON is not consumed by the optimizer loop."
        )

    set_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    optimize_query = args.optimize in {"query", "both"}
    optimize_lm = args.optimize in {"lm", "both"}
    if not optimize_query and not optimize_lm:
        raise ValueError("Nothing to optimize. Use --optimize query, lm, or both.")

    store = AutoChunkStore(args.sqlite_db)
    index = load_faiss_index(args.faiss_index, nprobe=args.nprobe, ef_search=args.hnsw_ef_search)
    logger.info(f"Loaded FAISS index from {args.faiss_index} | ntotal={index.ntotal}")

    logger.info("Loading QA examples from JSONL...")
    all_examples = load_qa_pairs_from_jsonl(args.train_jsonl, max_examples=args.max_examples)
    logger.info(f"Loaded {len(all_examples)} total examples before rank sharding")
    dataset = ShardedDataset(all_examples, seed=args.seed)

    query_encoder = BiEncoderSide(args.embed_model, side="query").to(device)
    doc_encoder = BiEncoderSide(args.embed_model, side="doc").to(device)
    doc_encoder.eval()
    set_requires_grad(doc_encoder, False)

    set_requires_grad(query_encoder, optimize_query)
    query_encoder.train() if optimize_query else query_encoder.eval()
    query_model = maybe_ddp(query_encoder, optimize_query, local_rank, device, static_graph=False)

    lm_tok = AutoTokenizer.from_pretrained(args.lm_name, use_fast=True)
    if lm_tok.pad_token_id is None:
        lm_tok.pad_token = lm_tok.eos_token

    lm_model, lm_is_peft = build_lm_model(args, device, optimize_lm, logger)
    lm_model = maybe_ddp(
        lm_model,
        optimize_lm,
        local_rank,
        device,
        static_graph=bool(optimize_lm and args.gradient_checkpointing),
    )

    if optimize_query:
        q_trainable, q_total = trainable_parameter_summary(unwrap_model(query_model))
        logger.info(f"Query encoder trainable params: {q_trainable}/{q_total}")
    if optimize_lm:
        lm_trainable, lm_total = trainable_parameter_summary(unwrap_model(lm_model))
        logger.info(f"LM trainable params: {lm_trainable}/{lm_total} | lora_enabled={lm_is_peft}")

    optim_groups = []
    if optimize_query:
        query_params = [p for p in query_model.parameters() if p.requires_grad]
        optim_groups.append({"params": query_params, "lr": args.lr_query})
    if optimize_lm:
        lm_params = [p for p in lm_model.parameters() if p.requires_grad]
        optim_groups.append({"params": lm_params, "lr": args.lr_lm})
    optimizer = AdamW(optim_groups, weight_decay=args.weight_decay)

    cfg = TrainConfig(
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
    )

    global_step = 0
    epoch = 0
    stop_training = False

    while not stop_training:
        epoch += 1
        logger.info(f"Starting epoch {epoch}")

        for batch in dataset.iter_batches(batch_size=args.batch_size, epoch=epoch):
            t_step0 = time.time()
            batch_q = [x for x, _ in batch]
            batch_a = [y for _, y in batch]
            loss_terms = []
            avg_retrieved = 0.0
            avg_target_entropy = 0.0
            avg_best_doc_rank = 0.0
            avg_gold_mass = 0.0

            t0 = time.time()
            retrieval_encoder = unwrap_model(query_model)
            q_vecs = retrieval_encoder.encode_numpy(
                batch_q, max_len=cfg.retr_max_len, batch_size=cfg.retrieval_encode_batch_size
            )
            _, vids_np = index.search(q_vecs.astype(np.float32), cfg.top_k)
            retrieval_time = time.time() - t0

            docs_per_example: List[List[str]] = []
            for i in range(len(batch_q)):
                vids = [int(v) for v in vids_np[i].tolist() if int(v) >= 0]
                docs_per_example.append(store.get_texts_by_vids(vids))

            query_time = 0.0
            lm_time = 0.0
            for q, a, docs in zip(batch_q, batch_a, docs_per_example):
                if len(docs) == 0:
                    continue

                avg_retrieved += len(docs)
                prompts = [format_qa_prompt(q, d, cfg) for d in docs]
                answers = [a] * len(docs)

                train_query_context = torch.enable_grad() if optimize_query else torch.no_grad()
                train_lm_context = torch.enable_grad() if optimize_lm else torch.no_grad()

                t1_i = time.time()
                with train_query_context:
                    q_emb = unwrap_model(query_model).encode([q], max_len=cfg.retr_max_len)
                with torch.no_grad():
                    d_emb = doc_encoder.encode(docs, max_len=cfg.retr_max_len)
                retr_logits = (q_emb @ d_emb.T).squeeze(0) / cfg.retr_temperature
                query_time += time.time() - t1_i

                t2_i = time.time()
                ll_chunks = []
                with train_lm_context:
                    for i in range(0, len(prompts), cfg.teacher_batch_size):
                        ll = compute_continuation_loglikelihood(
                            lm_model,
                            lm_tok,
                            prompts=prompts[i:i + cfg.teacher_batch_size],
                            continuations=answers[i:i + cfg.teacher_batch_size],
                            max_len=cfg.lm_max_len,
                        )
                        ll_chunks.append(ll)
                doc_ll = torch.cat(ll_chunks, dim=0)
                if cfg.normalize_by_y_length:
                    y_len = max(1, len(lm_tok(a, add_special_tokens=False).input_ids))
                    doc_ll = doc_ll / y_len
                lm_time += time.time() - t2_i

                loss_i = rag_sequence_marginal_nll(retr_logits, doc_ll)
                loss_terms.append(loss_i)

                with torch.no_grad():
                    log_p_eta = F.log_softmax(retr_logits.detach(), dim=0)
                    log_post = log_p_eta + doc_ll.detach()
                    log_post = log_post - torch.logsumexp(log_post, dim=0)
                    target_dist = log_post.exp()
                    entropy = -(target_dist * log_post).sum()
                    avg_target_entropy += float(entropy.item())
                    best_idx = int(torch.argmax(doc_ll.detach()).item())
                    avg_gold_mass += float(target_dist[best_idx].item())
                    rank_pos = int((torch.argsort(retr_logits.detach(), descending=True) == best_idx).nonzero(as_tuple=False)[0].item()) + 1
                    avg_best_doc_rank += rank_pos

            if not loss_terms:
                logger.warning("No usable examples in batch after retrieval; skipping")
                continue

            loss = torch.stack(loss_terms).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm_query = None
            grad_norm_lm = None
            if args.max_grad_norm > 0:
                if optimize_query:
                    grad_norm_query = torch.nn.utils.clip_grad_norm_([
                        p for p in query_model.parameters() if p.requires_grad
                    ], args.max_grad_norm)
                if optimize_lm:
                    grad_norm_lm = torch.nn.utils.clip_grad_norm_([
                        p for p in lm_model.parameters() if p.requires_grad
                    ], args.max_grad_norm)
            optimizer.step()

            global_step += 1
            mean_loss = gather_scalar(loss)
            denom = max(1.0, float(len(loss_terms)))
            mean_avg_retrieved = avg_retrieved / denom
            mean_target_entropy = avg_target_entropy / denom
            mean_best_doc_rank = avg_best_doc_rank / denom
            mean_gold_mass = avg_gold_mass / denom

            if global_step % args.log_every == 0:
                grad_norm_query_val = float(grad_norm_query.item()) if grad_norm_query is not None and torch.is_tensor(grad_norm_query) else None
                grad_norm_lm_val = float(grad_norm_lm.item()) if grad_norm_lm is not None and torch.is_tensor(grad_norm_lm) else None
                logger.info(
                    f"step={global_step} optimize={args.optimize} loss={mean_loss:.6f} batch={len(batch_q)} "
                    f"avg_retrieved={mean_avg_retrieved:.2f} best_doc_rank={mean_best_doc_rank:.2f} "
                    f"best_doc_target_mass={mean_gold_mass:.4f} target_entropy={mean_target_entropy:.4f} "
                    f"t_retr={retrieval_time:.2f}s t_query={query_time:.2f}s t_lm={lm_time:.2f}s "
                    f"t_total={time.time() - t_step0:.2f}s grad_norm_query={grad_norm_query_val} grad_norm_lm={grad_norm_lm_val}"
                )

            if global_step >= args.max_steps:
                stop_training = True
                break

    barrier()
    if is_main_process():
        if optimize_query:
            ckpt_dir = os.path.join(args.out_dir, "query_encoder_final")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrap_model(query_model).encoder.save_pretrained(ckpt_dir)
            unwrap_model(query_model).tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"Saved final query encoder to {ckpt_dir}")

        if optimize_lm:
            ckpt_dir = os.path.join(args.out_dir, "lm_final")
            save_lm_artifacts(lm_model, lm_tok, ckpt_dir, merge_lora=args.merge_lora, logger=logger)
            logger.info(f"Saved final LM to {ckpt_dir}")

    store.close()
    cleanup_distributed()


# ============================================================
# CLI
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Distributed RAG training with configurable optimization of query encoder, LM, or both")

    ap.add_argument("--train_jsonl", required=True, help="QA JSONL")
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--sqlite_db", required=True)

    ap.add_argument("--embed_model", default="intfloat/e5-large-v2")
    ap.add_argument("--lm_name", required=True)
    ap.add_argument("--out_dir", default="./rag_qa_out")

    ap.add_argument("--optimize", default="query", choices=["query", "lm", "both"], help="Which trainable component(s) to optimize")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--lr_query", type=float, default=2e-5)
    ap.add_argument("--lr_lm", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--top_k", type=int, default=16)
    ap.add_argument("--retr_temperature", type=float, default=1.0, help="RAG retrieval softmax temperature")

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
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--deepspeed_config", type=str, default=None, help="Accepted for compatibility/documentation only; this script still uses torchrun/DDP.")

    ap.add_argument("--lora_r", type=int, default=0, help="0 disables LoRA; >0 enables LM LoRA fine-tuning")
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", type=str, default=None, help="Comma-separated target modules (default: q_proj,k_proj,v_proj,o_proj)")
    ap.add_argument("--load_in_8bit", action="store_true", help="Load LM in 8-bit for LoRA training")
    ap.add_argument("--load_in_4bit", action="store_true", help="Load LM in 4-bit for LoRA/QLoRA-style training")
    ap.add_argument("--merge_lora", action="store_true", help="Merge LoRA adapters into base LM weights on save")

    ap.add_argument("--normalize_by_y_length", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    if args.load_in_4bit and args.load_in_8bit:
        ap.error("Choose only one of --load_in_4bit or --load_in_8bit (or neither).")
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
