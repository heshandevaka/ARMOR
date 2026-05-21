#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizerBase


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_docs_block(docs: List[Tuple[str, str]]) -> str:
    """
    docs: list of (doc_tag, doc_text)
    """
    out = []
    for i, (tag, text) in enumerate(docs, start=1):
        out.append(f"[Doc {i} | {tag}]\n{text}")
    return "\n\n".join(out)


class RaftMultiDocDataset(Dataset):
    """
    RAFT-style training dataset built from your JSONL.

    Each training instance:
      input:  Question + (Docs = golden + distractors OR distractors only) + Instruction
      target: "##Reason: ...\n##Answer: ..."

    We sample distractors from other examples' contexts.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        num_distractors: int = 4,
        p_golden: float = 0.8,
        seed: int = 42,
        # optional quality filtering if present
        require_keep: bool = True,
        min_quality: float = 0.0,
        require_supported: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_distractors = num_distractors
        self.p_golden = p_golden

        rng = random.Random(seed)

        rows = load_jsonl(jsonl_path)
        # Optional quality filtering
        filtered: List[Dict[str, Any]] = []
        for ex in rows:
            qmeta = ex.get("quality", {}) or {}
            if require_keep and ("keep" in qmeta) and not bool(qmeta["keep"]):
                continue
            if require_supported and ("supported_by_context" in qmeta) and not bool(qmeta["supported_by_context"]):
                continue
            if "overall_quality" in qmeta:
                try:
                    if float(qmeta["overall_quality"]) < min_quality:
                        continue
                except Exception:
                    continue

            question = _as_str(ex.get("question")).strip()
            answer = _as_str(ex.get("answer")).strip()
            context = _as_str(ex.get("context")).strip()
            
            # More lenient on reason and instruction for datasets that don't have them yet
            reason = _as_str(ex.get("reason")).strip()
            if not reason:
                reason = "Information found in the provided documents."
                
            instruction = _as_str(ex.get("instruction")).strip()
            if not instruction:
                instruction = "Given the documents below, answer the question based strictly on the information provided."

            if not (question and answer and context):
                continue
            
            # Put back the possibly defaulted fields into the dict for later use
            ex["question"] = question
            ex["answer"] = answer
            ex["reason"] = reason
            ex["context"] = context
            ex["instruction"] = instruction
            
            filtered.append(ex)

        if len(filtered) < 2:
            raise ValueError(
                f"Need at least 2 usable examples to sample distractors, got {len(filtered)}. "
                "Check if your data has 'question', 'answer', and 'context' fields."
            )

        # Build a pool of distractor contexts with tags
        # We tag docs with source/doc_id/chunk_id to mimic "documents" concept.
        self.pool: List[Tuple[str, str]] = []
        for ex in filtered:
            source = ex.get("source") or "unknown"
            doc_id = ex.get("doc_id") or "unknown"
            chunk_id = ex.get("chunk_id") or ex.get("source_chunk_id") or "unknown"
            tag = f"{_as_str(source)}:{_as_str(doc_id)}:{_as_str(chunk_id)}"
            self.pool.append((tag, _as_str(ex.get("context")).strip()))

        # Precompute tokenized prompt/target IDs for each example, but
        # distractors are sampled dynamically in __getitem__ to vary them each epoch.
        self.base: List[Dict[str, Any]] = []
        for ex in filtered:
            source = ex.get("source") or "unknown"
            doc_id = ex.get("doc_id") or "unknown"
            chunk_id = ex.get("chunk_id") or ex.get("source_chunk_id") or "unknown"
            tag = f"{_as_str(source)}:{_as_str(doc_id)}:{_as_str(chunk_id)}"
            
            ex_id = ex.get("id") or ex.get("example_id") or "unknown"
            
            base_item = {
                "id": _as_str(ex_id),
                "question": ex["question"],
                "answer": ex["answer"],
                "reason": ex["reason"],
                "gold_tag": tag,
                "gold_context": ex["context"],
                "instruction": ex["instruction"],
            }
            self.base.append(base_item)

        self.rng = rng

        # sanity print
        print(
            f"[RaftMultiDocDataset] loaded={len(self.base)} "
            f"num_distractors={num_distractors} p_golden={p_golden} max_length={max_length}"
        )

    def __len__(self) -> int:
        return len(self.base)

    def _sample_distractors(self, avoid_tag: str, k: int) -> List[Tuple[str, str]]:
        """
        Sample k distractor docs from pool excluding avoid_tag.
        """
        candidates = [x for x in self.pool if x[0] != avoid_tag]
        if not candidates:
            return []
        # sample without replacement if possible
        if len(candidates) >= k:
            return self.rng.sample(candidates, k)
        # else sample with replacement
        return [self.rng.choice(candidates) for _ in range(k)]

    @staticmethod
    def build_prompt(question: str, docs_block: str, instruction: str) -> str:
        # This is the RAFT-style “Q + Dk + Instruction” structure.
        return (
            f"Question: {question}\n\n"
            f"Documents:\n{docs_block}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"CoT Answer:\n"
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.base[idx]

        # Decide whether to include golden doc this time (P%).
        include_golden = (self.rng.random() < self.p_golden)

        distractors = self._sample_distractors(ex["gold_tag"], self.num_distractors)

        docs: List[Tuple[str, str]] = []
        if include_golden:
            docs.append((f"GOLD:{ex['gold_tag']}", ex["gold_context"]))
        for tag, text in distractors:
            docs.append((f"DISTRACTOR:{tag}", text))

        docs_block = build_docs_block(docs)
        prompt = self.build_prompt(ex["question"], docs_block, ex["instruction"])

        # Target is the RAFT CoT+Answer format.
        target = f"##Reason: {ex['reason']}\n##Answer: {ex['answer']}."

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # If too long, truncate documents block from the left (drop earliest docs)
        # while keeping question/instruction/target intact.
        # Simple approach: if over length, chop prompt_ids from the front.
        full_no_special = prompt_ids + target_ids
        # +2 is a safe-ish buffer for BOS/EOS, model dependent
        if len(full_no_special) + 2 > self.max_length:
            overflow = (len(full_no_special) + 2) - self.max_length
            # remove overflow tokens from the start of prompt_ids (not target)
            if overflow < len(prompt_ids):
                prompt_ids = prompt_ids[overflow:]
            else:
                # prompt got obliterated; as last resort, keep a minimal prefix
                prompt_ids = prompt_ids[-64:]

        full_no_special = prompt_ids + target_ids
        input_ids = self.tokenizer.build_inputs_with_special_tokens(full_no_special)

        # labels: mask prompt & specials, train on target tokens
        labels = [-100] * len(input_ids)
        target_len = len(target_ids)
        # assume specials are only at boundaries; supervise final target_len tokens
        start = len(input_ids) - target_len
        for i in range(start, len(input_ids)):
            labels[i] = input_ids[i]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_masks = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
        attention_mask_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.label_pad_token_id)

        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask_padded,
            "labels": labels_padded,
        }
