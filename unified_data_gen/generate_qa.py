#!/usr/bin/env python
from __future__ import annotations

import json
import uuid
import argparse
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ------------------------------------------------------------
# 1. Load docs (flat jsonl: doc_id, category, text, metadata)
# ------------------------------------------------------------

def load_docs_from_jsonl(path: str) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "text" in row and "doc_id" in row:
                content = row.get("text", "") or ""
                if not content.strip():
                    continue
                meta = row.get("metadata", {}) or {}
                docs.append(
                    {
                        "id": row["doc_id"],
                        "source": row.get("category", "unknown"),
                        "content": content,
                        "metadata": meta,
                    }
                )
    return docs

# ------------------------------------------------------------
# 2. Chunking
# ------------------------------------------------------------

def simple_word_chunk(text: str,
                      max_words: int = 200,
                      overlap_words: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_words
    return chunks

def chunk_docs(docs: List[Dict[str, Any]],
               max_words: int = 200,
               overlap_words: int = 40) -> List[Dict[str, Any]]:
    all_chunks: List[Dict[str, Any]] = []
    for doc in docs:
        chunks = simple_word_chunk(doc["content"], max_words, overlap_words)
        for i, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "doc_id": doc["id"],
                    "source": doc["source"],
                    "chunk_index": i,
                    "chunk_id": f"{doc['id']}__{i}",
                    "text": ch,
                    "metadata": doc["metadata"],
                }
            )
    return all_chunks

# ------------------------------------------------------------
# 3. vLLM-based QA generator
# ------------------------------------------------------------

class QAGenerator:
    def __init__(
        self,
        model_name: str,
        domain_expert_prompt: str,
        tp_size: int = 4,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.domain_expert_prompt = domain_expert_prompt
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
        self.params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def _chat_prompt(self, system: str, user: str) -> str:
        return (
            f"System:\n{system}\n\n"
            f"User:\n{user}\n\n"
            "Assistant:\n"
        )

    def _build_prompt(self, context: str, num_questions: int) -> str:
        system = (
            f"{self.domain_expert_prompt} "
            "Your task is to create high-quality QA training data for RAG systems. "
            "All questions and answers must be strictly grounded in the provided CONTEXT."
        )
        user = f"""
Given the CONTEXT below, generate {num_questions} high-quality technical QA pairs.

RULES:
1. Each question must be SELF-CONTAINED (don't say "according to the text").
2. Each question must be answerable SOLELY using the context.
3. The answer should be concise and technically accurate.

OUTPUT FORMAT:
Return ONLY a JSON list of objects:
[
  {{"question": "...", "answer": "..."}},
  ...
]

CONTEXT:
\"\"\"{context}\"\"\"
"""
        return self._chat_prompt(system, user)

    def generate_batch(self, contexts: List[str], num_questions: int = 1) -> List[List[Dict[str, str]]]:
        prompts = [self._build_prompt(c, num_questions) for c in contexts]
        outputs = self.llm.generate(prompts, self.params)
        all_results = []

        for out in outputs:
            text = out.outputs[0].text.strip()
            try:
                start = text.index("[")
                end = text.rindex("]") + 1
                data = json.loads(text[start:end])
                cleaned = []
                for item in data:
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    if q and a:
                        cleaned.append({"question": q, "answer": a})
                all_results.append(cleaned)
            except:
                all_results.append([])
        return all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--domain", default="isac", help="Domain to use for expert prompt")
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Dynamic Domain Loading for the expert prompt
    try:
        domain_prompts = importlib.import_module(f"domains.{args.domain}.prompts")
        expert_prompt = domain_prompts.QA_SYSTEM_PROMPT
    except (ImportError, AttributeError):
        expert_prompt = f"You are a domain expert in {args.domain.upper()}."

    docs = load_docs_from_jsonl(args.input_jsonl)
    chunks = chunk_docs(docs)
    print(f"Loaded {len(docs)} docs, {len(chunks)} chunks.")

    gen = QAGenerator(model_name=args.model, domain_expert_prompt=expert_prompt, tp_size=args.tp_size)
    
    with open(args.output_jsonl, "w") as out_f:
        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i:i+args.batch_size]
            contexts = [c["text"] for c in batch]
            results = gen.generate_batch(contexts, num_questions=args.num_questions)
            
            for chunk, qa_list in zip(batch, results):
                for qa in qa_list:
                    res = {
                        "id": str(uuid.uuid4()),
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "context": chunk["text"],
                        "doc_id": chunk["doc_id"],
                        "chunk_id": chunk["chunk_id"],
                        "metadata": chunk["metadata"]
                    }
                    out_f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
