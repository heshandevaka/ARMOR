# utils.py
import json
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def needs_e5_prefix(embed_model_name: str) -> bool:
    n = embed_model_name.lower()
    return "e5" in n


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_by_tokens(tokenizer, text: str, max_tokens: int = 384, overlap: int = 64) -> List[str]:
    """
    Token-window chunking (recommended for long arXiv/standards).
    """
    text = clean_text(text)
    if not text:
        return []

    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []

    chunks = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        end = min(len(ids), start + max_tokens)
        piece_ids = ids[start:end]
        piece = tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        if piece:
            chunks.append(piece)
        if end >= len(ids):
            break
    return chunks


def init_sqlite(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        vid INTEGER PRIMARY KEY,
        doc_id TEXT,
        category TEXT,
        chunk_id TEXT,
        text TEXT,
        metadata_json TEXT
    )
    """)
    conn.commit()
    return conn


def sqlite_insert_many(conn, rows: List[Tuple[int, str, str, str, str, str]]):
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO chunks (vid, doc_id, category, chunk_id, text, metadata_json) VALUES (?,?,?,?,?,?)",
        rows
    )


def sqlite_fetch_by_vids(db_path: str, vids: List[int]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    out = []
    for vid in vids:
        cur.execute("SELECT vid, doc_id, category, chunk_id, text, metadata_json FROM chunks WHERE vid=?",
                    (int(vid),))
        row = cur.fetchone()
        if row:
            out.append({
                "vid": row[0],
                "doc_id": row[1],
                "category": row[2],
                "chunk_id": row[3],
                "text": row[4],
                "metadata": json.loads(row[5]) if row[5] else {},
            })
    conn.close()
    return out


def build_prompt_plain(question: str, contexts: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """
    Plain prompt (works for base/instruct models; best results if your model is instruct-tuned).
    """
    blocks = []
    total = 0
    for i, c in enumerate(contexts, start=1):
        block = (
            f"[{i}] doc_id={c['doc_id']} category={c['category']} chunk_id={c['chunk_id']}\n"
            f"{c['text']}\n"
        )
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    ctx = "\n".join(blocks) if blocks else "None"

#     return f"""You are an expert assistant in Integrated Sensing and Communication (ISAC).
# Answer the user's question using ONLY the retrieved context.
# If the context is insufficient, say what is missing, then give a cautious best-effort answer.

# Question:
# {question}

# Retrieved context:
# {ctx}

# Answer (use citations like [1], [2] referring to the context blocks):
# """

    return f"""{ctx}
    {question}"""


def build_chat_input_if_available(tokenizer, user_prompt: str) -> str:
    """
    If the tokenizer has a chat template (common for Qwen/Llama instruct),
    wrap the prompt as a chat conversation. Otherwise, return prompt directly.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return user_prompt


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norm
