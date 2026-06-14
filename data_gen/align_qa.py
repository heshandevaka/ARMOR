#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import orjson
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def simple_tokens(text: str) -> List[str]:
    return TOKEN_RE.findall((text or "").lower())


def ngram_set(tokens: Sequence[str], n: int) -> set[str]:
    if len(tokens) < n:
        return set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


FIELD_CANDIDATES = {
    "id": ["id", "example_id", "qid"],
    "doc_id": ["doc_id", "document_id"],
    "chunk_id": ["chunk_id", "source_chunk_id"],
    "question": ["question", "prompt", "query", "input"],
    "answer": ["answer", "completion", "output", "target"],
    "reason": ["reason", "rationale", "explanation"],
    "context": ["context", "passage", "document", "doc_text"],
}


def get_field(obj: dict, key: str, default=None):
    for cand in FIELD_CANDIDATES[key]:
        if cand in obj and obj[cand] is not None:
            return obj[cand]
    return default


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


@dataclass
class ChunkRow:
    vid: int
    doc_id: str
    text: str
    chunk_id: Optional[str] = None
    category: Optional[str] = None


@dataclass
class ScoredChunk:
    row: ChunkRow
    score: float
    contains: bool
    token_jaccard: float
    trigram_jaccard: float
    exact_chunk_id_match: bool


class AutoChunkStore:
    def __init__(
        self,
        sqlite_path: str,
        table: Optional[str] = None,
        id_col: Optional[str] = None,
        text_col: Optional[str] = None,
        doc_id_col: Optional[str] = None,
        chunk_id_col: Optional[str] = None,
        category_col: Optional[str] = None,
    ):
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self.table, self.id_col, self.text_col, self.doc_id_col, self.chunk_id_col, self.category_col = self._discover(
            table, id_col, text_col, doc_id_col, chunk_id_col, category_col
        )

    def close(self):
        self.conn.close()

    def _discover(self, table, id_col, text_col, doc_id_col, chunk_id_col, category_col):
        if table and id_col and text_col and doc_id_col:
            return table, id_col, text_col, doc_id_col, chunk_id_col, category_col

        objs = self.conn.execute(
            """
            SELECT type, name
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' AND type IN ('table','view')
            ORDER BY name
            """
        ).fetchall()
        if not objs:
            raise RuntimeError("No user tables/views found in sqlite DB.")

        best = None
        best_score = -1
        for obj in objs:
            name = obj["name"]
            cols = self.conn.execute(f"PRAGMA table_info({qident(name)})").fetchall()
            colnames = [c["name"] for c in cols]
            score = 0
            if any(c.lower() == "vid" for c in colnames):
                score += 3
            if any(c.lower() == "text" for c in colnames):
                score += 3
            if any(c.lower() == "doc_id" for c in colnames):
                score += 3
            if any(c.lower() == "chunk_id" for c in colnames):
                score += 1
            if score > best_score:
                best_score = score
                best = (name, colnames)

        if best is None:
            raise RuntimeError("Could not infer chunk table from sqlite DB.")

        table_name, colnames = best
        lower = {c.lower(): c for c in colnames}
        rid = id_col or lower.get("vid") or lower.get("id")
        rtext = text_col or lower.get("text") or lower.get("content") or lower.get("body")
        rdoc = doc_id_col or lower.get("doc_id") or lower.get("document_id")
        rchunk = chunk_id_col or lower.get("chunk_id")
        rcat = category_col or lower.get("category") or lower.get("source")
        if not (rid and rtext and rdoc):
            raise RuntimeError(f"Could not infer required columns from {table_name}. Available: {colnames}")
        return table_name, rid, rtext, rdoc, rchunk, rcat

    def fetch_chunks_for_doc_ids(self, doc_ids: Iterable[str]) -> Dict[str, List[ChunkRow]]:
        doc_ids = list({str(d) for d in doc_ids if d is not None and str(d) != ""})
        if not doc_ids:
            return {}
        placeholders = ",".join(["?"] * len(doc_ids))
        extra_chunk = f", {qident(self.chunk_id_col)} AS chunk_id" if self.chunk_id_col else ", NULL AS chunk_id"
        extra_cat = f", {qident(self.category_col)} AS category" if self.category_col else ", NULL AS category"
        query = f"""
            SELECT
                {qident(self.id_col)} AS vid,
                {qident(self.doc_id_col)} AS doc_id,
                {qident(self.text_col)} AS text
                {extra_chunk}
                {extra_cat}
            FROM {qident(self.table)}
            WHERE {qident(self.doc_id_col)} IN ({placeholders})
            ORDER BY {qident(self.doc_id_col)}, {qident(self.id_col)}
        """
        rows = self.conn.execute(query, doc_ids).fetchall()
        out: Dict[str, List[ChunkRow]] = defaultdict(list)
        for r in rows:
            out[str(r["doc_id"])].append(
                ChunkRow(
                    vid=int(r["vid"]),
                    doc_id=str(r["doc_id"]),
                    text=str(r["text"]),
                    chunk_id=r["chunk_id"],
                    category=r["category"],
                )
            )
        return out


def score_chunk(context: str, row: ChunkRow, qa_chunk_id: Optional[str] = None) -> ScoredChunk:
    ctext = context or ""
    rtext = row.text or ""
    c_lower = ctext.lower()
    r_lower = rtext.lower()
    contains = (c_lower in r_lower) or (r_lower in c_lower)
    c_toks = simple_tokens(ctext)
    r_toks = simple_tokens(rtext)
    tok_j = jaccard(set(c_toks), set(r_toks))
    tri_j = jaccard(ngram_set(c_toks, 3), ngram_set(r_toks, 3))
    exact_chunk_id_match = bool(qa_chunk_id and row.chunk_id and str(qa_chunk_id) == str(row.chunk_id))
    score = (3.0 if exact_chunk_id_match else 0.0) + (2.0 if contains else 0.0) + tok_j + 2.0 * tri_j
    return ScoredChunk(row=row, score=score, contains=contains, token_jaccard=tok_j, trigram_jaccard=tri_j, exact_chunk_id_match=exact_chunk_id_match)


def main():
    ap = argparse.ArgumentParser(description="Create a contrastive-training-friendly dataset from QA data and indexed chunks.")
    ap.add_argument("--qa_jsonl", required=True)
    ap.add_argument("--sqlite_db", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--top_k_positive", type=int, default=3)
    ap.add_argument("--min_positive_score", type=float, default=0.05)
    ap.add_argument("--emit_candidate_details", action="store_true")
    ap.add_argument("--sqlite_table", default=None)
    ap.add_argument("--sqlite_id_col", default=None)
    ap.add_argument("--sqlite_text_col", default=None)
    ap.add_argument("--sqlite_doc_id_col", default=None)
    ap.add_argument("--sqlite_chunk_id_col", default=None)
    ap.add_argument("--sqlite_category_col", default=None)
    args = ap.parse_args()

    qa_rows = list(iter_jsonl(args.qa_jsonl))
    doc_ids = [str(get_field(row, "doc_id", "")) for row in qa_rows if get_field(row, "doc_id")]

    store = AutoChunkStore(
        args.sqlite_db,
        table=args.sqlite_table,
        id_col=args.sqlite_id_col,
        text_col=args.sqlite_text_col,
        doc_id_col=args.sqlite_doc_id_col,
        chunk_id_col=args.sqlite_chunk_id_col,
        category_col=args.sqlite_category_col,
    )
    chunks_by_doc = store.fetch_chunks_for_doc_ids(doc_ids)

    num_written = 0
    num_no_doc_match = 0
    num_no_positive = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(qa_rows):
            doc_id = get_field(row, "doc_id")
            question = get_field(row, "question", "")
            answer = get_field(row, "answer", "")
            reason = get_field(row, "reason", "")
            context = get_field(row, "context", "")
            qa_chunk_id = get_field(row, "chunk_id")
            ex_id = get_field(row, "id", i)

            candidates = chunks_by_doc.get(str(doc_id), []) if doc_id is not None else []
            if not candidates:
                num_no_doc_match += 1
                out = {
                    "example_id": ex_id,
                    "doc_id": doc_id,
                    "question": question,
                    "answer": answer,
                    "reason": reason,
                    "context": context,
                    "source_chunk_id": qa_chunk_id,
                    "positive_vids": [],
                    "positive_texts": [],
                    "positive_scores": [],
                    "top_positive_vid": None,
                    "top_positive_text": None,
                    "top_positive_score": None,
                    "num_same_doc_chunks": 0,
                }
                out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
                continue

            scored = [score_chunk(context, c, qa_chunk_id=qa_chunk_id) for c in candidates]
            scored.sort(key=lambda s: (s.score, s.exact_chunk_id_match, s.trigram_jaccard, s.token_jaccard, -int(s.row.vid)), reverse=True)
            keep = [s for s in scored if s.score >= args.min_positive_score][: args.top_k_positive]
            if not keep:
                num_no_positive += 1
                keep = scored[: args.top_k_positive]

            out = {
                "example_id": ex_id,
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "reason": reason,
                "context": context,
                "source_chunk_id": qa_chunk_id,
                "positive_vids": [s.row.vid for s in keep],
                "positive_texts": [s.row.text for s in keep],
                "positive_scores": [round(float(s.score), 6) for s in keep],
                "top_positive_vid": keep[0].row.vid if keep else None,
                "top_positive_text": keep[0].row.text if keep else None,
                "top_positive_score": round(float(keep[0].score), 6) if keep else None,
                "num_same_doc_chunks": len(candidates),
            }
            if args.emit_candidate_details:
                out["positive_candidates"] = [
                    {
                        "vid": s.row.vid,
                        "doc_id": s.row.doc_id,
                        "chunk_id": s.row.chunk_id,
                        "category": s.row.category,
                        "score": round(float(s.score), 6),
                        "contains": bool(s.contains),
                        "token_jaccard": round(float(s.token_jaccard), 6),
                        "trigram_jaccard": round(float(s.trigram_jaccard), 6),
                        "exact_chunk_id_match": bool(s.exact_chunk_id_match),
                        "text_preview": (s.row.text[:300] + "...") if len(s.row.text) > 300 else s.row.text,
                    }
                    for s in keep
                ]
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            num_written += 1

    store.close()
    print(json.dumps({
        "written": num_written,
        "num_examples": len(qa_rows),
        "num_no_doc_match": num_no_doc_match,
        "num_no_positive_above_threshold": num_no_positive,
        "output": args.out_jsonl,
    }, indent=2))


if __name__ == "__main__":
    main()
