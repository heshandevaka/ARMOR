import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import orjson
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from datasets import load_dataset

from utils_index import (
    chunk_by_tokens,
    init_sqlite,
    l2_normalize,
    needs_e5_prefix,
    sqlite_insert_many,
)


def iter_local_jsonl(isac_docs_jsonl: str) -> Iterable[Tuple[str, str, Dict[str, Any], str]]:
    """
    Each line:
      {"doc_id": "...", "category": "...", "metadata": {...}, "text": "..."}
    or compatible variants.

    Yields:
      (doc_id, category, metadata_dict, text)
    """
    with open(isac_docs_jsonl, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = orjson.loads(line)

            doc_id = (
                obj.get("doc_id")
                or obj.get("id")
                or obj.get("ID")
            )
            cat = (
                obj.get("category")
                or obj.get("Category")
                or obj.get("source_category")
                or "unknown"
            )
            meta = obj.get("metadata") or obj.get("Metadata") or {}
            text = (
                obj.get("text")
                or obj.get("content")
                or obj.get("Content")
                or ""
            )

            if doc_id and text:
                yield str(doc_id), str(cat), meta, str(text)


def iter_hf_dataset(dataset_name: str, split: str = "train") -> Iterable[Tuple[str, str, Dict[str, Any], str]]:
    """
    Supports datasets from the Hugging Face Hub, e.g.:
      AliMaatouk/Tele-Data

    Expected fields (case-insensitive / flexible):
      ID / id / doc_id
      Category / category
      Content / content / text
      Metadata / metadata
    """
    ds = load_dataset(dataset_name, split=split)

    for obj in ds:
        doc_id = (
            obj.get("doc_id")
            or obj.get("id")
            or obj.get("ID")
        )
        cat = (
            obj.get("category")
            or obj.get("Category")
            or obj.get("source_category")
            or "unknown"
        )
        meta = obj.get("metadata") or obj.get("Metadata") or {}
        text = (
            obj.get("text")
            or obj.get("content")
            or obj.get("Content")
            or ""
        )

        if doc_id and text:
            if not isinstance(meta, dict):
                meta = {"raw_metadata": meta}
            yield str(doc_id), str(cat), meta, str(text)


def iter_isac_docs(isac_docs_jsonl: str, hf_split: str = "train") -> Iterable[Tuple[str, str, Dict[str, Any], str]]:
    """
    Backward-compatible loader.

    If `isac_docs_jsonl` is a local path, read JSONL as before.
    Otherwise, treat it as a Hugging Face dataset repo id.
    """
    if os.path.exists(isac_docs_jsonl):
        yield from iter_local_jsonl(isac_docs_jsonl)
    else:
        yield from iter_hf_dataset(isac_docs_jsonl, split=hf_split)


def embed_passages(
    embedder: SentenceTransformer,
    embed_model_name: str,
    texts: List[str],
    pool=None,
) -> np.ndarray:
    if needs_e5_prefix(embed_model_name):
        texts = ["passage: " + t for t in texts]
    if pool is not None:
        vecs = embedder.encode_multi_process(texts, pool, normalize_embeddings=True)
    else:
        vecs = embedder.encode(
            texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
        )
    return np.asarray(vecs, dtype=np.float32)


def make_index(index_type: str, dim: int, nlist: int, pq_m: int, hnsw_m: int) -> faiss.Index:
    index_type = index_type.lower()

    if index_type == "flat":
        idx = faiss.IndexFlatIP(dim)
        return idx

    if index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        return idx

    if index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(dim)
        idx = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8)
        idx.metric_type = faiss.METRIC_INNER_PRODUCT
        return idx

    raise ValueError(f"Unknown index_type: {index_type}. Use flat|hnsw|ivfpq")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_jsonl", required=True,
                    help="Local jsonl path OR Hugging Face dataset repo id")
    ap.add_argument("--hf_split", default="train",
                    help="HF dataset split if --docs_jsonl is a dataset repo id")
    ap.add_argument("--out_faiss", default="index.faiss")
    ap.add_argument("--out_sqlite", default="chunks.sqlite")

    ap.add_argument("--embed_model", default="intfloat/e5-large-v2")
    ap.add_argument("--chunk_max_tokens", type=int, default=384)
    ap.add_argument("--chunk_overlap", type=int, default=64)

    ap.add_argument("--index_type", default="ivfpq", choices=["flat", "hnsw", "ivfpq"])
    ap.add_argument("--train_size", type=int, default=200000, help="Only for ivfpq: vectors used to train.")
    ap.add_argument("--nlist", type=int, default=4096, help="Only for ivfpq.")
    ap.add_argument("--pq_m", type=int, default=64, help="Only for ivfpq.")
    ap.add_argument("--hnsw_m", type=int, default=32, help="Only for hnsw.")
    ap.add_argument("--batch_chunks", type=int, default=2048)

    ap.add_argument("--gpu_devices", default="", help='Comma list like "cuda:0,cuda:1,cuda:2" for embedding')
    ap.add_argument("--min_text_chars", type=int, default=200)
    args = ap.parse_args()

    gpu_devices = [d.strip() for d in args.gpu_devices.split(",") if d.strip()] or None

    # Tokenizer for chunking
    tok = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)

    # Embedder
    device0 = gpu_devices[0] if gpu_devices else None
    embedder = SentenceTransformer(args.embed_model, device=device0, trust_remote_code=True)

    pool = None
    if gpu_devices and len(gpu_devices) > 1:
        try:
            pool = embedder.start_multi_process_pool(target_devices=gpu_devices)
        except RuntimeError as e:
            print(f"[WARN] Could not start multi-process pool ({e}). Falling back to single-process on {device0}.")
            pool = None

    # SQLite store
    import os
    if os.path.exists(args.out_sqlite):
        os.remove(args.out_sqlite)
    conn = init_sqlite(args.out_sqlite)

    # Buffers
    buf_texts: List[str] = []
    buf_rows: List[Tuple[str, str, str, str, str]] = []

    # IVF-PQ training cache
    train_vecs: List[np.ndarray] = []
    train_rows: List[Tuple[str, str, str, str, str]] = []

    index: Optional[faiss.Index] = None
    next_vid = 0
    dim = None

    def flush(vecs: np.ndarray, rows: List[Tuple[str, str, str, str, str]]):
        nonlocal index, next_vid
        assert index is not None

        index.add(vecs)

        to_insert = []
        for i, r in enumerate(rows):
            doc_id, cat, chunk_id, text, meta_json = r
            to_insert.append((next_vid + i, doc_id, cat, chunk_id, text, meta_json))
        sqlite_insert_many(conn, to_insert)
        conn.commit()

        next_vid += len(rows)

    doc_iter = iter_isac_docs(args.docs_jsonl, hf_split=args.hf_split)

    for doc_id, cat, meta, text in tqdm(doc_iter, desc="Read docs"):
        if not text or len(text) < args.min_text_chars:
            continue

        chunks = chunk_by_tokens(tok, text, max_tokens=args.chunk_max_tokens, overlap=args.chunk_overlap)
        if not chunks:
            continue

        meta_json = json.dumps(meta, ensure_ascii=False)

        for ci, ch in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk{ci}"
            buf_texts.append(ch)
            buf_rows.append((doc_id, cat, chunk_id, ch, meta_json))

            if len(buf_texts) >= args.batch_chunks:
                vecs = embed_passages(embedder, args.embed_model, buf_texts, pool=pool)

                if dim is None:
                    dim = vecs.shape[1]

                if args.index_type == "ivfpq" and index is None:
                    need = max(0, args.train_size - sum(v.shape[0] for v in train_vecs))
                    if need > 0:
                        take = min(need, vecs.shape[0])
                        train_vecs.append(vecs[:take])
                        train_rows.extend(buf_rows[:take])

                        remaining_vecs = vecs[take:]
                        remaining_rows = buf_rows[take:]

                        buf_texts, buf_rows = [], []

                        got = sum(v.shape[0] for v in train_vecs)
                        if got >= args.train_size:
                            train_mat = np.vstack(train_vecs)
                            index = make_index(args.index_type, dim, args.nlist, args.pq_m, args.hnsw_m)
                            index.train(train_mat)

                            flush(train_mat, train_rows)

                            if remaining_vecs.shape[0] > 0:
                                flush(remaining_vecs, remaining_rows)

                            train_vecs.clear()
                            train_rows.clear()
                        continue

                if index is None:
                    index = make_index(args.index_type, dim, args.nlist, args.pq_m, args.hnsw_m)

                flush(vecs, buf_rows)
                buf_texts, buf_rows = [], []

    if buf_texts:
        vecs = embed_passages(embedder, args.embed_model, buf_texts, pool=pool)
        if dim is None:
            dim = vecs.shape[1]

        if args.index_type == "ivfpq" and index is None:
            train_vecs.append(vecs)
            train_rows.extend(buf_rows)
            train_mat = np.vstack(train_vecs)
            index = make_index(args.index_type, dim, args.nlist, args.pq_m, args.hnsw_m)
            index.train(train_mat)
            flush(train_mat, train_rows)
        else:
            if index is None:
                index = make_index(args.index_type, dim, args.nlist, args.pq_m, args.hnsw_m)
            flush(vecs, buf_rows)

    if pool is not None:
        embedder.stop_multi_process_pool(pool)

    if index is None:
        raise RuntimeError("Index was not created. Check your input dataset / filters.")

    faiss.write_index(index, args.out_faiss)
    conn.close()

    print(f"\nSaved FAISS index:   {args.out_faiss}")
    print(f"Saved chunk store:   {args.out_sqlite}")
    print(f"Total vectors:       {index.ntotal}")
    if args.index_type == "ivfpq":
        print(f"IVF nlist:           {args.nlist}")
    if args.index_type == "hnsw":
        print(f"HNSW M:              {args.hnsw_m}")


if __name__ == "__main__":
    main()