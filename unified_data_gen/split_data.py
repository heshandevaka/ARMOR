#!/usr/bin/env python
import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Split RAFT-style data into train/val/test and also create QA-only splits."
    )
    p.add_argument("--input", required=True, help="Input RAFT-style JSONL file")
    p.add_argument("--out_dir", default="data/domain", help="Output directory")
    p.add_argument("--train_frac", type=float, default=0.9)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--test_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Bad JSON at line {i}: {e}\nLine: {line[:200]}") from e
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False)


def main():
    args = parse_args()
    assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6

    rows = read_jsonl(args.input)
    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    cleaned = []
    for r in rows:
        q = normalize_to_text(r.get("question", "")).strip()
        a = normalize_to_text(r.get("answer", "")).strip()
        if not (q and a):
            continue
        cleaned.append(r)

    if not cleaned:
        raise ValueError("After filtering for non-empty question/answer, dataset is empty.")

    random.seed(args.seed)
    random.shuffle(cleaned)

    n = len(cleaned)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)

    raft_train = cleaned[:n_train]
    raft_val = cleaned[n_train:n_train + n_val]
    raft_test = cleaned[n_train + n_val:]

    out_root = Path(args.out_dir)
    raft_dir = out_root / "raft"
    qa_dir = out_root / "ft"
    raft_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(raft_dir / "train.jsonl", raft_train)
    write_jsonl(raft_dir / "val.jsonl", raft_val)
    write_jsonl(raft_dir / "test.jsonl", raft_test)

    def make_qa_view(rows):
        qa_rows = []
        for r in rows:
            q = normalize_to_text(r.get("question", "")).strip()
            a = normalize_to_text(r.get("answer", "")).strip()
            if not (q and a):
                continue
            qa_rows.append(
                {
                    "prompt": q,
                    "completion": a,
                }
            )
        return qa_rows

    qa_train = make_qa_view(raft_train)
    qa_val = make_qa_view(raft_val)
    qa_test = make_qa_view(raft_test)

    write_jsonl(qa_dir / "train.jsonl", qa_train)
    write_jsonl(qa_dir / "val.jsonl", qa_val)
    write_jsonl(qa_dir / "test.jsonl", qa_test)

    print(
        f"Total RAFT examples (after basic filtering): {n}\n"
        f"  RAFT train: {len(raft_train)} | RAFT val: {len(raft_val)} | RAFT test: {len(raft_test)}\n"
        f"  QA train:   {len(qa_train)}   | QA val:   {len(qa_val)}   | QA test:   {len(qa_test)}"
    )


if __name__ == "__main__":
    main()
