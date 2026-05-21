import argparse
import os
from datasets import load_dataset, DatasetDict
from collections import Counter


def main():
    parser = argparse.ArgumentParser(
        description="Build in-domain and size-matched out-of-domain Tele-Eval splits using source IDs."
    )
    parser.add_argument(
        "--domain_jsonl",
        type=str,
        required=True,
        help="Path to local domain dataset JSONL (e.g. /path/to/domain_dataset.jsonl).",
    )
    parser.add_argument(
        "--tele_eval_name",
        type=str,
        default="AliMaatouk/Tele-Eval",
        help="HF dataset name for Tele-Eval (default: AliMaatouk/Tele-Eval).",
    )
    parser.add_argument(
        "--tele_split",
        type=str,
        default="data",
        help="Tele-Eval split name (default: 'data').",
    )
    parser.add_argument(
        "--domain_source_key",
        type=str,
        default="source_id",
        help="Source ID column in the domain dataset (default: 'source_id').",
    )
    parser.add_argument(
        "--tele_source_key",
        type=str,
        default="id",
        help="Source ID column in Tele-Eval (default: 'ID').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for out-of-domain sampling (default: 42).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tele_eval_domain_splits",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--save_jsonl_only",
        action="store_true",
        help="If set, only JSONL files are written (no HF save_to_disk).",
    )

    args = parser.parse_args()

    # 1) Load domain dataset JSONL
    print(f"Loading domain dataset from {args.domain_jsonl} ...")
    domain_ds = load_dataset("json", data_files=args.domain_jsonl, split="train")

    if args.domain_source_key not in domain_ds.column_names:
        raise KeyError(
            f"Domain dataset has no column '{args.domain_source_key}'. "
            f"Available columns: {domain_ds.column_names}"
        )

    domain_source_ids = set(domain_ds[args.domain_source_key])
    print(f"Domain dataset size: {len(domain_ds)}")
    print(f"Unique domain source IDs: {len(domain_source_ids)}")

    # 2) Load Tele-Eval
    print(f"Loading Tele-Eval from {args.tele_eval_name} ...")
    tele = load_dataset(args.tele_eval_name)
    tele_ds = tele[args.tele_split]

    if args.tele_source_key not in tele_ds.column_names:
        raise KeyError(
            f"Tele-Eval split '{args.tele_split}' has no column '{args.tele_source_key}'. "
            f"Available columns: {tele_ds.column_names}"
        )

    print(f"Tele-Eval split size: {len(tele_ds)}")

    # 3) Build in-domain and out-of-domain pool
    def is_in_domain(example):
        return example[args.tele_source_key] in domain_source_ids

    print("Splitting Tele-Eval into in-domain and out-of-domain sets...")
    in_domain = tele_ds.filter(is_in_domain)
    out_pool = tele_ds.filter(
        lambda ex: ex[args.tele_source_key] not in domain_source_ids
    )

    n_in = len(in_domain)
    n_out_pool = len(out_pool)

    print(f"In-domain Tele-Eval size: {n_in}")
    print(f"Out-of-domain pool size: {n_out_pool}")

    if n_in == 0:
        raise ValueError(
            "In-domain set is empty: no Tele-Eval rows matched domain source IDs."
        )
    if n_out_pool < n_in:
        raise ValueError(
            f"Out-of-domain pool ({n_out_pool}) is smaller than in-domain ({n_in}); "
            f"cannot sample equal sizes without replacement."
        )

    # 4) Sample out-of-domain to match in-domain size
    print(f"Sampling {n_in} out-of-domain examples (seed={args.seed})...")
    out_domain = out_pool.shuffle(seed=args.seed).select(range(n_in))

    # 5) Rename Tele-Eval fields for final output
    rename_map = {
        "Statement": "prompt",
        "Answer": "completion",
    }

    def rename_fields(example):
        for old, new in rename_map.items():
            if old in example:
                example[new] = example[old]
        return example

    in_domain = in_domain.map(rename_fields)
    out_domain = out_domain.map(rename_fields)

    # Optional: remove original Tele-Eval fields
    remove_cols = [c for c in ["Statement", "Answer"] if c in in_domain.column_names]
    if remove_cols:
        in_domain = in_domain.remove_columns(remove_cols)
        out_domain = out_domain.remove_columns(remove_cols)

    # 6) Save outputs
    os.makedirs(args.output_dir, exist_ok=True)

    if not args.save_jsonl_only:
        ds_dict = DatasetDict(
            {
                "in_domain": in_domain,
                "out_domain": out_domain,
            }
        )
        ds_dict.save_to_disk(args.output_dir)
        print(f"Saved HF DatasetDict to: {args.output_dir}")

    in_path = f"{args.output_dir}/in_domain.jsonl"
    out_path = f"{args.output_dir}/out_domain.jsonl"

    print(f"Writing JSONL files:\n  {in_path}\n  {out_path}")
    in_domain.to_json(in_path, orient="records", lines=True)
    out_domain.to_json(out_path, orient="records", lines=True)

    # 7) Sanity check
    in_sources = Counter(in_domain[args.tele_source_key])
    out_sources = Counter(out_domain[args.tele_source_key])
    overlap = set(in_sources.keys()) & set(out_sources.keys())

    print(
        f"Sanity check: source ID overlap between in-domain and out-domain = "
        f"{len(overlap)} (should be 0)."
    )
    if overlap:
        print(f"Overlapping IDs (first few): {list(overlap)[:10]}")


if __name__ == "__main__":
    main()
