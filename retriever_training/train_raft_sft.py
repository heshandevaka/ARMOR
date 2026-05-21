#!/usr/bin/env python3
"""
Unified fine-tuning script for:
  1) RAFT-style multi-document training (custom dataset/collator)
  2) SFT training (TRL SFTTrainer)

Key features:
- Single CLI with subcommands: `raft` and `sft`
- Shared hyperparameters (lr, batch sizes, steps, max_seq_length, precision, deepspeed, etc.)
- Unified data args: --train_path and --val_path (both REQUIRED)
- Optional PEFT (LoRA / QLoRA-style with 4bit/8bit load) for BOTH methods
"""

import os
import argparse
from typing import Optional, List, Dict, Any

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from datasets import load_dataset

# Method-specific dependencies
# RAFT-side (your code)
from raft_data import RaftMultiDocDataset, DataCollatorForCausalLM

# SFT-side
from trl import SFTTrainer, SFTConfig

# PEFT
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# -----------------------
# Shared utilities
# -----------------------
def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_wandb(project: Optional[str], run_name: Optional[str], entity: Optional[str]) -> None:
    if not project:
        os.environ["WANDB_MODE"] = "disabled"
        return
    os.environ["WANDB_PROJECT"] = project
    if run_name:
        os.environ["WANDB_NAME"] = run_name
    if entity:
        os.environ["WANDB_ENTITY"] = entity


def load_tokenizer(model_name_or_path: str):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_name_or_path: str,
    *,
    bf16: bool,
    fp16: bool,
    load_in_8bit: bool,
    load_in_4bit: bool,
) -> torch.nn.Module:
    dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else None)

    kwargs: Dict[str, Any] = dict(trust_remote_code=True)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    # bitsandbytes quantized loading (typically used with LoRA)
    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    if load_in_4bit:
        kwargs["load_in_4bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.config.use_cache = False
    return model


def enable_grad_ckpt(model: torch.nn.Module, enabled: bool) -> None:
    if enabled:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False


def parse_target_modules(s: Optional[str]) -> List[str]:
    if not s:
        return ["q_proj", "k_proj", "v_proj", "o_proj"] #, "gate_proj", "up_proj", "down_proj"
    return [x.strip() for x in s.split(",") if x.strip()]


def apply_lora(model: torch.nn.Module, args) -> torch.nn.Module:
    """
    Apply LoRA to the model if args.lora_r > 0.
    """
    if args.lora_r <= 0:
        return model

    # If quantized base model, prep for k-bit training
    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = parse_target_modules(args.lora_target_modules)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model


def save_model_and_tokenizer(model: torch.nn.Module, tokenizer, output_dir: str, merge_lora: bool) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # If it's a PEFT model and merge_lora is requested, merge adapters into base weights.
    if merge_lora and hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
    else:
        # Works for both normal and PEFT models.
        # (Trainer.save_model usually calls this, but we keep it explicit and consistent.)
        model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)


# -----------------------
# Argument groups
# -----------------------
def add_shared_args(p: argparse.ArgumentParser) -> None:
    # Data (unified) - REQUIRED for both methods
    p.add_argument("--train_path", required=True, help="Path to training JSON/JSONL")
    p.add_argument("--val_path", required=True, help="Path to validation JSON/JSONL")

    # Model I/O
    p.add_argument("--model_name_or_path", "--model", required=True)
    p.add_argument("--output_dir", required=True)

    # Core training hyperparameters
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=2048)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
    "--lr_scheduler",
    type=str,
    default="linear",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    help="HF scheduler type",
)
    p.add_argument("--lr_warmup_ratio", type=float, default=0.0, help="Warmup ratio (0.0-1.0)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=float, default=1.0)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--per_device_eval_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--logging_steps", type=int, default=25)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=1)

    # Precision/memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # DeepSpeed
    p.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config json (e.g., ZeRO-3)")

    # W&B
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    # PEFT / LoRA (shared)
    p.add_argument("--lora_r", type=int, default=0, help="0 disables LoRA; >0 enables LoRA")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default=None,
        help="Comma-separated target modules (default: q/k/v/o projections)",
    )

    p.add_argument("--load_in_8bit", action="store_true", help="bitsandbytes 8-bit load (recommended with LoRA)")
    p.add_argument("--load_in_4bit", action="store_true", help="bitsandbytes 4-bit load (QLoRA-style)")

    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA adapters into base weights on save")


def add_raft_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--num_distractors", type=int, default=0)
    p.add_argument("--p_golden", type=float, default=1.0)
    p.add_argument("--min_quality", type=float, default=0.0)
    p.add_argument("--no_require_supported", action="store_true")
    p.add_argument("--no_require_keep", action="store_true")
    p.add_argument("--val_seed", type=int, default=12345, help="Seed for val distractor sampling")


def add_sft_args(p: argparse.ArgumentParser) -> None:
    # Add SFT-specific knobs here if you want later (packing, formatting, etc.)
    p.add_argument("--packing", action="store_true", help="Enable TRL packing (if supported by your TRL version)")


# -----------------------
# Method runners
# -----------------------
def build_common_tokenizer_and_model(args):
    setup_wandb(args.wandb_project, args.wandb_run_name, args.wandb_entity)
    set_seed(args.seed)

    tok = load_tokenizer(args.model_name_or_path)

    # Quantized loading is typically only useful when LoRA is enabled, but we don't force that.
    model = load_model(
        args.model_name_or_path,
        bf16=args.bf16,
        fp16=args.fp16,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    enable_grad_ckpt(model, args.gradient_checkpointing)

    # Apply PEFT for either method if needed
    model = apply_lora(model, args)

    # Keep tokenizer/model sizes in sync if we created a pad token
    model.resize_token_embeddings(len(tok))

    return tok, model


def run_raft(args) -> None:
    tok, model = build_common_tokenizer_and_model(args)

    train_ds = RaftMultiDocDataset(
        args.train_path,
        tokenizer=tok,
        max_length=args.max_seq_length,
        num_distractors=args.num_distractors,
        p_golden=args.p_golden,
        min_quality=args.min_quality,
        require_supported=not args.no_require_supported,
        require_keep=not args.no_require_keep,
        seed=args.seed,
    )
    eval_ds = RaftMultiDocDataset(
        args.val_path,
        tokenizer=tok,
        max_length=args.max_seq_length,
        num_distractors=args.num_distractors,
        p_golden=args.p_golden,
        min_quality=args.min_quality,
        require_supported=not args.no_require_supported,
        require_keep=not args.no_require_keep,
        seed=args.val_seed,
    )

    collator = DataCollatorForCausalLM(tokenizer=tok)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.lr_warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb" if args.wandb_project else "none",
        deepspeed=args.deepspeed,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tok,
    )

    trainer.train()
    # Save best checkpoint model state via trainer, then normalize final artifacts
    trainer.save_model(args.output_dir)
    save_model_and_tokenizer(trainer.model, tok, args.output_dir, merge_lora=args.merge_lora)


def run_sft(args) -> None:
    tok, model = build_common_tokenizer_and_model(args)

    dataset = load_dataset(
        "json",
        data_files={"train": args.train_path, "validation": args.val_path},
    )
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]

    sft_cfg = SFTConfig(
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.lr_warmup_ratio,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="wandb" if args.wandb_project else "none",
        deepspeed=args.deepspeed,
        packing=bool(args.packing),
    )

    # IMPORTANT:
    # Since we already applied LoRA (if requested) to `model` above,
    # we do NOT pass peft_config here (to avoid double-wrapping).
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    save_model_and_tokenizer(trainer.model, tok, args.output_dir, merge_lora=args.merge_lora)


# -----------------------
# Entrypoint
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser("Unified RAFT + SFT/LoRA fine-tuning")
    sub = parser.add_subparsers(dest="method", required=True)

    p_raft = sub.add_parser("raft", help="RAFT-style multi-doc training")
    add_shared_args(p_raft)
    add_raft_args(p_raft)

    p_sft = sub.add_parser("sft", help="SFT training")
    add_shared_args(p_sft)
    add_sft_args(p_sft)

    args = parser.parse_args()

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16 (or neither).")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit (or neither).")

    if args.method == "raft":
        run_raft(args)
    elif args.method == "sft":
        run_sft(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")


if __name__ == "__main__":

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        
    # DEBUG
    print("RANK", os.environ.get("RANK"), "LOCAL_RANK", os.environ.get("LOCAL_RANK"),
        "cuda.current_device()", torch.cuda.current_device())
    main()
