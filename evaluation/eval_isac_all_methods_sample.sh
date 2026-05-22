#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DOMAIN="${DOMAIN:-isac}"
GEN_MODEL="${GEN_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EMB_MODEL="${EMB_MODEL:-intfloat/e5-large-v2}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2}"

ARMOR_ROOT="${ARMOR_ROOT:-/data/hdf/ARMOR}"
OLD_EVAL_ROOT="${OLD_EVAL_ROOT:-/data/hdf/telecom-co-scientist/eval}"

# Paths produced/used by the ARMOR training repo.
CKPT_ROOT="${CKPT_ROOT:-$ARMOR_ROOT/retriever_training/checkpoints}"
UNIFIED_INDEX="${UNIFIED_INDEX:-$ARMOR_ROOT/unified_data_gen/data/unified/unified_index.faiss}"
UNIFIED_DB="${UNIFIED_DB:-$ARMOR_ROOT/unified_data_gen/data/unified/unified_chunks.sqlite}"

# Extra eval sets used by telecom-co-scientist/eval wrappers.
DOMAIN_TEST="${DOMAIN_TEST:-/data/hdf/telecom-co-scientist/fine_tuning/data/$DOMAIN/test.jsonl}"
TELE_EVAL_IN="${TELE_EVAL_IN:-/data/hdf/telecom-co-scientist/eval/$DOMAIN/tele_eval_split/in_domain.jsonl}"
TELE_EVAL_OUT="${TELE_EVAL_OUT:-/data/hdf/telecom-co-scientist/eval/$DOMAIN/tele_eval_split/out_domain.jsonl}"

OUT_ROOT="${OUT_ROOT:-$ARMOR_ROOT/evaluation/results_$DOMAIN}"
TELE_EVAL_PY="$ARMOR_ROOT/evaluation/tele-eval/eval_tele_eval.py"
TELE_QNA_PY="$ARMOR_ROOT/evaluation/tele-qna/eval_tele_qna.py"

LR="${LR:-2e-5}"
BS="${BS:-4}"
TOP_K="${TOP_K:-16}"
RET_TEMP="${RET_TEMP:-1.0}"
CONTR_TEMP="${CONTR_TEMP:-1.0}"
TEMP_LR="${TEMP_LR:-1e-2}"
QUERY_DISTILL="${QUERY_DISTILL:-1.0}"
INIT_CONTR_TEMP="${INIT_CONTR_TEMP:-10}"
INIT_RETR_TEMP="${INIT_RETR_TEMP:-10}"
MIN_TEMP="${MIN_TEMP:-1.0}"
LORA_R="${LORA_R:-16}"
LR_SCHEDULER="${LR_SCHEDULER:-constant_with_warmup}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.0}"

SAFE_GEN_MODEL="${GEN_MODEL//\//-}"

RAG_QE="$CKPT_ROOT/$GEN_MODEL/${DOMAIN}_unified_corpus_rag_lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-retr_temperature=$RET_TEMP/query_encoder_final"
CONTRIEVER_QE="$CKPT_ROOT/${DOMAIN}_unified_corpus_contriever_lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-contrastive_temperature=$CONTR_TEMP/query_encoder_final"
MIX_STATIC_QE="$CKPT_ROOT/$GEN_MODEL/${DOMAIN}_unified_corpus_rag_contr_mix-static-lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-rag_lambd=1.0-contr_lambd=1.0-contrastive_temperature=$CONTR_TEMP/query_encoder_final"
MIX_ADAPTIVE_QE="$CKPT_ROOT/$GEN_MODEL/${DOMAIN}_unified_corpus_rag_contr_mix-adaptive-temp-sigmoid-bl-with-qdistill-lr=$LR-temp_lr=$TEMP_LR-batch_size=$BS-rag_top_k=$TOP_K-rag_lambd=1.0-contr_lambd=1.0-qdistill=$QUERY_DISTILL-init_contrastive_temperature=$INIT_CONTR_TEMP-init_retr_temperature=$INIT_RETR_TEMP-min_temp=$MIN_TEMP/query_encoder_final"
REPLUG_QE="$CKPT_ROOT/$GEN_MODEL/${DOMAIN}_unified_corpus_replug_lr=$LR-batch_size=$BS-top_k=$TOP_K/query_encoder_final"
RAG_LM_QE="$CKPT_ROOT/$SAFE_GEN_MODEL/${DOMAIN}_unified_rag_both_lora_lrq=$LR-lrlm=$LR-batch_size=$BS-rag_top_k=$TOP_K-retr_temperature=$RET_TEMP/query_encoder_final"
RAG_LM_MODEL="$CKPT_ROOT/$SAFE_GEN_MODEL/${DOMAIN}_unified_rag_both_lora_lrq=$LR-lrlm=$LR-batch_size=$BS-rag_top_k=$TOP_K-retr_temperature=$RET_TEMP/lm_final"
RAFT_MODEL="$CKPT_ROOT/$GEN_MODEL-lora_r=$LORA_R-bs=$BS-lr=$LR-$DOMAIN-raft-raft_unified_data"
SFT_MODEL="$CKPT_ROOT/$GEN_MODEL-lora_r=$LORA_R-bs=$BS-lr=$LR-$DOMAIN-sft-sft_unified_data"

run_tele_eval() {
  local method="$1"
  local model="$2"
  local mode="$3"
  local query_encoder="${4:-}"

  for split_name in domain_test tele_eval_in tele_eval_out; do
    local data_path=""
    case "$split_name" in
      domain_test) data_path="$DOMAIN_TEST" ;;
      tele_eval_in) data_path="$TELE_EVAL_IN" ;;
      tele_eval_out) data_path="$TELE_EVAL_OUT" ;;
    esac

    local out_dir="$OUT_ROOT/tele_eval/$split_name/$method"
    local args=(
      "$TELE_EVAL_PY"
      --model "$model"
      --data-path "$data_path"
      --output-dir "$out_dir"
      --retrieval-mode "$mode"
      --dtype bfloat16
      --device cuda
      --judge-model "$JUDGE_MODEL"
      --retrieval-top-k 8
      --max-context-chars 12000
    )

    if [[ "$mode" != "closed_book" ]]; then
      args+=(
        --faiss-index "$UNIFIED_INDEX"
        --sqlite-db "$UNIFIED_DB"
        --embed-model "$EMB_MODEL"
        --query-encoder-path "$query_encoder"
      )
    fi

    python "${args[@]}"
  done
}

run_tele_qna() {
  local method="$1"
  local model="$2"
  local mode="$3"
  local query_encoder="${4:-}"

  local args=(
    "$TELE_QNA_PY"
    --model "$model"
    --output-dir "$OUT_ROOT/tele_qna/$method"
    --retrieval-mode "$mode"
    --dtype bfloat16
    --device cuda
    --split train
    --query-with-options
  )

  if [[ "$mode" != "closed_book" ]]; then
    args+=(
      --faiss-index "$UNIFIED_INDEX"
      --sqlite-db "$UNIFIED_DB"
      --embed-model "$EMB_MODEL"
      --query-encoder-path "$query_encoder"
      --retrieval-top-k 8
      --max-context-chars 12000
    )
  fi

  python "${args[@]}"
}

run_raft_style_eval() {
  local method="$1"
  local model="$2"

  for split_name in domain_test tele_eval_in tele_eval_out; do
    local data_path=""
    case "$split_name" in
      domain_test) data_path="$DOMAIN_TEST" ;;
      tele_eval_in) data_path="$TELE_EVAL_IN" ;;
      tele_eval_out) data_path="$TELE_EVAL_OUT" ;;
    esac

    python "$OLD_EVAL_ROOT/$DOMAIN/eval_rag_raft.py" \
      --model "$model" \
      --tokenizer "$model" \
      --data-path "$data_path" \
      --output-dir "$OUT_ROOT/raft_style/$split_name/$method" \
      --faiss-index "$UNIFIED_INDEX" \
      --sqlite-db "$UNIFIED_DB" \
      --embed-model "$EMB_MODEL" \
      --rag-top-k 8 \
      --rag-max-context-chars 12000 \
      --dtype bfloat16 \
      --device cuda \
      --judge-model "$JUDGE_MODEL"
  done
}

# Frozen baselines.
run_tele_eval "base_closed_book" "$GEN_MODEL" "closed_book"
run_tele_qna "base_closed_book" "$GEN_MODEL" "closed_book"
run_tele_eval "base_rag" "$GEN_MODEL" "rag" "$EMB_MODEL"
run_tele_qna "base_rag" "$GEN_MODEL" "rag" "$EMB_MODEL"

# Retriever-trained methods from train_isac_all_methods.sh.
run_tele_eval "rag" "$GEN_MODEL" "rag" "$RAG_QE"
run_tele_qna "rag" "$GEN_MODEL" "rag" "$RAG_QE"

run_tele_eval "contriever" "$GEN_MODEL" "rag" "$CONTRIEVER_QE"
run_tele_qna "contriever" "$GEN_MODEL" "rag" "$CONTRIEVER_QE"

run_tele_eval "mix_static" "$GEN_MODEL" "rag" "$MIX_STATIC_QE"
run_tele_qna "mix_static" "$GEN_MODEL" "rag" "$MIX_STATIC_QE"

run_tele_eval "mix_adaptive" "$GEN_MODEL" "rag" "$MIX_ADAPTIVE_QE"
run_tele_qna "mix_adaptive" "$GEN_MODEL" "rag" "$MIX_ADAPTIVE_QE"

run_tele_eval "replug" "$GEN_MODEL" "replug" "$REPLUG_QE"
run_tele_qna "replug" "$GEN_MODEL" "replug" "$REPLUG_QE"

# Joint query-encoder + LM fine-tuning.
run_tele_eval "rag_lm_query_ft" "$RAG_LM_MODEL" "rag" "$RAG_LM_QE"
run_tele_qna "rag_lm_query_ft" "$RAG_LM_MODEL" "rag" "$RAG_LM_QE"

# Generator-tuned methods.
run_tele_eval "sft" "$SFT_MODEL" "closed_book"
run_tele_qna "sft" "$SFT_MODEL" "closed_book"

# RAFT is best evaluated with the older RAFT-style RAG prompt evaluator.
run_raft_style_eval "raft" "$RAFT_MODEL"

