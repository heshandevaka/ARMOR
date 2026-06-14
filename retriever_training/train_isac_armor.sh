#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${NPROC:-${#DEVICES[@]}}"

LR="${LR:-2e-5}"
BS="${BS:-4}"
TOP_K="${TOP_K:-16}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EMB_MODEL="${EMB_MODEL:-intfloat/e5-large-v2}"

UNIFIED_INDEX="../data_gen/data/unified/unified_index.faiss"
UNIFIED_DB="../data_gen/data/unified/unified_chunks.sqlite"
TRAIN_DATA="../data_gen/data/isac/aligned_train_unified.jsonl"
VAL_DATA="../data_gen/data/isac/aligned_val_unified.jsonl"

TEMP_LR=1e-2
LAMBD_RAG=1.0
LAMBD_CONTR=1.0
QUERY_DISTILL=1.0
INIT_CONTR_TEMP=10.0
INIT_RETR_TEMP=10.0
MIN_TEMP=1.0

# Rebrand mix_adaptive to armor
OUT_DIR="checkpoints/${MODEL}/isac_unified_corpus_armor-lr=${LR}-temp_lr=${TEMP_LR}-batch_size=${BS}-rag_top_k=${TOP_K}-rag_lambd=${LAMBD_RAG}-contr_lambd=${LAMBD_CONTR}-qdistill=${QUERY_DISTILL}-init_contrastive_temperature=${INIT_CONTR_TEMP}-init_retr_temperature=${INIT_RETR_TEMP}-min_temp=${MIN_TEMP}"

echo "Launching ARMOR query encoder training on ${MODEL} for ISAC domain..."
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Output directory: ${OUT_DIR}"

torchrun --master-port 29501 --nproc_per_node="${NPROC}" train_armor.py \
    --train_jsonl "${TRAIN_DATA}" \
    --val_jsonl "${VAL_DATA}" \
    --faiss_index "${UNIFIED_INDEX}" \
    --sqlite_db "${UNIFIED_DB}" \
    --embed_model "${EMB_MODEL}" \
    --lm_name "${MODEL}" \
    --lr "${LR}" \
    --temperature_lr "${TEMP_LR}" \
    --top_k "${TOP_K}" \
    --batch_size "${BS}" \
    --contrastive_num_negatives 3 \
    --contrastive_exclude_same_doc \
    --contrastive_temperature "${INIT_CONTR_TEMP}" \
    --retr_temperature "${INIT_RETR_TEMP}" \
    --temperature_min "${MIN_TEMP}" \
    --mix_rag_weight "${LAMBD_RAG}" \
    --mix_contriever_weight "${LAMBD_CONTR}" \
    --normalize_by_y_length \
    --regularizers query_distill \
    --query_distill_coef "${QUERY_DISTILL}" \
    --out_dir "${OUT_DIR}"
