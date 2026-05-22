#!/bin/bash
set -e

# --- CONFIGURATION ---
# ARMOR currently includes only the ISAC domain prompts/config under domains/isac.
DOMAIN="${1:-isac}"
if [[ "$DOMAIN" != "isac" ]]; then
  echo "Error: ARMOR/unified_data_gen currently includes only the 'isac' domain." >&2
  echo "Usage: $0 [isac]" >&2
  exit 1
fi
RAW_INPUT="AliMaatouk/Tele-Data"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
EMB_MODEL="intfloat/e5-large-v2"
INDEX_CHUNK_MAX_TOKENS=128
INDEX_CHUNK_OVERLAP=16
INDEX_TYPE="flat"
export TP_SIZE="${TP_SIZE:-2}"

# --- Runtime knobs ---
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Optional OpenAI post-filter
OPENAI_FILTER_MODEL="gpt-5.2"
OPENAI_FILTER_THRESHOLD="0.50"
OPENAI_FILTER_SNIPPET_CHARS="2000"
OPENAI_FILTER_TIMEOUT_S="120"
OPENAI_FILTER_MAX_RETRIES="6"
OPENAI_FILTER_BASE_BACKOFF_S="0.75"
OPENAI_FILTER_MAX_TO_JUDGE="0"

# ---- Env tuning (optional) ----
export MIN_DOC_CONF="${MIN_DOC_CONF:-0.70}"
export CHUNK_CHARS="${CHUNK_CHARS:-8000}"
export CHUNK_OVERLAP="${CHUNK_OVERLAP:-800}"
export GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-20}"
export JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-20}"
export MIN_TECH="${MIN_TECH:-0.70}"
export MIN_CLARITY="${MIN_CLARITY:-0.70}"

# Output locations (subfolder for organization)
OUT_DIR="data/$DOMAIN"
mkdir -p "$OUT_DIR"

# If you have clean docs already, set this.
# Otherwise set to "" to process from scratch.
CLEAN_DOCS_JSONL="" 
OPENAI_FILTERED_CLEAN_DOCS=""

OPENAI_ARGS=()
if [[ -n "${OPENAI_FILTER_MODEL}" ]]; then
  OPENAI_ARGS+=(
    --openai_doc_filter_model "$OPENAI_FILTER_MODEL"
    --openai_doc_filter_threshold "$OPENAI_FILTER_THRESHOLD"
    --openai_doc_filter_snippet_chars "$OPENAI_FILTER_SNIPPET_CHARS"
    --openai_doc_filter_timeout_s "$OPENAI_FILTER_TIMEOUT_S"
    --openai_doc_filter_max_retries "$OPENAI_FILTER_MAX_RETRIES"
    --openai_doc_filter_base_backoff_s "$OPENAI_FILTER_BASE_BACKOFF_S"
    --openai_doc_filter_max_to_judge "$OPENAI_FILTER_MAX_TO_JUDGE"
  )
fi

FROM_ARGS=()
if [[ -n "${CLEAN_DOCS_JSONL}" ]]; then
  FROM_ARGS+=( --from_clean_docs "$CLEAN_DOCS_JSONL" )
fi
if [[ -n "${OPENAI_FILTERED_CLEAN_DOCS}" ]]; then
  FROM_ARGS+=( --from_openai_filtered_clean_docs "$OPENAI_FILTERED_CLEAN_DOCS" )
fi

# --- STAGE 1: FILTERING & CLEAN DOC GENERATION ---
echo ">>> Running Stage 1: Filtering documents for domain: $DOMAIN..."
python filter_docs.py \
    --domain $DOMAIN \
    --input "$RAW_INPUT" \
    --out_dir "$OUT_DIR" \
    --model "$MODEL" \
    "${FROM_ARGS[@]}" \
    "${OPENAI_ARGS[@]}"

# --- STAGE 2: INDEX BUILDING ---
if [[ -n "${OPENAI_FILTER_MODEL}" ]]; then
    FINAL_CLEAN_DOCS="$OUT_DIR/${DOMAIN}_clean_docs_openai_filtered.jsonl"
else
    FINAL_CLEAN_DOCS="$OUT_DIR/${DOMAIN}_clean_docs.jsonl"
fi
FAISS_INDEX="$OUT_DIR/${DOMAIN}_index.faiss"
SQLITE_DB="$OUT_DIR/${DOMAIN}_chunks.sqlite"

echo ">>> Running Stage 2: Building FAISS/SQLite index using $FINAL_CLEAN_DOCS..."
python build_index.py \
    --docs_jsonl "$FINAL_CLEAN_DOCS" \
    --out_faiss "$FAISS_INDEX" \
    --out_sqlite "$SQLITE_DB" \
    --embed_model "$EMB_MODEL" \
    --index_type "$INDEX_TYPE" \
    --chunk_max_tokens $INDEX_CHUNK_MAX_TOKENS \
    --chunk_overlap $INDEX_CHUNK_OVERLAP

# The retriever training scripts expect a unified corpus path. For the
# ISAC-only ARMOR release, mirror the ISAC index/db into data/unified.
UNIFIED_DIR="data/unified"
mkdir -p "$UNIFIED_DIR"
cp "$FAISS_INDEX" "$UNIFIED_DIR/unified_index.faiss"
cp "$SQLITE_DB" "$UNIFIED_DIR/unified_chunks.sqlite"

# --- STAGE 3: QA PAIR GENERATION ---
QA_OUT="$OUT_DIR/${DOMAIN}_qa_pairs.jsonl"

echo ">>> Running Stage 3: Generating grounded QA pairs..."
python generate_qa.py \
    --domain $DOMAIN \
    --input_jsonl "$FINAL_CLEAN_DOCS" \
    --output_jsonl "$QA_OUT" \
    --model "$MODEL" \
    --tp_size $TP_SIZE

# --- STAGE 4: RETRIEVER ALIGNMENT ---
ALIGN_OUT="$OUT_DIR/contriever_aligned_dataset.jsonl"

echo ">>> Running Stage 4: Aligning QA pairs to index..."
python align_qa.py \
    --qa_jsonl "$QA_OUT" \
    --sqlite_db "$SQLITE_DB" \
    --out_jsonl "$ALIGN_OUT"

# --- STAGE 5: DATA SPLITTING ---
SPLIT_DIR="$OUT_DIR/splits"

echo ">>> Running Stage 5: Splitting aligned data into train/val/test sets..."
python split_data.py \
    --input "$ALIGN_OUT" \
    --out_dir "$SPLIT_DIR"

# Compatibility outputs consumed by retriever_training/train_isac_all_methods.sh.
cp "$SPLIT_DIR/raft/train.jsonl" "$OUT_DIR/aligned_train_unified.jsonl"
cp "$SPLIT_DIR/raft/val.jsonl" "$OUT_DIR/aligned_val_unified.jsonl"

echo ">>> Pipeline Complete!"
echo "Final dataset splits are saved in: $SPLIT_DIR"
echo "Retriever-aligned train/val files are saved in: $OUT_DIR"
echo "ISAC-only unified index/db are saved in: $UNIFIED_DIR"
