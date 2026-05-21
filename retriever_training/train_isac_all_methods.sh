#!/bin/bash
# train_isac_all_methods.sh - Unified training script for all retriever training methods in ISAC domain

METHOD=$1

if [ -z "$METHOD" ]; then
    echo "Usage: $0 <method>"
    echo "Methods: rag, contriever, mix_static, mix_adaptive, rag_lm_query_ft, raft, replug, sft"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1

LR=2e-5
BS=4
TOP_K=16
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
EMB_MODEL="intfloat/e5-large-v2"

UNIFIED_INDEX="../unified_data_gen/data/unified/unified_index.faiss"
UNIFIED_DB="../unified_data_gen/data/unified/unified_chunks.sqlite"
TRAIN_DATA="../unified_data_gen/data/isac/aligned_train_unified.jsonl"
VAL_DATA="../unified_data_gen/data/isac/aligned_val_unified.jsonl"
FT_TRAIN_PATH="../unified_data_gen/data/isac/splits/ft/train.jsonl"
FT_VAL_PATH="../unified_data_gen/data/isac/splits/ft/val.jsonl"
RAFT_TRAIN_PATH="../unified_data_gen/data/isac/splits/raft/train.jsonl"
RAFT_VAL_PATH="../unified_data_gen/data/isac/splits/raft/val.jsonl"

case $METHOD in
    rag)
        RET_TEMP=1.0
        torchrun --master-port 29601 --nproc_per_node=2 retriever_train_qa_ddp_with_contriever.py \
            --objective rag \
            --batch_size $BS \
            --lr $LR \
            --top_k $TOP_K \
            --retr_temperature $RET_TEMP \
            --train_jsonl $TRAIN_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --lm_name $MODEL \
            --normalize_by_y_length \
            --out_dir checkpoints/$MODEL/isac_unified_corpus_rag_lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-retr_temperature=$RET_TEMP
        ;;
    contriever)
        CONTR_TEMP=1.0
        torchrun --master-port 29604 --nproc_per_node=2 retriever_train_qa_ddp_with_contriever.py \
            --objective contriever \
            --batch_size $BS \
            --lr $LR \
            --train_jsonl $TRAIN_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --top_k $TOP_K \
            --contrastive_num_negatives 3 \
            --contrastive_temperature $CONTR_TEMP \
            --contrastive_exclude_same_doc \
            --out_dir checkpoints/isac_unified_corpus_contriever_lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-contrastive_temperature=$CONTR_TEMP
        ;;
    mix_static)
        LAMBD_RAG=1.0
        LAMBD_CONTR=1.0
        CONTR_TEMP=1.0
        MIXED_WEIGHT_STRAT=static
        torchrun --master-port 29607 --nproc_per_node=2 retriever_train_qa_ddp_with_mix_weighting_strategies.py \
            --objective rag_contriever \
            --mix_weighting_strategy $MIXED_WEIGHT_STRAT \
            --train_jsonl $TRAIN_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --lm_name $MODEL \
            --top_k $TOP_K \
            --batch_size $BS \
            --contrastive_num_negatives 3 \
            --contrastive_exclude_same_doc \
            --contrastive_temperature $CONTR_TEMP \
            --mix_rag_weight $LAMBD_RAG \
            --mix_contriever_weight $LAMBD_CONTR \
            --out_dir checkpoints/$MODEL/isac_unified_corpus_rag_contr_mix-$MIXED_WEIGHT_STRAT-lr=$LR-batch_size=$BS-rag_top_k=$TOP_K-rag_lambd=$LAMBD_RAG-contr_lambd=$LAMBD_CONTR-contrastive_temperature=$CONTR_TEMP
        ;;
    mix_adaptive)
        TEMP_LR=1e-2
        LAMBD_RAG=1.0
        LAMBD_CONTR=1.0
        QUERY_DISTILL=1.0
        INIT_CONTR_TEMP=10
        INIT_RETR_TEMP=10
        MIN_TEMP=1.0
        torchrun --master-port 29501 --nproc_per_node=2 retriever_train_qa_ddp_static_mix_adaptive_temp_sigmoid_bl_with_reg.py \
            --train_jsonl $TRAIN_DATA \
            --val_jsonl $VAL_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --lm_name $MODEL \
            --lr $LR \
            --temperature_lr $TEMP_LR \
            --top_k $TOP_K \
            --batch_size $BS \
            --contrastive_num_negatives 3 \
            --contrastive_exclude_same_doc \
            --contrastive_temperature $INIT_CONTR_TEMP \
            --retr_temperature $INIT_RETR_TEMP \
            --temperature_min $MIN_TEMP \
            --mix_rag_weight $LAMBD_RAG \
            --mix_contriever_weight $LAMBD_CONTR \
            --normalize_by_y_length \
            --regularizers query_distill \
            --query_distill_coef $QUERY_DISTILL \
            --out_dir checkpoints/$MODEL/isac_unified_corpus_rag_contr_mix-adaptive-temp-sigmoid-bl-with-qdistill-lr=$LR-temp_lr=$TEMP_LR-batch_size=$BS-rag_top_k=$TOP_K-rag_lambd=$LAMBD_RAG-contr_lambd=$LAMBD_CONTR-qdistill=$QUERY_DISTILL-init_contrastive_temperature=$INIT_CONTR_TEMP-init_retr_temperature=$INIT_RETR_TEMP-min_temp=$MIN_TEMP
        ;;
    rag_lm_query_ft)
        LR_QUERY=2e-5
        LR_LM=2e-5
        RET_TEMP=1.0
        SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '-')
        torchrun --master-port 29520 --nproc_per_node=2 retriever_train_qa_ddp_rag_lm_query_ft.py \
            --optimize both \
            --batch_size $BS \
            --lr_query $LR_QUERY \
            --lr_lm $LR_LM \
            --top_k $TOP_K \
            --retr_temperature $RET_TEMP \
            --train_jsonl $TRAIN_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --lm_name $MODEL \
            --normalize_by_y_length \
            --gradient_checkpointing \
            --lora_r 16 \
            --lora_alpha 16 \
            --lora_dropout 0.05 \
            --merge_lora \
            --out_dir checkpoints/$SAFE_MODEL_NAME/isac_unified_rag_both_lora_lrq=$LR_QUERY-lrlm=$LR_LM-batch_size=$BS-rag_top_k=$TOP_K-retr_temperature=$RET_TEMP
        ;;
    raft)
        IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        WS=${#DEVICES[@]}
        GAS=$((BS / WS))
        LORA_R=16
        LR_SCHEDULER="constant_with_warmup"
        LR_WARMUP_RATIO=0.0
        SUFF=raft_unified_data
        torchrun --nproc_per_node=$WS --master_port 29522 train_raft_sft.py raft \
            --model_name_or_path $MODEL \
            --train_path $RAFT_TRAIN_PATH \
            --val_path $RAFT_VAL_PATH \
            --lr $LR \
            --lr_scheduler $LR_SCHEDULER \
            --lr_warmup_ratio $LR_WARMUP_RATIO \
            --gradient_accumulation_steps $GAS \
            --epochs 16 \
            --lora_r $LORA_R \
            --load_in_4bit \
            --merge_lora \
            --deepspeed ds_config_zero3.json \
            --output_dir checkpoints/$MODEL-lora_r=$LORA_R-bs=$BS-lr=$LR-isac-raft-$SUFF \
            --wandb_run_name $MODEL-isac-raft-$SUFF \
            --wandb_project Tele_RAG_Opt
        ;;
    replug)
        torchrun --master-port 29520 --nproc_per_node=2 retriever_train_qa_ddp_with_mix_weighting_strategies.py \
            --objective replug \
            --batch_size $BS \
            --lr $LR \
            --top_k $TOP_K \
            --train_jsonl $TRAIN_DATA \
            --faiss_index $UNIFIED_INDEX \
            --sqlite_db $UNIFIED_DB \
            --embed_model $EMB_MODEL \
            --lm_name $MODEL \
            --normalize_by_y_length \
            --out_dir checkpoints/$MODEL/isac_unified_corpus_replug_lr=$LR-batch_size=$BS-top_k=$TOP_K
        ;;
    sft)
        IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"
        WS=${#DEVICES[@]}
        GAS=$((BS / WS))
        LORA_R=16
        LR_SCHEDULER="constant_with_warmup"
        LR_WARMUP_RATIO=0.0
        SUFF=sft_unified_data
        torchrun --nproc_per_node=$WS --master_port 29521 train_raft_sft.py sft \
            --model_name_or_path $MODEL \
            --train_path $FT_TRAIN_PATH \
            --val_path $FT_VAL_PATH \
            --lr $LR \
            --lr_scheduler $LR_SCHEDULER \
            --lr_warmup_ratio $LR_WARMUP_RATIO \
            --gradient_accumulation_steps $GAS \
            --epochs 16 \
            --lora_r $LORA_R \
            --load_in_4bit \
            --merge_lora \
            --deepspeed ds_config_zero3.json \
            --output_dir checkpoints/$MODEL-lora_r=$LORA_R-bs=$BS-lr=$LR-isac-sft-$SUFF \
            --wandb_run_name $MODEL-isac-sft-$SUFF \
            --wandb_project Tele_RAG_Opt
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Methods: rag, contriever, mix_static, mix_adaptive, rag_lm_query_ft, raft, replug, sft"
        exit 1
        ;;
esac
