#!/bin/bash
# bash hf2mcore_qwen1.5_moe_convertor.sh A2.7B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B-to-mcore-tp1-pp1-ep4 1 1 4 false
# bash hf2mcore_qwen1.5_moe_convertor.sh A2.7B /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B-to-mcore-tp1-pp1-ep4 /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B-to-hf 1 1 4 true /mnt/qwen-ckpts/Qwen1.5-MoE-A2.7B

set -e
export CUDA_VISIBLE_DEVICES=0
START_TIME=$SECONDS
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

SOURCE_CKPT_PATH=/root/work/huangxin/nanda/models/llama3.1-8b-instruct-qwenmoe-4exp
TARGET_CKPT_PATH=/root/work/huangxin/nanda/models/llama3.1-8b-instruct-qwenmoe-4exp-tp2-ep1
TP=2
PP=1
EP=1
mg2hf=false
HF_CKPT_PATH=none

MEGATRONN_PATH=/root/work/huangxin/nanda/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
NUM_LAYERS=32
INTERMEDIATE_SIZE=14336
MOE_INTERMEDIATE_SIZE=3584
SHARED_EXPERT_INTERMEDIATE_SIZE=14336
MAX_POSITION_EMBEDDINGS=131072
EXTRA_VOCAB_SIZE=0
NUM_EXPERTS=4
EXPERTS_TOPK=4
ROPE_THETA=500000
NUM_KEY_VALUE_HEADS=8

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups ${NUM_KEY_VALUE_HEADS}"

cpu_options=" \
            --use-cpu-initialization"


if [ $NUM_EXPERTS -gt 0 ]; then
    expert_options=" \
                --moe-router-topk ${EXPERTS_TOPK} \
                --num-experts ${NUM_EXPERTS} \
                --target-expert-model-parallel-size ${EP}"
fi

if [ $mg2hf = true ]; then
    convert_options=" \
                --convert-checkpoint-from-megatron-to-transformers \
                --hf-ckpt-path ${HF_CKPT_PATH}"

elif [ $mg2hf = false ]; then
    convert_options=""
fi


DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} $MEGATRONN_PATH/tools/checkpoint/qwen_moe_mcore.py \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size 1 \
    --save-interval 1 \
    --bf16 \
    --swiglu \
    --norm-epsilon 1e-5 \
    --tokenizer-model /root/work/huangxin/nanda/models/llama3.1-8b-instruct \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --moe-ffn-hidden-size ${MOE_INTERMEDIATE_SIZE} \
    --ffn-hidden-size ${INTERMEDIATE_SIZE} \
    --num-attention-heads ${NUM_ATTN_HEADS} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --seq-length 1 \
    --no-async-tensor-model-parallel-allreduce \
    --tokenizer-type HuggingFaceTokenizer \
    --untie-embeddings-and-output-weights \
    --no-rope-fusion \
    --use-rotary-position-embeddings \
    --transformer-impl transformer_engine \
    --disable-bias-linear \
    --normalization RMSNorm \
    --use-mcore-models \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --enable-shared-expert \
    --rotary-base ${ROPE_THETA} \
    ${expert_options} \
    ${convert_options} \
    ${gqa_options} \
    ${cpu_options}


ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"