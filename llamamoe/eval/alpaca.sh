#!/bin/bash

# Runs Mixtral 8x7B model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/root/work/huangxin/nanda/data/cache
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
export CUDA_HOME=/usr/local/cuda-12.1
export LD_LIBRARY_PATH=/root/work/huangxin/envs/nju-megatron/lib/gcc/x86_64-conda-linux-gnu/11.4.0:$LD_LIBRARY_PATH
# python path
WORK_DIR=/root/work/huangxin/nanda
MEGATRONN_PATH=$WORK_DIR/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

# args
CHECKPOINT_PATH=/root/work/huangxin/nanda/models/Qwen2.5-3B-Instruct-qwenmoe-mcore-exp12-TP1PP1EP4
TOKENIZER_MODEL=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct
DATA_PATH=$WORK_DIR/Megatron-LM/llamamoe/data/data-bin/id_llama3_8b_text_document
LPR_LOSS_COEFF=1e-3
AUX_LOSS_COEFF=1e-3
LPR_STAGE=1
NUM_EXPERTS=12
TOPK=4

LR=5e-5
MICRO_BSZ=4
GLOBAL_BSZ=1024
TRAIN_STEPS=5000
SAVE_STEPS=1000
EVAL_STEPS=1000
WARMUP_STEPS=300

TP=1
EP=4
PP=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

LM_EVAL_ARGS=(
    --num-tokens-to-generate 1024
    --temperature 0.1
    --max-batch-size 80
    --save-path /root/work/huangxin/nanda/Megatron-LM/llamamoe/eval/results/test.jsonl
    --prompts /root/work/huangxin/nanda/pkgs/FastChat-main/fastchat/llm_judge/data/mt_bench/model_answer/qwen2.5-3b-instruct-sparse-4exp.jsonl
)

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --transformer-impl transformer_engine
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 32768
    --num-layers 36
    --hidden-size 2048
    --ffn-hidden-size 11008
    --num-attention-heads 16
    --init-method-std 0.02
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --no-rope-fusion
    --use-rotary-position-embeddings
    --swiglu
    --group-query-attention
    --num-query-groups 2
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --ckpt-format torch
    --add-qkv-bias
)

MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk $TOPK
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff $AUX_LOSS_COEFF
    --moe-lpr-loss-coeff $LPR_LOSS_COEFF
    --moe-lpr-stage $LPR_STAGE
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-ffn-hidden-size 2752
    --enable-shared-expert
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH
    --split 99990,8,2
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BSZ
    --global-batch-size $GLOBAL_BSZ
    --lr $LR
    --train-iters $TRAIN_STEPS
    --lr-decay-iters $TRAIN_STEPS
    --lr-decay-style cosine
    --min-lr 0
    --weight-decay 0.1
    --lr-warmup-iters $WARMUP_STEPS
    --clip-grad 1.0
    --use-flash-attn
    --bf16
    --distributed-timeout-minutes 1000000
    --no-save-optim
    --no-save-rng
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --expert-model-parallel-size $EP
    --use-distributed-optimizer
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval $SAVE_STEPS \
    --eval-interval 1000 \
    --eval-iters $EVAL_STEPS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --no-load-optim \
    --no-load-rng
)

torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRONN_PATH}/llamamoe/eval/inference.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${LM_EVAL_ARGS[@]}