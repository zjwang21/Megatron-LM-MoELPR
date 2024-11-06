#!/bin/bash

# Runs Mixtral 8x7B model
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_DATASETS_CACHE=/root/work/huangxin/nanda/data/cache
# python path
WORK_DIR=/root/work/huangxin/nanda
MEGATRONN_PATH=$WORK_DIR/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

# args
CHECKPOINT_PATH=/root/work/huangxin/nanda/models/llama3.1-8b-instruct-mcore-tp2
TOKENIZER_MODEL=/root/work/huangxin/nanda/models/llama3.1-8b-instruct
DATA_PATH=$WORK_DIR/Megatron-LM/llamamoe/data/data-bin/id_llama3_8b_text_document

LR=5e-5
MICRO_BSZ=4
GLOBAL_BSZ=1024
TRAIN_STEPS=5000
SAVE_STEPS=1000
EVAL_STEPS=1000
WARMUP_STEPS=300

TP=2
EP=1
PP=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

LM_EVAL_ARGS=(
    --num-fewshot 25
    --results-path /root/work/huangxin/nanda/Megatron-LM/llamamoe/eval/results/mcore.json
    --task-list arc_challenge
    --adaptive-seq-len
    --bsz 32
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
    --seq-length 1024
    --max-position-embeddings 131072
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --no-rope-fusion
    --use-rotary-position-embeddings
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --rotary-base 500000
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
    --distributed-timeout-minutes 2
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

torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRONN_PATH}/llamamoe/eval/mcore_eval_harness.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${LM_EVAL_ARGS[@]}