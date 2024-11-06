#!/bin/bash

# Runs Mixtral 8x7B model
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
DATA_PATH=$WORK_DIR/data/review/tokenized/qwen2.5-5w
LPR_LOSS_COEFF=1e-2
AUX_LOSS_COEFF=1e-3
LPR_STAGE=2
NUM_EXPERTS=12
TOPK=4

LR=5e-5
MICRO_BSZ=4
GLOBAL_BSZ=512
TRAIN_STEPS=360
SAVE_STEPS=350
EVAL_STEPS=10000
WARMUP_STEPS=20

TP=1
EP=4
PP=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

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
    --data-cache-path $DATA_PATH
    --split 99998,1,1
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
    --distributed-timeout-minutes 100000
    --no-save-optim
    --no-save-rng
    --finetune
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
    --eval-interval 100000 \
    --eval-iters $EVAL_STEPS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng \
    --log-throughput \
    --log-progress
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"share_expert"}
        --wandb-exp-name ${WANDB_NAME:-"qwen2.5_3b_instruct_12exp_top4_tp1_ep4_review5w"}
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} ${MEGATRONN_PATH}/pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} > $WORK_DIR/Megatron-LM/llamamoe/elhutr30b_qwen2.5_3b_12exp_top4_share_review5w.out

exit