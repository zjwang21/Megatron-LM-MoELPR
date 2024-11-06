#!/bin/bash
echo "----------------"
sleep 5
work_wait() {
    TRAIN_FLAG="megatron"
    sleep 1500
    while true; do
        if  ps -ef | grep "$TRAIN_FLAG" | grep -v "grep" > /dev/null
        then
            echo "training..."
            sleep 30
        else
            exit 1
        fi
    done
}

echo $RANK
sleep 5

ENV_DIR=/root/work/huangxin/envs/nju-megatron
WORK_DIR=/root/work/huangxin/nanda/Megatron-LM/llamamoe

if [ ${RANK} -eq 0 ];
then
    export PATH=$PATH:$ENV_DIR/bin && \
    bash $WORK_DIR/eval/mcore_eval.sh arc long
    bash $WORK_DIR/eval/mcore_eval.sh belebele long
    bash $WORK_DIR/eval/mcore_eval.sh mmlu long
    bash $WORK_DIR/eval/mcore_eval.sh hellaswag long
    work_wait
fi