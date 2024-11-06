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
    bash $WORK_DIR/eval/mcore_eval.sh arc short
    bash $WORK_DIR/eval/mcore_eval.sh belebele short
    bash $WORK_DIR/eval/mcore_eval.sh mmlu short
    bash $WORK_DIR/eval/mcore_eval.sh hellaswag short
    work_wait
fi