WORK_DIR=/root/work/huangxin/nanda/Megatron-LM
export PYTHONPATH=${WORK_DIR}:$PYTHONPATH

cd $WORK_DIR
python $WORK_DIR/tools/preprocess_data.py \
    --input //root/work/huangxin/nanda/data/id_50b.jsonl \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /root/work/huangxin/ckpts/cm_3B \
    --append-eod \
    --output-prefix /root/work/huangxin/nanda/data/data-bin/id_cm_mc4_culturax \
    --workers 24 \
