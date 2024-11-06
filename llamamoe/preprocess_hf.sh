export WORK_DIR=/root/work/huangxin/nanda/Megatron-LM

python $WORK_DIR/tools/hf_pretrain_dataset_patch/preprocess_pretrain_data_hf.py \
       --data_dir $WORK_DIR/../data/review/5w \
       --save_dir $WORK_DIR/../data/review/tokenized/qwen2.5-5w \
       --tokenizer_name_or_path /root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct \
       --sequence_length 2048