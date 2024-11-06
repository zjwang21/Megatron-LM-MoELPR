#export HF_DATASETS_CACHE=/root/work/huangxin/nanda/data/cache
model_path=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-0.5B
lm_eval --model hf \
        --model_args pretrained=$model_path,dtype="bfloat16" \
        --tasks cmmlu \
        --device cuda:0 \
        --num_fewshot 5 \
        --batch_size 16