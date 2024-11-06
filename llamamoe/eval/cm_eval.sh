export HF_DATASETS_CACHE=/root/work/huangxin/nanda/data/cache

task=$1
device=$2

model_path=/root/work/huangxin/ckpts/cm_3B
declare -A task_dict=(
  ["arc"]="arc_challenge,arc_zh,arc_id"
  ["mmlu"]="m_mmlu_en,m_mmlu_zh,m_mmlu_id"
  ["hellaswag"]="hellaswag_en,hellaswag_zh,hellaswag_id"
  ["belebele"]="belebele_eng_Latn,belebele_zho_Hans,belebele_ind_Latn"
)

declare -A few_shot_dict=(
  ["arc"]=25
  ["mmlu"]=5
  ["hellaswag"]=10
  ["belebele"]=5
)

declare -A bsz_dict=(
  ["arc"]=16
  ["mmlu"]=16
  ["hellaswag"]=24
  ["belebele"]=16
)

lm_eval --model hf \
        --model_args pretrained=$model_path,dtype="bfloat16" \
        --tasks ${task_dict[$task]} \
        --device cuda:$device \
        --num_fewshot ${few_shot_dict[$task]} \
        --output_path /root/work/huangxin/nanda/Megatron-LM/llamamoe/eval/results \
        --batch_size ${bsz_dict[$task]}