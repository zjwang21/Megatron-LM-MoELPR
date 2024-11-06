export HF_DATASETS_CACHE=/root/work/huangxin/nanda/data/cache

model_path=/root/work/huangxin/ckpts/cm_3B
declare -A task_dict=(
  ["arc"]="arc_challenge,arc_zh,arc_es,arc_id,arc_fr,arc_pt,arc_tr"
  ["mmlu"]="m_mmlu_en,m_mmlu_zh,m_mmlu_es,m_mmlu_id,m_mmlu_fr,m_mmlu_pt,m_mmlu_tr"
  ["hellaswag"]="hellaswag_en,hellaswag_zh,hellaswag_es,hellaswag_id,hellaswag_fr,hellaswag_pt"
  ["belebele"]="belebele_eng_Latn,belebele_zho_Hans,belebele_spa_Latn,belebele_ind_Latn,belebele_fra_Latn,belebele_por_Latn,belebele_tur_Latn"
)

declare -A task_dict_long=(
  ["arc"]="arc_hu,arc_el"
  ["mmlu"]="m_mmlu_hu,m_mmlu_el"
  ["hellaswag"]="hellaswag_hu,hellaswag_el"
  ["belebele"]="belebele_hun_Latn,belebele_ell_Grek"
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
  ["hellaswag"]=16
  ["belebele"]=16
)

declare -A bsz_dict_long=(
  ["arc"]=6
  ["mmlu"]=6
  ["hellaswag"]=6
  ["belebele"]=6
)

task=$1
device=$2

lm_eval --model hf \
        --model_args pretrained=$model_path,dtype="bfloat16" \
        --tasks ${task_dict[$task]} \
        --device cuda:$device \
        --num_fewshot ${few_shot_dict[$task]} \
        --output_path /root/work/huangxin/nanda/Megatron-LM/llamamoe/eval/results \
        --batch_size ${bsz_dict[$task]}

lm_eval --model hf \
        --model_args pretrained=$model_path,dtype="bfloat16" \
        --tasks ${task_dict_long[$task]} \
        --device cuda:$device \
        --num_fewshot ${few_shot_dict[$task]} \
        --output_path /root/work/huangxin/nanda/Megatron-LM/llamamoe/eval/results \
        --batch_size ${bsz_dict_long[$task]}