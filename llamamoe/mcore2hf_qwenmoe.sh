MEGATRONN_PATH=/root/work/huangxin/nanda/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

LLAMA3_PATH=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct
NUM_EXPERTS=12
TARGET_TP_SIZE=1
TARGET_EP_SIZE=4
TARGET_PP_SIZE=1
HF_FORMAT_DIR=/root/work/huangxin/nanda/models/llama3-8b-instruct-id5b-mcore2hf-8exp-top2

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_FORMAT_DIR=/root/work/huangxin/nanda/models/llama3-8b-instruct-qwenmoe-mcore-exp8-TP1PP1EP8

echo "Convert mixtral mdoel to megatron mcore model with tp=${TARGET_TP_SIZE} ep=${TARGET_EP_SIZE} pp=${TARGET_PP_SIZE}......"
cd ../
python /root/work/huangxin/nanda/Megatron-LM/tools/checkpoint/convert.py \
--model-type GPT \
--loader mcore \
--saver loader_qwenmoe \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${LLAMA3_PATH} \