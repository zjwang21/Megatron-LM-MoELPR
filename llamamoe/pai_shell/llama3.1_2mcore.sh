MEGATRONN_PATH=/root/work/huangxin/nanda/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

TARGET_TP_SIZE=2
TARGET_PP_SIZE=1
HF_FORMAT_DIR=/root/work/huangxin/nanda/models/llama3.1-8b-instruct

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_FORMAT_DIR=/root/work/huangxin/nanda/models/llama3.1-8b-instruct-mcore-tp2

echo "Convert llama mdoel to megatron mcore model with tp=${TARGET_TP_SIZE} pp=${TARGET_PP_SIZE}......"
cd ../
python /root/work/huangxin/nanda/Megatron-LM/tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_llama_mistral \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${HF_FORMAT_DIR} \
--model-size llama3-8B \
--checkpoint-type hf