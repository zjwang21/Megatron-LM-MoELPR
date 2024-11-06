MEGATRONN_PATH=/root/work/huangxin/nanda/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

LLAMA3_PATH=/root/work/huangxin/ckpts/cm_3B
TARGET_TP_SIZE=1
TARGET_EP_SIZE=1
TARGET_PP_SIZE=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_FORMAT_DIR=/root/work/huangxin/nanda/models/cm3b-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

echo "Convert mixtral mdoel to megatron mcore model with tp=${TARGET_TP_SIZE} ep=${TARGET_EP_SIZE} pp=${TARGET_PP_SIZE}......"
cd ../
python /root/work/huangxin/nanda/Megatron-LM/tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_llama_mistral \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${LLAMA3_PATH} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${LLAMA3_PATH} \
--checkpoint-type hf \
--model-size llama2-7B