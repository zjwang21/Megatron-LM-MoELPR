MEGATRONN_PATH=/root/work/huangxin/nanda/Megatron-LM
HF_DATA_PATCH_PATH=$MEGATRONN_PATH/tools
export PYTHONPATH=${MEGATRONN_PATH}:${HF_DATA_PATCH_PATH}:$PYTHONPATH

LLAMA3_PATH=/root/work/huangxin/nanda/models/Qwen/Qwen2.5-3B-Instruct
NUM_EXPERTS=4
TARGET_TP_SIZE=1
TARGET_EP_SIZE=2
TARGET_PP_SIZE=1
HF_FORMAT_DIR=/root/work/huangxin/nanda/models/Qwen2.5-3B-Instruct-mixtral-${NUM_EXPERTS}exp

echo "Upcycling llama3 to mixtral with ${NUM_EXPERTS} experts......"
python upcycling.py \
--model_path $LLAMA3_PATH \
--output_path $HF_FORMAT_DIR \
--num_experts $NUM_EXPERTS \
--moe_type sparse

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_FORMAT_DIR=/root/work/huangxin/nanda/models/Qwen2.5-3B-Instruct-mixtral-mcore-exp${NUM_EXPERTS}-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}

echo "Convert mixtral mdoel to megatron mcore model with tp=${TARGET_TP_SIZE} ep=${TARGET_EP_SIZE} pp=${TARGET_PP_SIZE}......"
cd ../
python /root/work/huangxin/nanda/Megatron-LM/tools/checkpoint/convert.py \
--model-type GPT \
--loader loader_mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${LLAMA3_PATH} \


rm -rf $HF_FORMAT_DIR