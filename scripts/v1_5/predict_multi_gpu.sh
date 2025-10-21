#!/bin/bash

# 使用多个 GPU 并行执行 predict_with_multi_layers.py 脚本，以实现数据并行加速。

# --- 配置 ---
# GPUS: 要使用的 GPU ID 列表，以逗号分隔。
# MODEL_PATH, IMAGE_FOLDER, INPUT_JSON, OUTPUT_JSON, MM_VISION_LAYERS_TO_USE, BATCH_SIZE: 传递给 python 脚本的参数。
# ----------------

# --- 默认值 ---
GPUS="0,1,2,3,4,5,6,7"
MODEL_PATH=${1:-"./checkpoints/llava-v1.5-7b-multi-projector"}
IMAGE_FOLDER=${2:-"./playground/data/llava_v1_5_mix665k/train2017"}
INPUT_JSON=${3:-"./playground/data/llava_v1_5_mix665k/llava_v1_5_mix665k.json"}
OUTPUT_JSON=${4:-"./playground/data/llava_v1_5_mix665k/llava_v1_5_mix665k_output.json"}
MM_VISION_LAYERS_TO_USE=${5:-"6,12,18,23"}
BATCH_SIZE=${6:-8}
# ---------------

# 将 GPU 字符串转换为数组
IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

OUTPUT_DIR=$(dirname "${OUTPUT_JSON}")
BASE_NAME=$(basename "${OUTPUT_JSON}" .json)

echo "Starting distributed prediction on ${NUM_GPUS} GPUs..."

# 1. 并行启动多个预测进程
for i in "${!GPU_ARRAY[@]}"; do
    GPU_ID=${GPU_ARRAY[$i]}
    echo "Launching process for GPU ${GPU_ID} (Chunk ${i}/${NUM_GPUS})..."
    
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m llava.eval.predict_with_multi_layers \
        --model-path "${MODEL_PATH}" \
        --image-folder "${IMAGE_FOLDER}" \
        --input-json "${INPUT_JSON}" \
        --output-json "${OUTPUT_JSON}" \
        --mm-vision-layers-to-use "${MM_VISION_LAYERS_TO_USE}" \
        --batch-size ${BATCH_SIZE} \
        --num-chunks ${NUM_GPUS} \
        --chunk-idx ${i} &
done

# 等待所有后台进程完成
wait

echo "All prediction chunks have been generated."

# 2. 合并结果文件 (需要一个合并脚本)
echo "Merging prediction chunks..."
python -m llava.eval.merge_json_results \
    --input-dir "${OUTPUT_DIR}" \
    --filename-prefix "${BASE_NAME}_chunk" \
    --output-file "${OUTPUT_JSON}"

echo "Merging complete. Final result is at ${OUTPUT_JSON}"
echo "Distributed prediction finished."
