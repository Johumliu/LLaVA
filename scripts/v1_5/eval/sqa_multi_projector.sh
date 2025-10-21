#!/bin/bash

# 在 ScienceQA (SQA) 基准上评估多投影器模型，并指定使用的层

# --- 配置 ---
# MODEL_PATH: 您训练好的多投影器模型的路径
# DATA_ROOT: 数据集的根目录
# EVAL_LAYER_IDX: 要评估的层的索引 (0, 1, 2, 3, ...)。
# ----------------

# --- 默认值 ---
MODEL_PATH=${1:-./checkpoints/llava-v1.5-7b-multi-projector}
DATA_ROOT=${2:-./playground/data}
EVAL_LAYER_IDX=${3:-2} # 默认评估第 3 层 (索引 2)
# ---------------

# 基于模型路径和层索引，生成唯一的实验名称
CKPT_NAME=$(basename "${MODEL_PATH}")
if [ -d "${MODEL_PATH}/mm_projector.bin" ]; then
    # 如果是完整的模型目录
    EXP_NAME="${CKPT_NAME}_layer_${EVAL_LAYER_IDX}"
else
    # 如果是 checkpoint 目录
    PARENT_DIR_NAME=$(basename "$(dirname "$MODEL_PATH")")
    EXP_NAME="${PARENT_DIR_NAME}_${CKPT_NAME}_layer_${EVAL_LAYER_IDX}"
fi

CHUNKS=8
CONV_MODE=vicuna_v1
RESULT_DIR=${DATA_ROOT}/eval/scienceqa/answers/${EXP_NAME}

echo "Starting evaluation for ${EXP_NAME} on ScienceQA..."

# 1. 生成模型的预测结果
for IDX in $(seq 0 $((CHUNKS-1))); do
    python -m llava.eval.model_vqa_science \
        --model-path "${MODEL_PATH}" \
        --question-file ${DATA_ROOT}/eval/scienceqa/llava_test_CQM-A.json \
        --image-folder ${DATA_ROOT}/eval/scienceqa/images/test \
        --answers-file ${RESULT_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --conv-mode ${CONV_MODE} \
        --eval-layer-idx ${EVAL_LAYER_IDX}
done

# 2. 合并预测结果
output_file=${RESULT_DIR}/merge.jsonl
> "${output_file}" 
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${RESULT_DIR}/${CHUNKS}_${IDX}.jsonl >> "${output_file}"
done

# 3. 评估并计算准确率
python llava/eval/eval_science_qa.py \
    --base-dir ${DATA_ROOT}/eval/scienceqa \
    --result-file "${output_file}" \
    --output-file ${RESULT_DIR}/output.json \
    --output-result ${RESULT_DIR}/result.json

echo "Evaluation finished for ${EXP_NAME}. Results are in ${RESULT_DIR}"
