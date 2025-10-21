#!/bin/bash

# 基于 pope_custom_adaptive.sh 的两步评估流程，
# 在 POPE 基准上评估多投影器模型，并指定使用的层。

# --- 配置 ---
# MODEL_PATH: 您训练好的多投影器模型的路径
# EVAL_LAYER_IDX: 要评估的层的索引 (0, 1, 2, 3, ...)。
# ----------------

MODEL_PATH=${1:-./checkpoints/llava-v1.5-7b-multi-projector}
EVAL_LAYER_IDX=${2:-2} # 默认评估第 3 层 (索引 2)
DATA_ROOT=./playground/data

# 基于模型路径和层索引，生成唯一的实验名称
CKPT_NAME=$(basename "${MODEL_PATH}")
EXP_NAME="${CKPT_NAME}_layer_${EVAL_LAYER_IDX}"
ANSWERS_DIR=${DATA_ROOT}/eval/pope/answers
ANSWERS_FILE=${ANSWERS_DIR}/${EXP_NAME}.jsonl

mkdir -p "${ANSWERS_DIR}"

echo "Starting answer generation for ${EXP_NAME} on POPE..."

# 1. 使用 model_vqa_loader.py 生成模型的预测答案
python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file ${DATA_ROOT}/eval/pope/llava_pope_test.jsonl \
    --image-folder ${DATA_ROOT}/eval/pope/val2014 \
    --answers-file "${ANSWERS_FILE}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --eval-layer-idx ${EVAL_LAYER_IDX}

echo "Answer generation finished. Starting evaluation..."

# 2. 使用 eval_pope.py 评估生成的答案文件
python llava/eval/eval_pope.py \
    --annotation-dir ${DATA_ROOT}/eval/pope/coco \
    --question-file ${DATA_ROOT}/eval/pope/llava_pope_test.jsonl \
    --result-file "${ANSWERS_FILE}"

echo "Evaluation finished for ${EXP_NAME}. Results are in ${ANSWERS_FILE}"
