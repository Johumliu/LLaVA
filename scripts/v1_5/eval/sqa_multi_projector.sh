#!/bin/bash

# 基于 sqa_custom_adaptive.sh 的简洁结构，
# 在 ScienceQA (SQA) 基准上评估多投影器模型，并指定使用的层。

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
RESULT_FILE=${DATA_ROOT}/eval/scienceqa/answers/${EXP_NAME}.jsonl

echo "Starting evaluation for ${EXP_NAME} on ScienceQA..."

python -m llava.eval.model_vqa_science \
    --model-path "${MODEL_PATH}" \
    --question-file ${DATA_ROOT}/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ${DATA_ROOT}/eval/scienceqa/images/test \
    --answers-file "${RESULT_FILE}" \
    --conv-mode vicuna_v1 \
    --eval-layer-idx ${EVAL_LAYER_IDX}

python llava/eval/eval_science_qa.py \
    --base-dir ${DATA_ROOT}/eval/scienceqa \
    --result-file "${RESULT_FILE}" \
    --output-file ${DATA_ROOT}/eval/scienceqa/answers/${EXP_NAME}_output.json \
    --output-result ${DATA_ROOT}/eval/scienceqa/answers/${EXP_NAME}_result.json

echo "Evaluation finished for ${EXP_NAME}. Results are in ${DATA_ROOT}/eval/scienceqa/answers/"
