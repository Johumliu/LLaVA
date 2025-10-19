#!/bin/bash

# 在 MME 基准上评估多投影器模型，并指定使用的层

# --- 配置 ---
# MODEL_PATH: 您训练好的多投影器模型的路径
# EVAL_LAYER_IDX: 要评估的层的索引 (0, 1, 2, 3, ...)。-1 或不设置则使用最后一层。
# ----------------

MODEL_PATH=${1:-./checkpoints/llava-v1.5-7b-multi-projector}
DATA_ROOT=${2:-./playground/data}
EVAL_LAYER_IDX=${3:-2}  # 默认评估第 3 层 (索引 2)

# 基于模型路径和层索引，生成唯一的实验名称
CKPT_NAME=$(basename "${MODEL_PATH}")
EXP_NAME="${CKPT_NAME}_layer_${EVAL_LAYER_IDX}"

MME_QA_FILE=${DATA_ROOT}/eval/MME/llava_mme.jsonl
MME_IMAGE_DIR=${DATA_ROOT}/eval/MME/MME_Benchmark_release_version
ANSWERS_DIR=${DATA_ROOT}/eval/MME/answers
ANSWERS_FILE=${ANSWERS_DIR}/${EXP_NAME}.jsonl

mkdir -p "${ANSWERS_DIR}"

echo "Starting evaluation for ${EXP_NAME}..."

python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file "${MME_QA_FILE}" \
    --image-folder "${MME_IMAGE_DIR}" \
    --answers-file "${ANSWERS_FILE}" \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --eval-layer-idx ${EVAL_LAYER_IDX}

# 转换答案为MME官方评测输入格式
(
  cd "${DATA_ROOT}/eval/MME" && \
  python convert_answer_to_mme.py --experiment "${EXP_NAME}"
)

# 运行官方评测脚本，输出总分和各分项结果
(
  cd "${DATA_ROOT}/eval/MME/eval_tool" && \
  python calculation.py --results_dir "${ANSWERS_DIR}/${EXP_NAME}"
)

echo "Evaluation finished for ${EXP_NAME}. Results are in ${ANSWERS_DIR}/${EXP_NAME}"
