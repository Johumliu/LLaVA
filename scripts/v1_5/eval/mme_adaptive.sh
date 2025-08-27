#!/bin/bash

# 用已微调的Adaptive模型在 MME 基准上评估
# 可选：通过第一个参数覆盖模型路径；第二个参数覆盖数据根目录
# 默认模型路径与 scripts/v1_5/finetune_adaptive.sh 的输出一致

MODEL_PATH=${1:-./checkpoints/llava-v1.5-7b-adaptive}
DATA_ROOT=${2:-./playground/data}

MME_QA_FILE=${DATA_ROOT}/eval/MME/llava_mme.jsonl
MME_IMAGE_DIR=${DATA_ROOT}/eval/MME/MME_Benchmark_release_version
ANSWERS_DIR=${DATA_ROOT}/eval/MME/answers
EXP_NAME=llava-v1.5-7b-adaptive
ANSWERS_FILE=${ANSWERS_DIR}/${EXP_NAME}.jsonl

mkdir -p "${ANSWERS_DIR}"

python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file "${MME_QA_FILE}" \
    --image-folder "${MME_IMAGE_DIR}" \
    --answers-file "${ANSWERS_FILE}" \
    --temperature 0 \
    --conv-mode vicuna_v1

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
