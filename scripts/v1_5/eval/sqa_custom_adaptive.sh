#!/bin/bash

# 在 ScienceQA 基准上评估自定义训练的 adaptive 模型

MODEL_PATH=./checkpoints/llava-v1.5-7b-adaptive2
EXP_NAME=llava-v1.5-7b-adaptive2
ANSWERS_DIR=./playground/data/eval/scienceqa/answers

# 确保答案目录存在
mkdir -p "${ANSWERS_DIR}"

python -m llava.eval.model_vqa_science \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl" \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl" \
    --output-file "${ANSWERS_DIR}/${EXP_NAME}_output.jsonl" \
    --output-result "${ANSWERS_DIR}/${EXP_NAME}_result.json"
