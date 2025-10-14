#!/bin/bash

# 在 POPE 基准上评估自定义训练的 adaptive 模型

MODEL_PATH=./checkpoints/llava-v1.5-7b-adaptive2
EXP_NAME=llava-v1.5-7b-adaptive2
ANSWERS_DIR=./playground/data/eval/pope/answers

# 确保答案目录存在
mkdir -p "${ANSWERS_DIR}"

python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl"
