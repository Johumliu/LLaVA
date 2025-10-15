#!/bin/bash

# 在 TextVQA 基准上评估自定义训练的 adaptive 模型

MODEL_PATH=./checkpoints/llava-v1.5-7b-adaptive2
EXP_NAME=llava-v1.5-7b-adaptive2
ANSWERS_DIR=./playground/data/eval/textvqa/answers

# 确保答案目录存在
mkdir -p "${ANSWERS_DIR}"

python -m llava.eval.model_vqa_loader \
    --model-path "${MODEL_PATH}" \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl" \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${ANSWERS_DIR}/${EXP_NAME}.jsonl"
