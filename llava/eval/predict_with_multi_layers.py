import argparse
import torch
import os
import json
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image
import math
import random

def get_chunk(lst, n, k):
    """将列表分割成 n 个大致相等的块，并返回第 k 个块"""
    chunks = [lst[i::n] for i in range(n)]
    return chunks[k]

def predict_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    try:
        layers_to_predict = [int(x) for x in args.mm_vision_layers_to_use.split(',')]
        print(f"Will generate predictions for layers: {layers_to_predict}")
    except Exception as e:
        raise ValueError(f"Could not parse --mm-vision-layers-to-use: {e}")

    input_data = json.load(open(os.path.expanduser(args.input_json), "r"))
    
    # --- 修改：随机采样百分之一的数据 ---
    subset_size = len(input_data) // 100
    random.seed(42) # for reproducibility
    random.shuffle(input_data)
    input_data = input_data[:subset_size]
    print(f"Running prediction on a random subset of the data: {len(input_data)} items.")
    # -----------------------------------------

    # 1. 将数据扁平化为独立的预测任务
    tasks = []
    for item_idx, item in enumerate(input_data):
        if not item.get("image") or not item.get("conversations"):
            continue
        
        # 找到所有 "human" 回合的索引
        human_turn_indices = [i for i, turn in enumerate(item['conversations']) if turn['from'] == 'human']
        
        for turn_idx in human_turn_indices:
            tasks.append({'item_idx': item_idx, 'turn_idx': turn_idx})

    # 用于存储所有新生成的回合
    all_new_turns = [[] for _ in range(len(input_data))]

    # 2. 外层循环：遍历要预测的每一个层
    for layer_idx, layer_num in enumerate(tqdm(layers_to_predict, desc="Processing Layers")):
        
        # 3. 内层循环：按批次处理“任务”
        for i in tqdm(range(0, len(tasks), args.batch_size), desc=f"Layer {layer_num} Batches", leave=False):
            batch_tasks = tasks[i : i + args.batch_size]

            batch_images = []
            batch_prompts = []
            batch_image_sizes = []
            
            # 准备一个批次的数据
            for task in batch_tasks:
                item = input_data[task['item_idx']]
                turn_idx = task['turn_idx']
                
                image_file = item["image"]
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                
                # --- 修改：构建独立的、无上下文的对话 ---
                conv = conv_templates[args.conv_mode].copy()
                # 获取当前正在处理的人类提问
                question = item['conversations'][turn_idx]['value']
                
                # 确保问题中的 <image> 占位符被正确替换
                if '<image>' not in question:
                    # 如果用户的问题中没有 <image>，我们在开头添加
                    if model.config.mm_use_im_start_end:
                         question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                    else:
                         question = DEFAULT_IMAGE_TOKEN + '\n' + question
                else:
                    # 如果用户问题中已有 <image>，我们只替换它
                    if model.config.mm_use_im_start_end:
                        question = question.replace('<image>\n', DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n')
                    else:
                        question = question.replace('<image>\n', DEFAULT_IMAGE_TOKEN + '\n')
                
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                # -----------------------------------------
                
                batch_images.append(image)
                batch_image_sizes.append(image.size)
                batch_prompts.append(prompt)

            if not batch_images:
                continue

            # 批量处理图片和文本
            image_tensors = process_images(batch_images, image_processor, model.config)
            input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in batch_prompts]
            
            max_len = max(len(ids) for ids in input_ids)
            batch_input_ids = torch.full((len(input_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
            for j, ids in enumerate(input_ids):
                batch_input_ids[j, max_len - len(ids):] = ids

            # 批量生成预测
            with torch.inference_mode():
                batch_output_ids = model.generate(
                    batch_input_ids.cuda(),
                    images=image_tensors.to(dtype=torch.float16, device='cuda'),
                    image_sizes=batch_image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    eval_layer_idx=layer_idx
                )

            # 批量解码并将结果存回
            batch_full_outputs = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)

            for j, full_output in enumerate(batch_full_outputs):
                full_output = full_output.strip()
                
                task = batch_tasks[j]
                item_idx = task['item_idx']
                question_index = task['turn_idx'] // 2 # 计算这是第几个问题 (0-indexed)

                # 从完整输出中分离出答案
                # (注意：这里的 conv 是上一个循环的最后一个，但 sep 应该是通用的)
                conv = conv_templates[args.conv_mode].copy()
                sep = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                parts = full_output.rsplit(sep, 1)
                answer = parts[1].strip() if len(parts) == 2 else full_output
                
                new_turn = {
                    "from": f"output_layer_{layer_num}_q{question_index}",
                    "value": answer
                }
                all_new_turns[item_idx].append(new_turn)

    # 4. 所有层处理完毕后，将新生成的回合追加到原始数据中
    for item_idx, item in enumerate(input_data):
        item['conversations'].extend(all_new_turns[item_idx])

    # 5. 保存最终结果
    with open(os.path.expanduser(args.output_json), "w") as f:
        json.dump(input_data, f, indent=2)
    print(f"Prediction finished. Results are saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--mm-vision-layers-to-use", type=str, required=True, help="Comma-separated list of vision layer numbers to use for prediction, e.g., '6,12,18,23'")
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for prediction.")
    # 新增用于分布式预测的参数
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    predict_model(args)
