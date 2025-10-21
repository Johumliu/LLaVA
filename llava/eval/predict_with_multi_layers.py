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
    
    # 外层循环：遍历要预测的每一个层
    for layer_idx, layer_num in enumerate(tqdm(layers_to_predict, desc="Processing Layers")):
        
        # 内层循环：按批次处理数据
        for i in tqdm(range(0, len(input_data), args.batch_size), desc=f"Layer {layer_num} Batches", leave=False):
            batch_slice = slice(i, i + args.batch_size)
            batch_data = input_data[batch_slice]

            batch_images = []
            batch_prompts = []
            batch_image_sizes = []
            
            # 1. 准备一个批次的数据
            for item in batch_data:
                image_file = item.get("image")
                if not image_file or not os.path.exists(os.path.join(args.image_folder, image_file)):
                    # 跳过没有有效图片的项
                    continue
                
                image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
                
                qs = item['conversations'][0]['value'].replace('<image>\n', '').replace('\n<image>', '').strip()
                
                conv = conv_templates[args.conv_mode].copy()
                if model.config.mm_use_im_start_end:
                    qs_with_image = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs_with_image = DEFAULT_IMAGE_TOKEN + '\n' + qs
                
                conv.append_message(conv.roles[0], qs_with_image)
                conv.append_message(conv.roles[1], None)
                
                batch_images.append(image)
                batch_image_sizes.append(image.size)
                batch_prompts.append(conv.get_prompt())

            if not batch_images:
                continue

            # 2. 批量处理图片和文本
            image_tensors = process_images(batch_images, image_processor, model.config)
            input_ids = [tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt') for prompt in batch_prompts]
            
            max_len = max(len(ids) for ids in input_ids)
            batch_input_ids = torch.full((len(input_ids), max_len), tokenizer.pad_token_id, dtype=torch.long)
            for j, ids in enumerate(input_ids):
                batch_input_ids[j, max_len - len(ids):] = ids

            # 3. 批量生成预测
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
                    eval_layer_idx=layer_idx  # 使用 projector 的索引
                )

            # 4. 批量解码并将结果存回
            batch_full_outputs = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)

            for j, full_output in enumerate(batch_full_outputs):
                full_output = full_output.strip()
                
                conv = conv_templates[args.conv_mode].copy()
                sep = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                parts = full_output.rsplit(sep, 1)
                answer = parts[1].strip() if len(parts) == 2 else full_output

                original_item = batch_data[j]
                new_turn = {
                    "from": f"output_layer_{layer_num}",
                    "value": answer
                }
                original_item['conversations'].append(new_turn)

    # 所有层处理完毕后，保存最终结果
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    predict_model(args)
