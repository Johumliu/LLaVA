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

    # 解析需要进行预测的层
    try:
        layers_to_predict = [int(x) for x in args.mm_vision_layers_to_use.split(',')]
        print(f"Will generate predictions for layers: {layers_to_predict}")
    except Exception as e:
        raise ValueError(f"Could not parse --mm-vision-layers-to-use: {e}")

    # 加载输入数据
    input_data = json.load(open(os.path.expanduser(args.input_json), "r"))
    
    results = []
    
    for item in tqdm(input_data, desc="Processing items"):
        image_file = item.get("image")
        if not image_file:
            continue

        # 加载图片
        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        # 提取第一个 human 的对话
        if not item.get('conversations') or item['conversations'][0]['from'] != 'human':
            continue
        
        qs = item['conversations'][0]['value']
        # 移除 <image> 占位符，因为我们会通过模板来添加
        qs = qs.replace('<image>\n', '').replace('\n<image>', '').strip()
        
        # 为每一层生成预测
        for i, layer_num in enumerate(layers_to_predict):
            # 每次都重置对话模板
            conv = conv_templates[args.conv_mode].copy()
            if model.config.mm_use_im_start_end:
                qs_with_image = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs_with_image = DEFAULT_IMAGE_TOKEN + '\n' + qs
            
            conv.append_message(conv.roles[0], qs_with_image)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).to(dtype=torch.float16, device='cuda'),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    eval_layer_idx=i  # 使用 projector 的索引
                )

            # 解码并清理输出
            full_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 从完整输出中分离出答案
            sep = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            parts = full_output.rsplit(sep, 1)
            answer = parts[1].strip() if len(parts) == 2 else full_output

            # 构造新的对话条目
            new_turn = {
                "from": f"output_layer_{layer_num}",
                "value": answer
            }
            item['conversations'].append(new_turn)

        results.append(item)

    # 保存结果
    with open(os.path.expanduser(args.output_json), "w") as f:
        json.dump(results, f, indent=2)
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
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    predict_model(args)
