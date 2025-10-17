import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class MultiLayerCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # 从逗号分隔的字符串解析要使用的层级列表
        layers_str = getattr(args, 'mm_vision_layers_to_use', "-2")
        self.layers_to_use = [int(x) for x in layers_str.split(',')]

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                
                # 提取并处理所有指定层的特征
                current_image_features = self._process_hidden_states(image_forward_out.hidden_states)
                image_features.append(current_image_features)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self._process_hidden_states(image_forward_outs.hidden_states)
        
        return image_features

    def _process_hidden_states(self, hidden_states):
        """ 从 hidden_states 中提取、处理并返回指定的多层特征 """
        output_features = []
        
        # hidden_states 的第0个是 embedding layer 的输出, Transformer layer 从第1个开始
        # 将 Python 的负数索引转换为正数索引
        num_hidden_layers = len(hidden_states) - 1
        
        for layer_idx in self.layers_to_use:
            if layer_idx < 0:
                # 转换负数索引, -1 对应最后一个, -2 对应倒数第二个
                actual_idx = num_hidden_layers + layer_idx + 1
            else:
                # 正数索引，+1 是因为 hidden_states[0] 是 embedding
                actual_idx = layer_idx + 1

            if 0 <= actual_idx < len(hidden_states):
                layer_feature = hidden_states[actual_idx]
                
                if self.select_feature == 'patch':
                    feature = layer_feature[:, 1:]
                elif self.select_feature == 'cls_patch':
                    feature = layer_feature
                else:
                    raise ValueError(f"Unknown select feature: {self.select_feature}")
                
                output_features.append(feature)
        
        # 确保输出的数据类型与 vision_tower 的输入期望一致
        # (通常 vision_tower 是 float32, 但后续流程可能需要其他类型)
        output_features = [feat.to(self.dtype) for feat in output_features]

        return output_features


    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
