import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from llava.model.moe_layer import MoELayer


class AdaptiveCLIPVisionTower(nn.Module):
    """自适应CLIP视觉编码器，能够选择最合适的层"""
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # 定义要融合的固定层
        self.layers_to_fuse = [6, 15, 23]  # 对应第7, 16, 24层
        
        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        # 初始化用于层特征融合的MoE模块
        self.feature_fusion_moe = MoELayer(
            hidden_size=self.vision_tower.config.hidden_size,
            num_experts=len(self.layers_to_fuse),
            top_k=1  # 每个token只选择一个最相关的层特征
        )
        self.feature_fusion_moe.to(device=self.device, dtype=self.dtype)
        
        self.is_loaded = True
    
    def forward(self, images):
        """前向传播"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                
                # 提取指定层的特征
                selected_layer_features = []
                for layer_idx in self.layers_to_fuse:
                    # hidden_states 包含 embedding layer 的输出，所以索引要 +1
                    selected_layer_features.append(image_forward_out.hidden_states[layer_idx + 1])

                # 融合特征
                fused_features = self.fuse_features(selected_layer_features)
                image_features.append(fused_features)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            
            # 提取指定层的特征
            selected_layer_features = []
            for layer_idx in self.layers_to_fuse:
                selected_layer_features.append(image_forward_outs.hidden_states[layer_idx + 1])
            
            image_features = self.fuse_features(selected_layer_features)
        
        return image_features

    def fuse_features(self, layer_features):
        # 移除 CLS token 并堆叠
        stacked_features = []
        for features in layer_features:
            if self.select_feature == 'patch':
                # [batch_size, num_tokens, hidden_size]
                stacked_features.append(features[:, 1:])
            else: # 'cls_patch'
                stacked_features.append(features)
        
        # -> [batch_size, num_experts, num_tokens, hidden_size]
        stacked_features = torch.stack(stacked_features, dim=1)
        
        # 在 permute 之前获取正确的维度信息
        batch_size, _, num_tokens, hidden_size = stacked_features.shape
        
        # -> [batch_size, num_tokens, num_experts, hidden_size]
        stacked_features = stacked_features.permute(0, 2, 1, 3)

        # 通过 MoE 融合
        # MoE 的输出是 [batch_size * num_tokens, hidden_size]
        fused_features = self.feature_fusion_moe(stacked_features)
        
        # 使用之前保存的维度信息来恢复正确的形状
        fused_features = fused_features.reshape(batch_size, num_tokens, hidden_size)
        
        return fused_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return self.vision_tower.dtype
    
    @property
    def device(self):
        return self.vision_tower.device
    
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only
    
    @property
    def hidden_size(self):
        return self.config.hidden_size
    
    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size
    
    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
