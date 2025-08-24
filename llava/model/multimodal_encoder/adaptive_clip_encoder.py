import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class LayerClassifier(nn.Module):
    """轻量级分类器，用于选择最合适的视觉编码器层"""
    
    def __init__(self, vision_hidden_size, num_layers, classifier_hidden_size=256):
        super().__init__()
        self.num_layers = num_layers
        
        # 图像特征编码器
        self.image_encoder = nn.Sequential(
            nn.Linear(vision_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_size, classifier_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 文本特征编码器（如果有的话）
        self.text_encoder = nn.Sequential(
            nn.Linear(vision_hidden_size, classifier_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(classifier_hidden_size, classifier_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(classifier_hidden_size // 2, num_layers)
        )
        
    def forward(self, image_features, text_features=None):
        """
        Args:
            image_features: [batch_size, seq_len, hidden_size]
            text_features: [batch_size, seq_len, hidden_size] or None
        """
        # 对图像特征进行平均池化
        if len(image_features.shape) == 3:
            image_features = image_features.mean(dim=1)  # [batch_size, hidden_size]
        
        # 编码图像特征
        image_encoded = self.image_encoder(image_features)
        
        if text_features is not None:
            # 对文本特征进行平均池化
            if len(text_features.shape) == 3:
                text_features = text_features.mean(dim=1)  # [batch_size, hidden_size]
            
            # 编码文本特征
            text_encoded = self.text_encoder(text_features)
            
            # 融合特征
            fused = torch.cat([image_encoded, text_encoded], dim=-1)
        else:
            fused = image_encoded
        
        # 输出层概率分布
        layer_logits = self.fusion(fused)
        layer_probs = F.softmax(layer_logits, dim=-1)
        
        return layer_probs, layer_logits


class AdaptiveCLIPVisionTower(nn.Module):
    """自适应CLIP视觉编码器，能够选择最合适的层"""
    
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        
        # 新增参数
        self.use_adaptive_layer_selection = getattr(args, 'use_adaptive_layer_selection', True)
        self.top_k_layers = getattr(args, 'top_k_layers', 3)
        self.training_mode = getattr(args, 'training_mode', True)
        
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
        
        # 初始化层分类器
        if self.use_adaptive_layer_selection:
            num_layers = len(self.vision_tower.vision_model.encoder.layers)
            self.layer_classifier = LayerClassifier(
                vision_hidden_size=self.vision_tower.config.hidden_size,
                num_layers=num_layers
            )
        
        self.is_loaded = True
    
    def get_all_layer_features(self, images):
        """获取所有层的特征"""
        if type(images) is list:
            all_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0), 
                    output_hidden_states=True
                )
                # 获取所有层的特征
                layer_features = []
                for layer_idx in range(len(image_forward_out.hidden_states)):
                    features = image_forward_out.hidden_states[layer_idx]
                    if self.select_feature == 'patch':
                        features = features[:, 1:]  # 去掉CLS token
                    elif self.select_feature == 'cls_patch':
                        features = features  # 保留所有token
                    layer_features.append(features)
                all_features.append(layer_features)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype), 
                output_hidden_states=True
            )
            # 获取所有层的特征
            all_features = []
            for layer_idx in range(len(image_forward_outs.hidden_states)):
                features = image_forward_outs.hidden_states[layer_idx]
                if self.select_feature == 'patch':
                    features = features[:, 1:]  # 去掉CLS token
                elif self.select_feature == 'cls_patch':
                    features = features  # 保留所有token
                all_features.append(features)
            all_features = [all_features]  # 保持一致的格式
        
        return all_features
    
    def select_best_layer_features(self, all_features, text_features=None):
        """选择最佳层的特征"""
        if not self.use_adaptive_layer_selection:
            # 使用固定的层选择
            selected_features = []
            for features in all_features:
                selected_features.append(features[self.select_layer])
            return selected_features, None, None
        
        # 使用分类器选择最佳层
        selected_features = []
        layer_probs_list = []
        selected_layer_indices = []
        
        for features in all_features:
            # 使用第一层的特征作为分类器的输入
            classifier_input = features[0]  # [1, seq_len, hidden_size]
            
            # 获取层概率分布
            layer_probs, layer_logits = self.layer_classifier(classifier_input, text_features)
            layer_probs_list.append(layer_probs)
            
            if self.training_mode:
                # 训练模式：选择top-k层
                top_k_probs, top_k_indices = torch.topk(layer_probs, self.top_k_layers, dim=-1)
                # 随机选择其中一个作为当前的选择
                selected_idx = top_k_indices[0, torch.randint(0, self.top_k_layers, (1,)).item()]
            else:
                # 推理模式：选择概率最高的层
                selected_idx = torch.argmax(layer_probs, dim=-1).item()
            
            selected_layer_indices.append(selected_idx)
            selected_features.append(features[selected_idx])
        
        return selected_features, layer_probs_list, selected_layer_indices
    
    def forward(self, images, text_features=None):
        """前向传播"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # 获取所有层的特征
        all_features = self.get_all_layer_features(images)
        
        # 选择最佳层的特征
        selected_features, layer_probs, selected_indices = self.select_best_layer_features(
            all_features, text_features
        )
        
        # 转换为正确的数据类型
        selected_features = [feat.to(images.dtype) if hasattr(images, 'dtype') else feat for feat in selected_features]
        
        if len(selected_features) == 1:
            return selected_features[0]
        else:
            return selected_features
    
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
