import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class AdaptiveLLaVATrainer(Trainer):
    """支持自适应层选择和MOE投影器的LLaVA训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 训练参数
        self.use_adaptive_layer_selection = getattr(self.args, 'use_adaptive_layer_selection', True)
        self.top_k_layers = getattr(self.args, 'top_k_layers', 3)
        self.layer_classifier_loss_weight = getattr(self.args, 'layer_classifier_loss_weight', 0.1)
        self.moe_load_balancing_weight = getattr(self.args, 'moe_load_balancing_weight', 0.01)
        
        # 记录最佳层选择
        self.best_layer_selections = {}
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算损失，包括主要的语言建模损失和辅助损失
        """
        if not self.use_adaptive_layer_selection:
            # 使用标准训练
            return super().compute_loss(model, inputs, return_outputs)
        
        # 获取输入
        images = inputs.get('images', None)
        input_ids = inputs.get('input_ids', None)
        attention_mask = inputs.get('attention_mask', None)
        labels = inputs.get('labels', None)
        
        if images is None:
            return super().compute_loss(model, inputs, return_outputs)
        
        # 获取视觉编码器
        vision_tower = model.get_vision_tower()
        if not hasattr(vision_tower, 'get_all_layer_features'):
            # 如果不是自适应编码器，使用标准训练
            return super().compute_loss(model, inputs, return_outputs)
        
        # 获取所有层的特征
        all_features = vision_tower.get_all_layer_features(images)
        
        # 获取文本特征（如果有的话）
        text_features = None
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'embed_tokens'):
            text_embeddings = model.get_model().embed_tokens(input_ids)
            text_features = text_embeddings
        
        # 计算top-k层的损失
        top_k_losses = []
        top_k_layer_indices = []
        
        for k in range(self.top_k_layers):
            # 选择第k个最佳层
            selected_features = []
            for features in all_features:
                # 这里简化处理，实际应该根据分类器结果选择
                layer_idx = min(k, len(features) - 1)
                selected_features.append(features[layer_idx])
            
            # 使用选定的特征进行前向传播
            try:
                # 临时替换视觉特征
                original_image_features = getattr(model, '_temp_image_features', None)
                model._temp_image_features = selected_features
                
                # 前向传播
                outputs = model(**inputs)
                loss = outputs.loss
                
                top_k_losses.append(loss)
                top_k_layer_indices.append(k)
                
                # 恢复原始特征
                if original_image_features is not None:
                    model._temp_image_features = original_image_features
                else:
                    delattr(model, '_temp_image_features')
                    
            except Exception as e:
                logger.warning(f"计算第{k}层损失时出错: {e}")
                top_k_losses.append(torch.tensor(float('inf')))
                top_k_layer_indices.append(k)
        
        # 找到损失最小的层
        if top_k_losses:
            min_loss_idx = torch.argmin(torch.stack([loss for loss in top_k_losses if loss != float('inf')]))
            best_loss = top_k_losses[min_loss_idx]
            best_layer_idx = top_k_layer_indices[min_loss_idx]
        else:
            # 如果没有有效的损失，使用标准训练
            return super().compute_loss(model, inputs, return_outputs)
        
        # 计算层分类器的损失
        layer_classifier_loss = self._compute_layer_classifier_loss(
            vision_tower, all_features, best_layer_idx, text_features
        )
        
        # 计算MOE负载均衡损失
        moe_load_balancing_loss = self._compute_moe_load_balancing_loss(model)
        
        # 总损失
        total_loss = best_loss + self.layer_classifier_loss_weight * layer_classifier_loss + self.moe_load_balancing_weight * moe_load_balancing_loss
        
        # 记录最佳层选择
        batch_size = images.shape[0] if hasattr(images, 'shape') else 1
        for i in range(batch_size):
            if i not in self.best_layer_selections:
                self.best_layer_selections[i] = []
            self.best_layer_selections[i].append(best_layer_idx)
        
        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss
    
    def _compute_layer_classifier_loss(self, vision_tower, all_features, target_layer_idx, text_features):
        """计算层分类器的损失"""
        if not hasattr(vision_tower, 'layer_classifier'):
            return torch.tensor(0.0, device=next(vision_tower.parameters()).device)
        
        # 创建目标标签
        batch_size = len(all_features)
        target_labels = torch.full((batch_size,), target_layer_idx, 
                                 dtype=torch.long, device=next(vision_tower.parameters()).device)
        
        # 使用第一层特征作为分类器输入
        classifier_inputs = []
        for features in all_features:
            if len(features) > 0:
                classifier_inputs.append(features[0])
            else:
                # 如果特征为空，创建一个零张量
                dummy_feature = torch.zeros(1, vision_tower.hidden_size, 
                                         device=next(vision_tower.parameters()).device)
                classifier_inputs.append(dummy_feature)
        
        # 计算分类器损失
        total_loss = 0.0
        for i, features in enumerate(classifier_inputs):
            if features is not None and features.numel() > 0:
                # 计算分类器输出
                layer_probs, layer_logits = vision_tower.layer_classifier(features, text_features)
                
                # 计算交叉熵损失
                if i < len(target_labels):
                    loss = F.cross_entropy(layer_logits, target_labels[i:i+1])
                    total_loss += loss
        
        return total_loss / max(len(classifier_inputs), 1)
    
    def _compute_moe_load_balancing_loss(self, model):
        """计算MOE负载均衡损失"""
        total_loss = 0.0
        count = 0
        
        # 遍历模型中的所有MOE投影器
        for name, module in model.named_modules():
            if hasattr(module, 'get_load_balancing_loss'):
                loss = module.get_load_balancing_loss()
                if loss is not None and loss.numel() > 0:
                    total_loss += loss
                    count += 1
        
        return total_loss / max(count, 1)
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """重写日志记录，包含自适应层选择的信息"""
        # 调用父类方法
        super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        # 记录层选择统计
        if self.use_adaptive_layer_selection and self.best_layer_selections:
            self._log_layer_selection_stats()
    
    def _log_layer_selection_stats(self):
        """记录层选择统计信息"""
        if not self.best_layer_selections:
            return
        
        # 计算每个层的选择频率
        layer_counts = {}
        total_selections = 0
        
        for sample_selections in self.best_layer_selections.values():
            for layer_idx in sample_selections:
                layer_counts[layer_idx] = layer_counts.get(layer_idx, 0) + 1
                total_selections += 1
        
        # 记录到日志
        if total_selections > 0:
            logger.info("=== 层选择统计 ===")
            for layer_idx in sorted(layer_counts.keys()):
                frequency = layer_counts[layer_idx] / total_selections * 100
                logger.info(f"层 {layer_idx}: {frequency:.2f}% ({layer_counts[layer_idx]}/{total_selections})")
            logger.info("==================")
        
        # 清空统计（可选）
        # self.best_layer_selections.clear()
    
    def on_step_end(self):
        """每个训练步骤结束时的回调"""
        super().on_step_end()
        
        # 定期记录层选择统计
        if self.state.global_step % 100 == 0:
            self._log_layer_selection_stats()
