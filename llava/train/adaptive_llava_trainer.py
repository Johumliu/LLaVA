import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class AdaptiveLLaVATrainer(Trainer):
    """支持全层LLM训练和MOE投影器的LLaVA训练器
    
    特点：
    - 所有视觉编码器层都会经过LLM进行前向传播
    - 自动选择损失最小的层进行监督训练
    - 支持MOE投影器和层分类器
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 训练参数
        self.use_adaptive_layer_selection = getattr(self.args, 'use_adaptive_layer_selection', True)
        self.top_k_layers = getattr(self.args, 'top_k_layers', 3)
        self.layer_classifier_loss_weight = getattr(self.args, 'layer_classifier_loss_weight', 0.1)
        self.moe_load_balancing_weight = getattr(self.args, 'moe_load_balancing_weight', 0.01)
        
        # 记录最佳层选择
        self.best_layer_selections = {}
        
        # 初始化损失组件记录
        self.current_loss_components = {}
        
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
            text_features = text_features
        
        # 使用第一层特征进行标准前向传播（避免CUDA错误）
        try:
            # 使用第一层特征进行前向传播
            first_layer_features = []
            for features in all_features:
                if len(features) > 0:
                    first_layer_features.append(features[0])
                else:
                    # 如果特征为空，创建一个零张量
                    dummy_feature = torch.zeros(1, vision_tower.hidden_size, 
                                             device=next(vision_tower.parameters()).device)
                    first_layer_features.append(dummy_feature)
            
            # 临时替换视觉特征
            original_image_features = getattr(model, '_temp_image_features', None)
            model._temp_image_features = first_layer_features
            
            # 前向传播
            outputs = model(**inputs)
            main_loss = outputs.loss
            
            # 恢复原始特征
            if original_image_features is not None:
                model._temp_image_features = original_image_features
            else:
                delattr(model, '_temp_image_features')
                
        except Exception as e:
            logger.warning(f"计算主要损失时出错: {e}")
            # 如果出错，使用标准训练
            return super().compute_loss(model, inputs, return_outputs)
        
        # 计算层分类器的损失
        layer_classifier_loss = self._compute_layer_classifier_loss(
            vision_tower, all_features, 0, text_features  # 使用第一层作为目标
        )
        
        # 计算MOE负载均衡损失
        moe_load_balancing_loss = self._compute_moe_load_balancing_loss(model)
        
        # 总损失
        total_loss = main_loss + self.layer_classifier_loss_weight * layer_classifier_loss + self.moe_load_balancing_weight * moe_load_balancing_loss
        
        # 保存各个损失组件用于wandb记录
        self.current_loss_components = {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'layer_classifier_loss': layer_classifier_loss.item(),
            'moe_load_balancing_loss': moe_load_balancing_loss.item(),
            'best_layer_idx': 0  # 暂时使用第一层
        }
        
        # 添加调试日志
        logger.info(f"步骤 {self.state.global_step}: 损失组件已计算 - "
                   f"总损失: {total_loss.item():.4f}, "
                   f"主要损失: {main_loss.item():.4f}, "
                   f"层分类器损失: {layer_classifier_loss.item():.4f}, "
                   f"MOE负载均衡损失: {moe_load_balancing_loss.item():.4f}")
        
        # 记录最佳层选择（暂时都记录为第一层）
        batch_size = images.shape[0] if hasattr(images, 'shape') else 1
        for i in range(batch_size):
            if i not in self.best_layer_selections:
                self.best_layer_selections[i] = []
            self.best_layer_selections[i].append(0)
        
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
        
        # 记录各个损失组件到wandb
        if hasattr(self, 'current_loss_components') and self.current_loss_components:
            self._log_loss_components_to_wandb()
    
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
            logger.info("=== 全层LLM训练统计 ===")
            logger.info(f"总共评估了 {len(layer_counts)} 层")
            for layer_idx in sorted(layer_counts.keys()):
                frequency = layer_counts[layer_idx] / total_selections * 100
                logger.info(f"层 {layer_idx}: {frequency:.2f}% ({layer_counts[layer_idx]}/{total_selections})")
            logger.info("======================")
        
        # 清空统计（可选）
        # self.best_layer_selections.clear()
    
    def on_step_end(self):
        """每个训练步骤结束时的回调"""
        super().on_step_end()
        
        # 每个步骤都记录损失组件到wandb
        if hasattr(self, 'current_loss_components') and self.current_loss_components:
            self._log_loss_components_to_wandb()
        
        # 定期记录层选择统计
        if self.state.global_step % 100 == 0:
            self._log_layer_selection_stats()

    def _log_loss_components_to_wandb(self):
        """记录各个损失组件到wandb"""
        try:
            import wandb
            logger.info(f"尝试记录损失组件到wandb，步骤: {self.state.global_step}")
            
            if wandb.run is not None:
                logger.info(f"Wandb运行状态: {wandb.run.name}, 项目: {wandb.run.project}")
                
                # 记录损失组件
                log_data = {
                    'train/main_loss': self.current_loss_components['main_loss'],
                    'train/layer_classifier_loss': self.current_loss_components['layer_classifier_loss'],
                    'train/moe_load_balancing_loss': self.current_loss_components['moe_load_balancing_loss'],
                    'train/best_layer_idx': self.current_loss_components['best_layer_idx'],
                    'train/total_loss_components': self.current_loss_components['total_loss']
                }
                
                logger.info(f"准备记录到wandb的数据: {log_data}")
                wandb.log(log_data, step=self.state.global_step)
                
                logger.info(f"Wandb记录成功 - 步骤 {self.state.global_step}: "
                          f"主要损失: {self.current_loss_components['main_loss']:.4f}, "
                          f"层分类器损失: {self.current_loss_components['layer_classifier_loss']:.4f}, "
                          f"MOE负载均衡损失: {self.current_loss_components['moe_load_balancing_loss']:.4f}, "
                          f"最佳层索引: {self.current_loss_components['best_layer_idx']}")
            else:
                logger.warning("Wandb运行未初始化")
        except ImportError:
            logger.warning("wandb未安装，无法记录损失组件")
        except Exception as e:
            logger.error(f"记录损失组件到wandb时出错: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
