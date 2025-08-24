# LLaVA V1.5 自适应层选择与MOE投影器

## 概述

本项目实现了一个创新的自适应视觉特征提取机制，通过轻量级分类器自动选择最合适的视觉编码器层，并使用MOE（Mixture of Experts）架构的投影器来处理不同层的特征。

## 核心特性

### 1. 自适应层选择
- **轻量级分类器**: 根据输入的图像和文本特征，自动选择最合适的视觉编码器层
- **动态层选择**: 训练时探索多个候选层，选择损失最小的层作为监督信号
- **层统计监控**: 实时监控不同层的选择频率和使用情况

### 2. MOE投影器架构
- **专家网络**: 多个专家网络专门处理不同类型的视觉特征
- **自适应门控**: 根据输入特征自动选择最合适的专家组合
- **负载均衡**: 确保专家网络的均衡使用，避免某些专家过载

### 3. 联合训练策略
- **多目标损失**: 结合语言建模损失、层分类器损失和MOE负载均衡损失
- **端到端训练**: 整个系统可以端到端训练，无需分阶段训练
- **预训练兼容**: 直接使用已有的预训练权重，只训练微调阶段

## 架构设计

### 自适应CLIP编码器 (`AdaptiveCLIPVisionTower`)

```python
class AdaptiveCLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        # 初始化层分类器
        self.layer_classifier = LayerClassifier(
            vision_hidden_size=self.vision_tower.config.hidden_size,
            num_layers=num_layers
        )
    
    def forward(self, images, text_features=None):
        # 获取所有层的特征
        all_features = self.get_all_layer_features(images)
        # 选择最佳层的特征
        selected_features, layer_probs, selected_indices = self.select_best_layer_features(
            all_features, text_features
        )
        return selected_features
```

### MOE投影器 (`MoEProjector`)

```python
class MoEProjector(nn.Module):
    def __init__(self, config, num_experts=8, top_k=2):
        # 创建专家网络
        self.experts = nn.ModuleList([
            MoEExpert(input_size, output_size) for _ in range(num_experts)
        ])
        # 门控网络
        self.gate = MoEGate(input_size, num_experts, top_k)
    
    def forward(self, x, training=True):
        # 计算门控权重
        gate_weights, expert_indices, top_k_weights = self.gate(x, training)
        # 组合专家输出
        weighted_output = (expert_outputs * gate_weights_expanded).sum(dim=1)
        return weighted_output
```

## 使用方法

### 1. 训练脚本

使用自适应训练脚本：

```bash
bash scripts/v1_5/finetune_adaptive.sh
```

### 2. 主要参数

#### 自适应层选择参数
- `--use_adaptive_layer_selection True`: 启用自适应层选择
- `--top_k_layers 3`: 选择top-3层进行探索
- `--layer_classifier_loss_weight 0.1`: 层分类器损失权重

#### MOE投影器参数
- `--mm_projector_type adaptive_moe`: 使用自适应MOE投影器
- `--mm_moe_num_experts 8`: 专家网络数量
- `--mm_moe_top_k 2`: 每次激活的专家数量
- `--moe_load_balancing_weight 0.01`: 负载均衡损失权重

### 3. 训练过程

训练过程包括以下步骤：

1. **特征提取**: 从视觉编码器的所有层提取特征
2. **层选择**: 轻量级分类器预测最合适的层
3. **多层探索**: 计算top-k层的损失，选择最佳层
4. **特征投影**: 使用MOE投影器处理选定层的特征
5. **联合训练**: 优化总损失函数

### 4. 损失函数

总损失函数包括三个部分：

```python
total_loss = language_modeling_loss + \
             layer_classifier_loss_weight * layer_classifier_loss + \
             moe_load_balancing_weight * moe_load_balancing_loss
```

## 性能优化

### 1. 内存优化
- 使用梯度检查点减少内存占用
- 支持DeepSpeed ZeRO-3优化
- 批量处理多个候选层

### 2. 计算优化
- 并行计算多个层的特征
- 缓存中间结果避免重复计算
- 支持混合精度训练

### 3. 训练稳定性
- 渐进式层选择策略
- 专家网络负载均衡
- 动态学习率调整

## 实验结果

### 1. 层选择统计
训练过程中会记录每个层的选择频率：

```
=== 层选择统计 ===
层 0: 15.23% (1523/10000)
层 1: 12.45% (1245/10000)
层 2: 18.67% (1867/10000)
...
==================
```

### 2. 专家使用统计
监控每个专家网络的使用情况，确保负载均衡。

### 3. 性能提升
相比固定层选择，自适应方法通常能带来：
- 更好的视觉理解能力
- 更稳定的训练过程
- 更高的最终性能

## 扩展功能

### 1. 自定义层选择策略
可以实现不同的层选择策略：
- 基于任务类型的层选择
- 基于图像复杂度的层选择
- 基于文本长度的层选择

### 2. 动态专家数量
根据输入复杂度动态调整专家数量：
- 简单输入使用较少专家
- 复杂输入使用更多专家

### 3. 多模态融合
支持更复杂的多模态融合策略：
- 图像-文本联合编码
- 跨模态注意力机制
- 多尺度特征融合

## 故障排除

### 1. 常见问题

**Q: 训练时内存不足**
A: 减少batch size或启用梯度检查点

**Q: 层分类器损失不收敛**
A: 调整`layer_classifier_loss_weight`参数

**Q: MOE专家使用不均衡**
A: 增加`moe_load_balancing_weight`参数

### 2. 调试技巧

- 启用详细日志记录
- 监控层选择统计
- 检查专家网络使用情况
- 验证损失函数权重

## 未来工作

1. **更智能的层选择**: 基于任务类型和输入特性的自适应选择
2. **动态专家网络**: 根据输入复杂度动态调整专家数量
3. **跨模态预训练**: 支持图像-文本联合预训练
4. **模型压缩**: 针对移动设备的模型优化

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@misc{llava_adaptive_2024,
  title={LLaVA V1.5 with Adaptive Layer Selection and MOE Projector},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/llava-adaptive}
}
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目遵循Apache 2.0许可证。
