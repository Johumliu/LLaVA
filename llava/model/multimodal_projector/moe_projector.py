import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoEGate(nn.Module):
    """MOE门控网络，用于选择最合适的专家"""
    
    def __init__(self, input_size, num_experts, top_k=2, use_noise=True, noise_epsilon=1e-2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noise = use_noise
        self.noise_epsilon = noise_epsilon
        
        # 门控网络
        self.gate = nn.Linear(input_size, num_experts)
        
    def forward(self, x, training=True):
        """
        Args:
            x: [batch_size, input_size]
            training: 是否处于训练模式
        """
        # 计算门控权重
        gate_logits = self.gate(x)  # [batch_size, num_experts]
        
        if training and self.use_noise:
            # 训练时添加噪声
            noise = torch.randn_like(gate_logits) * self.noise_epsilon
            gate_logits = gate_logits + noise
        
        # 选择top-k专家
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # 计算softmax权重
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # 创建稀疏权重矩阵
        batch_size = x.shape[0]
        sparse_weights = torch.zeros(batch_size, self.num_experts, device=x.device, dtype=x.dtype)
        sparse_weights.scatter_(1, top_k_indices, top_k_weights)
        
        return sparse_weights, top_k_indices, top_k_weights


class MoEExpert(nn.Module):
    """单个专家网络"""
    
    def __init__(self, input_size, output_size, hidden_size=None, dropout=0.1):
        super().__init__()
        if hidden_size is None:
            hidden_size = max(input_size, output_size)
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)


class MoEProjector(nn.Module):
    """MOE架构的视觉特征投影器"""
    
    def __init__(self, config, num_experts=8, top_k=2, use_noise=True, 
                 expert_hidden_size=None, dropout=0.1):
        super().__init__()
        
        self.input_size = config.mm_hidden_size
        self.output_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noise = use_noise
        self.dropout = dropout
        
        if expert_hidden_size is None:
            expert_hidden_size = max(self.input_size, self.output_size)
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            MoEExpert(self.input_size, self.output_size, expert_hidden_size, dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(self.input_size, num_experts, top_k, use_noise)
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(self.input_size)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(self.output_size)
        
        # 残差连接
        self.use_residual = True
        
    def forward(self, x, training=True):
        """
        Args:
            x: [batch_size, seq_len, input_size] 或 [batch_size, input_size]
            training: 是否处于训练模式
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, input_size = original_shape
            x = x.view(-1, input_size)  # [batch_size * seq_len, input_size]
        else:
            batch_size, input_size = original_shape
            seq_len = 1
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 计算门控权重
        gate_weights, expert_indices, top_k_weights = self.gate(x_norm, training)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_output = self.experts[expert_idx](x_norm)  # [batch_size * seq_len, output_size]
            expert_outputs.append(expert_output)
        
        # 组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size * seq_len, num_experts, output_size]
        
        # 应用门控权重
        gate_weights_expanded = gate_weights.unsqueeze(-1).expand(-1, -1, self.output_size)
        weighted_output = (expert_outputs * gate_weights_expanded).sum(dim=1)  # [batch_size * seq_len, output_size]
        
        # 输出归一化
        output = self.output_norm(weighted_output)
        
        # 残差连接（如果输入和输出维度相同）
        if self.use_residual and self.input_size == self.output_size:
            # 需要将输入投影到输出维度
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(x)
            else:
                # 创建残差投影层
                self.residual_proj = nn.Linear(self.input_size, self.output_size).to(x.device)
                residual = self.residual_proj(x)
            output = output + residual
        
        # 恢复原始形状
        if len(original_shape) == 3:
            output = output.view(batch_size, seq_len, self.output_size)
        
        return output
    
    def get_expert_usage(self):
        """获取每个专家的使用统计"""
        usage = torch.zeros(self.num_experts)
        total_samples = 0
        
        # 这里需要在实际使用中统计，暂时返回随机值
        return usage
    
    def get_load_balancing_loss(self):
        """计算负载均衡损失"""
        # 这里需要在实际使用中计算，暂时返回0
        return torch.tensor(0.0)


class AdaptiveMoEProjector(nn.Module):
    """自适应MOE投影器，能够根据输入特征选择最合适的专家组合"""
    
    def __init__(self, config, num_experts=8, top_k=2, use_noise=True, 
                 expert_hidden_size=None, dropout=0.1):
        super().__init__()
        
        self.input_size = config.mm_hidden_size
        self.output_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 确保expert_hidden_size有合理的默认值
        if expert_hidden_size is None:
            expert_hidden_size = max(self.input_size, self.output_size)
        
        # 主要的MOE投影器
        self.main_projector = MoEProjector(
            config, num_experts, top_k, use_noise, expert_hidden_size, dropout
        )
        
        # 辅助投影器（用于不同层的特征）
        aux_expert_hidden_size = max(expert_hidden_size // 2, 512)  # 确保最小值
        self.aux_projectors = nn.ModuleList([
            MoEProjector(config, max(num_experts // 2, 2), top_k, use_noise, 
                        aux_expert_hidden_size, dropout)
            for _ in range(3)  # 为不同层提供不同的投影器
        ])
        
        # 层选择器
        self.layer_selector = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, len(self.aux_projectors))
        )
        
    def forward(self, x, layer_info=None, training=True):
        """
        Args:
            x: 输入特征
            layer_info: 层信息，用于选择辅助投影器
            training: 是否处于训练模式
        """
        if layer_info is not None:
            # 根据层信息选择投影器
            layer_logits = self.layer_selector(x.mean(dim=1) if len(x.shape) == 3 else x)
            layer_probs = F.softmax(layer_logits, dim=-1)
            
            # 选择最合适的辅助投影器
            selected_layer = torch.argmax(layer_probs, dim=-1)
            
            # 使用选定的辅助投影器
            outputs = []
            for i, layer_idx in enumerate(selected_layer):
                if layer_idx < len(self.aux_projectors):
                    output = self.aux_projectors[layer_idx](x[i:i+1], training)
                    outputs.append(output)
                else:
                    # 使用主投影器
                    output = self.main_projector(x[i:i+1], training)
                    outputs.append(output)
            
            if len(outputs) == 1:
                return outputs[0]
            else:
                return torch.cat(outputs, dim=0)
        else:
            # 使用主投影器
            return self.main_projector(x, training)


class CompatibleMoEProjector(nn.Module):
    """兼容预训练权重的MOE投影器，能够从简单线性层初始化"""
    
    def __init__(self, config, num_experts=8, top_k=2, use_noise=True, 
                 expert_hidden_size=None, dropout=0.1):
        super().__init__()
        
        self.input_size = config.mm_hidden_size
        self.output_size = config.hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_noise = use_noise
        self.dropout = dropout
        
        if expert_hidden_size is None:
            expert_hidden_size = max(self.input_size, self.output_size)
        
        # 创建专家网络
        self.experts = nn.ModuleList([
            MoEExpert(self.input_size, self.output_size, expert_hidden_size, dropout)
            for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = MoEGate(self.input_size, num_experts, top_k, use_noise)
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(self.input_size)
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(self.output_size)
        
        # 残差连接
        self.use_residual = True
        
        # 兼容性标志
        self.compatible_mode = False
        
    def load_pretrained_weights(self, pretrained_weights):
        """从预训练权重初始化，兼容简单的线性层"""
        if len(pretrained_weights) == 2:  # 只有weight和bias
            print("Loading pretrained weights into first expert for compatibility")
            # 将预训练权重加载到第一个专家
            first_expert = self.experts[0]
            if hasattr(first_expert, 'network') and len(first_expert.network) >= 1:
                # 加载到第一个线性层
                first_expert.network[0].weight.data = pretrained_weights['0.weight'].clone()
                first_expert.network[0].bias.data = pretrained_weights['0.bias'].clone()
                
                # 初始化其他专家为第一个专家的变体
                for i in range(1, self.num_experts):
                    self.experts[i].network[0].weight.data = pretrained_weights['0.weight'].clone() + torch.randn_like(pretrained_weights['0.weight']) * 0.01
                    self.experts[i].network[0].bias.data = pretrained_weights['0.bias'].clone() + torch.randn_like(pretrained_weights['0.bias']) * 0.01
                
                self.compatible_mode = True
                print("Successfully initialized MOE projector with pretrained weights")
            else:
                print("Warning: Expert structure not compatible with pretrained weights")
        else:
            print("Warning: Pretrained weights structure not recognized")
    
    def forward(self, x, training=True):
        """
        Args:
            x: [batch_size, seq_len, input_size] 或 [batch_size, input_size]
            training: 是否处于训练模式
        """
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, input_size = original_shape
            x = x.view(-1, input_size)  # [batch_size * seq_len, input_size]
        else:
            batch_size, input_size = original_shape
            seq_len = 1
        
        # 输入归一化
        x_norm = self.input_norm(x)
        
        # 计算门控权重
        gate_weights, expert_indices, top_k_weights = self.gate(x_norm, training)
        
        # 计算每个专家的输出
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            expert_output = self.experts[expert_idx](x_norm)  # [batch_size * seq_len, output_size]
            expert_outputs.append(expert_output)
        
        # 组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size * seq_len, num_experts, output_size]
        
        # 应用门控权重
        gate_weights_expanded = gate_weights.unsqueeze(-1).expand(-1, -1, self.output_size)
        weighted_output = (expert_outputs * gate_weights_expanded).sum(dim=1)  # [batch_size * seq_len, output_size]
        
        # 输出归一化
        output = self.output_norm(weighted_output)
        
        # 残差连接（如果输入和输出维度相同）
        if self.use_residual and self.input_size == self.output_size:
            # 需要将输入投影到输出维度
            if hasattr(self, 'residual_proj'):
                residual = self.residual_proj(x)
            else:
                # 创建残差投影层
                self.residual_proj = nn.Linear(self.input_size, self.output_size).to(x.device)
                residual = self.residual_proj(x)
            output = output + residual
        
        # 恢复原始形状
        if len(original_shape) == 3:
            output = output.view(batch_size, seq_len, self.output_size)
        
        return output
    
    def get_expert_usage(self):
        """获取每个专家的使用统计"""
        usage = torch.zeros(self.num_experts)
        total_samples = 0
        
        # 这里需要在实际使用中统计，暂时返回随机值
        return usage
    
    def get_load_balancing_loss(self):
        """计算负载均衡损失"""
        # 这里需要在实际使用中计算，暂时返回0
        return torch.tensor(0.0)
