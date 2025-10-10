import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELayer(nn.Module):
    """
    一个通用的混合专家（MoE）层，用于特征融合。
    它接收多个专家的输入，并通过一个门控网络计算权重，
    然后对专家们的输出进行加权求和。
    """
    def __init__(self, hidden_size, num_experts, top_k=1):
        super(MoELayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(hidden_size, num_experts)

        # 在这个融合场景中，专家网络可以是简单的恒等映射，
        # 因为我们只是想对不同层的特征进行加权求和。
        # 权重由门控网络根据输入特征动态决定。
        # 因此，我们不需要显式的专家网络参数。

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, num_tokens, num_experts, hidden_size]
                              或者 [batch_size * num_tokens, num_experts, hidden_size]
        
        Returns:
            torch.Tensor: 融合后的特征张量，形状为 [batch_size, num_tokens, hidden_size]
        """
        
        # 使用第一个专家的特征作为门控网络的输入
        # -> [batch_size * num_tokens, hidden_size]
        gate_input = x.reshape(-1, self.num_experts, self.hidden_size)[:, 0, :]

        # 计算门控权重
        # -> [batch_size * num_tokens, num_experts]
        gate_logits = self.gate(gate_input)
        
        # 选择 top_k 个专家
        # weights -> [batch_size * num_tokens, top_k]
        # indices -> [batch_size * num_tokens, top_k]
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        # 创建一个稀疏的权重掩码
        mask = torch.zeros_like(gate_logits).scatter_(1, indices, weights)
        
        # 将输入 reshape 以便进行矩阵乘法
        # x -> [batch_size * num_tokens, hidden_size, num_experts]
        x_reshaped = x.reshape(-1, self.num_experts, self.hidden_size).permute(0, 2, 1)

        # 将权重掩码扩展维度以匹配输入
        # mask -> [batch_size * num_tokens, 1, num_experts]
        mask_reshaped = mask.unsqueeze(1)
        
        # 进行加权求和
        # fused_output -> [batch_size * num_tokens, hidden_size, 1]
        fused_output = torch.bmm(x_reshaped, mask_reshaped.transpose(1, 2))
        
        # 移除多余的维度
        # -> [batch_size * num_tokens, hidden_size]
        fused_output = fused_output.squeeze(-1)
        
        return fused_output
