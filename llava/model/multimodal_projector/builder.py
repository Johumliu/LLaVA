import torch
import torch.nn as nn
import re
from .moe_projector import MoEProjector, AdaptiveMoEProjector, CompatibleMoEProjector
import os


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    # 环境变量强制切换为 adaptive_moe（不修改checkpoint）
    force_adaptive_moe = os.environ.get('LLAVA_FORCE_ADAPTIVE_MOE', '0') in ('1', 'true', 'True')
    if force_adaptive_moe and projector_type != 'adaptive_moe':
        try:
            setattr(config, 'mm_projector_type', 'adaptive_moe')
            projector_type = 'adaptive_moe'
            print('[LLaVA] Force using adaptive_moe projector via LLAVA_FORCE_ADAPTIVE_MOE=1')
        except Exception:
            pass

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()
    
    # 新增的MOE投影器类型
    if projector_type == 'moe':
        num_experts = getattr(config, 'mm_moe_num_experts', 8)
        top_k = getattr(config, 'mm_moe_top_k', 2)
        return MoEProjector(config, num_experts=num_experts, top_k=top_k)
    
    if projector_type == 'adaptive_moe':
        num_experts = getattr(config, 'mm_moe_num_experts', 8)
        top_k = getattr(config, 'mm_moe_top_k', 2)
        return AdaptiveMoEProjector(config, num_experts=num_experts, top_k=top_k)
    
    if projector_type == 'compatible_moe':
        num_experts = getattr(config, 'mm_moe_num_experts', 8)
        top_k = getattr(config, 'mm_moe_top_k', 2)
        return CompatibleMoEProjector(config, num_experts=num_experts, top_k=top_k)

    raise ValueError(f'Unknown projector type: {projector_type}')
