#!/usr/bin/env python3
"""
测试兼容的MOE投影器
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'llava'))

def test_compatible_moe():
    """测试兼容的MOE投影器"""
    print("=== 测试兼容的MOE投影器 ===")
    
    try:
        from llava.model.multimodal_projector.moe_projector import CompatibleMoEProjector
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.mm_hidden_size = 1024
                self.hidden_size = 4096
        
        config = MockConfig()
        
        # 创建兼容的MOE投影器
        print("创建兼容的MOE投影器...")
        projector = CompatibleMoEProjector(config, num_experts=4, top_k=2)
        print("✓ 投影器创建成功")
        
        # 创建模拟的预训练权重
        print("创建模拟预训练权重...")
        pretrained_weights = {
            '0.weight': torch.randn(4096, 1024),
            '0.bias': torch.randn(4096)
        }
        
        # 加载预训练权重
        print("加载预训练权重...")
        projector.load_pretrained_weights(pretrained_weights)
        print("✓ 预训练权重加载成功")
        
        # 测试前向传播
        print("测试前向传播...")
        x = torch.randn(2, 196, 1024)  # [batch_size, seq_len, hidden_size]
        output = projector(x, training=True)
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print("✓ 前向传播测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_builder():
    """测试构建器"""
    print("\n=== 测试构建器 ===")
    
    try:
        from llava.model.multimodal_projector.builder import build_vision_projector
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.mm_hidden_size = 1024
                self.hidden_size = 4096
                self.mm_projector_type = 'compatible_moe'
                self.mm_moe_num_experts = 4
                self.mm_moe_top_k = 2
        
        config = MockConfig()
        
        # 测试构建
        projector = build_vision_projector(config)
        print("✓ 构建器测试成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试兼容的MOE投影器...\n")
    
    success = True
    
    if not test_compatible_moe():
        success = False
    
    if not test_builder():
        success = False
    
    if success:
        print("\n🎉 所有测试通过！兼容的MOE投影器工作正常。")
        print("\n现在可以使用以下命令开始训练:")
        print("bash scripts/v1_5/finetune_adaptive.sh")
    else:
        print("\n❌ 部分测试失败，需要进一步修复。")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
