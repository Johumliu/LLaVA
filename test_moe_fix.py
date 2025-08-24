#!/usr/bin/env python3
"""
测试MOE投影器修复
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'llava'))

def test_moe_projector():
    """测试MOE投影器"""
    print("=== 测试MOE投影器修复 ===")
    
    # 创建模拟配置
    class MockConfig:
        def __init__(self):
            self.mm_hidden_size = 1024
            self.hidden_size = 4096
    
    config = MockConfig()
    
    try:
        from llava.model.multimodal_projector.moe_projector import MoEProjector, AdaptiveMoEProjector
        
        # 测试MoEProjector
        print("测试MoEProjector...")
        projector = MoEProjector(config, num_experts=4, top_k=2)
        print("✓ MoEProjector创建成功")
        
        # 测试AdaptiveMoEProjector
        print("测试AdaptiveMoEProjector...")
        adaptive_projector = AdaptiveMoEProjector(config, num_experts=4, top_k=2)
        print("✓ AdaptiveMoEProjector创建成功")
        
        # 测试前向传播
        print("测试前向传播...")
        x = torch.randn(2, 196, 1024)  # [batch_size, seq_len, hidden_size]
        
        output1 = projector(x, training=True)
        print(f"MoEProjector输出形状: {output1.shape}")
        
        output2 = adaptive_projector(x, training=True)
        print(f"AdaptiveMoEProjector输出形状: {output2.shape}")
        
        print("✓ 前向传播测试成功")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

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
                self.mm_projector_type = 'adaptive_moe'
                self.mm_moe_num_experts = 4
                self.mm_moe_top_k = 2
        
        config = MockConfig()
        
        # 测试构建
        projector = build_vision_projector(config)
        print("✓ 构建器测试成功")
        
    except Exception as e:
        print(f"❌ 构建器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主测试函数"""
    print("开始测试MOE投影器修复...\n")
    
    success = True
    
    if not test_moe_projector():
        success = False
    
    if not test_builder():
        success = False
    
    if success:
        print("\n🎉 所有测试通过！MOE投影器修复成功。")
    else:
        print("\n❌ 部分测试失败，需要进一步修复。")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
