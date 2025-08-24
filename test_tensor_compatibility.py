#!/usr/bin/env python3
"""
测试张量形状兼容性修复
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'llava'))

def test_tensor_reshape():
    """测试张量reshape的兼容性"""
    print("=== 测试张量reshape兼容性 ===")
    
    try:
        from llava.model.multimodal_projector.moe_projector import MoEProjector, CompatibleMoEProjector
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.mm_hidden_size = 1024
                self.hidden_size = 4096
        
        config = MockConfig()
        
        # 测试MoEProjector
        print("测试MoEProjector...")
        projector = MoEProjector(config, num_experts=4, top_k=2)
        
        # 创建不同形状的输入进行测试
        test_cases = [
            torch.randn(2, 196, 1024),  # 标准3D输入
            torch.randn(4, 256, 1024),  # 不同的batch_size和seq_len
            torch.randn(1, 100, 1024),  # 小的batch_size
        ]
        
        for i, x in enumerate(test_cases):
            print(f"测试用例 {i+1}: 输入形状 {x.shape}")
            try:
                # 测试前向传播
                output = projector(x, training=True)
                print(f"  输出形状: {output.shape}")
                print(f"  ✓ 测试用例 {i+1} 通过")
            except Exception as e:
                print(f"  ❌ 测试用例 {i+1} 失败: {e}")
                return False
        
        # 测试CompatibleMoEProjector
        print("\n测试CompatibleMoEProjector...")
        compatible_projector = CompatibleMoEProjector(config, num_experts=4, top_k=2)
        
        for i, x in enumerate(test_cases):
            print(f"测试用例 {i+1}: 输入形状 {x.shape}")
            try:
                # 测试前向传播
                output = compatible_projector(x, training=True)
                print(f"  输出形状: {output.shape}")
                print(f"  ✓ 测试用例 {i+1} 通过")
            except Exception as e:
                print(f"  ❌ 测试用例 {i+1} 失败: {e}")
                return False
        
        print("✓ 所有张量reshape测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_non_contiguous_tensor():
    """测试非连续张量的处理"""
    print("\n=== 测试非连续张量处理 ===")
    
    try:
        from llava.model.multimodal_projector.moe_projector import CompatibleMoEProjector
        
        # 创建模拟配置
        class MockConfig:
            def __init__(self):
                self.mm_hidden_size = 1024
                self.hidden_size = 4096
        
        config = MockConfig()
        projector = CompatibleMoEProjector(config, num_experts=4, top_k=2)
        
        # 创建非连续张量
        x = torch.randn(2, 196, 1024)
        x_non_contiguous = x.transpose(1, 2).transpose(1, 2)  # 创建非连续张量
        
        print(f"原始张量连续: {x.is_contiguous()}")
        print(f"非连续张量连续: {x_non_contiguous.is_contiguous()}")
        
        # 测试非连续张量
        try:
            output = projector(x_non_contiguous, training=True)
            print(f"非连续张量输出形状: {output.shape}")
            print("✓ 非连续张量处理成功")
            return True
        except Exception as e:
            print(f"❌ 非连续张量处理失败: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试张量形状兼容性修复...\n")
    
    success = True
    
    if not test_tensor_reshape():
        success = False
    
    if not test_non_contiguous_tensor():
        success = False
    
    if success:
        print("\n🎉 所有测试通过！张量形状兼容性修复成功。")
        print("\n现在可以重新运行训练脚本:")
        print("bash scripts/v1_5/finetune_adaptive.sh")
    else:
        print("\n❌ 部分测试失败，需要进一步修复。")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
