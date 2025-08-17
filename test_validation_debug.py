#!/usr/bin/env python3
"""
专门用于调试验证管线的测试脚本
检查数据加载、特征提取、模型前向传播、损失计算的每个步骤
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from torch.utils.data import DataLoader

# 添加src到路径
sys.path.append('src')

from src.unified_dataset import UnifiedSequenceDataset, create_unified_dataloaders
from src.model import EventDenoisingMamba

def load_config():
    """加载配置文件"""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def check_tensor_for_nan_inf(tensor, name):
    """检查张量中的NaN和Inf"""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        print(f"❌ {name}: 发现异常值")
        if has_nan:
            nan_count = torch.isnan(tensor).sum().item()
            print(f"   NaN数量: {nan_count}")
        if has_inf:
            inf_count = torch.isinf(tensor).sum().item()
            print(f"   Inf数量: {inf_count}")
        
        # 显示统计信息
        finite_mask = torch.isfinite(tensor)
        if finite_mask.any():
            finite_tensor = tensor[finite_mask]
            print(f"   有限值统计: min={finite_tensor.min():.6f}, max={finite_tensor.max():.6f}, mean={finite_tensor.mean():.6f}")
        
        return True
    else:
        print(f"✅ {name}: 正常 (min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f})")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    config = load_config()
    
    # 确保使用load模式
    if config['data_pipeline']['mode'] != 'load':
        print("❌ 配置文件中data_pipeline.mode不是'load'，请确保已生成验证数据")
        return False
    
    try:
        # 创建验证数据集
        val_dataset = UnifiedSequenceDataset(config, split='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        print(f"✅ 验证数据集创建成功，包含 {len(val_dataset)} 个序列")
        
        # 测试加载第一个样本
        features, labels = next(iter(val_loader))
        print(f"✅ 成功加载第一个样本")
        print(f"   Features shape: {features.shape}")
        print(f"   Labels shape: {labels.shape}")
        
        # 检查数据异常值
        features = features.squeeze(0)  # Remove batch dim
        labels = labels.squeeze(0)
        
        nan_found = False
        nan_found |= check_tensor_for_nan_inf(features, "Features")
        nan_found |= check_tensor_for_nan_inf(labels.float(), "Labels")
        
        # 检查特征的每个维度
        print("\n📊 各特征维度详细检查:")
        feature_names = ['x_norm', 'y_norm', 'dt', 'polarity']
        for i, name in enumerate(feature_names):
            if i < features.shape[-1]:
                feature_dim = features[:, i]
                nan_found |= check_tensor_for_nan_inf(feature_dim, f"Feature[{i}] {name}")
        
        return not nan_found
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_forward():
    """测试模型前向传播"""
    print("\n🔍 测试模型前向传播...")
    
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建模型
        model = EventDenoisingMamba(config).to(device)
        model.eval()
        
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建测试数据
        batch_size = 1
        seq_len = 100
        feature_dim = 4
        
        # 创建正常范围的测试特征
        test_features = torch.randn(batch_size, seq_len, feature_dim).to(device)
        test_features[:, :, 0] = torch.rand(batch_size, seq_len).to(device)  # x_norm [0,1]
        test_features[:, :, 1] = torch.rand(batch_size, seq_len).to(device)  # y_norm [0,1]  
        test_features[:, :, 2] = torch.rand(batch_size, seq_len).to(device) * 1000  # dt [0,1000]
        test_features[:, :, 3] = torch.randint(-1, 2, (batch_size, seq_len), dtype=torch.float).to(device)  # polarity {-1,1}
        
        print("✅ 测试特征创建成功")
        check_tensor_for_nan_inf(test_features, "Test Features")
        
        # 前向传播
        with torch.no_grad():
            model.reset_hidden_state()
            predictions = model(test_features)
        
        print(f"✅ 模型前向传播成功")
        print(f"   Predictions shape: {predictions.shape}")
        
        nan_found = check_tensor_for_nan_inf(predictions, "Model Predictions")
        
        return not nan_found
        
    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_calculation():
    """测试损失计算"""
    print("\n🔍 测试损失计算...")
    
    try:
        criterion = nn.BCEWithLogitsLoss()
        
        # 创建测试数据
        batch_size = 100
        predictions = torch.randn(batch_size, 1)  # logits
        labels = torch.randint(0, 2, (batch_size,), dtype=torch.float)
        
        print("✅ 测试数据创建成功")
        check_tensor_for_nan_inf(predictions, "Test Predictions")
        check_tensor_for_nan_inf(labels, "Test Labels")
        
        # 计算损失
        loss = criterion(predictions.squeeze(), labels)
        
        print(f"✅ 损失计算成功")
        print(f"   Loss value: {loss.item():.6f}")
        
        nan_found = check_tensor_for_nan_inf(loss.unsqueeze(0), "Loss")
        
        # 测试边界情况
        print("\n🔍 测试边界情况...")
        
        # 极大logits
        extreme_predictions = torch.tensor([100.0, -100.0, 0.0]).unsqueeze(1)
        extreme_labels = torch.tensor([1.0, 0.0, 0.5])
        extreme_loss = criterion(extreme_predictions.squeeze(), extreme_labels)
        print(f"   极值测试损失: {extreme_loss.item():.6f}")
        nan_found |= check_tensor_for_nan_inf(extreme_loss.unsqueeze(0), "Extreme Loss")
        
        return not nan_found
        
    except Exception as e:
        print(f"❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_pipeline():
    """测试完整验证管线"""
    print("\n🔍 测试完整验证管线...")
    
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建模型和数据
        model = EventDenoisingMamba(config).to(device)
        model.eval()
        
        val_dataset = UnifiedSequenceDataset(config, split='val')
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        criterion = nn.BCEWithLogitsLoss()
        chunk_size = config['training']['chunk_size']
        
        print(f"✅ 管线组件创建成功")
        print(f"   Chunk size: {chunk_size}")
        
        total_loss = 0
        total_chunks_processed = 0
        nan_found = False
        
        # 只处理前3个序列进行调试
        max_sequences = min(3, len(val_loader))
        
        with torch.no_grad():
            for seq_idx, (long_features, long_labels) in enumerate(val_loader):
                if seq_idx >= max_sequences:
                    break
                    
                print(f"\n--- 处理序列 {seq_idx + 1}/{max_sequences} ---")
                
                # 重置隐藏状态
                model.reset_hidden_state()
                
                # 解包批次维度
                long_features = long_features.squeeze(0).to(device)
                long_labels = long_labels.squeeze(0).to(device)
                
                print(f"   序列长度: {long_features.shape[0]}")
                
                # 检查输入数据
                seq_nan_found = False
                seq_nan_found |= check_tensor_for_nan_inf(long_features, f"Seq{seq_idx+1} Features")
                seq_nan_found |= check_tensor_for_nan_inf(long_labels.float(), f"Seq{seq_idx+1} Labels")
                
                if seq_nan_found:
                    nan_found = True
                    print(f"❌ 序列 {seq_idx + 1} 输入数据有异常值，跳过")
                    continue
                
                # 分块处理
                num_chunks = 0
                for i in range(0, long_features.shape[0], chunk_size):
                    chunk_features = long_features[i : i + chunk_size]
                    chunk_labels = long_labels[i : i + chunk_size]
                    
                    if chunk_features.shape[0] < 1:
                        continue
                    
                    num_chunks += 1
                    print(f"   Chunk {num_chunks}: {chunk_features.shape[0]} events")
                    
                    # 添加批次维度
                    chunk_features = chunk_features.unsqueeze(0)
                    
                    # 前向传播
                    predictions = model(chunk_features)
                    
                    # 检查预测结果
                    chunk_nan_found = check_tensor_for_nan_inf(predictions, f"Seq{seq_idx+1} Chunk{num_chunks} Predictions")
                    
                    if chunk_nan_found:
                        nan_found = True
                        print(f"❌ 序列 {seq_idx + 1} Chunk {num_chunks} 预测结果有异常值")
                        continue
                    
                    # 计算损失
                    loss = criterion(predictions.squeeze(), chunk_labels.float())
                    
                    print(f"     Loss: {loss.item():.6f}")
                    
                    # 检查损失
                    loss_nan_found = check_tensor_for_nan_inf(loss.unsqueeze(0), f"Seq{seq_idx+1} Chunk{num_chunks} Loss")
                    
                    if loss_nan_found:
                        nan_found = True
                        print(f"❌ 序列 {seq_idx + 1} Chunk {num_chunks} 损失计算有异常值")
                        continue
                    
                    # 累计损失
                    total_loss += loss.item() * len(chunk_features)
                    total_chunks_processed += len(chunk_features)
        
        # 计算平均损失
        if total_chunks_processed > 0:
            avg_loss = total_loss / total_chunks_processed
            print(f"\n✅ 验证管线完成")
            print(f"   总事件数: {total_chunks_processed}")
            print(f"   平均损失: {avg_loss:.6f}")
            
            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"❌ 最终平均损失异常: {avg_loss}")
                nan_found = True
        else:
            print(f"❌ 没有处理任何事件")
            nan_found = True
        
        return not nan_found
        
    except Exception as e:
        print(f"❌ 验证管线失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始验证管线NaN调试测试")
    print("=" * 60)
    
    # 运行各项测试
    tests = [
        ("数据加载", test_data_loading),
        ("模型前向传播", test_model_forward), 
        ("损失计算", test_loss_calculation),
        ("完整验证管线", test_validation_pipeline)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试失败: {e}")
            results[test_name] = False
    
    # 总结
    print(f"\n{'='*60}")
    print("🎯 测试总结:")
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"   {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！验证管线应该正常工作。")
    else:
        print("\n⚠️  发现问题，请检查上述失败的测试项。")

if __name__ == "__main__":
    main()