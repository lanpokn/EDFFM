#!/usr/bin/env python3
"""
专门用于服务器环境调试validation NaN问题的脚本
包含详细的环境检查和数据完整性验证
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import h5py
import glob
from torch.utils.data import DataLoader

sys.path.append('src')
from src.unified_dataset import UnifiedSequenceDataset, create_unified_dataloaders
from src.model import EventDenoisingMamba

def check_environment():
    """检查服务器环境信息"""
    print("🔍 环境信息检查:")
    print(f"   Python版本: {sys.version}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB")
    
    print(f"   NumPy版本: {np.__version__}")
    print(f"   工作目录: {os.getcwd()}")

def check_h5_data_integrity():
    """检查H5验证数据的完整性"""
    print("\n🔍 H5验证数据完整性检查:")
    
    config_path = 'configs/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    h5_archive_path = config['data_pipeline']['h5_archive_path']
    val_dir = os.path.join(h5_archive_path, 'val')
    
    print(f"   验证数据目录: {val_dir}")
    
    if not os.path.exists(val_dir):
        print(f"   ❌ 验证数据目录不存在: {val_dir}")
        return False
    
    h5_files = glob.glob(os.path.join(val_dir, '*.h5'))
    print(f"   找到 {len(h5_files)} 个H5文件")
    
    for i, h5_file in enumerate(h5_files):
        print(f"\n   检查文件 {i+1}: {os.path.basename(h5_file)}")
        print(f"     文件大小: {os.path.getsize(h5_file) / 1024**2:.1f}MB")
        
        try:
            with h5py.File(h5_file, 'r') as hf:
                features = hf['features']
                labels = hf['labels']
                
                print(f"     Features形状: {features.shape}")
                print(f"     Labels形状: {labels.shape}")
                
                # 检查一小部分数据的异常值
                sample_size = min(1000, features.shape[0])
                features_sample = features[:sample_size]
                labels_sample = labels[:sample_size]
                
                # 检查NaN和Inf
                has_nan_features = np.isnan(features_sample).any()
                has_inf_features = np.isinf(features_sample).any()
                has_nan_labels = np.isnan(labels_sample).any()
                has_inf_labels = np.isinf(labels_sample).any()
                
                if has_nan_features or has_inf_features:
                    print(f"     ❌ Features有异常值: NaN={has_nan_features}, Inf={has_inf_features}")
                    
                    # 详细分析
                    for dim in range(features_sample.shape[-1]):
                        dim_data = features_sample[:, dim]
                        dim_nan = np.isnan(dim_data).sum()
                        dim_inf = np.isinf(dim_data).sum()
                        if dim_nan > 0 or dim_inf > 0:
                            print(f"       维度{dim}: NaN={dim_nan}, Inf={dim_inf}")
                else:
                    print(f"     ✅ Features正常")
                    
                if has_nan_labels or has_inf_labels:
                    print(f"     ❌ Labels有异常值: NaN={has_nan_labels}, Inf={has_inf_labels}")
                else:
                    print(f"     ✅ Labels正常")
                    
                # 检查数据范围
                print(f"     Features范围: min={np.min(features_sample):.6f}, max={np.max(features_sample):.6f}")
                print(f"     Labels范围: min={np.min(labels_sample):.6f}, max={np.max(labels_sample):.6f}")
                
        except Exception as e:
            print(f"     ❌ 读取文件失败: {e}")
            return False
    
    return True

def test_model_numerical_stability():
    """测试模型在服务器环境下的数值稳定性"""
    print("\n🔍 模型数值稳定性测试:")
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   使用设备: {device}")
    
    # 创建模型
    model = EventDenoisingMamba(config).to(device)
    model.eval()
    
    criterion = nn.BCEWithLogitsLoss()
    
    # 测试各种数据范围
    test_cases = [
        ("正常数据", {
            'features': torch.randn(1, 100, 4) * 0.1,
            'labels': torch.randint(0, 2, (100,), dtype=torch.float)
        }),
        ("大数值", {
            'features': torch.randn(1, 100, 4) * 100,
            'labels': torch.randint(0, 2, (100,), dtype=torch.float)
        }),
        ("小数值", {
            'features': torch.randn(1, 100, 4) * 1e-6,
            'labels': torch.randint(0, 2, (100,), dtype=torch.float)
        }),
        ("极值测试", {
            'features': torch.tensor([[[1000.0, 1000.0, 1000.0, 1.0]]] * 100),
            'labels': torch.ones(100, dtype=torch.float)
        })
    ]
    
    for case_name, data in test_cases:
        print(f"\n   测试用例: {case_name}")
        
        features = data['features'].to(device)
        labels = data['labels'].to(device)
        
        try:
            with torch.no_grad():
                model.reset_hidden_state()
                predictions = model(features)
                loss = criterion(predictions.squeeze(), labels)
                
                # 检查结果
                pred_has_nan = torch.isnan(predictions).any()
                pred_has_inf = torch.isinf(predictions).any()
                loss_has_nan = torch.isnan(loss)
                loss_has_inf = torch.isinf(loss)
                
                if pred_has_nan or pred_has_inf or loss_has_nan or loss_has_inf:
                    print(f"     ❌ 异常值检测:")
                    print(f"       Predictions NaN: {pred_has_nan}, Inf: {pred_has_inf}")
                    print(f"       Loss NaN: {loss_has_nan}, Inf: {loss_has_inf}")
                    print(f"       Loss值: {loss.item() if not loss_has_nan else 'NaN'}")
                else:
                    print(f"     ✅ 正常, Loss: {loss.item():.6f}")
                    
        except Exception as e:
            print(f"     ❌ 测试失败: {e}")

def test_server_validation_pipeline():
    """测试服务器上的完整验证管线"""
    print("\n🔍 服务器验证管线测试:")
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 创建模型和数据加载器
        model = EventDenoisingMamba(config).to(device)
        model.eval()
        
        train_loader, val_loader = create_unified_dataloaders(config)
        criterion = nn.BCEWithLogitsLoss()
        chunk_size = config['training']['chunk_size']
        
        print(f"   验证序列数: {len(val_loader)}")
        print(f"   Chunk大小: {chunk_size}")
        
        total_loss = 0
        total_chunks_processed = 0
        
        with torch.no_grad():
            for seq_idx, (long_features, long_labels) in enumerate(val_loader):
                print(f"\n   处理序列 {seq_idx + 1}/{len(val_loader)}")
                
                # 重置状态
                model.reset_hidden_state()
                
                # 数据移动和形状检查
                long_features = long_features.squeeze(0).to(device)
                long_labels = long_labels.squeeze(0).to(device)
                
                print(f"     序列形状: features {long_features.shape}, labels {long_labels.shape}")
                
                # 检查输入数据异常值
                input_nan = torch.isnan(long_features).any() or torch.isnan(long_labels).any()
                input_inf = torch.isinf(long_features).any() or torch.isinf(long_labels).any()
                
                if input_nan or input_inf:
                    print(f"     ❌ 输入数据异常: NaN={input_nan}, Inf={input_inf}")
                    continue
                
                # 分块处理
                chunk_count = 0
                for i in range(0, long_features.shape[0], chunk_size):
                    chunk_features = long_features[i : i + chunk_size]
                    chunk_labels = long_labels[i : i + chunk_size]
                    
                    if chunk_features.shape[0] < 1:
                        continue
                    
                    chunk_count += 1
                    chunk_features = chunk_features.unsqueeze(0)
                    
                    # 前向传播
                    predictions = model(chunk_features)
                    
                    # 检查预测结果
                    pred_nan = torch.isnan(predictions).any()
                    pred_inf = torch.isinf(predictions).any()
                    
                    if pred_nan or pred_inf:
                        print(f"     ❌ Chunk {chunk_count} 预测异常: NaN={pred_nan}, Inf={pred_inf}")
                        continue
                    
                    # 计算损失
                    loss = criterion(predictions.squeeze(), chunk_labels.float())
                    
                    # 检查损失
                    loss_nan = torch.isnan(loss)
                    loss_inf = torch.isinf(loss)
                    
                    if loss_nan or loss_inf:
                        print(f"     ❌ Chunk {chunk_count} 损失异常: NaN={loss_nan}, Inf={loss_inf}")
                        print(f"       Predictions范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
                        print(f"       Labels范围: [{chunk_labels.min():.6f}, {chunk_labels.max():.6f}]")
                        continue
                    
                    # 累计 - 使用正确的权重
                    sequence_length = chunk_features.shape[1]  # 正确的权重
                    total_loss += loss.item() * sequence_length
                    total_chunks_processed += sequence_length
                    
                    if chunk_count <= 3:  # 只打印前几个chunk的详细信息
                        print(f"     Chunk {chunk_count}: Loss={loss.item():.6f}, 权重={sequence_length}")
                
                print(f"     序列处理完成，共 {chunk_count} 个chunks")
                
                # 限制只处理前几个序列进行调试
                if seq_idx >= 2:
                    break
        
        # 计算最终结果
        if total_chunks_processed > 0:
            avg_loss = total_loss / total_chunks_processed
            print(f"\n🎯 验证结果:")
            print(f"   总事件数: {total_chunks_processed}")
            print(f"   平均损失: {avg_loss:.6f}")
            
            # 最终检查
            if np.isnan(avg_loss) or np.isinf(avg_loss):
                print(f"   ❌ 最终结果异常: {avg_loss}")
                return False
            else:
                print(f"   ✅ 最终结果正常")
                return True
        else:
            print(f"   ❌ 没有处理任何事件")
            return False
            
    except Exception as e:
        print(f"   ❌ 验证管线失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主调试函数"""
    print("🚀 服务器环境验证NaN调试")
    print("=" * 60)
    
    # 运行各项检查
    tests = [
        ("环境检查", check_environment),
        ("H5数据完整性", check_h5_data_integrity),
        ("模型数值稳定性", test_model_numerical_stability),
        ("验证管线测试", test_server_validation_pipeline)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func is check_environment:
                test_func()  # 环境检查没有返回值
            else:
                result = test_func()
                if result is False:
                    print(f"❌ {test_name} 发现问题")
                else:
                    print(f"✅ {test_name} 正常")
        except Exception as e:
            print(f"❌ {test_name} 失败: {e}")

if __name__ == "__main__":
    main()