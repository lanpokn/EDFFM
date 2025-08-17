#!/usr/bin/env python3
"""
专门调试trainer中validation loss计算问题
"""

import yaml
import torch
import torch.nn as nn
import sys
sys.path.append('src')

from src.unified_dataset import create_unified_dataloaders
from src.model import EventDenoisingMamba

def debug_trainer_validation():
    """调试trainer validation中的具体问题"""
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型和数据
    model = EventDenoisingMamba(config).to(device)
    model.eval()
    
    train_loader, val_loader = create_unified_dataloaders(config)
    criterion = nn.BCEWithLogitsLoss()
    chunk_size = config['training']['chunk_size']
    
    print(f"🔍 调试trainer validation问题")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Validation sequences: {len(val_loader)}")
    
    total_loss = 0
    total_chunks_processed = 0
    
    with torch.no_grad():
        for seq_idx, (long_features, long_labels) in enumerate(val_loader):
            if seq_idx >= 1:  # 只处理第一个序列
                break
                
            print(f"\n--- 分析序列 {seq_idx + 1} ---")
            
            # 重置隐藏状态
            model.reset_hidden_state()
            
            # 解包
            long_features = long_features.squeeze(0).to(device)
            long_labels = long_labels.squeeze(0).to(device)
            
            print(f"   原始序列形状: features {long_features.shape}, labels {long_labels.shape}")
            
            # 分块处理
            for i in range(0, min(chunk_size * 3, long_features.shape[0]), chunk_size):  # 只处理前3块
                chunk_features = long_features[i : i + chunk_size]
                chunk_labels = long_labels[i : i + chunk_size]
                
                if chunk_features.shape[0] < 1:
                    continue
                
                print(f"\n   Chunk {i//chunk_size + 1}:")
                print(f"     chunk_features before unsqueeze: {chunk_features.shape}")
                
                # 添加批次维度 - 这是关键！
                chunk_features = chunk_features.unsqueeze(0)
                print(f"     chunk_features after unsqueeze: {chunk_features.shape}")
                print(f"     chunk_labels shape: {chunk_labels.shape}")
                
                # 前向传播
                predictions = model(chunk_features)
                print(f"     predictions shape: {predictions.shape}")
                
                # 计算损失
                loss = criterion(predictions.squeeze(), chunk_labels.float())
                print(f"     loss: {loss.item():.6f}")
                
                # 这里是问题所在！
                print(f"\n     🚨 调试计算:")
                print(f"       len(chunk_features) = {len(chunk_features)} (batch size)")
                print(f"       chunk_features.shape[0] = {chunk_features.shape[0]} (batch size)")  
                print(f"       chunk_features.shape[1] = {chunk_features.shape[1]} (sequence length)")
                print(f"       chunk_labels.shape[0] = {chunk_labels.shape[0]} (sequence length)")
                
                # 错误的计算方式 (当前trainer中的方式)
                wrong_weight = len(chunk_features)  # 这是batch_size=1
                wrong_contribution = loss.item() * wrong_weight
                
                # 正确的计算方式
                correct_weight = chunk_features.shape[1]  # 这是sequence_length
                correct_contribution = loss.item() * correct_weight
                
                print(f"       ❌ 错误权重: {wrong_weight} -> 贡献: {wrong_contribution:.6f}")
                print(f"       ✅ 正确权重: {correct_weight} -> 贡献: {correct_contribution:.6f}")
                
                # 按照错误方式累计（模拟当前trainer）
                total_loss += loss.item() * len(chunk_features)
                total_chunks_processed += len(chunk_features)
                
                print(f"       累计total_loss: {total_loss:.6f}")
                print(f"       累计total_chunks_processed: {total_chunks_processed}")
    
    # 最终计算
    if total_chunks_processed > 0:
        avg_loss = total_loss / total_chunks_processed
        print(f"\n🎯 最终结果 (错误的计算方式):")
        print(f"   total_loss: {total_loss:.6f}")
        print(f"   total_chunks_processed: {total_chunks_processed}")
        print(f"   avg_loss: {avg_loss:.6f}")
        
        if torch.isnan(torch.tensor(avg_loss)) or torch.isinf(torch.tensor(avg_loss)):
            print(f"   ❌ 结果异常: {avg_loss}")
        else:
            print(f"   ✅ 结果正常: {avg_loss}")
    else:
        print(f"❌ total_chunks_processed = 0，会导致除零错误或返回0")

if __name__ == "__main__":
    debug_trainer_validation()