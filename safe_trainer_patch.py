#!/usr/bin/env python3
"""
为trainer添加临时的NaN安全检查补丁
可以在服务器上使用这个替代validate_one_epoch函数
"""

import torch
import numpy as np

def safe_validate_one_epoch(trainer):
    """
    带有详细NaN检查的安全验证函数
    用于替代trainer.validate_one_epoch()
    """
    trainer.model.eval()
    total_loss = 0
    total_chunks_processed = 0
    nan_chunks = 0
    inf_chunks = 0
    
    print(f"🔍 开始安全验证...")
    
    with torch.no_grad():
        for seq_idx, (long_features, long_labels) in enumerate(trainer.val_loader):
            print(f"   序列 {seq_idx + 1}/{len(trainer.val_loader)}")
            
            # 重置隐藏状态
            trainer.model.reset_hidden_state()
            
            # 数据移动
            long_features = long_features.squeeze(0).to(trainer.device)
            long_labels = long_labels.squeeze(0).to(trainer.device)
            
            # 检查输入数据
            if torch.isnan(long_features).any():
                print(f"     ❌ 输入features有NaN，跳过序列")
                continue
            if torch.isinf(long_features).any():
                print(f"     ❌ 输入features有Inf，跳过序列")
                continue
            if torch.isnan(long_labels).any():
                print(f"     ❌ 输入labels有NaN，跳过序列")
                continue
            if torch.isinf(long_labels).any():
                print(f"     ❌ 输入labels有Inf，跳过序列")
                continue
            
            chunk_count = 0
            # 分块处理
            for i in range(0, long_features.shape[0], trainer.chunk_size):
                chunk_features = long_features[i : i + trainer.chunk_size]
                chunk_labels = long_labels[i : i + trainer.chunk_size]
                
                if chunk_features.shape[0] < 1:
                    continue
                
                chunk_count += 1
                chunk_features = chunk_features.unsqueeze(0)
                
                # 前向传播
                try:
                    predictions = trainer.model(chunk_features)
                    
                    # 检查预测结果
                    if torch.isnan(predictions).any():
                        print(f"     ❌ Chunk {chunk_count} 预测有NaN")
                        nan_chunks += 1
                        continue
                    if torch.isinf(predictions).any():
                        print(f"     ❌ Chunk {chunk_count} 预测有Inf")
                        inf_chunks += 1
                        continue
                    
                    # 计算损失
                    loss = trainer.criterion(predictions.squeeze(), chunk_labels.float())
                    
                    # 检查损失
                    if torch.isnan(loss):
                        print(f"     ❌ Chunk {chunk_count} 损失为NaN")
                        print(f"       Predictions范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
                        print(f"       Labels范围: [{chunk_labels.min():.6f}, {chunk_labels.max():.6f}]")
                        nan_chunks += 1
                        continue
                    if torch.isinf(loss):
                        print(f"     ❌ Chunk {chunk_count} 损失为Inf")
                        inf_chunks += 1
                        continue
                    
                    # 累计损失（使用修正的权重计算）
                    sequence_length = chunk_features.shape[1]
                    total_loss += loss.item() * sequence_length
                    total_chunks_processed += sequence_length
                    
                    if chunk_count <= 3:  # 打印前几个chunk的信息
                        print(f"     Chunk {chunk_count}: Loss={loss.item():.6f}, 长度={sequence_length}")
                    
                except Exception as e:
                    print(f"     ❌ Chunk {chunk_count} 计算失败: {e}")
                    continue
            
            print(f"     序列完成: {chunk_count} chunks")
    
    # 计算结果
    print(f"\n📊 验证统计:")
    print(f"   总事件数: {total_chunks_processed}")
    print(f"   NaN chunks: {nan_chunks}")
    print(f"   Inf chunks: {inf_chunks}")
    
    if total_chunks_processed > 0:
        avg_loss = total_loss / total_chunks_processed
        print(f"   平均损失: {avg_loss:.6f}")
        
        # 最终检查
        if np.isnan(avg_loss):
            print(f"   ❌ 最终平均损失为NaN!")
            return float('nan')
        elif np.isinf(avg_loss):
            print(f"   ❌ 最终平均损失为Inf!")
            return float('inf')
        else:
            print(f"   ✅ 验证成功")
            return avg_loss
    else:
        print(f"   ❌ 没有成功处理任何chunk")
        return float('nan')

def patch_trainer_validation():
    """
    创建一个使用安全验证的训练函数补丁
    """
    code = """
# 在你的训练脚本中使用这个替代验证函数
# 例如在main.py或单独的训练脚本中：

from safe_trainer_patch import safe_validate_one_epoch

# 在训练循环中替换：
# val_loss = trainer.validate_one_epoch()
# 改为：
# val_loss = safe_validate_one_epoch(trainer)

# 或者直接替换trainer的方法：
# trainer.validate_one_epoch = lambda: safe_validate_one_epoch(trainer)
"""
    print(code)

if __name__ == "__main__":
    patch_trainer_validation()