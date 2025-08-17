#!/usr/bin/env python3
"""
修复NaN问题的脚本
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import sys
sys.path.append('src')

from src.model import EventDenoisingMamba

def add_gradient_clipping_to_trainer():
    """在trainer.py中添加梯度裁剪"""
    
    trainer_code = '''
# 在trainer.py的train_one_epoch函数中，loss.backward()之后添加：

# 原代码：
# loss.backward()
# self.optimizer.step()

# 修改为：
loss.backward()

# 添加梯度裁剪防止梯度爆炸
torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

self.optimizer.step()
'''
    print("需要在trainer.py中添加梯度裁剪：")
    print(trainer_code)

def create_stable_model_patch():
    """创建数值稳定的模型补丁"""
    
    print("\n创建model.py的数值稳定补丁...")
    
    model_patch = '''
# 在model.py的forward函数中添加数值稳定性检查：

def forward(self, features):
    batch_size, sequence_length, feature_dim = features.shape
    
    # 验证输入特征维度为4维
    expected_dim = self.config['model']['input_feature_dim']
    assert feature_dim == expected_dim, f"Expected {expected_dim}D features, got {feature_dim}D"
    
    # 🔧 添加输入数值稳定性检查
    if torch.isnan(features).any() or torch.isinf(features).any():
        print(f"Warning: Input features contain NaN or Inf")
        # 替换NaN和Inf为有限值
        features = torch.where(torch.isnan(features), torch.zeros_like(features), features)
        features = torch.where(torch.isinf(features), torch.sign(features) * 10.0, features)
    
    # 🔧 限制输入范围防止数值爆炸
    features = torch.clamp(features, min=-10.0, max=10.0)
    
    # 直接进行Mamba处理，无需特征提取
    x = self.embedding(features)
    for layer in self.layers:
        x = layer(x)
        
        # 🔧 在每层之后检查数值稳定性
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Layer output contains NaN or Inf")
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            x = torch.where(torch.isinf(x), torch.sign(x) * 10.0, x)
    
    logits = self.classification_head(x)
    
    # 🔧 最终输出数值稳定性检查
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print(f"Warning: Output logits contain NaN or Inf")
        logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
        logits = torch.where(torch.isinf(logits), torch.sign(logits) * 10.0, logits)
    
    return logits
'''
    print(model_patch)

def create_safe_loss_function():
    """创建安全的损失函数"""
    
    safe_loss_code = '''
# 创建一个数值稳定的损失函数类
class SafeBCEWithLogitsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        # 检查输入
        if torch.isnan(predictions).any() or torch.isinf(predictions).any():
            print("Warning: Predictions contain NaN or Inf in loss calculation")
            predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)
            predictions = torch.where(torch.isinf(predictions), torch.sign(predictions) * 10.0, predictions)
        
        if torch.isnan(targets).any() or torch.isinf(targets).any():
            print("Warning: Targets contain NaN or Inf in loss calculation")
            targets = torch.where(torch.isnan(targets), torch.zeros_like(targets), targets)
        
        # 限制logits范围防止数值溢出
        predictions = torch.clamp(predictions, min=-50.0, max=50.0)
        
        loss = self.bce(predictions, targets)
        
        # 检查损失结果
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN or Inf, returning small positive value")
            return torch.tensor(0.693, device=loss.device, requires_grad=True)  # ln(2)
        
        return loss

# 在trainer.py中替换：
# self.criterion = nn.BCEWithLogitsLoss()
# 改为：
# self.criterion = SafeBCEWithLogitsLoss()
'''
    print(safe_loss_code)

def fix_feature_extraction():
    """修复特征提取中的数值问题"""
    
    feature_fix_code = '''
# 在feature_extractor.py中修复数值范围问题：

def process_sequence(self, raw_events):
    if len(raw_events) == 0:
        return np.empty((0, 4))
        
    num_events = raw_events.shape[0]
    feature_sequence = np.zeros((num_events, 4), dtype=np.float32)
    
    # Extract coordinates, timestamps, and polarities
    x = raw_events[:, 0].astype(np.float32)
    y = raw_events[:, 1].astype(np.float32)  
    t = raw_events[:, 2].astype(np.float64)
    p = raw_events[:, 3].astype(np.float32)
    
    # 🔧 修复归一化，确保范围在[0,1]
    x_norm = np.clip(x / max(self.w - 1, 1), 0.0, 1.0)
    y_norm = np.clip(y / max(self.h - 1, 1), 0.0, 1.0)
    
    # 🔧 修复dt计算，防止溢出
    dt = np.zeros_like(t, dtype=np.float32)
    if len(t) > 1:
        dt_raw = np.diff(t)
        dt[1:] = np.clip(dt_raw, 0.0, 1000.0).astype(np.float32)  # 限制最大dt
    dt[0] = 0.0
    
    # 🔧 确保polarity在{-1, 1}范围内
    p = np.clip(p, -1.0, 1.0)
    
    # Assemble features
    feature_sequence[:, 0] = x_norm
    feature_sequence[:, 1] = y_norm  
    feature_sequence[:, 2] = dt
    feature_sequence[:, 3] = p
    
    # 🔧 最终检查，移除任何异常值
    feature_sequence = np.where(np.isnan(feature_sequence), 0.0, feature_sequence)
    feature_sequence = np.where(np.isinf(feature_sequence), 0.0, feature_sequence)
    
    return feature_sequence
'''
    print(feature_fix_code)

def main():
    print("🔧 NaN问题修复方案")
    print("=" * 50)
    
    print("\n1. 梯度裁剪修复：")
    add_gradient_clipping_to_trainer()
    
    print("\n2. 模型数值稳定性修复：")
    create_stable_model_patch()
    
    print("\n3. 安全损失函数：")
    create_safe_loss_function()
    
    print("\n4. 特征提取修复：")
    fix_feature_extraction()
    
    print("\n🎯 立即可用的修复：")
    print("1. 重新运行特征生成以修复数据范围问题")
    print("2. 在config.yaml中减小chunk_size从32768到8192")
    print("3. 添加数值稳定性检查到模型forward函数")

if __name__ == "__main__":
    main()