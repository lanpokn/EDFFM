import torch
import torch.nn as nn
import numpy as np
try:
    from mamba_ssm import Mamba
except ImportError:
    from .utils.mock_mamba import Mamba

from .feature_extractor import FeatureExtractor

class EventDenoisingMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        
        # 集成特征提取器
        self.feature_extractor = FeatureExtractor(config)
        
        # 从特征提取器获取特征维度
        input_feature_dim = model_config['input_feature_dim']  # 13维
        d_model = model_config['d_model']
        
        self.embedding = nn.Linear(input_feature_dim, d_model)
        
        n_layers = model_config['n_layers']
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=model_config['d_state'], 
                d_conv=model_config['d_conv'],
                expand=model_config['expand']
            )
            for _ in range(n_layers)
        ])
        
        self.classification_head = nn.Linear(d_model, 1)

    def forward(self, raw_events):
        """
        Args:
            raw_events (Tensor): Shape: [batch_size, sequence_length, 4] - 原始事件
        Returns:
            probabilities (Tensor): Shape: [batch_size, sequence_length, 1] - 去噪概率
        """
        batch_size, sequence_length, _ = raw_events.shape
        
        # 恢复原始PFD特征提取（含完整理论依据的13维特征）
        features_list = []
        for b in range(batch_size):
            # 转换为numpy进行PFD特征提取
            events_np = raw_events[b].cpu().numpy()
            features_np = self.feature_extractor.process_sequence(events_np)
            features_tensor = torch.tensor(features_np, dtype=torch.float32, device=raw_events.device)
            features_list.append(features_tensor)
        
        # 堆叠成批量张量
        features = torch.stack(features_list, dim=0)  # [batch_size, sequence_length, 13]
        
        # Mamba处理
        x = self.embedding(features)
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities