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
        
        # 特征提取器已移除，现在直接接收预提取的特征
        # 从配置获取特征维度
        input_feature_dim = model_config['input_feature_dim']  # 11维
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

    def forward(self, features):
        """
        Args:
            features (Tensor): Shape: [batch_size, sequence_length, 11] - 预提取的PFD特征
        Returns:
            probabilities (Tensor): Shape: [batch_size, sequence_length, 1] - 去噪概率
        """
        # 直接接收预提取的特征，无需在forward中提取
        # features shape: [batch_size, sequence_length, 11]
        
        # Mamba处理
        x = self.embedding(features)
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities