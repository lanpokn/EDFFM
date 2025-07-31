import torch
import torch.nn as nn
import numpy as np
try:
    from mamba_ssm import Mamba
except ImportError:
    from .utils.mock_mamba import Mamba

# ✅ 移除：特征提取器现在在数据集阶段使用
# from .feature_extractor import FeatureExtractor

class EventDenoisingMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config['model']
        
        # ✅ 移除：特征提取器现在在数据集阶段处理，模型直接接收13维特征
        # self.feature_extractor = FeatureExtractor(config)  # 不再需要
        
        # 从特征提取器获取特征维度
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
        ✅ 修正：直接接收13维特征，无需内部特征提取
        
        Args:
            features (Tensor): Shape: [batch_size, sequence_length, 13] - 13维PFD特征
        Returns:
            probabilities (Tensor): Shape: [batch_size, sequence_length, 1] - 去噪概率
        """
        batch_size, sequence_length, feature_dim = features.shape
        
        # 验证输入特征维度
        assert feature_dim == 11, f"Expected 11D features, got {feature_dim}D"
        
        # 直接进行Mamba处理，无需特征提取
        x = self.embedding(features)  # [batch_size, sequence_length, 11] → [batch_size, sequence_length, d_model]
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        probabilities = torch.sigmoid(logits)
        
        return probabilities