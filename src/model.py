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
        
        # ✅ 特征提取器现在在数据集阶段处理，模型直接接收4维特征
        # self.feature_extractor = FeatureExtractor(config)  # 不再需要
        
        # 从配置获取特征维度 (当前为4维：x_norm, y_norm, dt, polarity)
        input_feature_dim = model_config['input_feature_dim']  # 4维
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
        Forward pass with 4D features (PFD temporarily disabled for performance)
        
        Args:
            features (Tensor): Shape: [batch_size, sequence_length, 4] - 4维特征 [x_norm, y_norm, dt, polarity]
        Returns:
            probabilities (Tensor): Shape: [batch_size, sequence_length, 1] - 去噪概率
        """
        batch_size, sequence_length, feature_dim = features.shape
        
        # 验证输入特征维度为4维
        expected_dim = self.config['model']['input_feature_dim']
        assert feature_dim == expected_dim, f"Expected {expected_dim}D features, got {feature_dim}D"
        
        # 直接进行Mamba处理，无需特征提取
        x = self.embedding(features)  # [batch_size, sequence_length, 4] → [batch_size, sequence_length, d_model]
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        # 注意：现在返回logits而不是probabilities，配合BCEWithLogitsLoss使用
        
        return logits

    # ### BEGIN BUGFIX ###
    def reset_hidden_state(self):
        """
        重置模型中所有Mamba层的内部隐藏状态。
        这对于在处理新的独立序列（无论是训练、验证还是测试）之前至关重要。
        """
        # 这是一个健壮的实现，它会遍历模型的所有子模块
        # 并为找到的每一个Mamba层重置状态。
        try:
            for module in self.modules():
                if isinstance(module, Mamba):
                    # mamba-ssm库通过重置inference_params来清除状态
                    if hasattr(module, 'inference_params'):
                        module.inference_params = None
        except Exception as e:
            print(f"Warning: Could not reset Mamba states: {e}")
    # ### END BUGFIX ###