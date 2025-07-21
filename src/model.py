import torch
import torch.nn as nn
from mamba_ssm import Mamba

class EventDenoisingMamba(nn.Module):
    def __init__(self, config): # 只需要模型相关的配置
        super().__init__()
        model_config = config['model']
        # 模型现在假设输入已经是增强过的特征了
        input_feature_dim = model_config['input_feature_dim']
        d_model = model_config['d_model']
        
        self.embedding = nn.Linear(input_feature_dim, d_model)
        
        self.layers = nn.ModuleList([...]) # Mamba层不变
        
        self.classification_head = nn.Linear(d_model, 1)

    def forward(self, features):
        """
        Args:
            features (Tensor): Shape: [batch_size, sequence_length, input_feature_dim]
        """
        x = self.embedding(features)
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        probabilities = torch.sigmoid(logits)
        return probabilities