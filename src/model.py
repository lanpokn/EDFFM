import torch
import torch.nn as nn
try:
    from mamba_ssm import Mamba
except ImportError:
    from .utils.mock_mamba import Mamba

class EventDenoisingMamba(nn.Module):
    def __init__(self, config): # 只需要模型相关的配置
        super().__init__()
        model_config = config['model']
        # 模型现在假设输入已经是增强过的特征了
        input_feature_dim = model_config['input_feature_dim']
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
            features (Tensor): Shape: [batch_size, sequence_length, input_feature_dim]
        """
        x = self.embedding(features)
        for layer in self.layers:
            x = layer(x)
        logits = self.classification_head(x)
        probabilities = torch.sigmoid(logits)
        return probabilities