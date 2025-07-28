import torch
import torch.nn as nn

class Mamba(nn.Module):
    """Mock Mamba implementation for testing without mamba-ssm dependency"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expand)
        self.conv1d = nn.Conv1d(d_model * expand, d_model * expand, kernel_size=d_conv, padding=d_conv-1, groups=d_model * expand)
        self.linear2 = nn.Linear(d_model * expand, d_model)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        # x shape: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.silu(x)
        
        # Apply 1D convolution (need to transpose for conv1d)
        x = x.transpose(1, 2)  # [batch, d_model*expand, seq_len]
        x = self.conv1d(x)[:, :, :-self.conv1d.kernel_size[0]+1]  # Remove padding
        x = x.transpose(1, 2)  # [batch, seq_len, d_model*expand]
        
        x = self.linear2(x)
        return x + residual