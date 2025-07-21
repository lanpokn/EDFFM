import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from src.feature_extractor import FeatureExtractor # 导入您的特征提取器

class EventDenoisingDataset(Dataset):
    def __init__(self, file_path, config):
        super().__init__()
        self.file_path = file_path
        self.sequence_length = config['data']['sequence_length']
        self.config = config
        
        # 我们现在需要一个特征提取器的实例
        # 注意：这个实例将在每个DataLoader worker进程中被创建
        self.feature_extractor = FeatureExtractor(config)
        
        self.all_events = np.loadtxt(file_path, delimiter=' ')
        print(f"Loaded {len(self.all_events)} events from {file_path}.")

    def __len__(self):
        return len(self.all_events) - self.sequence_length + 1

    def __getitem__(self, idx):
        # 1. 切分出原始事件序列和标签
        sequence_data = self.all_events[idx : idx + self.sequence_length]
        raw_events = sequence_data[:, :4] # (x, y, t, p)
        labels = sequence_data[:, 4]
        
        # 2. 在这里进行特征提取！
        # 输入是NumPy数组，输出也是NumPy数组
        # 特征提取器需要被设计成处理单个序列
        feature_sequence = self.feature_extractor.process_sequence(raw_events)
        
        # 3. 转换为torch.Tensor
        features_tensor = torch.tensor(feature_sequence, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        
        return features_tensor, labels_tensor


def create_dataloaders(config):
    """
    Creates train, validation, and test dataloaders.
    """
    train_dataset = EventDenoisingDataset(config['data']['train_path'], config)
    val_dataset = EventDenoisingDataset(config['data']['val_path'], config)
    test_dataset = EventDenoisingDataset(config['data']['test_path'], config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    return train_loader, val_loader, test_loader