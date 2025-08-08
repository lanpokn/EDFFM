# src/unified_dataset.py
import os
import h5py
import time
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 假设您原来的数据生成逻辑封装在 'src.epoch_iteration_dataset' 中
from src.epoch_iteration_dataset import EpochIterationDataset as GenerativeBackend

class UnifiedSequenceDataset(Dataset):
    """
    一个统一的数据集，支持两种操作模式：
    1. 'generate': 实时生成数据并自动存档。
    2. 'load': 从磁盘加载预先存档的数据。
    """
    def __init__(self, config: dict, split: str):
        self.config = config
        self.split = split
        self.mode = config['data_pipeline']['mode'] # 'generate' or 'load'
        
        data_config = config['data_pipeline']
        self.h5_archive_dir = os.path.join(data_config['h5_archive_path'], split)
        os.makedirs(self.h5_archive_dir, exist_ok=True)

        if self.mode == 'generate':
            self.generator = GenerativeBackend(config, split)
            self.num_sequences_per_epoch = config['training' if split == 'train' else 'evaluation']['num_long_sequences_per_epoch']
            # print(f"UnifiedDataset ({split}) in 'generate' mode. Will generate {self.num_sequences_per_epoch} sequences per epoch.")
        
        elif self.mode == 'load':
            self.file_list = sorted(glob.glob(os.path.join(self.h5_archive_dir, '*.h5')))
            if not self.file_list:
                raise FileNotFoundError(f"No .h5 files found in {self.h5_archive_dir}. Please run in 'generate' mode first.")
            self.num_sequences_per_epoch = len(self.file_list)
            # print(f"UnifiedDataset ({split}) in 'load' mode. Found {self.num_sequences_per_epoch} pre-generated files.")
        
        else:
            raise ValueError(f"Unknown mode in data_pipeline: {self.mode}")

    def __len__(self):
        return self.num_sequences_per_epoch

    def __getitem__(self, idx):
        if self.mode == 'generate':
            # 1. 实时生成一个长序列
            features, labels = self.generator._generate_one_long_sequence()
            
            # 2. 自动存档到磁盘 - 防重名机制
            import random
            timestamp = int(time.time() * 1000)
            base_filename = f'sequence_{timestamp}_{idx:05d}'
            
            # 防重名：检查文件是否存在，如果存在则添加随机后缀
            attempt = 0
            while attempt < 10:  # 最多尝试10次
                if attempt == 0:
                    filename = f'{base_filename}.h5'
                else:
                    random_suffix = random.randint(1000, 9999)
                    filename = f'{base_filename}_{random_suffix}.h5'
                
                file_path = os.path.join(self.h5_archive_dir, filename)
                
                # 检查文件是否已存在
                if not os.path.exists(file_path):
                    break
                attempt += 1
            else:
                # 如果10次都失败，使用UUID确保唯一性
                import uuid
                unique_id = str(uuid.uuid4())[:8]
                filename = f'{base_filename}_{unique_id}.h5'
                file_path = os.path.join(self.h5_archive_dir, filename)
            
            try:
                with h5py.File(file_path, 'w') as hf:
                    hf.create_dataset('features', data=features)
                    hf.create_dataset('labels', data=labels)
                # print(f"💾 Archived sequence to: {filename} (shape: {features.shape})")
            except Exception as e:
                print(f"Warning: Failed to save archive file {file_path}. Error: {e}")

            # 3. 转换为Tensor并返回
            return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

        elif self.mode == 'load':
            # 直接从磁盘读取文件
            file_path = self.file_list[idx]
            with h5py.File(file_path, 'r') as hf:
                features = hf['features'][:]
                labels = hf['labels'][:]
            
            filename = os.path.basename(file_path)
            # print(f"📂 Loaded sequence from: {filename} (shape: {features.shape})")
            return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

def create_unified_dataloaders(config):
    """根据配置创建统一的Dataloader"""
    train_dataset = UnifiedSequenceDataset(config, split='train')
    val_dataset = UnifiedSequenceDataset(config, split='val')
    
    # ### BEGIN BUGFIX 4: DATALOADER CONFIG ###
    # 对于有状态TBPTT，永远不要shuffle，并且使用单进程加载以保证顺序和生成安全
    shuffle = False
    num_workers = 0
    # ### END BUGFIX 4 ###
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    print(f"📊 Created Unified Dataloaders in '{config['data_pipeline']['mode']}' mode")
    print(f"  - Train: {len(train_loader)} sequences per epoch")
    print(f"  - Val:   {len(val_loader)} sequences per epoch")
    print(f"  - Shuffle: {shuffle}, Workers: {num_workers} (Fixed for TBPTT)")

    return train_loader, val_loader