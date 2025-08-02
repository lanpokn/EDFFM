"""
Mixed Flare DataLoaders for EventMamba-FX
Creates DataLoaders that combine DSEC background events with synthetic flare events
"""

import torch
from torch.utils.data import DataLoader
from src.mixed_flare_datasets import MixedFlareDataset
import numpy as np


def variable_length_collate_fn(batch, sequence_length):
    """
    Custom collate function to handle variable-length 11D feature sequences.
    ✅ 修正：现在处理11维特征而不是原始事件
    
    Args:
        batch: List of (features_tensor, labels_tensor) tuples
        sequence_length: Target sequence length from config
        
    Returns:
        Tuple of (batched_features, batched_labels)
        batched_features: [batch_size, sequence_length, 11]
        batched_labels: [batch_size, sequence_length]
    """
    batch_size = len(batch)
    
    # Initialize output tensors for 11D features
    batched_features = torch.zeros((batch_size, sequence_length, 11), dtype=torch.float32)
    batched_labels = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    
    for i, (features, labels) in enumerate(batch):
        n_features = len(features)
        
        if n_features == 0:
            # Empty sequence - keep as zeros (padding)
            continue
        elif n_features <= sequence_length:
            # Pad with zeros if sequence is shorter
            batched_features[i, :n_features] = features
            batched_labels[i, :n_features] = labels
            # Remaining positions stay as zero padding
        else:
            # Truncate if sequence is longer - sample uniformly
            # Use uniform sampling to preserve temporal distribution
            indices = np.linspace(0, n_features-1, sequence_length, dtype=int)
            batched_features[i] = features[indices]
            batched_labels[i] = labels[indices]
    
    return batched_features, batched_labels


def create_mixed_flare_dataloaders(config):
    """Create DataLoaders for mixed flare training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)  # 使用data部分的配置
    sequence_length = config['data']['sequence_length']
    
    # Create custom collate function with sequence_length parameter
    collate_fn = lambda batch: variable_length_collate_fn(batch, sequence_length)
    
    # Create datasets (all use same DSEC data for now, differ by random seed)
    train_dataset = MixedFlareDataset(config, split='train')
    val_dataset = MixedFlareDataset(config, split='val')  
    test_dataset = MixedFlareDataset(config, split='test')
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    print(f"Created mixed flare dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader