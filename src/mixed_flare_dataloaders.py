"""
Mixed Flare DataLoaders for EventMamba-FX
Creates DataLoaders that combine DSEC background events with synthetic flare events
"""

import torch
from torch.utils.data import DataLoader
from src.mixed_flare_datasets import MixedFlareDataset


def create_mixed_flare_dataloaders(config):
    """Create DataLoaders for mixed flare training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)  # 使用data部分的配置
    
    # Create datasets (all use same DSEC data for now, differ by random seed)
    train_dataset = MixedFlareDataset(config, split='train')
    val_dataset = MixedFlareDataset(config, split='val')  
    test_dataset = MixedFlareDataset(config, split='test')
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Created mixed flare dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader