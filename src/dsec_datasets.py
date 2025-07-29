#!/usr/bin/env python3
"""
DSEC Dataset Loader for EventMamba-FX
Loads events from DSEC format H5 files and samples 1-second windows
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import hdf5plugin  # Required for DSEC compressed H5 files
import numpy as np
import os
import glob
import random
from typing import List, Tuple, Optional


class DSECEventDataset(Dataset):
    """
    Dataset for loading DSEC events with 1-second time windows
    """
    
    def __init__(self, dsec_path: str, flare_path: str, time_window_us: int = 1000000, 
                 sequence_length: int = 64, camera_side: str = 'left', max_files: int = 3):
        """
        Args:
            dsec_path: Path to DSEC train folder
            flare_path: Path to flare events (not used for now)
            time_window_us: Time window in microseconds (1 second = 1000000)
            sequence_length: Number of events to sample from the window
            camera_side: 'left' or 'right' camera
            max_files: Maximum number of files to load for testing
        """
        self.dsec_path = dsec_path
        self.flare_path = flare_path
        self.time_window_us = time_window_us
        self.sequence_length = sequence_length
        self.camera_side = camera_side
        self.max_files = max_files
        
        # Find all available H5 files
        self.h5_files = self._find_h5_files()
        print(f"Found {len(self.h5_files)} DSEC event files, using first {min(len(self.h5_files), max_files)}")
        
        # Limit files for testing
        self.h5_files = self.h5_files[:max_files]
        
        # Pre-load file metadata to determine valid time ranges
        self.file_metadata = self._load_file_metadata()
        
    def _find_h5_files(self) -> List[str]:
        """Find all events.h5 files in DSEC dataset"""
        pattern = os.path.join(self.dsec_path, "*/events", self.camera_side, "events.h5")
        files = glob.glob(pattern)
        return sorted(files)
    
    def _load_file_metadata(self) -> List[dict]:
        """Load metadata for each H5 file to determine valid time ranges"""
        metadata = []
        for h5_file in self.h5_files:
            try:
                with h5py.File(h5_file, 'r') as f:
                    t_data = f['events/t'][:]
                    t_min, t_max = t_data[0], t_data[-1]
                    n_events = len(t_data)
                    
                    # Calculate number of valid 1-second windows
                    duration = t_max - t_min
                    n_windows = max(1, int(duration // self.time_window_us))
                    
                    metadata.append({
                        'file': h5_file,
                        't_min': t_min,
                        't_max': t_max,
                        'n_events': n_events,
                        'n_windows': n_windows,
                        'duration': duration
                    })
                    print(f"File: {os.path.basename(h5_file)}, Events: {n_events}, Windows: {n_windows}")
            except Exception as e:
                print(f"Error loading {h5_file}: {e}")
                
        return metadata
    
    def __len__(self) -> int:
        """Return total number of 1-second windows across all files"""
        return sum(meta['n_windows'] for meta in self.file_metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a 1-second window of events
        Returns:
            events: Tensor of shape (sequence_length, 4) - [x, y, t, p]
            labels: Tensor of shape (sequence_length,) - all zeros for now (no flare)
        """
        # Find which file and window this index corresponds to
        file_idx, window_idx = self._idx_to_file_window(idx)
        metadata = self.file_metadata[file_idx]
        
        # Load events from the selected file
        with h5py.File(metadata['file'], 'r') as f:
            # Load all event data
            x = f['events/x'][:]
            y = f['events/y'][:]
            t = f['events/t'][:]
            p = f['events/p'][:]
            
            # Convert DSEC polarity to model format (-1,1)
            # DSEC uses 0=negative, 255=positive (or sometimes 0/1)
            p = np.where(p > 0, 1, -1).astype(np.float32)
            
            # Calculate time window boundaries
            t_start = metadata['t_min'] + window_idx * self.time_window_us
            t_end = t_start + self.time_window_us
            
            # Find events in this time window
            mask = (t >= t_start) & (t < t_end)
            
            if not np.any(mask):
                # If no events in window, return empty sequence
                events = np.zeros((self.sequence_length, 4), dtype=np.float32)
                labels = np.zeros(self.sequence_length, dtype=np.float32)
            else:
                # Extract events in window
                x_win = x[mask]
                y_win = y[mask]
                t_win = t[mask]
                p_win = p[mask]
                
                # Normalize time to start from 0
                t_win = t_win - t_win[0]
                
                # Combine into events array
                events_win = np.column_stack([x_win, y_win, t_win, p_win]).astype(np.float32)
                
                # Sample or pad to sequence_length
                if len(events_win) >= self.sequence_length:
                    # Random sampling if too many events
                    indices = np.random.choice(len(events_win), self.sequence_length, replace=False)
                    indices = np.sort(indices)  # Keep temporal order
                    events = events_win[indices]
                else:
                    # Pad with zeros if too few events
                    events = np.zeros((self.sequence_length, 4), dtype=np.float32)
                    events[:len(events_win)] = events_win
                
                # For now, all labels are 0 (no flare)
                labels = np.zeros(self.sequence_length, dtype=np.float32)
        
        return torch.from_numpy(events), torch.from_numpy(labels)
    
    def _idx_to_file_window(self, idx: int) -> Tuple[int, int]:
        """Convert global index to (file_index, window_index)"""
        cumulative = 0
        for file_idx, metadata in enumerate(self.file_metadata):
            if idx < cumulative + metadata['n_windows']:
                window_idx = idx - cumulative
                return file_idx, window_idx
            cumulative += metadata['n_windows']
        
        # Fallback to last window of last file
        return len(self.file_metadata) - 1, self.file_metadata[-1]['n_windows'] - 1


def create_dsec_dataloaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DSEC-based dataloaders for train/val/test
    For now, we'll use the same dataset for all three splits
    """
    
    # Create datasets (limit files for testing)
    train_dataset = DSECEventDataset(
        dsec_path=config['data']['dsec_path'],
        flare_path=config['data']['flare_path'],
        time_window_us=config['data']['time_window_us'],
        sequence_length=config['data']['sequence_length'],
        camera_side='left',
        max_files=2  # Limit for testing
    )
    
    # For validation and test, we can use a subset of the same data
    # This is temporary - in production you'd want separate datasets
    val_dataset = train_dataset
    test_dataset = train_dataset
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    import yaml
    
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    dataset = DSECEventDataset(
        dsec_path=config['data']['dsec_path'],
        flare_path=config['data']['flare_path'],
        time_window_us=config['data']['time_window_us'],
        sequence_length=config['data']['sequence_length']
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test first sample
    events, labels = dataset[0]
    print(f"Events shape: {events.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Event range - X: [{events[:, 0].min():.1f}, {events[:, 0].max():.1f}]")
    print(f"Event range - Y: [{events[:, 1].min():.1f}, {events[:, 1].max():.1f}]")
    print(f"Event range - T: [{events[:, 2].min():.1f}, {events[:, 2].max():.1f}]")
    print(f"Event polarities: {torch.unique(events[:, 3])}")