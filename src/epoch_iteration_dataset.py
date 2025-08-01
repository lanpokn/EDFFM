"""
Epoch-Iteration Dataset for EventMamba-FX

Implements the correct data pipeline:
- Epoch: Generate one long_sequence with background + flare events, then extract features
- Iteration: Sliding window sampling from the long_feature_sequence

This addresses the core requirement: "先在完整序列上提取物理特征，再进行分段学习"
"""

import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Union

# Import existing components
from src.dsec_efficient import DSECEventDatasetEfficient
from src.dvs_flare_integration import create_flare_event_generator
from src.feature_extractor import FeatureExtractor
from src.event_visualization_utils import EventVisualizer


class EpochIterationDataset(Dataset):
    """
    Dataset implementing Epoch-Iteration architecture:
    
    Epoch Level (Data Generation):
    - Load background events (0.1-0.3s duration)
    - Generate flare events (0.1-0.3s duration)  
    - Merge and sort → long_sequence [N, 4]
    - Extract features → long_feature_sequence [N, 11]
    
    Iteration Level (Model Training):
    - Sliding window sampling from long_feature_sequence
    - Return fixed-length batches (e.g., 64 consecutive events)
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """Initialize the Epoch-Iteration dataset.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # Initialize DSEC background loader
        self.dsec_dataset = DSECEventDatasetEfficient(
            dsec_path=config['data']['dsec_path'],
            flare_path="",  # Not used for background events
            time_window_us=config['data']['time_window_us'],
            sequence_length=config['data']['sequence_length']
        )
        
        # Initialize flare event generator
        self.flare_generator = create_flare_event_generator(config)
        
        # Initialize feature extractor (CRITICAL: at dataset level)
        self.feature_extractor = FeatureExtractor(config)
        
        # Training parameters
        self.sequence_length = config['data']['sequence_length']  # e.g., 64
        self.min_duration_ms = 100  # 0.1s minimum duration
        self.max_duration_ms = 300  # 0.3s maximum duration
        
        # Debug mode
        self.debug_mode = config.get('debug_mode', False)
        self.max_samples = config['data'].get('max_samples_debug', None)
        
        # Epoch-level data (regenerated each epoch)
        self.current_epoch = -1
        self.long_feature_sequence = None  # [N, 11] - the core feature sequence
        self.long_labels = None           # [N] - corresponding labels
        self.num_iterations = 0           # Number of iterations possible in current epoch
        
        # Initialize event visualizer (debug mode only)
        self.event_visualizer = None
        if self.debug_mode:
            viz_output_dir = os.path.join(config.get('debug_output_dir', './output/debug'), 'epoch_iteration_analysis')
            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.event_visualizer = EventVisualizer(viz_output_dir, resolution)
            print(f"🚨 DEBUG MODE: Epoch-Iteration analysis enabled, output: {viz_output_dir}")
        
        print(f"Initialized EpochIterationDataset ({split}): {len(self.dsec_dataset)} background samples available")
    
    def new_epoch(self):
        """
        Generate new epoch-level data:
        1. Sample background events (0.1-0.3s)
        2. Generate flare events (0.1-0.3s)
        3. Merge & sort → long_sequence
        4. Extract features → long_feature_sequence
        """
        print(f"\n🔄 Generating new epoch data (Epoch {self.current_epoch + 1})")
        epoch_start_time = time.time()
        
        self.current_epoch += 1
        
        # Step 1: Generate background events
        print("  Step 1: Loading background events...")
        background_events = self._generate_background_events()
        
        # Step 2: Generate flare events  
        print("  Step 2: Generating flare events...")
        flare_events = self._generate_flare_events()
        
        # Step 3: Merge and sort events
        print("  Step 3: Merging and sorting events...")
        long_sequence, labels = self._merge_and_sort_events(background_events, flare_events)
        
        # Step 4: CRITICAL - Extract features on complete sequence
        print("  Step 4: Extracting PFD features on complete sequence...")
        feature_start_time = time.time()
        
        if len(long_sequence) == 0:
            print("    Warning: Empty sequence, creating minimal fallback")
            self.long_feature_sequence = np.zeros((1, 11), dtype=np.float32)
            self.long_labels = np.zeros(1, dtype=np.int64)
            self.num_iterations = 1
        else:
            # ✅ CORE REQUIREMENT: Extract features on complete sequence FIRST
            self.long_feature_sequence = self.feature_extractor.process_sequence(long_sequence)  # [N, 4] → [N, 11]
            self.long_labels = labels
            
            # Calculate number of possible iterations (sliding window)
            total_events = len(self.long_feature_sequence)
            self.num_iterations = max(1, total_events - self.sequence_length + 1)
        
        feature_time = time.time() - feature_start_time
        epoch_time = time.time() - epoch_start_time
        
        print(f"  ✅ Epoch generation complete:")
        print(f"    - Background events: {len(background_events) if len(background_events) > 0 else 0}")
        print(f"    - Flare events: {len(flare_events) if len(flare_events) > 0 else 0}")
        print(f"    - Total events: {len(long_sequence) if len(long_sequence) > 0 else 0}")
        print(f"    - Feature extraction: {feature_time:.3f}s")
        print(f"    - Available iterations: {self.num_iterations}")
        print(f"    - Total epoch time: {epoch_time:.3f}s")
        
        # Debug visualization (only for first few epochs)
        if self.event_visualizer is not None and self.current_epoch < 3:
            self._debug_visualize_epoch(background_events, flare_events, long_sequence, labels)
    
    def _generate_background_events(self) -> np.ndarray:
        """Generate random background events (0.1-0.3s duration)."""
        # Random duration
        duration_ms = random.uniform(self.min_duration_ms, self.max_duration_ms)
        duration_us = int(duration_ms * 1000)
        
        # Random DSEC sample
        idx = random.randint(0, len(self.dsec_dataset) - 1)
        background_data = self.dsec_dataset[idx]
        
        if isinstance(background_data, tuple):
            background_events = background_data[0]
        else:
            background_events = background_data
            
        if isinstance(background_events, torch.Tensor):
            background_events = background_events.numpy()
        
        # Crop to desired duration
        if len(background_events) > 0:
            t_min = background_events[:, 2].min()
            t_max = background_events[:, 2].max()
            current_duration = t_max - t_min
            
            if current_duration > duration_us:
                # Random time window within the sequence
                max_start_offset = current_duration - duration_us
                start_offset = random.uniform(0, max_start_offset)
                start_time = t_min + start_offset
                end_time = start_time + duration_us
                
                # Filter events in time window
                mask = (background_events[:, 2] >= start_time) & (background_events[:, 2] < end_time)
                background_events = background_events[mask]
        
        return background_events if len(background_events) > 0 else np.empty((0, 4))
    
    def _generate_flare_events(self) -> np.ndarray:
        """Generate random flare events (0.1-0.3s duration)."""
        # Random duration
        duration_ms = random.uniform(self.min_duration_ms, self.max_duration_ms)
        
        # Temporarily modify config for desired duration
        original_duration = self.config['data']['flare_synthesis'].get('duration_sec', 0.3)
        self.config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000.0
        
        try:
            # Generate flare events using DVS simulator
            flare_events, metadata = self.flare_generator.generate_flare_events(cleanup=True)
            
            # Handle empty generation
            if len(flare_events) == 0:
                print("    Warning: DVS simulation generated no events")
                return np.empty((0, 4))
            
            return flare_events
            
        except Exception as e:
            print(f"    Error in flare generation: {e}")
            return np.empty((0, 4))
        finally:
            # Restore original config
            self.config['data']['flare_synthesis']['duration_sec'] = original_duration
    
    def _merge_and_sort_events(self, background_events: np.ndarray, 
                              flare_events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge background and flare events with proper temporal alignment."""
        
        # Handle empty cases
        if len(background_events) == 0 and len(flare_events) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=np.int64)
        elif len(background_events) == 0:
            # Only flare events
            flare_formatted = self._format_flare_events(flare_events)
            labels = np.ones(len(flare_formatted), dtype=np.int64)
            return flare_formatted, labels
        elif len(flare_events) == 0:
            # Only background events
            labels = np.zeros(len(background_events), dtype=np.int64)
            return background_events, labels
        
        # Both exist - format flare events and merge
        flare_formatted = self._format_flare_events(flare_events)
        
        # Create labels
        bg_labels = np.zeros(len(background_events), dtype=np.int64)
        flare_labels = np.ones(len(flare_formatted), dtype=np.int64)
        
        # Merge using linear merge (O(n+m))
        combined_events, combined_labels = self._linear_merge_events(
            background_events, bg_labels, flare_formatted, flare_labels
        )
        
        # Normalize timestamps to start from 0
        if len(combined_events) > 0:
            t_min = combined_events[:, 2].min()
            combined_events[:, 2] = combined_events[:, 2] - t_min
        
        return combined_events, combined_labels
    
    def _format_flare_events(self, flare_events: np.ndarray) -> np.ndarray:
        """Convert DVS format [t, x, y, p] to project format [x, y, t, p]."""
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        # Convert format
        formatted_events = np.zeros_like(flare_events, dtype=np.float64)
        formatted_events[:, 0] = flare_events[:, 1]  # x
        formatted_events[:, 1] = flare_events[:, 2]  # y 
        formatted_events[:, 2] = flare_events[:, 0]  # t (no offset for epoch-level)
        formatted_events[:, 3] = flare_events[:, 3]  # p
        
        # Convert polarity from DVS format (1/0) to DSEC format (1/-1)
        formatted_events[:, 3] = np.where(formatted_events[:, 3] > 0, 1, -1)
        
        return formatted_events
    
    def _linear_merge_events(self, events1: np.ndarray, labels1: np.ndarray,
                           events2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Linear merge of two sorted event streams (like merge sort)."""
        if len(events1) == 0:
            return events2, labels2
        if len(events2) == 0:
            return events1, labels1
            
        # Initialize result arrays
        total_events = len(events1) + len(events2)
        merged_events = np.zeros((total_events, 4), dtype=events1.dtype)
        merged_labels = np.zeros(total_events, dtype=labels1.dtype)
        
        # Merge using dual pointers
        i, j, k = 0, 0, 0
        while i < len(events1) and j < len(events2):
            if events1[i, 2] <= events2[j, 2]:  # Compare timestamps
                merged_events[k] = events1[i]
                merged_labels[k] = labels1[i]
                i += 1
            else:
                merged_events[k] = events2[j]
                merged_labels[k] = labels2[j]
                j += 1
            k += 1
        
        # Add remaining events
        while i < len(events1):
            merged_events[k] = events1[i]
            merged_labels[k] = labels1[i]
            i += 1
            k += 1
            
        while j < len(events2):
            merged_events[k] = events2[j]
            merged_labels[k] = labels2[j]
            j += 1
            k += 1
            
        return merged_events, merged_labels
    
    def _debug_visualize_epoch(self, background_events: np.ndarray, flare_events: np.ndarray,
                              merged_events: np.ndarray, labels: np.ndarray):
        """Debug visualization for epoch-level data generation."""
        try:
            # Convert flare events to project format for visualization
            flare_formatted = self._format_flare_events(flare_events) if len(flare_events) > 0 else np.empty((0, 4))
            
            # Analyze and visualize
            self.event_visualizer.analyze_and_visualize_pipeline(
                background_events, flare_formatted, merged_events, labels, self.current_epoch
            )
        except Exception as e:
            print(f"    Warning: Debug visualization failed: {e}")
    
    def __len__(self) -> int:
        """Return number of possible iterations in current epoch."""
        if self.long_feature_sequence is None:
            return 0
        return self.num_iterations
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get iteration-level sample via sliding window.
        
        Args:
            idx: Iteration index (0 to num_iterations-1)
            
        Returns:
            Tuple of (features_tensor, labels_tensor)
            features_tensor: [sequence_length, 11] - fixed-length feature window
            labels_tensor: [sequence_length] - corresponding labels
        """
        # Ensure epoch data is generated
        if self.long_feature_sequence is None:
            self.new_epoch()
        
        # Handle edge cases
        if len(self.long_feature_sequence) == 0:
            # Return empty tensors
            empty_features = torch.zeros((self.sequence_length, 11), dtype=torch.float32)
            empty_labels = torch.zeros(self.sequence_length, dtype=torch.long)
            return empty_features, empty_labels
        
        # Sliding window sampling
        total_features = len(self.long_feature_sequence)
        
        if total_features <= self.sequence_length:
            # Sequence is shorter than required - pad with zeros
            features = np.zeros((self.sequence_length, 11), dtype=np.float32)
            labels = np.zeros(self.sequence_length, dtype=np.int64)
            
            features[:total_features] = self.long_feature_sequence
            labels[:total_features] = self.long_labels
        else:
            # Sliding window - sequential, non-overlapping sampling
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            
            # Clamp to valid range
            if end_idx > total_features:
                start_idx = total_features - self.sequence_length
                end_idx = total_features
            
            features = self.long_feature_sequence[start_idx:end_idx]
            labels = self.long_labels[start_idx:end_idx]
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return features_tensor, labels_tensor


class EpochIterationDataLoader:
    """
    Custom DataLoader implementing Epoch-Iteration architecture.
    
    Handles epoch regeneration and provides standard DataLoader interface.
    """
    
    def __init__(self, dataset: EpochIterationDataset, batch_size: int = 2, 
                 shuffle: bool = True, num_workers: int = 0):
        """Initialize Epoch-Iteration DataLoader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # Generate initial epoch
        self.dataset.new_epoch()
        
        # Calculate batches per epoch
        self.batches_per_epoch = max(1, len(self.dataset) // batch_size)
        
        print(f"EpochIterationDataLoader initialized:")
        print(f"  - Iterations per epoch: {len(self.dataset)}")
        print(f"  - Batches per epoch: {self.batches_per_epoch}")
        print(f"  - Batch size: {batch_size}")
    
    def __len__(self):
        """Return number of batches per epoch."""
        return self.batches_per_epoch
    
    def __iter__(self):
        """Iterate through batches, regenerating epoch data when needed."""
        # Generate new epoch data
        self.dataset.new_epoch()
        
        # Update batch count
        self.batches_per_epoch = max(1, len(self.dataset) // self.batch_size)
        
        # Create iteration indices
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        # Yield batches
        batch_count = 0
        for i in range(0, len(indices), self.batch_size):
            if batch_count >= self.batches_per_epoch:
                break
                
            batch_indices = indices[i:i + self.batch_size]
            
            # Collect batch data
            batch_features = []
            batch_labels = []
            
            for idx in batch_indices:
                features, labels = self.dataset[idx]
                batch_features.append(features)
                batch_labels.append(labels)
            
            # Stack into batch tensors
            batch_features_tensor = torch.stack(batch_features)  # [batch_size, sequence_length, 11]
            batch_labels_tensor = torch.stack(batch_labels)      # [batch_size, sequence_length]
            
            yield batch_features_tensor, batch_labels_tensor
            batch_count += 1


def create_epoch_iteration_dataloaders(config: Dict) -> Tuple[EpochIterationDataLoader, EpochIterationDataLoader, EpochIterationDataLoader]:
    """Create Epoch-Iteration DataLoaders for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    
    # Create datasets with epoch-iteration architecture
    train_dataset = EpochIterationDataset(config, split='train')
    val_dataset = EpochIterationDataset(config, split='val')  
    test_dataset = EpochIterationDataset(config, split='test')
    
    # Create custom DataLoaders
    train_loader = EpochIterationDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    val_loader = EpochIterationDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    test_loader = EpochIterationDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print(f"Created Epoch-Iteration dataloaders:")
    print(f"  Train: {len(train_loader)} batches per epoch")
    print(f"  Val: {len(val_loader)} batches per epoch")
    print(f"  Test: {len(test_loader)} batches per epoch")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the Epoch-Iteration dataset
    import yaml
    
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug mode for testing
    config['debug_mode'] = True
    config['data']['max_samples_debug'] = 4
    
    print("Testing Epoch-Iteration Dataset...")
    
    # Create dataset
    dataset = EpochIterationDataset(config, split='train')
    
    # Test epoch generation
    print("\nTesting epoch generation...")
    dataset.new_epoch()
    
    print(f"Generated epoch with {len(dataset)} iterations")
    
    # Test iteration sampling
    print("\nTesting iteration sampling...")
    for i in range(min(3, len(dataset))):
        features, labels = dataset[i]
        print(f"  Iteration {i}: features {features.shape}, labels {labels.shape}")
        print(f"    Background events: {torch.sum(labels == 0).item()}")
        print(f"    Flare events: {torch.sum(labels == 1).item()}")
    
    # Test DataLoader
    print("\nTesting DataLoader...")
    train_loader, _, _ = create_epoch_iteration_dataloaders(config)
    
    batch_count = 0
    for batch_features, batch_labels in train_loader:
        print(f"  Batch {batch_count}: {batch_features.shape}, {batch_labels.shape}")
        batch_count += 1
        if batch_count >= 2:  # Test first 2 batches
            break
    
    print("\n✅ Epoch-Iteration Dataset test completed!")