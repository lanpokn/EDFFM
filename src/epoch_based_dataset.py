"""
Epoch-Based Dataset for EventMamba-FX
Implements the correct "Epoch vs. Iteration" architecture:
- Epoch: Generate long_sequence and extract features once
- Iteration: Sliding window sampling from pre-computed features
"""

import os
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, List
import psutil
import gc

# Import existing components
from src.dsec_efficient import DSECEventDatasetEfficient
from src.dvs_flare_integration import DVSFlareEventGenerator
from src.feature_extractor import FeatureExtractor


class EpochBasedEventDataset(Dataset):
    """
    Dataset implementing correct Epoch-based architecture:
    - Epoch-level: Data generation + feature extraction on long sequences
    - Iteration-level: Sliding window sampling from pre-computed features
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """Initialize epoch-based dataset.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # Memory monitoring
        self.process = psutil.Process()
        self.max_memory_mb = config['data'].get('max_memory_mb', 4000)  # 4GB limit
        
        # Initialize DSEC background loader
        self.dsec_dataset = DSECEventDatasetEfficient(
            dsec_path=config['data']['dsec_path'],
            flare_path="",  # Not used for background
            time_window_us=config['data']['time_window_us'],
            sequence_length=config['data']['sequence_length'],
            max_files=3  # Limit files for memory safety
        )
        
        # Initialize flare event generator
        self.flare_generator = DVSFlareEventGenerator(config)
        
        # Initialize PFD feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Epoch-level data storage
        self.current_epoch_data = None
        self.current_epoch_features = None
        self.current_epoch_labels = None
        self.epoch_counter = -1
        
        # Iteration parameters
        self.sequence_length = config['data']['sequence_length']
        self.sliding_window_step = max(1, self.sequence_length // 4)  # 75% overlap
        
        print(f"Initialized EpochBasedEventDataset ({split})")
        print(f"  DSEC samples: {len(self.dsec_dataset)}")
        print(f"  Memory limit: {self.max_memory_mb} MB")
        print(f"  Sliding window step: {self.sliding_window_step}")
    
    def generate_epoch_data(self, epoch: int):
        """Generate data for a new epoch (Epoch-level operation)."""
        print(f"\n=== EPOCH {epoch} DATA GENERATION ===")
        start_time = time.time()
        
        try:
            # 1. Generate randomized event data
            long_sequence, labels = self._generate_long_sequence_with_flare()
            
            # 2. Extract PFD features on the complete long sequence
            print(f"Extracting PFD features on sequence of {len(long_sequence)} events...")
            long_feature_sequence = self.feature_extractor.process_sequence(long_sequence)
            
            # 3. Store epoch data
            self.current_epoch_data = long_sequence
            self.current_epoch_features = long_feature_sequence
            self.current_epoch_labels = labels
            self.epoch_counter = epoch
            
            generation_time = time.time() - start_time
            print(f"Epoch {epoch} data generation completed:")
            print(f"  Events: {len(long_sequence):,}")
            print(f"  Features: {long_feature_sequence.shape}")
            print(f"  Labels: {len(labels):,} (Background: {np.sum(labels==0):,}, Flare: {np.sum(labels==1):,})")
            print(f"  Time: {generation_time:.2f}s")
                    
        except Exception as e:
            print(f"❌ ERROR in epoch data generation: {e}")
            print("Falling back to safe minimal data...")
            self._generate_safe_fallback_data()
    
    def _generate_long_sequence_with_flare(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate long sequence combining background and flare events."""
        config = self.config['data']['randomized_training']
        
        # 1. Random scenario selection  
        scenario = self._select_random_scenario()
        print(f"Selected scenario: {scenario}")
        
        # 2. Generate background events
        background_events = np.empty((0, 4))
        if scenario in ["background_with_flare", "background_only"]:
            background_events = self._generate_random_background_events(config)
            background_events = self._validate_event_coordinates(background_events, "background_events")
            print(f"Generated background events: {len(background_events):,}")
        
        # 3. Generate flare events
        flare_events = np.empty((0, 4))
        if scenario in ["background_with_flare", "flare_only"]:
            flare_events = self._generate_random_flare_events(config)
            flare_events = self._validate_event_coordinates(flare_events, "flare_events")
            print(f"Generated flare events: {len(flare_events):,}")
        
        # 4. Apply random offsets and merge
        combined_events, labels = self._apply_random_offsets_and_merge(
            background_events, flare_events, config
        )
        
        # 5. Apply final processing
        combined_events, labels = self._apply_final_processing(combined_events, labels)
        
        # 6. Final coordinate validation
        combined_events = self._validate_event_coordinates(combined_events, "combined_events")
        
        return combined_events, labels
    
    def _select_random_scenario(self) -> str:
        """Select random training scenario."""
        rand_val = random.random()
        config = self.config['data']['randomized_training']
        
        if rand_val < config['background_contains_flare_prob']:
            return "background_with_flare"  # 75%
        elif rand_val < config['background_contains_flare_prob'] + config['flare_only_prob']:
            return "flare_only"  # 10%
        else:
            return "background_only"  # 15%
    
    def _generate_random_background_events(self, config: Dict) -> np.ndarray:
        """Generate random background events from DSEC."""
        bg_duration = random.uniform(*config['background_duration_range'])
        dsec_idx = random.randint(0, len(self.dsec_dataset) - 1)
        background_data = self.dsec_dataset[dsec_idx]
        
        if isinstance(background_data, tuple):
            background_events = background_data[0]
        else:
            background_events = background_data
            
        if isinstance(background_events, torch.Tensor):
            background_events = background_events.numpy()
        
        # Truncate to desired duration
        if len(background_events) > 0:
            bg_duration_us = bg_duration * 1e6
            t_min = background_events[:, 2].min()
            t_max = background_events[:, 2].max()
            current_duration = t_max - t_min
            
            if current_duration > bg_duration_us:
                max_start_offset = current_duration - bg_duration_us
                start_offset = random.uniform(0, max_start_offset)
                start_time = t_min + start_offset
                end_time = start_time + bg_duration_us
                
                mask = (background_events[:, 2] >= start_time) & (background_events[:, 2] < end_time)
                background_events = background_events[mask]
        
        return background_events
    
    def _generate_random_flare_events(self, config: Dict) -> np.ndarray:
        """Generate random flare events using DVS simulation."""
        flare_duration = random.uniform(*config['flare_duration_range'])
        
        # Temporarily modify config
        original_duration = self.config['data']['flare_synthesis'].get('duration_sec', 0.3)
        self.config['data']['flare_synthesis']['duration_sec'] = flare_duration
        
        try:
            flare_events, _ = self.flare_generator.generate_flare_events(cleanup=True)
            
            # Safety check: limit max frames
            if len(flare_events) > config['max_flare_frames']:
                print(f"Warning: Flare events ({len(flare_events)}) exceed limit, sampling...")
                indices = np.random.choice(len(flare_events), config['max_flare_frames'], replace=False)
                indices = np.sort(indices)
                flare_events = flare_events[indices]
                
        except Exception as e:
            print(f"❌ CRITICAL ERROR in flare generation: {e}")
            print("FALLBACK DISABLED - DVS simulation must work!")
            print("Detailed error information:")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"DVS simulation failed and fallback is disabled: {e}")
        finally:
            self.config['data']['flare_synthesis']['duration_sec'] = original_duration
        
        return flare_events
    
    def _validate_event_coordinates(self, events: np.ndarray, event_type: str) -> np.ndarray:
        """Validate and fix event coordinates to be within resolution bounds.
        
        Args:
            events: Event array [N, 4] with [x, y, t, p] format
            event_type: Type description for logging
            
        Returns:
            Validated events with coordinates clamped to resolution
        """
        if len(events) == 0:
            return events
            
        # Get resolution from config
        max_x = self.config['data']['resolution_w'] - 1  # 639 for 640x480
        max_y = self.config['data']['resolution_h'] - 1  # 479 for 640x480
        
        # Check for invalid coordinates
        invalid_x = (events[:, 0] < 0) | (events[:, 0] > max_x)
        invalid_y = (events[:, 1] < 0) | (events[:, 1] > max_y)
        
        if np.any(invalid_x) or np.any(invalid_y):
            print(f"⚠️  COORDINATE BUG DETECTED in {event_type}:")
            print(f"   Resolution: {max_x+1}x{max_y+1}")
            print(f"   Invalid X coords: {np.sum(invalid_x)} events")
            print(f"   Invalid Y coords: {np.sum(invalid_y)} events")
            print(f"   X range: [{events[:, 0].min():.1f}, {events[:, 0].max():.1f}]")
            print(f"   Y range: [{events[:, 1].min():.1f}, {events[:, 1].max():.1f}]")
            
            # Clamp coordinates to valid range
            events[:, 0] = np.clip(events[:, 0], 0, max_x)
            events[:, 1] = np.clip(events[:, 1], 0, max_y)
            print(f"   ✅ Coordinates clamped to valid range")
        
        return events
    
    
    def _apply_random_offsets_and_merge(self, background_events: np.ndarray, 
                                      flare_events: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random temporal offsets and merge events."""
        # Apply background offset
        if len(background_events) > 0:
            bg_offset = random.uniform(*config['background_offset_range']) * 1e6
            background_events = background_events.copy()
            background_events[:, 2] += bg_offset
        
        # Apply flare offset and format conversion
        if len(flare_events) > 0:
            flare_offset = random.uniform(*config['flare_offset_range']) * 1e6
            flare_formatted = self._format_flare_events(flare_events, flare_offset)
        else:
            flare_formatted = np.empty((0, 4))
        
        # Merge events
        if len(background_events) == 0 and len(flare_formatted) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=np.int64)
        elif len(background_events) == 0:
            labels = np.ones(len(flare_formatted), dtype=np.int64)
            return flare_formatted, labels
        elif len(flare_formatted) == 0:
            labels = np.zeros(len(background_events), dtype=np.int64)
            return background_events, labels
        else:
            bg_labels = np.zeros(len(background_events), dtype=np.int64)
            flare_labels = np.ones(len(flare_formatted), dtype=np.int64)
            
            combined_events, combined_labels = self._merge_sorted_events(
                background_events, bg_labels, flare_formatted, flare_labels
            )
            return combined_events, combined_labels
    
    def _format_flare_events(self, flare_events: np.ndarray, time_offset: float) -> np.ndarray:
        """Format flare events from DVS [t,x,y,p] to EventMamba [x,y,t,p] format."""
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        formatted_events = np.zeros_like(flare_events, dtype=np.float64)
        formatted_events[:, 0] = flare_events[:, 1]  # x
        formatted_events[:, 1] = flare_events[:, 2]  # y
        formatted_events[:, 2] = flare_events[:, 0] + time_offset  # t + offset
        formatted_events[:, 3] = flare_events[:, 3]  # p
        
        # Convert polarity format if needed
        unique_polarities = np.unique(formatted_events[:, 3])
        if np.any(unique_polarities > 1):
            formatted_events[:, 3] = np.where(formatted_events[:, 3] > 0, 1, -1)
        
        return formatted_events
    
    def _merge_sorted_events(self, events1: np.ndarray, labels1: np.ndarray,
                           events2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two sorted event streams using linear merge algorithm."""
        if len(events1) == 0:
            return events2, labels2
        if len(events2) == 0:
            return events1, labels1
            
        i, j = 0, 0
        total_events = len(events1) + len(events2)
        merged_events = np.zeros((total_events, 4), dtype=events1.dtype)
        merged_labels = np.zeros(total_events, dtype=labels1.dtype)
        k = 0
        
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
    
    def _apply_final_processing(self, combined_events: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply final processing: duration limits and safety checks."""
        if len(combined_events) == 0:
            return combined_events, labels
        
        config = self.config['data']['randomized_training']
        
        # Random final duration
        final_duration = random.uniform(*config['final_duration_range'])
        final_duration_us = min(final_duration * 1e6, config['max_total_duration'] * 1e6)
        
        # Truncate to final duration
        t_min = combined_events[:, 2].min()
        t_max = combined_events[:, 2].max()
        current_duration = t_max - t_min
        
        if current_duration > final_duration_us:
            max_start_offset = current_duration - final_duration_us
            start_offset = random.uniform(0, max_start_offset)
            start_time = t_min + start_offset
            end_time = start_time + final_duration_us
            
            mask = (combined_events[:, 2] >= start_time) & (combined_events[:, 2] < end_time)
            combined_events = combined_events[mask]
            labels = labels[mask]
        
        # Limit total events for memory safety
        max_events = 50000  # Hard limit
        if len(combined_events) > max_events:
            print(f"Warning: Truncating {len(combined_events)} events to {max_events} for memory safety")
            indices = np.linspace(0, len(combined_events)-1, max_events, dtype=int)
            combined_events = combined_events[indices]
            labels = labels[indices]
        
        return combined_events, labels
    
    def _generate_safe_fallback_data(self):
        """Generate minimal safe data as fallback."""
        print("Generating safe fallback data...")
        
        num_events = min(1000, self.sequence_length * 10)
        events = np.random.rand(num_events, 4).astype(np.float32)
        events[:, 0] *= 640  # x
        events[:, 1] *= 480  # y
        events[:, 2] *= 1000000  # t (1 second)
        events[:, 3] = np.random.choice([-1, 1], num_events)  # p
        
        events = events[np.argsort(events[:, 2])]  # Sort by time
        
        features = self.feature_extractor.process_sequence(events)
        labels = np.zeros(len(events), dtype=np.int64)
        
        self.current_epoch_data = events
        self.current_epoch_features = features
        self.current_epoch_labels = labels
    
    def __len__(self):
        """Return number of iterations per epoch (sliding windows)."""
        if self.current_epoch_features is None:
            return 100  # Default before first epoch
        
        total_length = len(self.current_epoch_features)
        if total_length <= self.sequence_length:
            return 1
        
        num_windows = (total_length - self.sequence_length) // self.sliding_window_step + 1
        return max(1, num_windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get iteration-level sample (sliding window from epoch features)."""
        if self.current_epoch_features is None:
            print("Warning: No epoch data available, generating fallback...")
            self._generate_safe_fallback_data()
        
        # Calculate sliding window boundaries
        start_idx = idx * self.sliding_window_step
        end_idx = start_idx + self.sequence_length
        
        # Handle boundary cases
        if end_idx > len(self.current_epoch_features):
            start_idx = max(0, len(self.current_epoch_features) - self.sequence_length)
            end_idx = len(self.current_epoch_features)
        
        if start_idx >= len(self.current_epoch_features):
            start_idx = 0
            end_idx = min(self.sequence_length, len(self.current_epoch_features))
        
        # Extract sliding window
        features = self.current_epoch_features[start_idx:end_idx]
        labels = self.current_epoch_labels[start_idx:end_idx]
        
        # Pad if necessary
        if len(features) < self.sequence_length:
            padding_size = self.sequence_length - len(features)
            feature_dim = features.shape[1] if len(features) > 0 else 11  # 11D after removing last 2
            
            feature_padding = np.zeros((padding_size, feature_dim), dtype=features.dtype if len(features) > 0 else np.float32)
            label_padding = np.zeros(padding_size, dtype=labels.dtype if len(labels) > 0 else np.int64)
            
            if len(features) > 0:
                features = np.vstack([features, feature_padding])
                labels = np.concatenate([labels, label_padding])
            else:
                features = feature_padding
                labels = label_padding
        
        # Convert to tensors
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return feature_tensor, labels_tensor