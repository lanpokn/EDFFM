"""
Real-time Evaluator for EventMamba-FX
Implements time-window based evaluation as specified:
- Read 0.3s windows sequentially
- Run PFD feature extraction
- Pass to network for flare probability
- Output denoised events (remove flare events)
- Maintain temporal continuity
"""

import os
import time
import numpy as np
import torch
from typing import List, Tuple, Dict
import h5py

from src.feature_extractor import FeatureExtractor
from src.model import EventDenoisingMamba


class RealtimeEventEvaluator:
    """
    Real-time evaluator that processes events in time windows:
    1. Read 0.3s time windows sequentially
    2. Extract PFD features for each window
    3. Use network to predict flare probabilities
    4. Output denoised events (temporal continuity preserved)
    """
    
    def __init__(self, model_path: str, config: Dict, device: str = 'cuda'):
        """Initialize real-time evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config: Configuration dictionary
            device: Device for inference
        """
        self.config = config
        self.device = device
        self.time_window_us = 300000  # 0.3 seconds in microseconds
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config)
        
        # Initialize and load model
        self.model = EventDenoisingMamba(config).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        print(f"Initialized RealtimeEventEvaluator:")
        print(f"  Model: {model_path}")
        print(f"  Device: {device}")
        print(f"  Time window: {self.time_window_us/1e6:.1f}s")
        print(f"  Feature dimensions: {config['model']['input_feature_dim']}")
        
    def evaluate_event_stream(self, input_file: str, output_file: str, 
                            flare_threshold: float = 0.5) -> Dict:
        """Evaluate an event stream with real-time processing.
        
        Args:
            input_file: Input H5 file path
            output_file: Output H5 file path for denoised events
            flare_threshold: Threshold for flare detection (0.5 = 50%)
            
        Returns:
            Dictionary with evaluation statistics
        """
        print(f"\n🚀 Starting real-time evaluation:")
        print(f"  Input: {input_file}")
        print(f"  Output: {output_file}")
        print(f"  Flare threshold: {flare_threshold}")
        
        start_time = time.time()
        
        # Load input events
        events = self._load_events(input_file)
        print(f"  Loaded {len(events):,} events")
        
        if len(events) == 0:
            print("❌ No events to process")
            return {'processed_events': 0, 'denoised_events': 0}
        
        # Process in time windows
        denoised_events, stats = self._process_time_windows(events, flare_threshold)
        
        # Save denoised events
        self._save_events(denoised_events, output_file)
        
        total_time = time.time() - start_time
        
        # Print results
        print(f"\n✅ Real-time evaluation completed:")
        print(f"  Processing time: {total_time:.2f}s")
        print(f"  Input events: {len(events):,}")
        print(f"  Output events: {len(denoised_events):,}")
        print(f"  Removed events: {len(events) - len(denoised_events):,}")
        print(f"  Removal rate: {(len(events) - len(denoised_events))/len(events)*100:.1f}%")
        print(f"  Processing speed: {len(events)/total_time:.0f} events/sec")
        
        return {
            'input_events': len(events),
            'output_events': len(denoised_events),
            'removed_events': len(events) - len(denoised_events),
            'removal_rate': (len(events) - len(denoised_events))/len(events),
            'processing_time': total_time,
            'events_per_second': len(events)/total_time,
            **stats
        }
    
    def _load_events(self, input_file: str) -> np.ndarray:
        """Load events from H5 file.
        
        Args:
            input_file: Path to H5 file
            
        Returns:
            Events array [N, 4] with [x, y, t, p] format
        """
        try:
            with h5py.File(input_file, 'r') as f:
                x = f['events/x'][:]
                y = f['events/y'][:]
                t = f['events/t'][:]
                p = f['events/p'][:]
                
                # Convert polarity format if needed
                if np.any(p > 1):  # DSEC format (0/255)
                    p = np.where(p > 0, 1, -1)
                
                # Stack into [x, y, t, p] format
                events = np.column_stack([x, y, t, p]).astype(np.float32)
                
                return events
                
        except Exception as e:
            print(f"❌ Error loading events: {e}")
            return np.empty((0, 4), dtype=np.float32)
    
    def _process_time_windows(self, events: np.ndarray, 
                            flare_threshold: float) -> Tuple[np.ndarray, Dict]:
        """Process events in sequential time windows.
        
        Args:
            events: Input events [N, 4]
            flare_threshold: Threshold for flare detection
            
        Returns:
            Tuple of (denoised_events, statistics)
        """
        if len(events) == 0:
            return events, {}
        
        # Sort events by timestamp
        events = events[np.argsort(events[:, 2])]
        
        # Calculate time windows
        t_min = events[0, 2]
        t_max = events[-1, 2]
        total_duration = t_max - t_min
        
        num_windows = int(np.ceil(total_duration / self.time_window_us))
        
        print(f"  Time span: {total_duration/1e6:.2f}s")
        print(f"  Processing {num_windows} windows of {self.time_window_us/1e6:.1f}s each")
        
        denoised_events_list = []
        window_stats = []
        
        for window_idx in range(num_windows):
            window_start = t_min + window_idx * self.time_window_us
            window_end = window_start + self.time_window_us
            
            # Extract events in this time window
            mask = (events[:, 2] >= window_start) & (events[:, 2] < window_end)
            window_events = events[mask]
            
            if len(window_events) == 0:
                continue
            
            # Process this window
            denoised_window, window_stat = self._process_single_window(
                window_events, flare_threshold, window_idx
            )
            
            denoised_events_list.append(denoised_window)
            window_stats.append(window_stat)
            
            # Progress update
            if (window_idx + 1) % 10 == 0 or window_idx == num_windows - 1:
                processed_events = sum(s['input_events'] for s in window_stats)
                print(f"    Processed {window_idx+1}/{num_windows} windows, "
                      f"{processed_events:,} events")
        
        # Combine all denoised events
        if denoised_events_list:
            denoised_events = np.vstack(denoised_events_list)
        else:
            denoised_events = np.empty((0, 4), dtype=np.float32)
        
        # Aggregate statistics
        stats = {
            'num_windows': num_windows,
            'avg_events_per_window': np.mean([s['input_events'] for s in window_stats]),
            'avg_flare_rate': np.mean([s['flare_rate'] for s in window_stats]),
            'processing_time_per_window': np.mean([s['processing_time'] for s in window_stats])
        }
        
        return denoised_events, stats
    
    def _process_single_window(self, window_events: np.ndarray, 
                             flare_threshold: float, window_idx: int) -> Tuple[np.ndarray, Dict]:
        """Process a single time window.
        
        Args:
            window_events: Events in this window [N, 4]
            flare_threshold: Flare detection threshold
            window_idx: Window index for logging
            
        Returns:
            Tuple of (denoised_events, window_statistics)
        """
        window_start_time = time.time()
        
        # 1. Extract PFD features for this window
        features = self.feature_extractor.process_sequence(window_events)
        
        # 2. Run network inference
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            feature_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Get flare probabilities
            probabilities = self.model(feature_tensor)  # [1, N, 1]
            prob_values = probabilities.squeeze().cpu().numpy()  # [N]
            
            # Apply threshold to identify flare events
            is_flare = prob_values > flare_threshold
            
            # Keep only non-flare events (temporal continuity preserved)
            denoised_events = window_events[~is_flare]
        
        processing_time = time.time() - window_start_time
        
        # Statistics for this window
        stats = {
            'window_idx': window_idx,
            'input_events': len(window_events),
            'output_events': len(denoised_events),
            'removed_events': len(window_events) - len(denoised_events),
            'flare_rate': np.sum(is_flare) / len(window_events) if len(window_events) > 0 else 0,
            'processing_time': processing_time
        }
        
        return denoised_events, stats
    
    def _save_events(self, events: np.ndarray, output_file: str):
        """Save denoised events to H5 file.
        
        Args:
            events: Denoised events [N, 4]
            output_file: Output file path
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with h5py.File(output_file, 'w') as f:
                # Create events group
                events_group = f.create_group('events')
                
                if len(events) > 0:
                    events_group.create_dataset('x', data=events[:, 0].astype(np.uint16))
                    events_group.create_dataset('y', data=events[:, 1].astype(np.uint16))
                    events_group.create_dataset('t', data=events[:, 2].astype(np.int64))
                    events_group.create_dataset('p', data=events[:, 3].astype(np.int8))
                else:
                    # Empty datasets
                    events_group.create_dataset('x', data=np.array([], dtype=np.uint16))
                    events_group.create_dataset('y', data=np.array([], dtype=np.uint16))
                    events_group.create_dataset('t', data=np.array([], dtype=np.int64))
                    events_group.create_dataset('p', data=np.array([], dtype=np.int8))
                
                print(f"  Saved {len(events):,} denoised events to {output_file}")
                
        except Exception as e:
            print(f"❌ Error saving events: {e}")


def create_realtime_evaluator(model_path: str, config: Dict, device: str = 'cuda'):
    """Factory function to create real-time evaluator."""
    return RealtimeEventEvaluator(model_path, config, device)


def test_realtime_evaluation(config_path: str = "configs/config.yaml",
                           model_path: str = "./checkpoints/best_model.pth"):
    """Test real-time evaluation functionality."""
    import yaml
    
    print("🧪 Testing Real-time Event Evaluation")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first by running: python main.py --config configs/config.yaml")
        return False
    
    # Create evaluator
    evaluator = create_realtime_evaluator(model_path, config)
    
    # For testing, we'll create synthetic events
    print("\n🔧 Creating synthetic test events...")
    
    # Generate synthetic mixed events (background + flare)
    num_bg_events = 10000
    num_flare_events = 2000
    duration_us = 1000000  # 1 second
    
    # Background events (should be kept)
    bg_events = np.random.rand(num_bg_events, 4).astype(np.float32)
    bg_events[:, 0] *= 639  # x
    bg_events[:, 1] *= 479  # y  
    bg_events[:, 2] *= duration_us  # t
    bg_events[:, 3] = np.random.choice([-1, 1], num_bg_events)  # p
    
    # Flare events (should be removed) - clustered in space and time
    flare_events = np.random.rand(num_flare_events, 4).astype(np.float32)
    flare_events[:, 0] = 200 + flare_events[:, 0] * 200  # x: clustered around 200-400
    flare_events[:, 1] = 150 + flare_events[:, 1] * 150  # y: clustered around 150-300
    flare_events[:, 2] = 300000 + flare_events[:, 2] * 400000  # t: clustered in middle
    flare_events[:, 3] = np.random.choice([-1, 1], num_flare_events)  # p
    
    # Combine and sort
    all_events = np.vstack([bg_events, flare_events])
    all_events = all_events[np.argsort(all_events[:, 2])]
    
    # Save test events
    test_input = "./test_input_events.h5"
    test_output = "./test_output_events.h5"
    
    with h5py.File(test_input, 'w') as f:
        events_group = f.create_group('events')
        events_group.create_dataset('x', data=all_events[:, 0].astype(np.uint16))
        events_group.create_dataset('y', data=all_events[:, 1].astype(np.uint16))
        events_group.create_dataset('t', data=all_events[:, 2].astype(np.int64))
        events_group.create_dataset('p', data=all_events[:, 3].astype(np.int8))
    
    print(f"  Created {len(all_events):,} synthetic events")
    print(f"  Background: {num_bg_events:,}, Flare: {num_flare_events:,}")
    
    # Run evaluation
    stats = evaluator.evaluate_event_stream(test_input, test_output, flare_threshold=0.3)
    
    # Cleanup test files
    os.remove(test_input)
    if os.path.exists(test_output):
        os.remove(test_output)
    
    print(f"\n✅ Real-time evaluation test completed!")
    print(f"  Processed {stats['input_events']:,} events")
    print(f"  Removed {stats['removed_events']:,} events ({stats['removal_rate']*100:.1f}%)")
    print(f"  Speed: {stats['events_per_second']:.0f} events/sec")
    
    return True


if __name__ == "__main__":
    # Run test
    test_realtime_evaluation()