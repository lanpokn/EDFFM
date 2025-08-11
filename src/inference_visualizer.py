#!/usr/bin/env python3
"""
Inference Event Visualizer for EventMamba-FX
Creates 1000 visualization frames at 1ms intervals for inference results.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import hdf5plugin
from tqdm import tqdm
from typing import Tuple, Optional

class InferenceEventVisualizer:
    """Creates temporal visualizations of event data at 1ms intervals."""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        """
        Args:
            resolution: Image resolution (width, height)
        """
        self.width, self.height = resolution
        
        # Set matplotlib to use non-interactive backend
        plt.switch_backend('Agg')
        
    def visualize_inference_comparison(self, 
                                     original_h5_path: str,
                                     clean_h5_path: str, 
                                     output_dir: str,
                                     time_limit_us: int = 100_000):  # 100ms default
        """
        Create 1000 comparison visualizations (1ms each) showing original vs cleaned events.
        
        Args:
            original_h5_path: Path to original H5 file
            clean_h5_path: Path to cleaned H5 file  
            output_dir: Output directory for visualizations
            time_limit_us: Time limit in microseconds (default: 1 second)
        """
        print(f"\n📊 Creating inference comparison visualizations...")
        print(f"  - Original file: {original_h5_path}")
        print(f"  - Clean file: {clean_h5_path}")
        print(f"  - Output dir: {output_dir}")
        print(f"  - Time limit: {time_limit_us/1000:.0f}ms")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        original_dir = os.path.join(output_dir, "original")
        clean_dir = os.path.join(output_dir, "clean") 
        comparison_dir = os.path.join(output_dir, "comparison")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Load event data
        print("📥 Loading event data...")
        original_events = self._load_events_in_time_range(original_h5_path, time_limit_us)
        clean_events = self._load_events_in_time_range(clean_h5_path, time_limit_us)
        
        if len(original_events) == 0:
            print("❌ No events found in the specified time range!")
            return
            
        print(f"  - Original events: {len(original_events):,}")
        print(f"  - Clean events: {len(clean_events):,}")
        print(f"  - Removed events: {len(original_events) - len(clean_events):,}")
        
        # Get time range
        start_time = original_events[0, 2]  # First timestamp
        end_time = start_time + time_limit_us
        
        # Create 100 frames at 1ms intervals for 100ms
        num_frames = min(1000, int(time_limit_us / 1000))  # 1 frame per ms
        time_intervals = np.linspace(start_time, end_time, num_frames + 1)
        
        print(f"🎬 Creating {num_frames} visualization frames...")
        for i in tqdm(range(num_frames), desc="Generating frames"):
            t_start = time_intervals[i]
            t_end = time_intervals[i + 1]
            
            # Filter events for this time window
            original_window = self._filter_events_by_time(original_events, t_start, t_end)
            clean_window = self._filter_events_by_time(clean_events, t_start, t_end)
            
            # Create visualizations
            frame_name = f"frame_{i:04d}"
            
            # Original events visualization
            self._create_single_frame_visualization(
                original_window, 
                os.path.join(original_dir, f"{frame_name}.png"),
                f"Original Events - Frame {i+1}/1000",
                t_start, t_end, start_time
            )
            
            # Clean events visualization
            self._create_single_frame_visualization(
                clean_window,
                os.path.join(clean_dir, f"{frame_name}.png"), 
                f"Clean Events - Frame {i+1}/1000",
                t_start, t_end, start_time
            )
            
            # Side-by-side comparison
            self._create_comparison_visualization(
                original_window, clean_window,
                os.path.join(comparison_dir, f"{frame_name}.png"),
                f"Inference Comparison - Frame {i+1}/1000", 
                t_start, t_end, start_time
            )
        
        print(f"✅ Visualization complete! {num_frames*3} total frames saved to: {output_dir}")
        print(f"  - Original frames: {original_dir}")
        print(f"  - Clean frames: {clean_dir}")  
        print(f"  - Comparison frames: {comparison_dir}")
        
    def _load_events_in_time_range(self, h5_path: str, time_limit_us: int) -> np.ndarray:
        """Load events within the specified time range."""
        with h5py.File(h5_path, 'r') as f:
            # Get first timestamp to establish time reference
            first_timestamp = f['events/t'][0]
            cutoff_timestamp = first_timestamp + time_limit_us
            
            # Find cutoff index using binary search
            timestamps = f['events/t']
            cutoff_idx = self._find_cutoff_index(timestamps, cutoff_timestamp)
            
            # Load events up to cutoff
            if cutoff_idx > 0:
                x = f['events/x'][:cutoff_idx]
                y = f['events/y'][:cutoff_idx] 
                t = f['events/t'][:cutoff_idx]
                p = f['events/p'][:cutoff_idx]
                
                # Convert polarity to 1/-1 format
                p = np.where(p > 0, 1, -1)
                
                # Stack into [N, 4] format
                events = np.column_stack([x, y, t, p]).astype(np.float64)
                return events
            else:
                return np.array([]).reshape(0, 4)
                
    def _find_cutoff_index(self, timestamps, cutoff_timestamp):
        """Binary search to find cutoff index."""
        left, right = 0, len(timestamps) - 1
        result = len(timestamps)
        
        while left <= right:
            mid = (left + right) // 2
            if timestamps[mid] <= cutoff_timestamp:
                left = mid + 1
            else:
                result = mid
                right = mid - 1
                
        return result
        
    def _filter_events_by_time(self, events: np.ndarray, t_start: float, t_end: float) -> np.ndarray:
        """Filter events within time window."""
        if len(events) == 0:
            return events
            
        timestamps = events[:, 2]
        mask = (timestamps >= t_start) & (timestamps < t_end)
        return events[mask]
        
    def _create_single_frame_visualization(self, events: np.ndarray, output_path: str, 
                                         title: str, t_start: float, t_end: float, 
                                         reference_time: float):
        """Create visualization for a single time window."""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
        ax.set_facecolor('black')
        
        if len(events) > 0:
            x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
            
            # Separate positive and negative events
            pos_mask = p > 0
            neg_mask = p <= 0
            
            # Plot events with bright colors on black background
            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c='red', s=2, alpha=0.8, 
                          label=f'ON ({np.sum(pos_mask)})')
                          
            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c='cyan', s=2, alpha=0.8,
                          label=f'OFF ({np.sum(neg_mask)})')
        else:
            ax.text(self.width/2, self.height/2, 'No events', 
                   ha='center', va='center', color='white', fontsize=14)
        
        # Set plot properties
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)  # Flip Y axis to match image coordinates
        ax.set_xlabel('X (pixels)', color='white')
        ax.set_ylabel('Y (pixels)', color='white')
        
        # Time info
        time_ms = (t_start - reference_time) / 1000.0
        ax.set_title(f'{title}\nTime: {time_ms:.1f}ms ({len(events)} events)', 
                    color='white', fontsize=10)
                    
        if len(events) > 0:
            ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
                     labelcolor='white')
                     
        # Style the plot
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, facecolor='black', edgecolor='white')
        plt.close()
        
    def _create_comparison_visualization(self, original_events: np.ndarray, 
                                       clean_events: np.ndarray,
                                       output_path: str, title: str,
                                       t_start: float, t_end: float, 
                                       reference_time: float):
        """Create side-by-side comparison visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor='black')
        
        for ax in [ax1, ax2]:
            ax.set_facecolor('black')
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # Plot original events
        if len(original_events) > 0:
            x, y, t, p = original_events[:, 0], original_events[:, 1], original_events[:, 2], original_events[:, 3]
            pos_mask = p > 0
            neg_mask = p <= 0
            
            if np.any(pos_mask):
                ax1.scatter(x[pos_mask], y[pos_mask], c='red', s=2, alpha=0.8)
            if np.any(neg_mask):
                ax1.scatter(x[neg_mask], y[neg_mask], c='cyan', s=2, alpha=0.8)
        else:
            ax1.text(self.width/2, self.height/2, 'No events', 
                    ha='center', va='center', color='white', fontsize=12)
        
        # Plot clean events  
        if len(clean_events) > 0:
            x, y, t, p = clean_events[:, 0], clean_events[:, 1], clean_events[:, 2], clean_events[:, 3]
            pos_mask = p > 0
            neg_mask = p <= 0
            
            if np.any(pos_mask):
                ax2.scatter(x[pos_mask], y[pos_mask], c='red', s=2, alpha=0.8)
            if np.any(neg_mask):
                ax2.scatter(x[neg_mask], y[neg_mask], c='cyan', s=2, alpha=0.8)
        else:
            ax2.text(self.width/2, self.height/2, 'No events',
                    ha='center', va='center', color='white', fontsize=12)
        
        # Labels and titles
        time_ms = (t_start - reference_time) / 1000.0
        ax1.set_title(f'Original ({len(original_events)} events)', color='white')
        ax2.set_title(f'Denoised ({len(clean_events)} events)', color='white')
        
        ax1.set_xlabel('X (pixels)', color='white')
        ax1.set_ylabel('Y (pixels)', color='white')
        ax2.set_xlabel('X (pixels)', color='white')
        
        fig.suptitle(f'{title}\nTime: {time_ms:.1f}ms', color='white', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, facecolor='black', edgecolor='white')
        plt.close()