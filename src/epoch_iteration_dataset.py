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
import cv2
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
            time_window_us=config['data']['time_window_us']
        )
        
        # Initialize flare event generator
        self.flare_generator = create_flare_event_generator(config)
        
        # Initialize feature extractor (CRITICAL: at dataset level)
        self.feature_extractor = FeatureExtractor(config)
        
        # Training parameters
        self.sequence_length = config['data']['sequence_length']  # e.g., 64
        
        # Background duration from config
        bg_range = config['data']['randomized_training']['background_duration_range']
        self.bg_min_duration_ms = bg_range[0] * 1000  # Convert to ms
        self.bg_max_duration_ms = bg_range[1] * 1000  # Convert to ms
        
        # Flare duration from flare_synthesis (no longer from randomized_training)
        flare_range = config['data']['flare_synthesis']['duration_range']
        self.flare_min_duration_ms = flare_range[0] * 1000  # Convert to ms
        self.flare_max_duration_ms = flare_range[1] * 1000  # Convert to ms
        
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
        
        # 🔍 ENHANCED DEBUG OUTPUT: Event counts and ranges analysis
        print(f"  ✅ Epoch generation complete:")
        print(f"    - Background events: {len(background_events) if len(background_events) > 0 else 0}")
        print(f"    - Flare events: {len(flare_events) if len(flare_events) > 0 else 0}")
        print(f"    - Total merged events: {len(long_sequence) if len(long_sequence) > 0 else 0}")
        print(f"    - Feature extraction: {feature_time:.3f}s")
        print(f"    - Available iterations: {self.num_iterations}")
        print(f"    - Total epoch time: {epoch_time:.3f}s")
        
        # 🔍 DETAILED RANGE ANALYSIS: Sample last 1000 events for debugging
        self._debug_event_ranges(background_events, flare_events, long_sequence, labels)
        
        # 🎯 NEW: DVS-style debug visualization for background and merged events
        if self.debug_mode and self.current_epoch < 3:
            self._save_epoch_debug_visualizations(background_events, flare_events, long_sequence, labels)
        
        # Debug visualization (only for first few epochs)
        if self.event_visualizer is not None and self.current_epoch < 3:
            self._debug_visualize_epoch(background_events, flare_events, long_sequence, labels)
    
    def _generate_background_events(self) -> np.ndarray:
        """Generate random background events using config duration range."""
        # Random duration from background config
        duration_ms = random.uniform(self.bg_min_duration_ms, self.bg_max_duration_ms)
        duration_us = int(duration_ms * 1000)
        
        # Random DSEC sample - now returns numpy array directly
        idx = random.randint(0, len(self.dsec_dataset) - 1)
        background_events = self.dsec_dataset[idx]  # Returns np.ndarray [N, 4]
        
        # 🚨 CRITICAL FIX: Now background_events should contain ALL events in time window
        # Crop to desired duration if current window is longer than desired
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
        
        print(f"    Background events loaded: {len(background_events)} events, duration: {duration_ms:.1f}ms")
        return background_events if len(background_events) > 0 else np.empty((0, 4))
    
    def _generate_flare_events(self) -> np.ndarray:
        """Generate random flare events using flare_synthesis duration range."""
        # Random duration from flare_synthesis config
        duration_ms = random.uniform(self.flare_min_duration_ms, self.flare_max_duration_ms)
        duration_sec = duration_ms / 1000.0
        
        # Temporarily modify config for desired duration
        # 🚨 FIX: Use duration_range instead of deprecated duration_sec
        original_duration_range = self.config['data']['flare_synthesis'].get('duration_range', [0.05, 0.15])
        
        # Set a fixed duration for this generation (within the range)
        temp_duration_range = [duration_sec, duration_sec]  # Fixed duration for this call
        self.config['data']['flare_synthesis']['duration_range'] = temp_duration_range
        
        try:
            # Generate flare events using DVS simulator
            flare_events, metadata = self.flare_generator.generate_flare_events(cleanup=True)
            
            # Handle empty generation
            if len(flare_events) == 0:
                print(f"    Warning: DVS simulation generated no events (duration: {duration_ms:.1f}ms)")
                return np.empty((0, 4))
            
            print(f"    Flare events generated: {len(flare_events)} events, duration: {duration_ms:.1f}ms")
            return flare_events
            
        except Exception as e:
            print(f"    Error in flare generation: {e}")
            return np.empty((0, 4))
        finally:
            # Restore original config
            self.config['data']['flare_synthesis']['duration_range'] = original_duration_range
    
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
    
    def _debug_event_ranges(self, background_events: np.ndarray, flare_events: np.ndarray, 
                           merged_events: np.ndarray, labels: np.ndarray):
        """Debug analysis of event ranges (x, y, t) for last 1000 events."""
        print(f"\n  🔍 DEBUG RANGE ANALYSIS:")
        
        def analyze_events(events, name, sample_size=1000):
            if len(events) == 0:
                print(f"    {name}: EMPTY")
                return
                
            # Sample last N events
            sample = events[-sample_size:] if len(events) > sample_size else events
            
            x_range = (sample[:, 0].min(), sample[:, 0].max())
            y_range = (sample[:, 1].min(), sample[:, 1].max()) 
            t_range = (sample[:, 2].min(), sample[:, 2].max())
            t_duration_ms = (t_range[1] - t_range[0]) / 1000.0  # Convert μs to ms
            
            print(f"    {name} (last {len(sample)} events):")
            print(f"      X range: [{x_range[0]:.1f}, {x_range[1]:.1f}]")
            print(f"      Y range: [{y_range[0]:.1f}, {y_range[1]:.1f}]") 
            print(f"      T range: [{t_range[0]:.0f}, {t_range[1]:.0f}] μs")
            print(f"      Duration: {t_duration_ms:.2f} ms")
        
        # Analyze each event type
        analyze_events(background_events, "Background Events")
        
        if len(flare_events) > 0:
            # Format flare events for analysis
            flare_formatted = self._format_flare_events(flare_events)
            analyze_events(flare_formatted, "Flare Events")
        
        analyze_events(merged_events, "Merged Events")
        
        # Label distribution
        if len(labels) > 0:
            bg_count = np.sum(labels == 0)
            flare_count = np.sum(labels == 1)
            print(f"    Label distribution: Background={bg_count}, Flare={flare_count}")

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

    def _save_epoch_debug_visualizations(self, background_events: np.ndarray, flare_events: np.ndarray,
                                       merged_events: np.ndarray, labels: np.ndarray):
        """Save DVS-style debug visualizations for background and merged events.
        
        Creates similar multi-resolution temporal visualizations as DVS flare sequences,
        but for background events and merged events using black frames as placeholders.
        """
        print(f"  🎯 Saving epoch debug visualizations (Epoch {self.current_epoch})...")
        
        # Create output directory
        epoch_debug_dir = os.path.join(self.config.get('debug_output_dir', './output/debug'), 
                                      f"epoch_{self.current_epoch:03d}")
        os.makedirs(epoch_debug_dir, exist_ok=True)
        
        try:
            # 1. Background events visualization
            if len(background_events) > 0:
                print(f"    Creating background events visualization ({len(background_events)} events)...")
                self._create_event_sequence_visualization(
                    background_events, 
                    os.path.join(epoch_debug_dir, "background_events"),
                    "Background Events",
                    event_type="background"
                )
            
            # 2. Merged events visualization  
            if len(merged_events) > 0:
                print(f"    Creating merged events visualization ({len(merged_events)} events)...")
                self._create_event_sequence_visualization(
                    merged_events,
                    os.path.join(epoch_debug_dir, "merged_events"), 
                    "Merged Events",
                    event_type="merged",
                    labels=labels
                )
            
            # 3. Save epoch metadata
            self._save_epoch_metadata(epoch_debug_dir, background_events, flare_events, merged_events, labels)
            
            print(f"    ✅ Epoch debug visualizations saved to: {epoch_debug_dir}")
            
        except Exception as e:
            print(f"    ❌ Error saving epoch debug visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _create_event_sequence_visualization(self, events: np.ndarray, output_dir: str, 
                                           title: str, event_type: str = "background", 
                                           labels: Optional[np.ndarray] = None):
        """Create DVS-style multi-resolution temporal visualizations for event sequences.
        
        Args:
            events: Event array [N, 4] in format [x, y, t, p]
            output_dir: Output directory for visualizations
            title: Title for the visualization
            event_type: Type of events ("background", "merged")
            labels: Optional labels for merged events [N] (0=background, 1=flare)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if len(events) == 0:
            print(f"    Warning: No events to visualize for {title}")
            return
            
        # Get time range and resolution
        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        duration_ms = (t_max - t_min) / 1000.0
        resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])
        
        # Multi-resolution strategies (same as DVS flare visualization)
        resolution_strategies = [0.5, 1, 2, 4]
        
        print(f"      Duration: {duration_ms:.1f}ms, Events: {len(events)}")
        
        for scale in resolution_strategies:
            scale_dir = os.path.join(output_dir, f"temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
            
            # Calculate temporal window size
            base_window_ms = 10.0  # Base window: 10ms
            window_duration_ms = base_window_ms / scale
            window_duration_us = window_duration_ms * 1000
            
            # Generate frame sequence
            num_frames = max(10, int(duration_ms / window_duration_ms))
            frame_step = (t_max - t_min) / num_frames if num_frames > 1 else 0
            
            for frame_idx in range(min(num_frames, 50)):  # Limit to 50 frames
                frame_start_time = t_min + frame_idx * frame_step
                frame_end_time = frame_start_time + window_duration_us
                
                # Filter events in this time window
                time_mask = (events[:, 2] >= frame_start_time) & (events[:, 2] < frame_end_time)
                frame_events = events[time_mask]
                
                # Create black background frame (pure black as requested)
                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                
                if len(frame_events) > 0:
                    # Overlay events on black frame
                    self._overlay_events_on_frame(frame, frame_events, event_type, 
                                                labels[time_mask] if labels is not None else None)
                
                # Save frame
                frame_filename = f"frame_{frame_idx:03d}_{scale}x_events.png"
                frame_path = os.path.join(scale_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
        print(f"      ✅ Multi-resolution visualization saved to: {output_dir}")

    def _overlay_events_on_frame(self, frame: np.ndarray, events: np.ndarray, 
                               event_type: str, labels: Optional[np.ndarray] = None):
        """Overlay events on frame with appropriate colors.
        
        Args:
            frame: BGR frame to overlay events on
            events: Events to overlay [N, 4]
            event_type: Type of events ("background", "merged")
            labels: Optional labels for merged events
        """
        if len(events) == 0:
            return
            
        # Color scheme (BGR format for OpenCV)
        colors = {
            'background_pos': (0, 0, 255),    # Red for positive background events
            'background_neg': (255, 0, 0),    # Blue for negative background events  
            'flare_pos': (0, 255, 255),       # Yellow for positive flare events
            'flare_neg': (0, 128, 255),       # Orange for negative flare events
        }
        
        for i, event in enumerate(events):
            x, y, t, p = event
            x, y = int(x), int(y)
            
            # Skip events outside frame bounds
            if x < 0 or x >= frame.shape[1] or y < 0 or y >= frame.shape[0]:
                continue
                
            # Determine color based on event type and polarity
            if event_type == "merged" and labels is not None:
                # Use labels to distinguish background vs flare in merged events
                if labels[i] == 0:  # Background
                    color = colors['background_pos'] if p > 0 else colors['background_neg']
                else:  # Flare
                    color = colors['flare_pos'] if p > 0 else colors['flare_neg']
            else:
                # Background events or unlabeled
                color = colors['background_pos'] if p > 0 else colors['background_neg']
                
            # Draw event as small circle
            cv2.circle(frame, (x, y), 1, color, -1)

    def _save_epoch_metadata(self, output_dir: str, background_events: np.ndarray, 
                            flare_events: np.ndarray, merged_events: np.ndarray, 
                            labels: np.ndarray):
        """Save epoch metadata similar to DVS flare metadata.
        
        Args:
            output_dir: Output directory
            background_events: Background events array
            flare_events: Flare events array
            merged_events: Merged events array  
            labels: Event labels
        """
        metadata_path = os.path.join(output_dir, "epoch_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Epoch Debug Information\n")
            f.write("======================\n\n")
            
            # Background events info
            if len(background_events) > 0:
                bg_t_min, bg_t_max = background_events[:, 2].min(), background_events[:, 2].max()
                bg_duration_ms = (bg_t_max - bg_t_min) / 1000.0
                bg_pos_count = np.sum(background_events[:, 3] > 0)
                bg_neg_count = np.sum(background_events[:, 3] <= 0)
                
                f.write(f"Background Events:\n")
                f.write(f"  Total events: {len(background_events)}\n")
                f.write(f"  Time range: {bg_t_min:.0f} - {bg_t_max:.0f} μs\n")
                f.write(f"  Duration: {bg_duration_ms:.1f} ms\n")
                f.write(f"  Event rate: {len(background_events) / (bg_duration_ms / 1000):.1f} events/s\n")
                f.write(f"  Polarity: {bg_pos_count} ON ({bg_pos_count/len(background_events)*100:.1f}%), ")
                f.write(f"{bg_neg_count} OFF ({bg_neg_count/len(background_events)*100:.1f}%)\n\n")
            
            # Flare events info  
            if len(flare_events) > 0:
                # Convert flare events to project format for analysis
                flare_formatted = self._format_flare_events(flare_events)
                if len(flare_formatted) > 0:
                    fl_t_min, fl_t_max = flare_formatted[:, 2].min(), flare_formatted[:, 2].max()
                    fl_duration_ms = (fl_t_max - fl_t_min) / 1000.0
                    fl_pos_count = np.sum(flare_formatted[:, 3] > 0)
                    fl_neg_count = np.sum(flare_formatted[:, 3] <= 0)
                    
                    f.write(f"Flare Events:\n")
                    f.write(f"  Total events: {len(flare_formatted)}\n")
                    f.write(f"  Time range: {fl_t_min:.0f} - {fl_t_max:.0f} μs\n")
                    f.write(f"  Duration: {fl_duration_ms:.1f} ms\n")
                    f.write(f"  Event rate: {len(flare_formatted) / (fl_duration_ms / 1000):.1f} events/s\n")
                    f.write(f"  Polarity: {fl_pos_count} ON ({fl_pos_count/len(flare_formatted)*100:.1f}%), ")
                    f.write(f"{fl_neg_count} OFF ({fl_neg_count/len(flare_formatted)*100:.1f}%)\n\n")
            
            # Merged events info
            if len(merged_events) > 0:
                mg_t_min, mg_t_max = merged_events[:, 2].min(), merged_events[:, 2].max()
                mg_duration_ms = (mg_t_max - mg_t_min) / 1000.0
                mg_pos_count = np.sum(merged_events[:, 3] > 0)
                mg_neg_count = np.sum(merged_events[:, 3] <= 0)
                
                f.write(f"Merged Events:\n")
                f.write(f"  Total events: {len(merged_events)}\n")
                f.write(f"  Time range: {mg_t_min:.0f} - {mg_t_max:.0f} μs\n")
                f.write(f"  Duration: {mg_duration_ms:.1f} ms\n")
                f.write(f"  Event rate: {len(merged_events) / (mg_duration_ms / 1000):.1f} events/s\n")
                f.write(f"  Polarity: {mg_pos_count} ON ({mg_pos_count/len(merged_events)*100:.1f}%), ")
                f.write(f"{mg_neg_count} OFF ({mg_neg_count/len(merged_events)*100:.1f}%)\n\n")
                
                # Label distribution for merged events
                if len(labels) > 0:
                    bg_label_count = np.sum(labels == 0)
                    flare_label_count = np.sum(labels == 1)
                    f.write(f"Label Distribution:\n")
                    f.write(f"  Background: {bg_label_count} ({bg_label_count/len(labels)*100:.1f}%)\n")
                    f.write(f"  Flare: {flare_label_count} ({flare_label_count/len(labels)*100:.1f}%)\n\n")
            
            # Epoch configuration
            f.write(f"Epoch Configuration:\n")
            f.write(f"  Epoch index: {self.current_epoch}\n")
            f.write(f"  Resolution: {self.config['data']['resolution_w']}x{self.config['data']['resolution_h']}\n")
            f.write(f"  Sequence length: {self.sequence_length}\n")
            bg_range = self.config['data']['randomized_training']['background_duration_range']
            f.write(f"  Background duration range: {bg_range[0]*1000:.0f}-{bg_range[1]*1000:.0f}ms\n")
            flare_range = self.config['data']['flare_synthesis']['duration_range']
            f.write(f"  Flare duration range: {flare_range[0]*1000:.0f}-{flare_range[1]*1000:.0f}ms\n")
    
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