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
    Dataset implementing TBPTT (Truncated Backpropagation Through Time) architecture:
    
    Long Sequence Factory:
    - Each __getitem__ call generates one complete long sequence
    - Load background events (0.1-0.3s duration)
    - Generate flare events (0.1-0.3s duration)  
    - Merge and sort → long_sequence [N, 4]
    - Extract features → long_feature_sequence [N, 11]
    - Return complete feature sequence for TBPTT chunking in Trainer
    
    No sliding window sampling - that's handled by Trainer's TBPTT logic
    """
    
    def __init__(self, config: Dict, split: str = 'train'):
        """Initialize the TBPTT Long Sequence Factory dataset.
        
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
        
        # TBPTT parameters - different for train/val splits
        if split == 'train':
            self.num_long_sequences_per_epoch = config['training'].get('num_long_sequences_per_epoch', 100)
        elif split == 'val':
            self.num_long_sequences_per_epoch = config['evaluation'].get('num_long_sequences_per_epoch', 20)
        else:  # test
            self.num_long_sequences_per_epoch = config['evaluation'].get('num_long_sequences_per_epoch', 20)
            
        # Keep sequence_length for backward compatibility with debug system
        self.sequence_length = config['data'].get('sequence_length', 64)
        
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
        
        # Remove epoch-level state - each __getitem__ generates fresh data
        
        # Initialize event visualizer (debug mode only)
        self.event_visualizer = None
        if self.debug_mode:
            viz_output_dir = os.path.join(config.get('debug_output_dir', './output/debug'), 'epoch_iteration_analysis')
            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.event_visualizer = EventVisualizer(viz_output_dir, resolution)
            print(f"🚨 DEBUG MODE: Epoch-Iteration analysis enabled, output: {viz_output_dir}")
        
        print(f"Initialized TBPTT Long Sequence Factory ({split}): {len(self.dsec_dataset)} background samples available")
        print(f"  - Long sequences per epoch: {self.num_long_sequences_per_epoch}")
    
    def _generate_one_long_sequence(self):
        """
        Generate one complete long sequence for TBPTT:
        1. Sample background events (0.1-0.3s)
        2. Generate flare events (0.1-0.3s)
        3. Merge & sort → long_sequence
        4. Extract features → long_feature_sequence
        
        Returns:
            Tuple of (long_feature_sequence, long_labels)
        """
        # Only print debug info occasionally to avoid spam
        debug_print = False  # Disabled to reduce output noise
        # if debug_print:
        #     print(f"\n🔄 Generating one long sequence")
        #     print(f"🔍 DEBUG: _generate_one_long_sequence() started, debug_mode={self.debug_mode}")
        sequence_start_time = time.time()
        
        # Step 1: Generate background events
        # if debug_print:
        #     print("  Step 1: Loading background events...")
        background_events = self._generate_background_events(debug_print)
        
        # Step 2: Generate flare events  
        # if debug_print:
        #     print("  Step 2: Generating flare events...")
        #     print(f"🔍 DEBUG: About to call _generate_flare_events()")
        flare_events = self._generate_flare_events(debug_print)
        # if debug_print:
        #     print(f"🔍 DEBUG: _generate_flare_events() completed, got {len(flare_events)} events")
        
        # Step 3: Merge and sort events
        # if debug_print:
        #     print("  Step 3: Merging and sorting events...")
        #     print(f"🔍 DEBUG: About to merge {len(background_events)} bg + {len(flare_events)} flare events")
        long_sequence, labels = self._merge_and_sort_events(background_events, flare_events)
        # if debug_print:
        #     print(f"🔍 DEBUG: Merge completed, got {len(long_sequence)} total events")
        
        # 🚨 IMMEDIATE DEBUG VISUALIZATION: Call right after merge, before feature extraction  
        sequence_count = getattr(self, '_sequence_count', 0)
        if self.debug_mode and sequence_count < 3:
            try:
                # Re-enable debug visualization now that feature extraction is fixed
                # Temporarily set current_epoch for compatibility
                self.current_epoch = sequence_count
                self._save_unified_debug_visualizations(background_events, flare_events, long_sequence, labels)
            except Exception as e:
                print(f"Debug visualization failed: {e}")
        
        # Step 4: CRITICAL - Extract features on complete sequence
        # if debug_print:
        #     print("  Step 4: Extracting PFD features on complete sequence...")
        #     print(f"🔍 DEBUG: About to extract features from {len(long_sequence)} events")
        feature_start_time = time.time()
        
        if len(long_sequence) == 0:
            # if debug_print:
            #     print("    Warning: Empty sequence, creating minimal fallback")
            long_feature_sequence = np.zeros((1, 11), dtype=np.float32)
            long_labels = np.zeros(1, dtype=np.int64)
        else:
            # ✅ CORE REQUIREMENT: Extract features on complete sequence FIRST
            # if debug_print:
            #     print(f"🔍 DEBUG: Calling feature_extractor.process_sequence()...")
            long_feature_sequence = self.feature_extractor.process_sequence(long_sequence)  # [N, 4] → [N, 11]
            # if debug_print:
            #     print(f"🔍 DEBUG: Feature extraction completed, got {len(long_feature_sequence)} feature vectors")
            long_labels = labels
        
        feature_time = time.time() - feature_start_time
        sequence_time = time.time() - sequence_start_time
        
        # 🔍 ENHANCED DEBUG OUTPUT: Event counts and ranges analysis
        # if debug_print:
        #     print(f"  ✅ Long sequence generation complete:")
        #     print(f"    - Background events: {len(background_events) if len(background_events) > 0 else 0}")
        #     print(f"    - Flare events: {len(flare_events) if len(flare_events) > 0 else 0}")
        #     print(f"    - Total merged events: {len(long_sequence) if len(long_sequence) > 0 else 0}")
        #     print(f"    - Feature extraction: {feature_time:.3f}s")
        #     print(f"    - Sequence length: {len(long_feature_sequence)}")
        #     print(f"    - Total sequence time: {sequence_time:.3f}s")
        
        # 🔍 DETAILED RANGE ANALYSIS: Sample last 1000 events for debugging
        # if debug_print:
        #     self._debug_event_ranges(background_events, flare_events, long_sequence, long_labels)
        
        # Remove problematic debug prints that reference non-existent attributes
        
        # Debug visualization only for first few sequences (avoid spam)
        sequence_count = getattr(self, '_sequence_count', 0)
        self._sequence_count = sequence_count + 1
        
        # 🎯 RESTORE ORIGINAL DEBUG VISUALIZATION: 恢复原本的可视化系统
        if self.event_visualizer is not None and sequence_count < 3:
            # Temporarily set current_epoch for compatibility with existing debug methods
            self.current_epoch = sequence_count
            self._debug_visualize_epoch(background_events, flare_events, long_sequence, long_labels)
        
        # 🎯 UNIFIED DEBUG VISUALIZATION: 统一可视化系统 (补充)
        # if debug_print:
        #     print(f"  🔍 Debug state: debug_mode={self.debug_mode}, sequence_count={sequence_count}")
        if self.debug_mode and sequence_count < 3:
            # Temporarily set current_epoch for compatibility with existing debug methods
            self.current_epoch = sequence_count
            # TEMP: Disable to isolate hanging issue
            # self._save_unified_debug_visualizations(background_events, flare_events, long_sequence, long_labels)
            pass
        
        # if debug_print:
        #     print(f"🔍 DEBUG: _generate_one_long_sequence() completed successfully")
        
        return long_feature_sequence, long_labels
    
    def _generate_background_events(self, debug_print: bool = False) -> np.ndarray:
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
        
        # 🚨 CRITICAL: Re-normalize background events after time window cropping
        if len(background_events) > 0:
            t_min_bg = background_events[:, 2].min()
            background_events[:, 2] = background_events[:, 2] - t_min_bg
        
        # if debug_print:
        #     print(f"    Background events loaded: {len(background_events)} events, duration: {duration_ms:.1f}ms")
        return background_events if len(background_events) > 0 else np.empty((0, 4))
    
    def _generate_flare_events(self, debug_print: bool = False) -> np.ndarray:
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
            flare_events, metadata, flare_video_frames = self.flare_generator.generate_flare_events(cleanup=True)
            
            # Handle empty generation
            if len(flare_events) == 0:
                # if debug_print:
                #     print(f"    Warning: DVS simulation generated no events (duration: {duration_ms:.1f}ms)")
                self.current_flare_video_frames = []  # Set empty list for consistency
                return np.empty((0, 4))
            
            # if debug_print:
            #     print(f"    Flare events generated: {len(flare_events)} events, duration: {duration_ms:.1f}ms")
            
            # Store flare video frames for debug visualization
            self.current_flare_video_frames = flare_video_frames
            
            return flare_events
            
        except Exception as e:
            # if debug_print:
            #     print(f"    Error in flare generation: {e}")
            self.current_flare_video_frames = []  # Set empty list for consistency
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
        
        # Timestamps already normalized: DSEC events start from 0, DVS events start from 0
        # No additional normalization needed since both streams are pre-normalized
        
        return combined_events, combined_labels
    
    def _format_flare_events(self, flare_events: np.ndarray) -> np.ndarray:
        """Convert DVS format [t, x, y, p] to project format [x, y, t, p]."""
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        # Convert format
        formatted_events = np.zeros_like(flare_events, dtype=np.float64)
        formatted_events[:, 0] = flare_events[:, 1]  # x
        formatted_events[:, 1] = flare_events[:, 2]  # y 
        formatted_events[:, 2] = flare_events[:, 0]  # t
        formatted_events[:, 3] = flare_events[:, 3]  # p
        
        # 🚨 CRITICAL: Normalize flare timestamps to start from 0
        if len(formatted_events) > 0:
            t_min = formatted_events[:, 2].min()
            formatted_events[:, 2] = formatted_events[:, 2] - t_min
        
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

    def _save_unified_debug_visualizations(self, background_events: np.ndarray, flare_events: np.ndarray,
                                         merged_events: np.ndarray, labels: np.ndarray):
        """Save unified debug visualizations with clear structure.
        
        Creates clean output structure:
        - background_events/: DSEC背景事件可视化
        - flare_events/: DVS炫光事件可视化  
        - merged_events/: 合成事件可视化
        """
        print(f"  🎯 Saving unified debug visualizations (Epoch {self.current_epoch})...")
        
        # Create clean output directory
        epoch_debug_dir = os.path.join('./output', f"debug_epoch_{self.current_epoch:03d}")
        os.makedirs(epoch_debug_dir, exist_ok=True)
        
        try:
            # 1. Background events visualization
            if len(background_events) > 0:
                print(f"    📊 Background events: {len(background_events)} events")
                self._create_event_sequence_visualization(
                    background_events, 
                    os.path.join(epoch_debug_dir, "background_events"),
                    "Background Events (DSEC)",
                    event_type="background"
                )
            
            # 2. Flare events visualization (单独显示炫光事件)
            if len(flare_events) > 0:
                print(f"    ✨ Flare events: {len(flare_events)} events")
                # 🚨 BUG FIX: Convert DVS format [t,x,y,p] to project format [x,y,t,p] for visualization
                flare_formatted = self._format_flare_events(flare_events)
                self._create_event_sequence_visualization(
                    flare_formatted,
                    os.path.join(epoch_debug_dir, "flare_events"), 
                    "Flare Events (DVS)",
                    event_type="flare"
                )
            
            # 3. Merged events visualization  
            if len(merged_events) > 0:
                print(f"    🔗 Merged events: {len(merged_events)} events")
                self._create_event_sequence_visualization(
                    merged_events,
                    os.path.join(epoch_debug_dir, "merged_events"), 
                    "Merged Events (Background + Flare)",
                    event_type="merged",
                    labels=labels
                )
            
            # 4. Save flare sequence original frames (if available)
            if hasattr(self, 'current_flare_video_frames') and self.current_flare_video_frames:
                print(f"    🎬 Flare sequence: {len(self.current_flare_video_frames)} frames")
                self._save_flare_sequence_frames(epoch_debug_dir, self.current_flare_video_frames)
            
            # 5. Save epoch metadata
            self._save_epoch_metadata(epoch_debug_dir, background_events, flare_events, merged_events, labels)
            
            print(f"    ✅ Unified debug visualizations saved to: {epoch_debug_dir}")
            
        except Exception as e:
            print(f"    ❌ Error saving unified debug visualizations: {e}")
            import traceback
            traceback.print_exc()

    def _save_flare_sequence_frames(self, epoch_debug_dir: str, video_frames: List[np.ndarray]):
        """Save original flare sequence frames to debug directory.
        
        Args:
            epoch_debug_dir: Epoch debug directory
            video_frames: List of RGB video frames from flare synthesis
        """
        frames_dir = os.path.join(epoch_debug_dir, "flare_sequence_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        for i, frame in enumerate(video_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
        
        print(f"      ✅ Saved {len(video_frames)} flare sequence frames to: {frames_dir}")

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
            elif event_type == "flare":
                # Pure flare events (yellow/orange)
                color = colors['flare_pos'] if p > 0 else colors['flare_neg']
            else:
                # Background events (red/blue)
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
            f.write(f"  Sequence index: {getattr(self, 'current_epoch', 0)}\n")
            f.write(f"  Resolution: {self.config['data']['resolution_w']}x{self.config['data']['resolution_h']}\n")
            f.write(f"  TBPTT chunk size: {self.config['training'].get('chunk_size', 'Not set')}\n")
            f.write(f"  Long sequences per epoch: {self.num_long_sequences_per_epoch}\n")
            bg_range = self.config['data']['randomized_training']['background_duration_range']
            f.write(f"  Background duration range: {bg_range[0]*1000:.0f}-{bg_range[1]*1000:.0f}ms\n")
            flare_range = self.config['data']['flare_synthesis']['duration_range']
            f.write(f"  Flare duration range: {flare_range[0]*1000:.0f}-{flare_range[1]*1000:.0f}ms\n")
    
    def __len__(self) -> int:
        """Return number of long sequences per epoch for TBPTT."""
        return self.num_long_sequences_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate one complete long sequence for TBPTT.
        
        Args:
            idx: Index (0 to num_long_sequences_per_epoch-1) - used as random seed
            
        Returns:
            Tuple of (long_features_tensor, long_labels_tensor)
            long_features_tensor: [L, feature_dim] - complete feature sequence
            long_labels_tensor: [L] - corresponding labels
        """
        # Set random seed based on index for reproducibility during the same epoch
        random_state = random.getstate()
        random.seed(idx)
        
        try:
            # Generate one complete long sequence
            long_features, long_labels = self._generate_one_long_sequence()
            
            # Convert to tensors
            features_tensor = torch.tensor(long_features, dtype=torch.float32)
            labels_tensor = torch.tensor(long_labels, dtype=torch.long)
            
            return features_tensor, labels_tensor
            
        finally:
            # Restore random state
            random.setstate(random_state)


# EpochIterationDataLoader class removed - now using standard PyTorch DataLoader
# The TBPTT chunking logic is moved to Trainer class


def create_epoch_iteration_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create TBPTT Long Sequence Factory DataLoaders for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets with TBPTT long sequence factory architecture
    train_dataset = EpochIterationDataset(config, split='train')
    val_dataset = EpochIterationDataset(config, split='val')  
    test_dataset = EpochIterationDataset(config, split='test')
    
    # Create standard PyTorch DataLoaders with TBPTT configuration
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1,      # Always 1 for TBPTT - we process one long sequence at a time
        shuffle=False,     # No shuffling needed since each __getitem__ generates random data
        num_workers=0      # Recommended for complex data generation logic
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Created TBPTT Long Sequence Factory dataloaders:")
    print(f"  Train: {len(train_loader)} long sequences per epoch")
    print(f"  Val: {len(val_loader)} long sequences per epoch")
    print(f"  Test: {len(test_loader)} long sequences per epoch")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the TBPTT Long Sequence Factory dataset
    import yaml
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    config_path = "configs/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug mode for testing
    config['debug_mode'] = True
    config['data']['max_samples_debug'] = 4
    config['training']['num_long_sequences_per_epoch'] = 3  # Test with 3 sequences
    
    print("Testing TBPTT Long Sequence Factory Dataset...")
    
    # Create dataset
    dataset = EpochIterationDataset(config, split='train')
    
    print(f"Dataset length: {len(dataset)} long sequences per epoch")
    
    # Test long sequence generation
    print("\nTesting long sequence generation...")
    for i in range(min(2, len(dataset))):
        features, labels = dataset[i]
        print(f"  Sequence {i}: features {features.shape}, labels {labels.shape}")
        print(f"    Background events: {torch.sum(labels == 0).item()}")
        print(f"    Flare events: {torch.sum(labels == 1).item()}")
        print(f"    Sequence length: {len(features)}")
    
    # Test DataLoader
    print("\nTesting TBPTT DataLoader...")
    train_loader, _, _ = create_epoch_iteration_dataloaders(config)
    
    sequence_count = 0
    for long_features, long_labels in train_loader:
        # DataLoader returns batch_size=1, so squeeze to get actual sequence
        long_features = long_features.squeeze(0)
        long_labels = long_labels.squeeze(0)
        print(f"  Long Sequence {sequence_count}: features {long_features.shape}, labels {long_labels.shape}")
        print(f"    Background events: {torch.sum(long_labels == 0).item()}")
        print(f"    Flare events: {torch.sum(long_labels == 1).item()}")
        sequence_count += 1
        if sequence_count >= 2:  # Test first 2 sequences
            break
    
    print("\n✅ TBPTT Long Sequence Factory Dataset test completed!")