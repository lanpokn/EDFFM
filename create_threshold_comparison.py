#!/usr/bin/env python3
"""
Create single threshold comparison visualization for EventMamba-FX
Shows original, denoised, and removed events in a single image
"""

import h5py
import hdf5plugin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

def load_events_from_h5(h5_path):
    """Load events from H5 file"""
    print(f"Loading events from: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            x = f['events/x'][:]
            y = f['events/y'][:]
            t = f['events/t'][:]
            p = f['events/p'][:]
        print(f"  Loaded {len(x)} events")
        return x, y, t, p
    except Exception as e:
        print(f"  Error loading: {e}")
        return None, None, None, None

def create_event_frame(x, y, p, width=640, height=480):
    """Create a frame representation of events"""
    if x is None or len(x) == 0:
        return np.zeros((height, width, 3))
        
    frame = np.zeros((height, width, 3))
    
    # Positive events in cyan, negative in red
    pos_mask = p == 1
    neg_mask = p == 0
    
    if np.any(pos_mask):
        pos_x, pos_y = x[pos_mask], y[pos_mask]
        valid_pos = (pos_x >= 0) & (pos_x < width) & (pos_y >= 0) & (pos_y < height)
        frame[pos_y[valid_pos], pos_x[valid_pos]] = [0, 1, 1]  # Cyan
        
    if np.any(neg_mask):
        neg_x, neg_y = x[neg_mask], y[neg_mask]
        valid_neg = (neg_x >= 0) & (neg_x < width) & (neg_y >= 0) & (neg_y < height)
        frame[neg_y[valid_neg], neg_x[valid_neg]] = [1, 0, 0]  # Red
    
    return frame

def calculate_removed_events(x_orig, y_orig, p_orig, x_clean, y_clean, p_clean):
    """Calculate which events were removed"""
    if x_orig is None or x_clean is None:
        return np.array([]), np.array([]), np.array([])
        
    # Create sets for efficient comparison
    orig_events = set(zip(x_orig, y_orig, p_orig))
    clean_events = set(zip(x_clean, y_clean, p_clean))
    
    # Find removed events
    removed_events = orig_events - clean_events
    
    if len(removed_events) > 0:
        removed_x, removed_y, removed_p = zip(*removed_events)
        return np.array(removed_x), np.array(removed_y), np.array(removed_p)
    else:
        return np.array([]), np.array([]), np.array([])

def create_removed_frame(removed_x, removed_y, removed_p, width=640, height=480):
    """Create frame showing only removed events in yellow"""
    frame = np.zeros((height, width, 3))
    
    if len(removed_x) > 0:
        valid_mask = (removed_x >= 0) & (removed_x < width) & (removed_y >= 0) & (removed_y < height)
        if np.any(valid_mask):
            frame[removed_y[valid_mask], removed_x[valid_mask]] = [1, 1, 0]  # Yellow
    
    return frame

def create_threshold_comparison(threshold, checkpoint_name, original_file, clean_file, output_file):
    """Create threshold comparison visualization"""
    print(f"Creating threshold comparison for {threshold} using {checkpoint_name}")
    
    # Load events
    x_orig, y_orig, t_orig, p_orig = load_events_from_h5(original_file)
    x_clean, y_clean, t_clean, p_clean = load_events_from_h5(clean_file)
    
    if x_orig is None or x_clean is None:
        print("Failed to load events!")
        return
    
    # Calculate statistics
    orig_count = len(x_orig)
    clean_count = len(x_clean)
    removed_count = orig_count - clean_count
    removal_rate = (removed_count / orig_count) * 100
    
    print(f"Statistics:")
    print(f"  Original: {orig_count} events")
    print(f"  Remaining: {clean_count} events") 
    print(f"  Removed: {removed_count} events ({removal_rate:.2f}%)")
    
    # Calculate removed events
    removed_x, removed_y, removed_p = calculate_removed_events(x_orig, y_orig, p_orig, x_clean, y_clean, p_clean)
    
    # Create frames
    frame_orig = create_event_frame(x_orig, y_orig, p_orig)
    frame_clean = create_event_frame(x_clean, y_clean, p_clean)
    frame_removed = create_removed_frame(removed_x, removed_y, removed_p)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original events
    axes[0].imshow(frame_orig)
    axes[0].set_title(f'Original Events\n({orig_count:,} total)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # After denoising
    axes[1].imshow(frame_clean) 
    axes[1].set_title(f'After Denoising\n({clean_count:,} remaining)', fontsize=14)
    axes[1].axis('off')
    
    # Removed events
    axes[2].imshow(frame_removed)
    axes[2].set_title(f'Removed Events\n({removed_count:,} events)', fontsize=14)
    axes[2].axis('off')
    
    # Add synthetic flare region indicator
    flare_center = (320, 240)
    flare_radius = 60
    
    for ax in axes:
        circle = Circle(flare_center, flare_radius, fill=False, color='yellow', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    # Overall title
    title = f'EventMamba-FX Denoising Results - Threshold {threshold}\n'
    title += f'Removal Rate: {removal_rate:.2f}% | {removed_count:,} of {orig_count:,} Events | {checkpoint_name}'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualization saved: {output_file}")

if __name__ == "__main__":
    # Default parameters
    threshold = "0.5"
    checkpoint_name = "ckpt_step_00065000.pth"
    original_file = "data/inference/test_synthetic_flare.h5"
    clean_file = "data/inference/test_synthetic_flare_clean_65k_05.h5"
    output_file = "data/inference/threshold_05_comparison_65k.png"
    
    # Allow command line override
    if len(sys.argv) > 1:
        threshold = sys.argv[1]
        clean_file = f"data/inference/test_synthetic_flare_clean_65k_{threshold.replace('.', '')}.h5"
        output_file = f"data/inference/threshold_{threshold.replace('.', '')}_comparison_65k.png"
    
    create_threshold_comparison(threshold, checkpoint_name, original_file, clean_file, output_file)