#!/usr/bin/env python3
"""
Checkpoint Comparison Visualizer for EventMamba-FX Inference Results
Compares performance between best_model.pth and ckpt_step_00065000.pth
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

def load_events_from_h5(h5_path):
    """Load events from H5 file"""
    with h5py.File(h5_path, 'r') as f:
        x = f['events/x'][:]
        y = f['events/y'][:]
        t = f['events/t'][:]
        p = f['events/p'][:]
    return x, y, t, p

def create_event_frame(x, y, p, width=640, height=480, time_window=None):
    """Create a frame representation of events"""
    frame = np.zeros((height, width, 3))
    
    # Filter by time window if provided
    if time_window is not None:
        start_time, end_time = time_window
        mask = (t >= start_time) & (t < end_time)
        x, y, p = x[mask], y[mask], p[mask]
    
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

def create_checkpoint_comparison():
    """Create comparison visualizations between two checkpoints"""
    
    # File paths
    original_file = 'data/inference/test_synthetic_flare.h5'
    
    # Best model results
    best_model_files = {
        '0.5': 'data/inference/test_synthetic_flare_clean_05.h5',
        '0.4': 'data/inference/test_synthetic_flare_clean_04.h5', 
        '0.3': 'data/inference/test_synthetic_flare_clean_03.h5',
        '0.2': 'data/inference/test_synthetic_flare_clean_02.h5'
    }
    
    # Checkpoint 65k results
    ckpt_65k_files = {
        '0.5': 'data/inference/test_synthetic_flare_clean_65k_05.h5',
        '0.4': 'data/inference/test_synthetic_flare_clean_65k_04.h5',
        '0.3': 'data/inference/test_synthetic_flare_clean_65k_03.h5', 
        '0.2': 'data/inference/test_synthetic_flare_clean_65k_02.h5'
    }
    
    # Load original events
    x_orig, y_orig, t_orig, p_orig = load_events_from_h5(original_file)
    orig_count = len(x_orig)
    print(f"Original events: {orig_count}")
    
    # Calculate removal statistics for both checkpoints
    best_stats = {}
    ckpt_65k_stats = {}
    
    for threshold in ['0.5', '0.4', '0.3', '0.2']:
        # Best model stats
        if os.path.exists(best_model_files[threshold]):
            x_best, y_best, t_best, p_best = load_events_from_h5(best_model_files[threshold])
            best_count = len(x_best)
            best_removed = orig_count - best_count
            best_removal_rate = (best_removed / orig_count) * 100
            best_stats[threshold] = {
                'remaining': best_count,
                'removed': best_removed,
                'removal_rate': best_removal_rate
            }
            print(f"Best model @ {threshold}: {best_count} remaining, {best_removed} removed ({best_removal_rate:.2f}%)")
        
        # Checkpoint 65k stats  
        if os.path.exists(ckpt_65k_files[threshold]):
            x_65k, y_65k, t_65k, p_65k = load_events_from_h5(ckpt_65k_files[threshold])
            ckpt_65k_count = len(x_65k)
            ckpt_65k_removed = orig_count - ckpt_65k_count
            ckpt_65k_removal_rate = (ckpt_65k_removed / orig_count) * 100
            ckpt_65k_stats[threshold] = {
                'remaining': ckpt_65k_count,
                'removed': ckpt_65k_removed, 
                'removal_rate': ckpt_65k_removal_rate
            }
            print(f"Ckpt 65k @ {threshold}: {ckpt_65k_count} remaining, {ckpt_65k_removed} removed ({ckpt_65k_removal_rate:.2f}%)")
    
    # Create comparative bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    thresholds = ['0.5', '0.4', '0.3', '0.2']
    x_pos = np.arange(len(thresholds))
    width = 0.35
    
    # Removal counts comparison
    best_removed = [best_stats[t]['removed'] if t in best_stats else 0 for t in thresholds]
    ckpt_65k_removed = [ckpt_65k_stats[t]['removed'] if t in ckpt_65k_stats else 0 for t in thresholds]
    
    ax1.bar(x_pos - width/2, best_removed, width, label='best_model.pth', color='skyblue', alpha=0.8)
    ax1.bar(x_pos + width/2, ckpt_65k_removed, width, label='ckpt_step_00065000.pth', color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Events Removed')
    ax1.set_title('EventMamba-FX Checkpoint Comparison: Events Removed')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(thresholds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (best, ckpt) in enumerate(zip(best_removed, ckpt_65k_removed)):
        ax1.text(i - width/2, best + 50, str(best), ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, ckpt + 50, str(ckpt), ha='center', va='bottom', fontsize=9)
    
    # Removal rates comparison  
    best_rates = [best_stats[t]['removal_rate'] if t in best_stats else 0 for t in thresholds]
    ckpt_65k_rates = [ckpt_65k_stats[t]['removal_rate'] if t in ckpt_65k_stats else 0 for t in thresholds]
    
    ax2.bar(x_pos - width/2, best_rates, width, label='best_model.pth', color='skyblue', alpha=0.8)
    ax2.bar(x_pos + width/2, ckpt_65k_rates, width, label='ckpt_step_00065000.pth', color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Threshold') 
    ax2.set_ylabel('Removal Rate (%)')
    ax2.set_title('EventMamba-FX Checkpoint Comparison: Removal Rates')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(thresholds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (best_rate, ckpt_rate) in enumerate(zip(best_rates, ckpt_65k_rates)):
        ax2.text(i - width/2, best_rate + 0.2, f'{best_rate:.1f}%', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, ckpt_rate + 0.2, f'{ckpt_rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data/inference/checkpoint_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ Checkpoint comparison saved: data/inference/checkpoint_comparison.png")
    
    # Create side-by-side visual comparison for threshold 0.3 (good middle ground)
    if os.path.exists(best_model_files['0.3']) and os.path.exists(ckpt_65k_files['0.3']):
        create_visual_comparison('0.3', best_model_files['0.3'], ckpt_65k_files['0.3'])

def create_visual_comparison(threshold, best_file, ckpt_65k_file):
    """Create visual side-by-side comparison"""
    
    # Load all datasets
    x_orig, y_orig, t_orig, p_orig = load_events_from_h5('data/inference/test_synthetic_flare.h5')
    x_best, y_best, t_best, p_best = load_events_from_h5(best_file)  
    x_65k, y_65k, t_65k, p_65k = load_events_from_h5(ckpt_65k_file)
    
    # Create frames
    frame_orig = create_event_frame(x_orig, y_orig, p_orig)
    frame_best = create_event_frame(x_best, y_best, p_best)
    frame_65k = create_event_frame(x_65k, y_65k, p_65k)
    
    # Calculate removed events by subtraction
    removed_best = calculate_removed_events(x_orig, y_orig, p_orig, x_best, y_best, p_best)
    removed_65k = calculate_removed_events(x_orig, y_orig, p_orig, x_65k, y_65k, p_65k)
    
    frame_removed_best = create_removed_frame(removed_best, width=640, height=480)
    frame_removed_65k = create_removed_frame(removed_65k, width=640, height=480)
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: best_model.pth
    axes[0,0].imshow(frame_orig)
    axes[0,0].set_title(f'Original Events\n({len(x_orig)} events)', fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(frame_best)
    axes[0,1].set_title(f'best_model.pth @ {threshold}\n({len(x_best)} events, {len(x_orig)-len(x_best)} removed)', fontsize=12)
    axes[0,1].axis('off')
    
    axes[0,2].imshow(frame_removed_best)
    axes[0,2].set_title(f'Removed Events\n({len(x_orig)-len(x_best)} events)', fontsize=12)  
    axes[0,2].axis('off')
    
    # Row 2: ckpt_step_00065000.pth
    axes[1,0].imshow(frame_orig)
    axes[1,0].set_title(f'Original Events\n({len(x_orig)} events)', fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(frame_65k)
    axes[1,1].set_title(f'ckpt_step_00065000.pth @ {threshold}\n({len(x_65k)} events, {len(x_orig)-len(x_65k)} removed)', fontsize=12)
    axes[1,1].axis('off')
    
    axes[1,2].imshow(frame_removed_65k)
    axes[1,2].set_title(f'Removed Events\n({len(x_orig)-len(x_65k)} events)', fontsize=12)
    axes[1,2].axis('off')
    
    # Add synthetic flare region indicator
    flare_center = (320, 240)
    flare_radius = 60
    
    for ax in axes.flat:
        circle = Circle(flare_center, flare_radius, fill=False, color='yellow', linewidth=2, linestyle='--')
        ax.add_patch(circle)
    
    # Overall title
    fig.suptitle(f'EventMamba-FX Checkpoint Comparison @ Threshold {threshold}\nSynthetic Flare Region (Yellow Circle)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'data/inference/checkpoint_visual_comparison_{threshold}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visual comparison saved: data/inference/checkpoint_visual_comparison_{threshold}.png")

def calculate_removed_events(x_orig, y_orig, p_orig, x_clean, y_clean, p_clean):
    """Calculate which events were removed by finding the difference"""
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

def create_removed_frame(removed_events, width=640, height=480):
    """Create frame showing only removed events"""
    frame = np.zeros((height, width, 3))
    
    if len(removed_events) == 3 and len(removed_events[0]) > 0:
        x, y, p = removed_events
        
        # All removed events in yellow for visibility
        valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if np.any(valid_mask):
            frame[y[valid_mask], x[valid_mask]] = [1, 1, 0]  # Yellow
    
    return frame

if __name__ == "__main__":
    create_checkpoint_comparison()