#!/usr/bin/env python3
"""
Create visualization based on known statistics when H5 reading fails
"""

import numpy as np
import matplotlib.pyplot as plt

def create_stats_comparison():
    """Create comparison chart based on known statistics"""
    
    # Known statistics from previous runs
    stats = {
        'best_model.pth': {
            '0.5': {'removed': 3174, 'removal_rate': 5.77},
            '0.4': {'removed': 3633, 'removal_rate': 6.61}, 
            '0.3': {'removed': 4175, 'removal_rate': 7.59},
            '0.2': {'removed': 4946, 'removal_rate': 8.99}
        },
        'ckpt_step_00065000.pth': {
            '0.5': {'removed': 7403, 'removal_rate': 13.46},
            '0.4': {'removed': 7904, 'removal_rate': 14.37},
            '0.3': {'removed': 8482, 'removal_rate': 15.42}, 
            '0.2': {'removed': 9254, 'removal_rate': 16.83}
        }
    }
    
    total_events = 55000  # Total synthetic events
    
    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    thresholds = ['0.5', '0.4', '0.3', '0.2']
    x_pos = np.arange(len(thresholds))
    width = 0.35
    
    # Removal counts comparison
    best_removed = [stats['best_model.pth'][t]['removed'] for t in thresholds]
    ckpt_65k_removed = [stats['ckpt_step_00065000.pth'][t]['removed'] for t in thresholds]
    
    bars1 = ax1.bar(x_pos - width/2, best_removed, width, label='best_model.pth', 
                    color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, ckpt_65k_removed, width, label='ckpt_step_00065000.pth', 
                    color='lightcoral', alpha=0.8)
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Events Removed')
    ax1.set_title('EventMamba-FX Checkpoint Comparison: Events Removed\n(Synthetic Flare Test Data: 55,000 total events)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(thresholds)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2, height1 + 50, 
                f'{int(height1)}', ha='center', va='bottom', fontsize=9)
        ax1.text(bar2.get_x() + bar2.get_width()/2, height2 + 50, 
                f'{int(height2)}', ha='center', va='bottom', fontsize=9)
    
    # Removal rates comparison  
    best_rates = [stats['best_model.pth'][t]['removal_rate'] for t in thresholds]
    ckpt_65k_rates = [stats['ckpt_step_00065000.pth'][t]['removal_rate'] for t in thresholds]
    
    bars3 = ax2.bar(x_pos - width/2, best_rates, width, label='best_model.pth', 
                    color='skyblue', alpha=0.8)
    bars4 = ax2.bar(x_pos + width/2, ckpt_65k_rates, width, label='ckpt_step_00065000.pth', 
                    color='lightcoral', alpha=0.8)
    
    ax2.set_xlabel('Threshold') 
    ax2.set_ylabel('Removal Rate (%)')
    ax2.set_title('EventMamba-FX Checkpoint Comparison: Removal Rates')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(thresholds)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        ax2.text(bar3.get_x() + bar3.get_width()/2, height3 + 0.2, 
                f'{height3:.1f}%', ha='center', va='bottom', fontsize=9)
        ax2.text(bar4.get_x() + bar4.get_width()/2, height4 + 0.2, 
                f'{height4:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('data/inference/checkpoint_comparison_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Statistics-based comparison saved: data/inference/checkpoint_comparison_statistics.png")
    
    # Create focused comparison for threshold 0.5
    create_threshold_05_focused()

def create_threshold_05_focused():
    """Create focused comparison for threshold 0.5"""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    models = ['best_model.pth', 'ckpt_step_00065000.pth']
    removed_counts = [3174, 7403]
    removal_rates = [5.77, 13.46]
    colors = ['skyblue', 'lightcoral']
    
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, removed_counts, color=colors, alpha=0.8)
    
    ax.set_xlabel('Model Checkpoint')
    ax.set_ylabel('Events Removed')
    ax.set_title('EventMamba-FX Performance Comparison @ Threshold 0.5\nSynthetic Flare Test (55,000 total events)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels with rates
    for i, (bar, rate) in enumerate(zip(bars, removal_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 100, 
               f'{int(height)} events\n({rate}%)', ha='center', va='bottom', fontsize=10)
    
    # Add summary text
    improvement = 7403 - 3174
    improvement_pct = ((7403 - 3174) / 3174) * 100
    
    summary_text = f"ckpt_step_00065000.pth removes {improvement} more events than best_model.pth\n"
    summary_text += f"({improvement_pct:.1f}% improvement in flare detection sensitivity)"
    
    ax.text(0.5, 0.95, summary_text, transform=ax.transAxes, 
           ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)
    
    plt.tight_layout()
    plt.savefig('data/inference/threshold_05_checkpoint_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Threshold 0.5 focused comparison saved: data/inference/threshold_05_checkpoint_comparison.png")

if __name__ == "__main__":
    create_stats_comparison()