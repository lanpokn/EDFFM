#!/usr/bin/env python3
"""
完整数据管线分析

分析完整的训练数据生成流程：
1. DSEC背景事件读取与分析
2. DVS炫光事件仿真与分析  
3. 事件合并与最终分析
4. 三组件可视化对比

目标：理解每个组件的贡献和合理性
"""

import sys
import os
import yaml
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

def create_event_visualization(events, title, save_path, resolution=(640, 480)):
    """创建事件可视化图像."""
    if len(events) == 0:
        print(f"⚠️  No events to visualize for {title}")
        return
    
    # Extract coordinates and polarities
    x = events[:, 1].astype(int)
    y = events[:, 2].astype(int) 
    p = events[:, 3].astype(int)
    
    # Filter valid coordinates
    valid = (x >= 0) & (x < resolution[0]) & (y >= 0) & (y < resolution[1])
    x, y, p = x[valid], y[valid], p[valid]
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot positive and negative events
    pos_mask = p > 0
    neg_mask = p <= 0
    
    if np.any(pos_mask):
        ax.scatter(x[pos_mask], y[pos_mask], c='red', s=0.1, alpha=0.6, label=f'ON ({np.sum(pos_mask)})')
    if np.any(neg_mask):
        ax.scatter(x[neg_mask], y[neg_mask], c='blue', s=0.1, alpha=0.6, label=f'OFF ({np.sum(neg_mask)})')
    
    ax.set_xlim(0, resolution[0])
    ax.set_ylim(resolution[1], 0)  # Flip Y axis
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'{title}\nTotal Events: {len(events):,}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization: {save_path}")

def analyze_dsec_background():
    """分析DSEC背景事件."""
    print("📊 Analyzing DSEC Background Events")
    print("=" * 50)
    
    try:
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Import DSEC efficient loader
        from dsec_efficient import DSECEfficientDataset
        
        # Create dataset
        dataset = DSECEfficientDataset(
            dsec_path=config['data']['dsec_path'],
            sequence_length=config['data']['sequence_length'],
            time_window_us=config['data']['time_window_us']
        )
        
        print(f"DSEC dataset loaded: {len(dataset)} sequences")
        
        # Sample a few sequences
        background_events_list = []
        for i in range(min(3, len(dataset))):
            try:
                events, _ = dataset[i]  # Get events and metadata
                background_events_list.append(events)
                
                # Calculate time span and density
                if len(events) > 0:
                    time_span_us = events[-1, 0] - events[0, 0]  # First column is timestamp
                    time_span_ms = time_span_us / 1000.0
                    density = len(events) / time_span_ms if time_span_ms > 0 else 0
                    
                    print(f"Sample {i+1}:")
                    print(f"  Events: {len(events):,}")
                    print(f"  Time span: {time_span_ms:.1f}ms")
                    print(f"  Density: {density:.1f} events/ms")
                    print(f"  Spatial range: X({events[:,1].min():.0f}-{events[:,1].max():.0f}), Y({events[:,2].min():.0f}-{events[:,2].max():.0f})")
                    print(f"  Polarity dist: {np.unique(events[:,3], return_counts=True)}")
                    
                    # Create visualization for first sample
                    if i == 0:
                        os.makedirs("output/pipeline_analysis", exist_ok=True)
                        create_event_visualization(
                            events, 
                            "DSEC Background Events",
                            "output/pipeline_analysis/dsec_background_events.png"
                        )
                else:
                    print(f"Sample {i+1}: No events")
                    
            except Exception as e:
                print(f"Error loading DSEC sample {i+1}: {e}")
                continue
        
        # Return first valid sample for merging
        for events in background_events_list:
            if len(events) > 0:
                return events
        
        return None
        
    except Exception as e:
        print(f"❌ DSEC analysis failed: {e}")
        traceback.print_exc()
        return None

def analyze_dvs_flare():
    """分析DVS炫光事件."""
    print("\n🔬 Analyzing DVS Flare Events")
    print("=" * 50)
    
    try:
        # Import the integration module
        from dvs_flare_integration import create_flare_event_generator
        
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['event_simulator']['type'] = 'dvs_voltmeter'
        config['debug_mode'] = False
        
        # Create generator
        generator = create_flare_event_generator(config)
        
        # Generate flare events
        flare_events, timing_info = generator.generate_flare_events()
        
        print(f"DVS Flare Events Generated:")
        print(f"  Events: {len(flare_events):,}")
        
        if len(flare_events) > 0:
            duration_sec = timing_info.get('duration_sec', 0.1)
            duration_ms = duration_sec * 1000
            density = len(flare_events) / duration_ms if duration_ms > 0 else 0
            
            print(f"  Duration: {duration_ms:.1f}ms")
            print(f"  Density: {density:.1f} events/ms")
            print(f"  Spatial range: X({flare_events[:,1].min():.0f}-{flare_events[:,1].max():.0f}), Y({flare_events[:,2].min():.0f}-{flare_events[:,2].max():.0f})")
            print(f"  Temporal range: {flare_events[:,0].min():.0f}-{flare_events[:,0].max():.0f} μs")
            print(f"  Polarity dist: {np.unique(flare_events[:,3], return_counts=True)}")
            
            # Create visualization
            os.makedirs("output/pipeline_analysis", exist_ok=True)
            create_event_visualization(
                flare_events,
                "DVS Flare Events", 
                "output/pipeline_analysis/dvs_flare_events.png"
            )
            
            return flare_events
        else:
            print("  ❌ No flare events generated!")
            return None
            
    except Exception as e:
        print(f"❌ DVS flare analysis failed: {e}")
        traceback.print_exc()
        return None

def merge_and_analyze_events(background_events, flare_events):
    """合并并分析完整事件序列."""
    print("\n🔄 Merging and Analyzing Complete Event Sequence")
    print("=" * 60)
    
    if background_events is None or flare_events is None:
        print("❌ Cannot merge - missing background or flare events")
        return None
    
    try:
        print(f"Before merging:")
        print(f"  Background events: {len(background_events):,}")
        print(f"  Flare events: {len(flare_events):,}")
        
        # Simple merge by timestamp (should use proper alignment in real training)
        merged_events = np.vstack([background_events, flare_events])
        
        # Sort by timestamp
        time_sorted_idx = np.argsort(merged_events[:, 0])
        merged_events = merged_events[time_sorted_idx]
        
        print(f"After merging:")
        print(f"  Total events: {len(merged_events):,}")
        
        # Calculate merged statistics
        if len(merged_events) > 0:
            time_span_us = merged_events[-1, 0] - merged_events[0, 0]
            time_span_ms = time_span_us / 1000.0
            density = len(merged_events) / time_span_ms if time_span_ms > 0 else 0
            
            print(f"  Time span: {time_span_ms:.1f}ms")
            print(f"  Merged density: {density:.1f} events/ms")
            print(f"  Spatial range: X({merged_events[:,1].min():.0f}-{merged_events[:,1].max():.0f}), Y({merged_events[:,2].min():.0f}-{merged_events[:,2].max():.0f})")
            
            # Analyze composition
            bg_count = len(background_events)
            flare_count = len(flare_events)
            bg_ratio = bg_count / len(merged_events) * 100
            flare_ratio = flare_count / len(merged_events) * 100
            
            print(f"  Composition: {bg_ratio:.1f}% background, {flare_ratio:.1f}% flare")
            
            # Create visualization
            create_event_visualization(
                merged_events,
                "Merged Events (Background + Flare)",
                "output/pipeline_analysis/merged_events.png"
            )
            
            return merged_events, {
                'background_count': bg_count,
                'flare_count': flare_count,
                'total_count': len(merged_events),
                'density': density,
                'time_span_ms': time_span_ms
            }
        
    except Exception as e:
        print(f"❌ Event merging failed: {e}")
        traceback.print_exc()
        return None, None

def create_comparison_summary(background_events, flare_events, merged_info):
    """创建对比分析总结."""
    print("\n📈 Pipeline Component Analysis Summary")
    print("=" * 70)
    
    # Component densities
    bg_density = len(background_events) / 1000 if background_events is not None else 0  # Approximate
    flare_density = len(flare_events) / 100 if flare_events is not None else 0  # 100ms typical
    merged_density = merged_info['density'] if merged_info else 0
    
    print(f"📊 Event Density Comparison:")
    print(f"  DSEC Background:  ~{bg_density:.1f} events/ms")
    print(f"  DVS Flare:        ~{flare_density:.1f} events/ms")  
    print(f"  Merged Total:     {merged_density:.1f} events/ms")
    print()
    
    if merged_info:
        print(f"📊 Event Count Breakdown:")
        print(f"  Background: {merged_info['background_count']:,} ({merged_info['background_count']/merged_info['total_count']*100:.1f}%)")
        print(f"  Flare:      {merged_info['flare_count']:,} ({merged_info['flare_count']/merged_info['total_count']*100:.1f}%)")
        print(f"  Total:      {merged_info['total_count']:,}")
        print()
    
    # Assessment
    print(f"✅ Pipeline Assessment:")
    if merged_density > 10000:
        print(f"  ⚠️  HIGH: {merged_density:.1f} events/ms - may need optimization")
    elif merged_density < 100:
        print(f"  ⚠️  LOW: {merged_density:.1f} events/ms - may need more sensitivity")
    else:
        print(f"  ✅ GOOD: {merged_density:.1f} events/ms - reasonable for training")
    
    print(f"\n💡 Recommendations:")
    if bg_density > flare_density * 10:
        print(f"  - Flare events are overwhelmed by background ({bg_density/flare_density:.1f}x difference)")
        print(f"  - Consider reducing background time window or increasing flare intensity")
    elif flare_density > bg_density * 10:
        print(f"  - Flare events dominate background ({flare_density/bg_density:.1f}x difference)")
        print(f"  - Consider reducing flare sensitivity or increasing background window")
    else:
        print(f"  - Good balance between background and flare events")
    
    print(f"\n📁 Visualizations saved to: output/pipeline_analysis/")
    print(f"  - dsec_background_events.png")
    print(f"  - dvs_flare_events.png") 
    print(f"  - merged_events.png")

def main():
    """运行完整管线分析."""
    print("EventMamba-FX Complete Pipeline Analysis")
    print("=" * 70)
    print("Analyzing: DSEC Background → DVS Flare → Merged Training Data")
    print("=" * 70)
    
    # Step 1: Analyze DSEC background events
    background_events = analyze_dsec_background()
    
    # Step 2: Analyze DVS flare events
    flare_events = analyze_dvs_flare()
    
    # Step 3: Merge and analyze complete sequence
    merged_events, merged_info = merge_and_analyze_events(background_events, flare_events)
    
    # Step 4: Create comparison summary
    create_comparison_summary(background_events, flare_events, merged_info)
    
    print("\n" + "=" * 70)
    print("COMPLETE PIPELINE ANALYSIS FINISHED")
    print("=" * 70)

if __name__ == "__main__":
    main()