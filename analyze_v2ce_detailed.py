#!/usr/bin/env python3
"""
详细分析V2CE输出的时间戳、坐标和分辨率问题
"""
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.dvs_flare_integration import V2CEFlareEventGenerator

def analyze_v2ce_output():
    """详细分析V2CE的输出特征"""
    print("=" * 60)
    print("V2CE Detailed Analysis")
    print("=" * 60)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/v2ce_analysis'
    config['data']['flare_synthesis']['duration_sec'] = 0.03  # 短序列便于分析
    
    # 创建V2CE生成器
    generator = V2CEFlareEventGenerator(config)
    events, timing = generator.generate_flare_events()
    
    if len(events) == 0:
        print("❌ No events generated")
        return
    
    print(f"📊 Basic Statistics:")
    print(f"  Total events: {len(events)}")
    print(f"  Event format: [timestamp_us, x, y, polarity]")
    
    # 分析时间戳分布
    timestamps = events[:, 0]
    print(f"\n⏰ Timestamp Analysis:")
    print(f"  Range: {timestamps.min()} - {timestamps.max()} μs")
    print(f"  Duration: {(timestamps.max() - timestamps.min()) / 1000:.1f} ms")
    print(f"  Unique timestamps: {len(np.unique(timestamps))}")
    
    # 时间戳分布直方图
    time_bins = np.linspace(timestamps.min(), timestamps.max(), 20)
    hist, _ = np.histogram(timestamps, bins=time_bins)
    print(f"  Time distribution (events per bin):")
    for i, count in enumerate(hist[:10]):  # 显示前10个bin
        bin_start = time_bins[i]
        bin_end = time_bins[i+1]
        print(f"    {bin_start:.0f}-{bin_end:.0f}μs: {count} events")
    
    # 分析空间分布
    x_coords = events[:, 1]
    y_coords = events[:, 2]
    print(f"\n📍 Spatial Analysis:")
    print(f"  X range: {x_coords.min()} - {x_coords.max()}")
    print(f"  Y range: {y_coords.min()} - {y_coords.max()}")
    print(f"  Unique X positions: {len(np.unique(x_coords))}")
    print(f"  Unique Y positions: {len(np.unique(y_coords))}")
    
    # 检查坐标分布模式
    x_hist, x_bins = np.histogram(x_coords, bins=10)
    y_hist, y_bins = np.histogram(y_coords, bins=10)
    print(f"  X distribution (events per region):")
    for i, count in enumerate(x_hist):
        print(f"    x={x_bins[i]:.0f}-{x_bins[i+1]:.0f}: {count} events")
    
    # 分析极性分布
    polarities = events[:, 3]
    unique_pols = np.unique(polarities)
    print(f"\n⚡ Polarity Analysis:")
    print(f"  Unique polarities: {unique_pols}")
    for pol in unique_pols:
        count = np.sum(polarities == pol)
        print(f"    Polarity {pol}: {count} events ({count/len(events)*100:.1f}%)")
    
    # 分析V2CE特定问题
    print(f"\n🔍 V2CE Specific Analysis:")
    
    # 1. 检查分辨率问题
    target_w, target_h = 640, 480
    v2ce_w, v2ce_h = 346, 260
    print(f"  V2CE resolution: {v2ce_w}x{v2ce_h}")
    print(f"  Target resolution: {target_w}x{target_h}")
    print(f"  Scale factors: x={target_w/v2ce_w:.3f}, y={target_h/v2ce_h:.3f}")
    
    # 检查是否有超出V2CE原始分辨率的坐标
    x_in_v2ce = (x_coords >= 0) & (x_coords < target_w)
    y_in_v2ce = (y_coords >= 0) & (y_coords < target_h)
    valid_coords = x_in_v2ce & y_in_v2ce
    print(f"  Valid coordinates: {np.sum(valid_coords)}/{len(events)} ({np.sum(valid_coords)/len(events)*100:.1f}%)")
    
    # 2. 检查时间戳的帧率一致性
    frame_duration = 1.0 / 800 * 1e6  # 800 FPS in microseconds
    expected_frame_times = np.arange(0, timing.get('duration_sec', 0.03) * 1e6, frame_duration)
    print(f"  Expected frame duration: {frame_duration:.0f}μs")
    print(f"  Expected frame count: {len(expected_frame_times)}")
    
    # 检查事件是否集中在特定时间点
    time_clusters = []
    for exp_time in expected_frame_times[:10]:  # 检查前10帧
        nearby_events = np.sum(np.abs(timestamps - exp_time) < frame_duration/2)
        time_clusters.append(nearby_events)
        if nearby_events > 0:
            print(f"    Near frame time {exp_time:.0f}μs: {nearby_events} events")
    
    # 3. 检查事件的时空相关性
    print(f"\n🎯 Event Clustering Analysis:")
    early_events = events[timestamps < timestamps.min() + (timestamps.max() - timestamps.min()) * 0.1]
    late_events = events[timestamps > timestamps.max() - (timestamps.max() - timestamps.min()) * 0.1]
    
    print(f"  Early 10% events: {len(early_events)}")
    print(f"  Late 10% events: {len(late_events)}")
    
    if len(early_events) > 0:
        print(f"  Early events X range: {early_events[:,1].min():.0f}-{early_events[:,1].max():.0f}")
        print(f"  Early events Y range: {early_events[:,2].min():.0f}-{early_events[:,2].max():.0f}")
    
    # 4. 检查是否存在V2CE的seq_len影响
    seq_len = config['data']['event_simulator']['v2ce']['seq_len']
    print(f"\n📚 Sequence Length Analysis:")
    print(f"  V2CE seq_len: {seq_len}")
    
    # 按序列分段分析事件分布
    total_duration = timestamps.max() - timestamps.min()
    segment_duration = total_duration / seq_len
    
    for i in range(min(seq_len, 5)):  # 检查前5个段
        seg_start = timestamps.min() + i * segment_duration
        seg_end = seg_start + segment_duration
        seg_events = np.sum((timestamps >= seg_start) & (timestamps < seg_end))
        print(f"  Segment {i} ({seg_start:.0f}-{seg_end:.0f}μs): {seg_events} events")
    
    return events, timing

if __name__ == "__main__":
    events, timing = analyze_v2ce_output()
    print(f"\n✅ Analysis completed")