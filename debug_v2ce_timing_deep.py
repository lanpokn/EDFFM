#!/usr/bin/env python3
"""
深度理解V2CE的时间戳生成机制
"""
import sys
import torch
import numpy as np
from functools import partial

sys.path.insert(0, "/mnt/e/2025/event_flick_flare/main/simulator/V2CE-Toolbox-master")
from scripts.LDATI import sample_voxel_statistical

def analyze_v2ce_timing():
    """分析V2CE的时间戳生成机制"""
    
    print("🔍 V2CE时间戳生成机制分析")
    print("="*60)
    
    # 测试不同的fps值
    test_fps_values = [30, 60, 200, 400, 800, 1600]
    
    for fps in test_fps_values:
        print(f"\n测试 fps = {fps}")
        print("-" * 30)
        
        # 创建简单的测试voxel (单个时间步)
        # Shape: [batch=1, polarity=2, time_bins=10, height=260, width=346]
        test_voxel = torch.randn(1, 2, 10, 260, 346).cuda() * 5.0  # 增大幅度确保有事件
        
        # 使用LDATI生成事件
        ldati = partial(
            sample_voxel_statistical, 
            fps=fps,
            bidirectional=False, 
            additional_events_strategy='slope'
        )
        
        try:
            events_list = ldati(test_voxel)
            
            if len(events_list) > 0 and len(events_list[0]) > 0:
                events = events_list[0]
                timestamps = events['timestamp']
                
                print(f"  生成事件数: {len(events)}")
                print(f"  时间戳范围: {timestamps.min()} - {timestamps.max()} μs")
                print(f"  时间跨度: {(timestamps.max() - timestamps.min()) / 1000:.1f} ms")
                
                # 计算理论值
                # 在V2CE中，1个voxel对应1帧的时间间隔
                frame_interval_us = 1.0 / fps * 1e6
                print(f"  理论帧间隔: {frame_interval_us:.1f} μs")
                
                # 分析时间戳分布
                unique_times = np.unique(timestamps)
                if len(unique_times) > 1:
                    time_diffs = np.diff(unique_times)
                    print(f"  最小时间间隔: {time_diffs.min():.1f} μs")
                    print(f"  平均时间间隔: {time_diffs.mean():.1f} μs")
                    print(f"  最大时间间隔: {time_diffs.max():.1f} μs")
                
            else:
                print(f"  ❌ 没有生成事件")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    print(f"\n🎯 多帧序列测试")
    print("="*40)
    
    # 测试多帧序列
    fps = 800  # 使用我们当前的fps
    num_frames = 48  # 对应30ms@1600fps的帧数
    
    print(f"测试参数: fps={fps}, frames={num_frames}")
    
    # 创建多帧voxel序列
    multi_voxel = torch.randn(num_frames, 2, 10, 260, 346).cuda() * 3.0
    
    ldati = partial(
        sample_voxel_statistical, 
        fps=fps,
        bidirectional=False, 
        additional_events_strategy='slope'
    )
    
    # 模拟V2CE主循环的时间戳处理
    event_stream_per_frame = []
    for i in range(0, multi_voxel.shape[0], 24):  # 使用batch_size=24
        batch_events = ldati(multi_voxel[i:i+24])
        event_stream_per_frame.extend(batch_events)
    
    # 模拟时间戳累加 (来自v2ce.py第369行)
    total_events = 0
    all_timestamps = []
    
    for i in range(len(event_stream_per_frame)):
        if len(event_stream_per_frame[i]) > 0:
            # 每帧的时间戳基础偏移
            frame_offset = int(i * 1 / fps * 1e6)  # 第i帧的时间戳偏移
            adjusted_timestamps = event_stream_per_frame[i]['timestamp'] + frame_offset
            all_timestamps.extend(adjusted_timestamps)
            total_events += len(event_stream_per_frame[i])
    
    if len(all_timestamps) > 0:
        all_timestamps = np.array(all_timestamps)
        print(f"  总事件数: {total_events}")
        print(f"  时间戳范围: {all_timestamps.min()} - {all_timestamps.max()} μs")
        print(f"  总时间跨度: {(all_timestamps.max() - all_timestamps.min()) / 1000:.1f} ms")
        
        # 计算期望的时间跨度
        expected_duration_ms = (num_frames - 1) * (1 / fps) * 1000
        print(f"  期望时间跨度: {expected_duration_ms:.1f} ms")
        print(f"  时间匹配度: {(all_timestamps.max() - all_timestamps.min()) / 1000 / expected_duration_ms * 100:.1f}%")

if __name__ == "__main__":
    analyze_v2ce_timing()