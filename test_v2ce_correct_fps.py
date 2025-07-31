#!/usr/bin/env python3
"""
Test V2CE with correct FPS understanding
V2CE的fps应该匹配输入视频的实际帧率，而不是我们想要的事件帧率
"""
import yaml
from src.dvs_flare_integration import V2CEFlareEventGenerator

def test_v2ce_correct_fps():
    """Test V2CE with correct FPS parameter understanding"""
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/v2ce_correct_fps'
    
    # 关键修正：V2CE的fps应该匹配我们生成的视频帧率
    # 我们的flare_synthesis生成的视频是1600fps，所以V2CE也应该用相近的fps
    flare_config = config['data']['flare_synthesis']
    flare_fps = flare_config.get('max_fps', 1600)  # 获取实际的flare视频fps
    
    # 但是V2CE可能无法处理太高的fps，我们试试几个合理的值
    test_fps_values = [30, 60, 120, 200, 400]
    
    print(f"Testing V2CE with different FPS values...")
    print(f"Our flare video is generated at ~{flare_fps} fps")
    
    for test_fps in test_fps_values:
        print(f"\n" + "="*50)
        print(f"Testing with fps = {test_fps}")
        print(f"="*50)
        
        # 临时修改配置
        config['data']['event_simulator']['v2ce']['base_fps'] = test_fps
        config['debug_output_dir'] = f'./output/v2ce_fps_{test_fps}'
        
        try:
            generator = V2CEFlareEventGenerator(config)
            
            # 直接调用LDATI测试，绕过我们的动态fps计算
            import sys
            import os
            v2ce_path = "/mnt/e/2025/event_flick_flare/main/simulator/V2CE-Toolbox-master"
            sys.path.insert(0, v2ce_path)
            
            from scripts.LDATI import sample_voxel_statistical
            from functools import partial
            import torch
            
            # 创建测试用的dummy voxel
            dummy_voxel = torch.randn(2, 2, 10, 260, 346).cuda()
            
            # 使用测试的fps
            ldati = partial(
                sample_voxel_statistical, 
                fps=test_fps,
                bidirectional=False, 
                additional_events_strategy='slope'
            )
            
            events_list = ldati(dummy_voxel)
            
            total_events = sum(len(events) for events in events_list)
            print(f"  Total events: {total_events}")
            
            if total_events > 0:
                # 分析第一个事件流的时间戳
                first_events = events_list[0]
                if len(first_events) > 0:
                    timestamps = first_events['timestamp']
                    print(f"  Time range: {timestamps.min()} - {timestamps.max()} μs")
                    print(f"  Duration: {(timestamps.max() - timestamps.min()) / 1000:.1f} ms")
                    
                    # 计算期望的时间间隔
                    expected_frame_interval = 1.0 / test_fps * 1e6  # 微秒
                    expected_voxel_interval = expected_frame_interval / 9  # 9个时间片
                    
                    print(f"  Expected frame interval: {expected_frame_interval:.0f} μs")
                    print(f"  Expected voxel interval: {expected_voxel_interval:.0f} μs")
                    
                    # 检查时间戳分布
                    unique_times = len(np.unique(timestamps))
                    print(f"  Unique timestamps: {unique_times}")
            
        except Exception as e:
            print(f"  ❌ Error with fps={test_fps}: {e}")
    
    print(f"\n🎯 Conclusion:")
    print(f"The optimal fps for V2CE should match the temporal resolution we want")
    print(f"Lower fps = longer time intervals = less temporal precision")
    print(f"Higher fps = shorter time intervals = better temporal precision")

if __name__ == "__main__":
    import numpy as np
    test_v2ce_correct_fps()