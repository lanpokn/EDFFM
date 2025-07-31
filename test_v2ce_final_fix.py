#!/usr/bin/env python3
"""
Test V2CE with corrected FPS calculation
"""
import yaml
from src.dvs_flare_integration import create_flare_event_generator

def test_v2ce_final():
    """Test V2CE with corrected FPS that matches video duration"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure for test
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/v2ce_final_fix'
    
    # Test with different durations to verify FPS calculation
    test_durations = [0.03, 0.05, 0.1]  # 30ms, 50ms, 100ms
    
    for duration in test_durations:
        print(f"\n" + "="*60)
        print(f"Testing V2CE with {duration*1000:.0f}ms video duration")
        print(f"="*60)
        
        config['data']['flare_synthesis']['duration_sec'] = duration
        config['debug_output_dir'] = f'./output/v2ce_final_{int(duration*1000)}ms'
        
        try:
            generator = create_flare_event_generator(config)
            events, timing = generator.generate_flare_events()
            
            print(f"\n📊 Results for {duration*1000:.0f}ms video:")
            print(f"  Events generated: {len(events)}")
            print(f"  Total time: {timing.get('total_pipeline_sec', 0):.3f}s")
            
            if len(events) > 0:
                event_duration_ms = (events[-1,0] - events[0,0]) / 1000
                print(f"  ✅ SUCCESS!")
                print(f"  Event time span: {event_duration_ms:.1f} ms")
                print(f"  Expected duration: {duration*1000:.0f} ms")
                print(f"  Time coverage: {event_duration_ms/(duration*1000)*100:.1f}%")
                print(f"  Event rate: {len(events) / (event_duration_ms/1000):.0f} events/sec")
                print(f"  Spatial range: x=[{events[:,1].min():.0f}-{events[:,1].max():.0f}], y=[{events[:,2].min():.0f}-{events[:,2].max():.0f}]")
                
                # 检查时间分布
                timestamps = events[:, 0]
                time_bins = 10
                import numpy as np
                hist, bin_edges = np.histogram(timestamps, bins=time_bins)
                print(f"  Time distribution (events per {event_duration_ms/time_bins:.1f}ms bin):")
                
                non_zero_bins = 0
                for i, count in enumerate(hist):
                    if count > 0:
                        non_zero_bins += 1
                        if i < 5:  # 只显示前5个非零bin
                            bin_start = (bin_edges[i] - timestamps.min()) / 1000
                            bin_end = (bin_edges[i+1] - timestamps.min()) / 1000
                            print(f"    {bin_start:.1f}-{bin_end:.1f}ms: {count} events")
                
                print(f"  Non-zero time bins: {non_zero_bins}/{time_bins} ({non_zero_bins/time_bins*100:.0f}%)")
                
                # 分析空间分布均匀性
                x_std = np.std(events[:,1])
                y_std = np.std(events[:,2])
                print(f"  Spatial spread: σ_x={x_std:.1f}, σ_y={y_std:.1f}")
                
            else:
                print(f"  ❌ No events generated")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎯 Final Assessment:")
    print(f"V2CE integration should now:")
    print(f"  1. Generate events covering the full video duration")
    print(f"  2. Have more uniform temporal distribution")
    print(f"  3. Use appropriate FPS matching video length")

if __name__ == "__main__":
    test_v2ce_final()