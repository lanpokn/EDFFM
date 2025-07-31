#!/usr/bin/env python3
"""
测试优化后的DVS-Voltmeter参数
"""
import yaml
import time
from src.dvs_flare_integration import create_flare_event_generator

def test_dvs_optimized():
    """测试优化后的DVS参数配置"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保使用DVS
    config['data']['event_simulator']['type'] = 'dvs_voltmeter'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/dvs_optimized_test'
    
    # 测试不同时长
    test_durations = [30, 50]  # ms
    
    print("🚀 DVS优化参数测试")
    print("=" * 50)
    print("🎯 优化目标: 减少事件数量，降低帧率")
    print("📊 DVS参数: 2x阈值提升")
    print("📈 帧率限制: 400fps (降低自1600fps)")
    print("🔄 采样点数: 12/周期 (降低自24)")
    
    results = []
    
    for duration_ms in test_durations:
        print(f"\n{'='*40}")
        print(f"测试时长: {duration_ms}ms")
        print(f"{'='*40}")
        
        config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000
        config['debug_output_dir'] = f'./output/dvs_optimized_{duration_ms}ms'
        
        try:
            start_time = time.time()
            generator = create_flare_event_generator(config)
            events, timing = generator.generate_flare_events()
            total_time = time.time() - start_time
            
            if len(events) > 0:
                timestamps = events[:, 0]
                actual_duration = (timestamps.max() - timestamps.min()) / 1000
                
                result = {
                    'duration_ms': duration_ms,
                    'actual_duration_ms': actual_duration,
                    'total_events': len(events),
                    'event_density_per_ms': len(events) / actual_duration,
                    'processing_time_sec': total_time,
                    'frames_generated': timing.get('total_frames', 'N/A'),
                    'fps_used': timing.get('fps', 'N/A'),
                    'dvs_simulation_sec': timing.get('dvs_simulation_sec', 'N/A')
                }
                
                results.append(result)
                
                print(f"✅ 成功生成 {len(events):,} 个事件")
                print(f"📊 事件密度: {result['event_density_per_ms']:.0f} events/ms")
                print(f"🎞️  生成帧数: {result['frames_generated']}")
                print(f"📈 使用帧率: {result['fps_used']} fps")
                print(f"⏱️  DVS仿真时间: {result['dvs_simulation_sec']:.2f}s")
                print(f"🕒 总处理时间: {total_time:.2f}s")
                print(f"🎯 时间精度: {duration_ms/actual_duration*100:.1f}%")
                
                # 极性分析
                pos_events = (events[:, 3] > 0).sum()
                neg_events = (events[:, 3] <= 0).sum()
                print(f"⚖️  极性分布: {pos_events}正/{neg_events}负 ({pos_events/len(events)*100:.1f}%/{neg_events/len(events)*100:.1f}%)")
                
                # 检查debug文件
                debug_dir = f'./output/dvs_optimized_{duration_ms}ms'
                print(f"📁 Debug输出: {debug_dir}/flare_seq_000/")
                print(f"   - original_frames/: {result['frames_generated']} 帧")
                print(f"   - event_visualizations/: 多分辨率(0.5x/1x/2x/4x)")
                
            else:
                print("❌ 未生成事件")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 对比分析
    if len(results) >= 2:
        print(f"\n🔍 对比分析")
        print(f"{'='*50}")
        print(f"{'时长':<8} {'事件数':<10} {'密度':<12} {'帧数':<8} {'FPS':<8}")
        print(f"{'-'*50}")
        
        for r in results:
            print(f"{r['duration_ms']:>5}ms {r['total_events']:>9,} "
                  f"{r['event_density_per_ms']:>9.0f}/ms {r['frames_generated']:>6} {r['fps_used']:>6}")
        
        # 评估改进效果
        avg_density = sum(r['event_density_per_ms'] for r in results) / len(results)
        total_events = sum(r['total_events'] for r in results)
        
        print(f"\n📈 优化效果评估:")
        print(f"   平均事件密度: {avg_density:.0f} events/ms")
        print(f"   总事件数: {total_events:,}")
        print(f"   预期改进: 相比之前应该显著减少")
        
        # 与之前V2CE对比的参考
        print(f"\n📊 参考对比 (V2CE vs 优化DVS):")
        print(f"   V2CE 30ms: ~90K events (~3000 events/ms)")
        print(f"   优化DVS 30ms: {results[0]['total_events']:,} events ({results[0]['event_density_per_ms']:.0f} events/ms)")
        
        if results[0]['event_density_per_ms'] < 2000:
            print(f"   ✅ 成功：事件数量显著减少")
        else:
            print(f"   ⚠️ 仍需进一步优化参数")

if __name__ == "__main__":
    test_dvs_optimized()