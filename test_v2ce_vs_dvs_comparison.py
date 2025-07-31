#!/usr/bin/env python3
"""
V2CE vs DVS-Voltmeter 对比测试
"""
import yaml
import time
from src.dvs_flare_integration import create_flare_event_generator

def test_simulators_comparison():
    """对比V2CE和DVS-Voltmeter的性能和输出质量"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 测试参数
    duration_ms = 30
    config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000
    config['debug_mode'] = True
    
    simulators = ['v2ce', 'dvs_voltmeter']
    results = {}
    
    for sim_type in simulators:
        print(f"\n{'='*60}")
        print(f"测试 {sim_type.upper()} 仿真器")
        print(f"{'='*60}")
        
        config['data']['event_simulator']['type'] = sim_type
        config['debug_output_dir'] = f'./output/{sim_type}_comparison_{duration_ms}ms'
        
        try:
            start_time = time.time()
            generator = create_flare_event_generator(config)
            events, timing = generator.generate_flare_events()
            total_time = time.time() - start_time
            
            # 收集结果
            result = {
                'simulator': sim_type,
                'total_events': len(events),
                'total_time_sec': total_time,
                'timing_info': timing
            }
            
            if len(events) > 0:
                timestamps = events[:, 0] 
                duration_actual = (timestamps.max() - timestamps.min()) / 1000
                result.update({
                    'time_span_ms': duration_actual,
                    'time_accuracy_percent': duration_ms / duration_actual * 100,
                    'event_rate_per_sec': len(events) / (duration_actual / 1000),
                    'spatial_range_x': [events[:, 1].min(), events[:, 1].max()],
                    'spatial_range_y': [events[:, 2].min(), events[:, 2].max()],
                    'polarity_stats': {
                        'positive': int((events[:, 3] > 0).sum()),
                        'negative': int((events[:, 3] <= 0).sum())
                    }
                })
                
                # 时间分布分析
                import numpy as np
                hist, _ = np.histogram(timestamps, bins=5)
                uniformity = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else float('inf')
                result['temporal_uniformity'] = uniformity
                
            results[sim_type] = result
            
            # 显示结果
            print(f"✅ 成功生成 {len(events)} 个事件")
            print(f"⏱️  总耗时: {total_time:.2f}s")
            if len(events) > 0:
                print(f"📊 时间跨度: {result['time_span_ms']:.1f}ms (期望: {duration_ms}ms)")
                print(f"🎯 时间精度: {result['time_accuracy_percent']:.1f}%")
                print(f"⚡ 事件率: {result['event_rate_per_sec']:.0f} events/sec")
                print(f"🔴 正极性: {result['polarity_stats']['positive']} ({result['polarity_stats']['positive']/len(events)*100:.1f}%)")
                print(f"🔵 负极性: {result['polarity_stats']['negative']} ({result['polarity_stats']['negative']/len(events)*100:.1f}%)")
                print(f"📈 时间均匀性: {result['temporal_uniformity']:.3f} (越小越均匀)")
                
        except Exception as e:
            print(f"❌ {sim_type} 测试失败: {e}")
            results[sim_type] = {'error': str(e)}
    
    # 对比总结
    print(f"\n{'='*60}")
    print(f"🏆 对比总结")
    print(f"{'='*60}")
    
    if len(results) == 2 and all('error' not in r for r in results.values()):
        v2ce_result = results['v2ce']
        dvs_result = results['dvs_voltmeter']
        
        print(f"📊 事件数量:")
        print(f"  V2CE: {v2ce_result['total_events']:,}")
        print(f"  DVS:  {dvs_result['total_events']:,}")
        print(f"  比率: {v2ce_result['total_events']/dvs_result['total_events']:.2f}x (V2CE/DVS)")
        
        print(f"\n⏱️  处理时间:")
        print(f"  V2CE: {v2ce_result['total_time_sec']:.2f}s")
        print(f"  DVS:  {dvs_result['total_time_sec']:.2f}s")
        print(f"  比率: {dvs_result['total_time_sec']/v2ce_result['total_time_sec']:.2f}x (DVS/V2CE)")
        
        print(f"\n🎯 时间精度:")
        print(f"  V2CE: {v2ce_result['time_accuracy_percent']:.1f}%")
        print(f"  DVS:  {dvs_result['time_accuracy_percent']:.1f}%")
        
        print(f"\n📈 时间均匀性 (标准差/均值):")
        print(f"  V2CE: {v2ce_result['temporal_uniformity']:.3f}")
        print(f"  DVS:  {dvs_result['temporal_uniformity']:.3f}")
        
        # 推荐
        print(f"\n🚀 推荐:")
        if v2ce_result['time_accuracy_percent'] > 95 and v2ce_result['total_time_sec'] < dvs_result['total_time_sec']:
            print(f"  ✅ V2CE: 更高时间精度 + 更快速度")
        elif dvs_result['total_events'] > v2ce_result['total_events'] * 1.5:
            print(f"  ✅ DVS: 更多事件数量")
        else:
            print(f"  ⚖️  两者各有优势，根据需求选择")

if __name__ == "__main__":
    test_simulators_comparison()