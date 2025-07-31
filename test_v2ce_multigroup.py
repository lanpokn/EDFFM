#!/usr/bin/env python3
"""
V2CE多组测试 - 不同时长和频率组合
"""
import yaml
import time
from src.dvs_flare_integration import create_flare_event_generator

def test_v2ce_multiple_groups():
    """测试V2CE在不同配置下的表现"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 测试配置组合
    test_groups = [
        {'duration_ms': 30, 'description': '短时炫光 (典型)'},
        {'duration_ms': 50, 'description': '中时炫光 (扩展)'},
        {'duration_ms': 100, 'description': '长时炫光 (极端)'},
    ]
    
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    
    print("🚀 V2CE多组测试开始")
    print("=" * 60)
    
    results = []
    
    for i, test_group in enumerate(test_groups):
        duration_ms = test_group['duration_ms']
        description = test_group['description']
        
        print(f"\n📊 测试组 {i+1}: {description}")
        print(f"   时长: {duration_ms}ms")
        print("-" * 40)
        
        config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000
        config['debug_output_dir'] = f'./output/v2ce_multitest_{duration_ms}ms'
        
        try:
            start_time = time.time()
            generator = create_flare_event_generator(config)
            events, timing = generator.generate_flare_events()
            total_time = time.time() - start_time
            
            if len(events) > 0:
                timestamps = events[:, 0]
                actual_duration = (timestamps.max() - timestamps.min()) / 1000
                
                # 时间分布分析
                import numpy as np
                hist, bin_edges = np.histogram(timestamps, bins=8)
                non_zero_bins = np.sum(hist > 0)
                temporal_coverage = non_zero_bins / len(hist) * 100
                
                result = {
                    'group': i + 1,
                    'duration_target_ms': duration_ms,
                    'duration_actual_ms': actual_duration,
                    'time_accuracy_percent': duration_ms / actual_duration * 100,
                    'total_events': len(events),
                    'event_density_per_ms': len(events) / actual_duration,
                    'processing_time_sec': total_time,
                    'temporal_coverage_percent': temporal_coverage,
                    'polarity_balance': {
                        'positive': int((events[:, 3] > 0).sum()),
                        'negative': int((events[:, 3] <= 0).sum())
                    },
                    'metadata': timing
                }
                
                results.append(result)
                
                print(f"✅ 事件数量: {len(events):,}")
                print(f"⏱️  实际时长: {actual_duration:.1f}ms (精度: {result['time_accuracy_percent']:.1f}%)")
                print(f"📈 事件密度: {result['event_density_per_ms']:.0f} events/ms")
                print(f"🕒 处理时间: {total_time:.2f}s")
                print(f"📊 时间覆盖: {temporal_coverage:.1f}% (8个时间区间)")
                print(f"⚖️  极性平衡: {result['polarity_balance']['positive']}正/{result['polarity_balance']['negative']}负")
                
                # 显示多分辨率可视化信息
                vis_counts = {
                    '0.5x': len(events) // 2,  # 估算
                    '1x': len(events),
                    '2x': len(events) * 2,      # 估算  
                    '4x': len(events) * 4       # 估算
                }
                print(f"🎨 多分辨率可视化: 0.5x/1x/2x/4x 已生成到 ./output/v2ce_multitest_{duration_ms}ms/")
                
            else:
                print("❌ 未生成事件")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    # 生成总结报告
    if results:
        print(f"\n🎯 多组测试总结")
        print("=" * 60)
        print(f"{'组别':<4} {'时长':<8} {'事件数':<10} {'精度':<8} {'密度':<12} {'处理时间':<8}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['group']:<4} {r['duration_target_ms']:>5}ms {r['total_events']:>9,} "
                  f"{r['time_accuracy_percent']:>6.1f}% {r['event_density_per_ms']:>9.0f}/ms {r['processing_time_sec']:>6.2f}s")
        
        # 性能指标
        avg_accuracy = sum(r['time_accuracy_percent'] for r in results) / len(results)
        avg_density = sum(r['event_density_per_ms'] for r in results) / len(results)
        total_events = sum(r['total_events'] for r in results)
        
        print(f"\n📈 平均指标:")
        print(f"   时间精度: {avg_accuracy:.1f}%")
        print(f"   事件密度: {avg_density:.0f} events/ms")
        print(f"   总事件数: {total_events:,}")
        
        print(f"\n🎨 可视化输出:")
        print(f"   每组生成4种分辨率 (0.5x, 1x, 2x, 4x)")
        print(f"   输出目录: ./output/v2ce_multitest_[duration]ms/flare_seq_v2ce_000/v2ce_event_visualizations/")
        print(f"   文件结构: temporal_[resolution]x/ 子目录")

if __name__ == "__main__":
    test_v2ce_multiple_groups()