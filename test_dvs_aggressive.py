#!/usr/bin/env python3
"""
测试激进优化的DVS参数 (大幅减少事件数量)
"""
import yaml
import time
from src.dvs_flare_integration import create_flare_event_generator

def test_dvs_aggressive():
    """测试激进优化的DVS参数配置"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保使用DVS
    config['data']['event_simulator']['type'] = 'dvs_voltmeter'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/dvs_aggressive_test'
    
    print("🚀 DVS激进优化参数测试")
    print("=" * 50)
    print("🎯 激进优化: 大幅减少事件数量")
    print("📊 DVS参数: 4x+阈值提升")
    print("📈 帧率限制: 200fps (从1600fps大幅降低)")
    print("🔄 采样点数: 8/周期 (最小值)")
    print("⚠️  目标: 获得与V2CE类似的事件数量级")
    
    # 单次30ms测试
    duration_ms = 30
    config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000
    config['debug_output_dir'] = f'./output/dvs_aggressive_{duration_ms}ms'
    
    try:
        start_time = time.time()
        generator = create_flare_event_generator(config)
        events, timing = generator.generate_flare_events()
        total_time = time.time() - start_time
        
        if len(events) > 0:
            timestamps = events[:, 0]
            actual_duration = (timestamps.max() - timestamps.min()) / 1000
            
            print(f"\n✅ 成功生成 {len(events):,} 个事件")
            print(f"📊 事件密度: {len(events)/actual_duration:.0f} events/ms")
            print(f"🎞️  生成帧数: {timing.get('total_frames', 'N/A')}")
            print(f"📈 使用帧率: {timing.get('fps', 'N/A')} fps")
            print(f"⏱️  DVS仿真时间: {timing.get('dvs_simulation_sec', 'N/A'):.2f}s")
            print(f"🕒 总处理时间: {total_time:.2f}s")
            print(f"🎯 时间精度: {duration_ms/actual_duration*100:.1f}%")
            
            # 极性分析
            pos_events = (events[:, 3] > 0).sum()
            neg_events = (events[:, 3] <= 0).sum()
            print(f"⚖️  极性分布: {pos_events}正/{neg_events}负 ({pos_events/len(events)*100:.1f}%/{neg_events/len(events)*100:.1f}%)")
            
            # 对比评估
            print(f"\n📊 与V2CE对比:")
            print(f"   V2CE 30ms: ~90,000 events (~3,000 events/ms)")
            print(f"   激进DVS 30ms: {len(events):,} events ({len(events)/actual_duration:.0f} events/ms)")
            
            # 改进评估
            improvement_factor = 196855 / (len(events)/actual_duration) if len(events) > 0 else 0
            print(f"   改进倍数: {improvement_factor:.1f}x减少")
            
            if len(events)/actual_duration < 10000:
                print(f"   ✅ 成功：事件数量已接近合理范围")
            elif len(events)/actual_duration < 50000:
                print(f"   🔶 改善：事件数量显著减少但仍偏高")
            else:
                print(f"   ❌ 仍需更激进的参数调整")
            
            # Debug文件信息
            debug_dir = f'./output/dvs_aggressive_{duration_ms}ms'
            print(f"\n📁 Debug输出: {debug_dir}/flare_seq_000/")
            print(f"   - original_frames/: {timing.get('total_frames', 'N/A')} 帧")
            print(f"   - event_visualizations/: 多分辨率(0.5x/1x/2x/4x)")
            
        else:
            print("❌ 未生成事件 - 参数可能过于激进")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print(f"\n🎯 参数调优建议:")
    print(f"   如果事件数量仍过多，可进一步：")
    print(f"   1. 提高k1参数 (当前21.06 → 30+)")
    print(f"   2. 提高k2参数 (当前35 → 50+)")
    print(f"   3. 提高k3参数 (当前0.001 → 0.01+)")
    print(f"   4. 降低帧率至100fps")

if __name__ == "__main__":
    test_dvs_aggressive()