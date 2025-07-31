#!/usr/bin/env python3
"""
测试超激进DVS参数 (匹配V2CE事件数量级)
"""
import yaml
import time
from src.dvs_flare_integration import create_flare_event_generator

def test_dvs_ultra():
    """测试超激进DVS参数以匹配V2CE事件数量"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保使用DVS
    config['data']['event_simulator']['type'] = 'dvs_voltmeter'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/dvs_ultra_test'
    
    print("🚀 DVS超激进参数测试")
    print("=" * 50)
    print("🎯 超激进优化: 匹配V2CE事件数量级")
    print("📊 DVS参数: 10x+阈值提升")
    print("📈 帧率限制: 100fps (极低)")
    print("🔄 采样点数: 6/周期 (最小)")
    print("🎯 目标事件密度: ~3,000 events/ms (V2CE水平)")
    
    # 快速30ms测试
    duration_ms = 30
    config['data']['flare_synthesis']['duration_sec'] = duration_ms / 1000
    config['debug_output_dir'] = f'./output/dvs_ultra_{duration_ms}ms'
    
    try:
        start_time = time.time()
        generator = create_flare_event_generator(config)
        events, timing = generator.generate_flare_events()
        total_time = time.time() - start_time
        
        if len(events) > 0:
            timestamps = events[:, 0]
            actual_duration = (timestamps.max() - timestamps.min()) / 1000
            event_density = len(events) / actual_duration
            
            print(f"\n✅ 成功生成 {len(events):,} 个事件")
            print(f"📊 事件密度: {event_density:.0f} events/ms")
            print(f"🎞️  生成帧数: {timing.get('total_frames', 'N/A')}")
            print(f"📈 使用帧率: {timing.get('fps', 'N/A')} fps")
            print(f"⏱️  DVS仿真时间: {timing.get('dvs_simulation_sec', 'N/A'):.2f}s")
            print(f"🕒 总处理时间: {total_time:.2f}s")
            print(f"🎯 时间精度: {duration_ms/actual_duration*100:.1f}%")
            
            # 极性分析
            pos_events = (events[:, 3] > 0).sum()
            neg_events = (events[:, 3] <= 0).sum()
            print(f"⚖️  极性分布: {pos_events}正/{neg_events}负 ({pos_events/len(events)*100:.1f}%/{neg_events/len(events)*100:.1f}%)")
            
            # 关键对比评估
            print(f"\n🎯 关键对比:")
            print(f"   目标 (V2CE): ~90,000 events (~3,000 events/ms)")
            print(f"   当前 (DVS): {len(events):,} events ({event_density:.0f} events/ms)")
            
            ratio_to_target = event_density / 3000
            if ratio_to_target <= 2.0:
                print(f"   ✅ 优秀: {ratio_to_target:.1f}x V2CE密度，接近目标")
                success = True
            elif ratio_to_target <= 5.0:
                print(f"   🔶 良好: {ratio_to_target:.1f}x V2CE密度，基本可接受")
                success = True
            elif ratio_to_target <= 10.0:
                print(f"   ⚠️ 偏高: {ratio_to_target:.1f}x V2CE密度，仍需优化")
                success = False
            else:
                print(f"   ❌ 过高: {ratio_to_target:.1f}x V2CE密度，需要更激进参数")
                success = False
            
            # 时间分布分析
            import numpy as np
            hist, _ = np.histogram(timestamps, bins=5)
            uniformity = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else float('inf')
            print(f"   📊 时间分布均匀性: {uniformity:.3f} (越小越好)")
            
            # Debug文件信息
            debug_dir = f'./output/dvs_ultra_{duration_ms}ms'
            print(f"\n📁 Debug输出: {debug_dir}/flare_seq_000/")
            print(f"   - original_frames/: {timing.get('total_frames', 'N/A')} 帧")
            print(f"   - event_visualizations/: 多分辨率(0.5x/1x/2x/4x)")
            
            # 最终评估
            print(f"\n🏆 最终评估:")
            if success:
                print(f"   ✅ DVS参数调优成功！")
                print(f"   ✅ 事件数量已降至合理范围")
                print(f"   ✅ 可以用于训练，泛化性应优于V2CE")
            else:
                print(f"   ⚠️ 需要进一步参数调优")
                print(f"   💡 建议: k1→80+, k2→120+, k3→0.1+")
            
        else:
            print("❌ 未生成事件 - 参数过于激进，请适当降低阈值")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("💡 可能需要检查DVS参数格式或降低阈值")

if __name__ == "__main__":
    test_dvs_ultra()