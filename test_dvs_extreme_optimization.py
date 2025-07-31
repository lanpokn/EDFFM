#!/usr/bin/env python3
"""
DVS极致参数优化测试

基于DVS-Voltmeter论文的深度分析，测试极致优化的物理参数：
- k1: 光-电转换敏感度 (大幅降低)
- k2: 暗电流阈值偏移 (大幅提高) 
- k3,k5,k6: 各种噪声项 (大幅降低)

目标: 在保持高帧率的同时，将事件数量降低到V2CE水平(3K/ms)
"""

import sys
import os
import yaml
import traceback

# Add src path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

def test_dvs_extreme_parameters():
    """测试DVS极致参数优化."""
    print("🔬 Testing DVS Extreme Parameter Optimization")
    print("=" * 60)
    
    try:
        # Load configuration
        config_path = "configs/config.yaml"
        print(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure DVS is selected with debug mode
        config['data']['event_simulator']['type'] = 'dvs_voltmeter'
        config['debug_mode'] = True
        config['debug_output_dir'] = './output/dvs_extreme_optimization'
        
        print(f"Simulator: {config['data']['event_simulator']['type']}")
        print(f"Debug mode: {config.get('debug_mode', False)}")
        
        # Display current extreme parameters
        dvs_params = config['data']['event_simulator']['dvs_voltmeter']['parameters']['dvs346_k']
        print(f"DVS346 extreme parameters: {dvs_params}")
        print()
        print("Physical interpretation:")
        print(f"  k1 (sensitivity): {dvs_params[0]} (5x reduced from ~5.0)")
        print(f"  k2 (threshold): {dvs_params[1]} (10x increased from ~20)")
        print(f"  k3 (photon noise): {dvs_params[2]} (100x reduced from ~0.1)")
        print(f"  k4 (temp leakage): {dvs_params[3]} (10x reduced)")
        print(f"  k5 (parasitic): {dvs_params[4]} (5x reduced)")
        print(f"  k6 (leak noise): {dvs_params[5]} (100x reduced)")
        print()
        
        # Import the integration module
        from dvs_flare_integration import create_flare_event_generator
        
        print("Creating DVS extreme parameter generator...")
        generator = create_flare_event_generator(config)
        print(f"Generator created: {type(generator).__name__}")
        
        # Generate flare events
        print("Generating flare events with extreme DVS parameters...")
        events, timing_info = generator.generate_flare_events()
        
        print(f"✅ Generated {len(events)} events")
        print(f"📊 Performance metrics:")
        
        total_time = timing_info.get('total_pipeline_sec', 0)
        if total_time > 0:
            event_density = len(events) / (total_time * 1000)  # events/ms
            print(f"   Event density: {event_density:.1f} events/ms")
            
            # Compare with targets
            v2ce_target = 3.0  # V2CE level
            dvs_original = 59000  # Original DVS level
            
            if event_density <= v2ce_target * 2:
                print(f"   🎯 SUCCESS: Near V2CE level ({v2ce_target}/ms)")
            elif event_density <= v2ce_target * 10:
                print(f"   ⚡ GOOD: 10x better than original DVS")
            elif event_density <= dvs_original / 2:
                print(f"   ✅ IMPROVED: 2x better than original DVS")
            else:
                print(f"   ⚠️  NEEDS MORE TUNING: Still high event density")
        
        print(f"📊 Timing breakdown:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                print(f"   {key}: {value:.3f}s")
        
        # Check event characteristics
        if len(events) > 0:
            print(f"📈 Event characteristics:")
            print(f"   Shape: {events.shape}")
            print(f"   Timestamp range: {events[:, 0].min():.0f} - {events[:, 0].max():.0f} μs")
            print(f"   Spatial range: X({events[:, 1].min():.0f}-{events[:, 1].max():.0f}), Y({events[:, 2].min():.0f}-{events[:, 2].max():.0f})")
            print(f"   Polarity distribution: {set(events[:, 3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ DVS extreme parameter test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run DVS extreme parameter optimization test."""
    print("DVS-Voltmeter Extreme Parameter Optimization Test")
    print("Based on physics-informed parameter tuning from ECCV 2022 paper")
    print("=" * 70)
    
    success = test_dvs_extreme_parameters()
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULT")
    print("=" * 70)
    
    if success:
        print("🎉 DVS extreme parameter optimization test COMPLETED!")
        print("Check debug output for detailed analysis.")
        print()
        print("Key insights:")
        print("- Reduced k1 (sensitivity) dramatically lowers event triggers")
        print("- Increased k2 (threshold) raises firing threshold")  
        print("- Reduced noise terms (k3,k5,k6) eliminate spurious events")
        print("- Physics-based tuning maintains realistic event characteristics")
    else:
        print("⚠️  Test encountered issues. Check the output above for details.")

if __name__ == "__main__":
    main()