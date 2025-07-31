#!/usr/bin/env python3
"""
快速DVS参数测试
"""

import sys
import os
import yaml

# Add src path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

def test_dvs_quick():
    """快速测试DVS参数."""
    print("🔬 Quick DVS Parameter Test")
    print("=" * 40)
    
    try:
        from dvs_flare_integration import create_flare_event_generator
        
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        config['data']['event_simulator']['type'] = 'dvs_voltmeter'
        config['debug_mode'] = False
        
        # Display current parameters
        dvs_params = config['data']['event_simulator']['dvs_voltmeter']['parameters']['dvs346_k']
        print(f"DVS346 parameters: {dvs_params}")
        
        generator = create_flare_event_generator(config)
        
        # Single test
        print("Generating events...")
        events, timing_info = generator.generate_flare_events()
        
        duration_sec = timing_info.get('duration_sec', 0.1)
        duration_ms = duration_sec * 1000
        
        print(f"Events: {len(events)}")
        print(f"Duration: {duration_ms:.1f}ms")
        
        if len(events) > 0 and duration_ms > 0:
            density = len(events) / duration_ms
            print(f"Density: {density:.1f} events/ms")
            
            if density < 100:
                print("⚠️  Too low - increase k1 or decrease k2")
            elif density > 20000:
                print("⚠️  Too high - decrease k1 or increase k2")
            else:
                print("✅ Reasonable range")
        else:
            print("❌ No events generated!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dvs_quick()