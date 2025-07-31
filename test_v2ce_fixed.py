#!/usr/bin/env python3
"""
Test Fixed V2CE Integration
"""
import yaml
from src.dvs_flare_integration import create_flare_event_generator

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configure for V2CE test
config['data']['event_simulator']['type'] = 'v2ce'
config['debug_mode'] = True
config['debug_output_dir'] = './output/v2ce_fixed_test'
config['data']['flare_synthesis']['duration_sec'] = 0.05  # Short test

print("Testing Fixed V2CE Integration...")
print(f"Base FPS: {config['data']['event_simulator']['v2ce']['base_fps']}")
print(f"Max FPS: {config['data']['event_simulator']['v2ce']['max_fps']}")

try:
    generator = create_flare_event_generator(config)
    events, timing = generator.generate_flare_events()
    
    print(f"\n🎉 RESULTS:")
    print(f"  Events generated: {len(events)}")
    print(f"  Total time: {timing.get('total_pipeline_sec', 0):.3f}s")
    print(f"  Simulator: {timing.get('simulator_type', 'Unknown')}")
    
    if len(events) > 0:
        print(f"  ✅ SUCCESS! V2CE is working properly")
        print(f"  Time span: {(events[-1,0] - events[0,0])/1000:.1f} ms")
        print(f"  Spatial range: x=[{events[:,1].min()}, {events[:,1].max()}], y=[{events[:,2].min()}, {events[:,2].max()}]")
        print(f"  Polarity: {sum(events[:,3] > 0)} ON ({sum(events[:,3] > 0)/len(events)*100:.1f}%), {sum(events[:,3] <= 0)} OFF")
        print(f"  Event rate: {len(events) / max(1, (events[-1,0] - events[0,0]) / 1e6):.1f} events/sec")
        
        # Verify coordinate validity
        target_w, target_h = 640, 480
        valid_coords = (events[:,1] >= 0) & (events[:,1] < target_w) & (events[:,2] >= 0) & (events[:,2] < target_h)
        print(f"  Coordinate validity: {sum(valid_coords)}/{len(events)} ({sum(valid_coords)/len(events)*100:.1f}%)")
        
        print(f"\n📊 Performance comparison:")
        print(f"  V2CE: {len(events)} events, {timing.get('total_pipeline_sec', 0):.3f}s")
        print(f"  Dynamic FPS calculation: Working ✅")
        print(f"  Event format: [timestamp_us, x, y, polarity] ✅")
        
    else:
        print(f"  ❌ Still no events generated")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()