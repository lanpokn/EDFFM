#!/usr/bin/env python3
"""
Quick V2CE Test Script
"""
import yaml
from src.dvs_flare_integration import create_flare_event_generator

# Load and modify config for quick test
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# V2CE configuration for quick test
config['data']['event_simulator']['type'] = 'v2ce'
config['debug_mode'] = True
config['debug_output_dir'] = './output/quick_v2ce_test'
config['data']['flare_synthesis']['duration_sec'] = 0.03  # Short duration
config['data']['event_simulator']['v2ce']['seq_len'] = 2  # Small seq_len

print("Quick V2CE Test...")
print(f"Simulator: {config['data']['event_simulator']['type']}")
print(f"Duration: {config['data']['flare_synthesis']['duration_sec']}s")

try:
    generator = create_flare_event_generator(config)
    events, timing = generator.generate_flare_events()
    
    print(f"\nResults:")
    print(f"  Events: {len(events)}")
    print(f"  Time: {timing.get('total_pipeline_sec', 0):.3f}s")
    
    if len(events) > 0:
        print(f"  ✅ SUCCESS! V2CE generated events")
        print(f"  Range: x=[{events[:,1].min()}-{events[:,1].max()}], y=[{events[:,2].min()}-{events[:,2].max()}]")
        print(f"  Polarity: {sum(events[:,3] > 0)} ON, {sum(events[:,3] <= 0)} OFF")
    else:
        print(f"  ❌ No events generated")
        
except Exception as e:
    print(f"❌ Error: {e}")