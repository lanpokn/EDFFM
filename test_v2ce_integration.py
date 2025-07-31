#!/usr/bin/env python3
"""
V2CE Integration Test Script for EventMamba-FX

This script tests the V2CE simulator integration with debug output enabled.
It verifies that:
1. V2CE simulator can be initialized
2. Flare events can be generated 
3. Debug visualizations are saved with V2CE suffix
4. Events are properly formatted for EventMamba-FX training
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_v2ce_basic_integration():
    """Test basic V2CE integration functionality."""
    print("=" * 60)
    print("V2CE Integration Test")
    print("=" * 60)
    
    # Load config and enable debug mode
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force V2CE simulator and enable debug
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/test_debug_v2ce'
    
    # Reduce parameters for quick testing
    config['data']['max_samples_debug'] = 1  # Only 1 sample
    config['training']['epochs'] = 1
    config['data']['flare_synthesis']['duration_sec'] = 0.05  # Very short duration
    
    print(f"Configuration:")
    print(f"  Simulator: {config['data']['event_simulator']['type']}")
    print(f"  Debug mode: {config.get('debug_mode', False)}")
    print(f"  Debug output: {config.get('debug_output_dir', 'N/A')}")
    print(f"  V2CE model: {config['data']['event_simulator']['v2ce']['model_path']}")
    
    try:
        # Test V2CE generator creation
        print("\n1. Testing V2CE generator creation...")
        from src.dvs_flare_integration import create_flare_event_generator
        
        generator = create_flare_event_generator(config)
        print(f"   ✅ V2CE generator created successfully")
        print(f"   Type: {type(generator).__name__}")
        
        # Test flare event generation
        print("\n2. Testing flare event generation...")
        events, timing_info = generator.generate_flare_events()
        
        print(f"   ✅ Event generation completed")
        print(f"   Generated events: {len(events)}")
        print(f"   Simulator type: {timing_info.get('simulator_type', 'Unknown')}")
        
        if len(events) > 0:
            print(f"   Time range: {events[0, 0]} - {events[-1, 0]} μs")
            print(f"   Duration: {(events[-1, 0] - events[0, 0]) / 1000:.1f} ms")
            print(f"   Spatial range: x=[{events[:, 1].min()}, {events[:, 1].max()}], "
                  f"y=[{events[:, 2].min()}, {events[:, 2].max()}]")
            
            # Polarity analysis
            pos_events = np.sum(events[:, 3] > 0)
            neg_events = len(events) - pos_events
            print(f"   Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), "
                  f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)")
        
        # Test debug output verification
        print("\n3. Verifying debug output...")
        debug_dir = config.get('debug_output_dir', './output/test_debug_v2ce')
        
        if os.path.exists(debug_dir):
            debug_folders = [f for f in os.listdir(debug_dir) if f.startswith('flare_seq_v2ce_')]
            print(f"   ✅ Debug directory exists: {debug_dir}")
            print(f"   V2CE debug folders found: {len(debug_folders)}")
            
            if debug_folders:
                latest_folder = os.path.join(debug_dir, debug_folders[-1])
                folder_contents = os.listdir(latest_folder)
                print(f"   Latest folder contents: {folder_contents}")
                
                # Check for expected V2CE-specific files
                expected_files = ['v2ce_metadata.txt', 'v2ce_input_frames', 'v2ce_event_visualizations']
                for expected in expected_files:
                    if expected in folder_contents:
                        print(f"   ✅ Found: {expected}")
                    else:
                        print(f"   ❌ Missing: {expected}")
        else:
            print(f"   ❌ Debug directory not found: {debug_dir}")
        
        # Test timing analysis
        print("\n4. Timing analysis:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                print(f"   {key}: {value:.3f}s")
        
        total_time = timing_info.get('total_pipeline_sec', 0)
        print(f"   Total pipeline time: {total_time:.3f}s")
        
        return True, events, timing_info
        
    except Exception as e:
        print(f"\n❌ Error during V2CE integration test: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_v2ce_vs_dvs_comparison():
    """Compare V2CE and DVS-Voltmeter outputs (if both available)."""
    print("\n" + "=" * 60)  
    print("V2CE vs DVS-Voltmeter Comparison")
    print("=" * 60)
    
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Test parameters
    base_config['debug_mode'] = True
    base_config['data']['max_samples_debug'] = 1
    base_config['data']['flare_synthesis']['duration_sec'] = 0.05
    
    results = {}
    
    for sim_type in ['v2ce', 'dvs_voltmeter']:
        print(f"\nTesting {sim_type.upper()} simulator...")
        
        config = base_config.copy()
        config['data']['event_simulator']['type'] = sim_type
        config['debug_output_dir'] = f'./output/test_debug_{sim_type}'
        
        try:
            from src.dvs_flare_integration import create_flare_event_generator
            
            generator = create_flare_event_generator(config)
            events, timing = generator.generate_flare_events()
            
            results[sim_type] = {
                'success': True,
                'events_count': len(events),
                'total_time': timing.get('total_pipeline_sec', 0),
                'simulator_type': timing.get('simulator_type', sim_type)
            }
            
            print(f"  ✅ {sim_type}: {len(events)} events in {timing.get('total_pipeline_sec', 0):.3f}s")
            
        except Exception as e:
            results[sim_type] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ {sim_type}: {e}")
    
    # Comparison summary
    if len(results) > 1:
        print(f"\nComparison Summary:")
        for sim_type, result in results.items():
            if result['success']:
                print(f"  {sim_type.upper()}: {result['events_count']} events, {result['total_time']:.3f}s")
            else:
                print(f"  {sim_type.upper()}: FAILED - {result['error']}")
    
    return results

def main():
    """Main test function."""
    print("Starting V2CE Integration Tests...")
    
    # Test 1: Basic V2CE integration
    success, events, timing = test_v2ce_basic_integration()
    
    if success:
        print(f"\n✅ V2CE integration test PASSED")
        
        # Test 2: Comparison with DVS (optional)
        try:
            comparison_results = test_v2ce_vs_dvs_comparison()
            print(f"\n✅ Simulator comparison completed")
        except Exception as e:
            print(f"\n⚠️  Simulator comparison skipped: {e}")
    else:
        print(f"\n❌ V2CE integration test FAILED")
        return 1
    
    print(f"\n" + "=" * 60)
    print("V2CE Integration Tests Completed")
    print("=" * 60)
    print(f"Debug outputs can be found in:")
    print(f"  - ./output/test_debug_v2ce/")
    print(f"  - ./output/test_debug_dvs_voltmeter/ (if DVS test ran)")
    print(f"\nYou can examine the debug visualizations to compare V2CE vs DVS outputs.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)