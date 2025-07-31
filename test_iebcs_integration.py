#!/usr/bin/env python3
"""
IEBCS Integration Test Script

This script tests the complete IEBCS integration including:
1. Loading IEBCS configuration
2. Initializing IEBCS simulator
3. Generating flare events with debug output
4. Checking dependencies
"""

import sys
import os
import yaml
import traceback

# Add src path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

def test_iebcs_dependencies():
    """Test IEBCS library dependencies."""
    print("🔍 Checking IEBCS dependencies...")
    
    # Check if IEBCS directory exists
    iebcs_path = "/mnt/e/2025/event_flick_flare/main/simulator/IEBCS-main"
    iebcs_src_path = os.path.join(iebcs_path, 'src')
    
    print(f"  IEBCS path: {iebcs_path}")
    print(f"  IEBCS exists: {os.path.exists(iebcs_path)}")
    print(f"  IEBCS src path: {iebcs_src_path}")
    print(f"  IEBCS src exists: {os.path.exists(iebcs_src_path)}")
    
    if not os.path.exists(iebcs_src_path):
        print("❌ IEBCS source directory not found!")
        return False
    
    # Add IEBCS to path and test imports
    if iebcs_src_path not in sys.path:
        sys.path.append(iebcs_src_path)
    
    try:
        print("  Testing IEBCS module imports...")
        from event_buffer import EventBuffer
        from dvs_sensor import DvsSensor
        print("  ✅ EventBuffer imported successfully")
        print("  ✅ DvsSensor imported successfully")
        
        # Test basic functionality
        print("  Testing EventBuffer creation...")
        buffer = EventBuffer(100)
        print(f"  ✅ EventBuffer created with size 100")
        
        print("  Testing DvsSensor creation...")
        sensor = DvsSensor("TestSensor")
        print(f"  ✅ DvsSensor 'TestSensor' created")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import IEBCS modules: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing IEBCS functionality: {e}")
        return False

def test_iebcs_flare_generation():
    """Test IEBCS flare event generation."""
    print("\n🧪 Testing IEBCS flare event generation...")
    
    try:
        # Load configuration
        config_path = "configs/config.yaml"
        print(f"  Loading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure IEBCS is selected
        config['data']['event_simulator']['type'] = 'iebcs'
        config['debug_mode'] = True  # Test with debug visualization
        config['debug_output_dir'] = './output/debug_visualizations_iebcs'
        
        print(f"  Simulator type: {config['data']['event_simulator']['type']}")
        print(f"  Debug mode: {config.get('debug_mode', False)}")
        
        # Import the integration module
        from dvs_flare_integration import create_flare_event_generator
        
        print("  Creating IEBCS flare event generator...")
        generator = create_flare_event_generator(config)
        print(f"  ✅ Generator created: {type(generator).__name__}")
        
        # Generate flare events
        print("  Generating flare events with IEBCS...")
        events, timing_info = generator.generate_flare_events()
        
        print(f"  ✅ Generated {len(events)} events")
        print(f"  📊 Timing info:")
        for key, value in timing_info.items():
            if key.endswith('_sec'):
                print(f"    {key}: {value:.3f}s")
        
        # Check event format
        if len(events) > 0:
            print(f"  Event format check:")
            print(f"    Shape: {events.shape}")
            print(f"    Columns: timestamp_us, x, y, polarity")
            print(f"    Sample event: {events[0]}")
            
            # Verify event ranges
            print(f"  Event statistics:")
            print(f"    Timestamp range: {events[:, 0].min():.0f} - {events[:, 0].max():.0f} μs")
            print(f"    X range: {events[:, 1].min():.0f} - {events[:, 1].max():.0f}")
            print(f"    Y range: {events[:, 2].min():.0f} - {events[:, 2].max():.0f}")
            print(f"    Polarity values: {set(events[:, 3])}")
        
        return True
        
    except Exception as e:
        print(f"❌ IEBCS flare generation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all IEBCS integration tests."""
    print("=" * 60)
    print("IEBCS Integration Test Suite")
    print("=" * 60)
    
    # Test 1: Dependencies
    deps_ok = test_iebcs_dependencies()
    
    if not deps_ok:
        print("\n❌ Dependency test failed. Cannot proceed with integration test.")
        return
    
    # Test 2: Flare generation
    generation_ok = test_iebcs_flare_generation()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Dependencies:      {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Flare Generation:  {'✅ PASS' if generation_ok else '❌ FAIL'}")
    
    if deps_ok and generation_ok:
        print("\n🎉 All IEBCS integration tests PASSED!")
        print("Check debug output in: ./output/debug_visualizations_iebcs/")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()