#!/usr/bin/env python3
"""
Test script for the new two-step decoupled system
"""

import yaml
import os
import sys

def test_step1_flare_generation():
    """Test Step 1: Flare Event Generation"""
    print("🧪 Testing Step 1: Flare Event Generation")
    print("=" * 50)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug mode with minimal sequences
    config['debug_mode'] = True
    config['generation']['num_train_sequences'] = 2
    config['generation']['num_val_sequences'] = 1
    
    try:
        from src.flare_event_generator import FlareEventGenerator
        
        generator = FlareEventGenerator(config)
        files = generator.generate_batch(3)
        
        print(f"✅ Step 1 test passed: Generated {len(files)} files")
        return True
        
    except Exception as e:
        print(f"❌ Step 1 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step2_event_composition():
    """Test Step 2: Event Composition"""
    print("\n🧪 Testing Step 2: Event Composition")
    print("=" * 50)
    
    # Check if Step 1 output exists
    flare_dir = os.path.join('output', 'data', 'flare_events')
    if not os.path.exists(flare_dir) or not os.listdir(flare_dir):
        print("⚠️  No flare events found, running Step 1 first...")
        if not test_step1_flare_generation():
            return False
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Enable debug mode
    config['debug_mode'] = True
    
    try:
        from src.event_composer import EventComposer
        
        composer = EventComposer(config)
        bg_files, merge_files = composer.compose_batch(max_sequences=3)
        
        print(f"✅ Step 2 test passed: Generated {len(bg_files)} bg + {len(merge_files)} merge files")
        return True
        
    except Exception as e:
        print(f"❌ Step 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_h5_output_format():
    """Test H5 output format compliance"""
    print("\n🧪 Testing H5 Output Format")
    print("=" * 50)
    
    import h5py
    import glob
    
    # Check flare events format
    flare_files = glob.glob('output/data/flare_events/*.h5')
    if flare_files:
        test_file = flare_files[0]
        print(f"Checking flare events format: {os.path.basename(test_file)}")
        
        try:
            with h5py.File(test_file, 'r') as f:
                required_datasets = ['events/t', 'events/x', 'events/y', 'events/p']
                for dataset in required_datasets:
                    if dataset not in f:
                        print(f"❌ Missing dataset: {dataset}")
                        return False
                    print(f"  ✓ {dataset}: {f[dataset].shape} {f[dataset].dtype}")
                
                print("✅ Flare events H5 format is correct")
        except Exception as e:
            print(f"❌ Error reading flare events H5: {e}")
            return False
    
    # Check merge events format
    merge_files = glob.glob('output/data/merge_events/*.h5')
    if merge_files:
        test_file = merge_files[0]
        print(f"Checking merge events format: {os.path.basename(test_file)}")
        
        try:
            with h5py.File(test_file, 'r') as f:
                required_datasets = ['events/t', 'events/x', 'events/y', 'events/p']
                for dataset in required_datasets:
                    if dataset not in f:
                        print(f"❌ Missing dataset: {dataset}")
                        return False
                    print(f"  ✓ {dataset}: {f[dataset].shape} {f[dataset].dtype}")
                
                print("✅ Merge events H5 format is correct")
        except Exception as e:
            print(f"❌ Error reading merge events H5: {e}")
            return False
    
    return True

def test_debug_visualizations():
    """Test debug visualization outputs"""
    print("\n🧪 Testing Debug Visualizations")
    print("=" * 50)
    
    debug_dirs = [
        'output/debug/flare_generation',
        'output/debug/event_composition'
    ]
    
    for debug_dir in debug_dirs:
        if os.path.exists(debug_dir):
            files = []
            for root, dirs, filenames in os.walk(debug_dir):
                files.extend([os.path.join(root, f) for f in filenames if f.endswith('.png')])
            
            print(f"  {debug_dir}: {len(files)} visualization files")
            if len(files) > 0:
                print("    ✓ Debug visualizations generated")
        else:
            print(f"  {debug_dir}: Not found (may be normal if debug was not run)")
    
    return True

def main():
    """Run complete system test"""
    print("🚀 EventMamba-FX Two-Step System Test")
    print("=" * 60)
    
    success = True
    
    # Test Step 1
    if not test_step1_flare_generation():
        success = False
    
    # Test Step 2
    if not test_step2_event_composition():
        success = False
    
    # Test H5 format
    if not test_h5_output_format():
        success = False
    
    # Test debug visualizations
    if not test_debug_visualizations():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! The new two-step system is working correctly.")
        print("\n📂 Generated outputs:")
        print("  - Flare events: output/data/flare_events/")
        print("  - Background events: output/data/bg_events/")
        print("  - Merged events: output/data/merge_events/")
        print("  - Debug visualizations: output/debug/")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    return success

if __name__ == "__main__":
    # Add current directory to path for imports
    sys.path.append('.')
    
    success = main()
    sys.exit(0 if success else 1)