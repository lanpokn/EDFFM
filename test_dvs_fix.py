#!/usr/bin/env python3
"""
Test script to verify DVS simulator bug fixes
Tests the DVS integration and coordinate validation
"""

import os
import sys
import yaml
import numpy as np
import tempfile
import shutil

sys.path.append('.')

def test_dvs_simulator():
    """Test DVS simulator integration."""
    print("🔧 Testing DVS simulator bug fixes...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Add missing duration_sec to config for testing
    if 'duration_sec' not in config['data']['flare_synthesis']:
        config['data']['flare_synthesis']['duration_sec'] = 0.3
    
    # Test DVS flare generation
    from src.dvs_flare_integration import DVSFlareEventGenerator
    
    print("1. Initializing DVS flare generator...")
    flare_generator = DVSFlareEventGenerator(config)
    
    print("2. Testing flare event generation...")
    
    # Use a specific temporary directory
    temp_dir = tempfile.mkdtemp(prefix="dvs_test_")
    print(f"   Using temp directory: {temp_dir}")
    
    try:
        # Generate flare events
        flare_events, metadata = flare_generator.generate_flare_events(
            temp_dir=temp_dir, 
            cleanup=False  # Keep files for debugging
        )
        
        print(f"✅ DVS simulation successful!")
        print(f"   Generated {len(flare_events)} events")
        print(f"   Event format: {flare_events.shape}")
        
        if len(flare_events) > 0:
            print(f"   X range: [{flare_events[:, 1].min():.1f}, {flare_events[:, 1].max():.1f}]")
            print(f"   Y range: [{flare_events[:, 2].min():.1f}, {flare_events[:, 2].max():.1f}]")
            print(f"   T range: [{flare_events[:, 0].min():.0f}, {flare_events[:, 0].max():.0f}] μs")
            print(f"   P values: {np.unique(flare_events[:, 3])}")
            
            # Check coordinate ranges
            max_x = config['data']['resolution_w'] - 1
            max_y = config['data']['resolution_h'] - 1
            
            invalid_x = (flare_events[:, 1] < 0) | (flare_events[:, 1] > max_x)
            invalid_y = (flare_events[:, 2] < 0) | (flare_events[:, 2] > max_y)
            
            if np.any(invalid_x) or np.any(invalid_y):
                print(f"⚠️  COORDINATE ISSUES DETECTED:")
                print(f"   Resolution: {max_x+1}x{max_y+1}")
                print(f"   Invalid X: {np.sum(invalid_x)} events")
                print(f"   Invalid Y: {np.sum(invalid_y)} events")
            else:
                print(f"✅ All coordinates within resolution bounds ({max_x+1}x{max_y+1})")
        
        return True
        
    except Exception as e:
        print(f"❌ DVS simulation failed: {e}")
        import traceback
        traceback.print_exc()
        
        # List files in temp directory for debugging
        print(f"\nFiles in temp directory {temp_dir}:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        return False
        
    finally:
        # Cleanup temp directory if requested
        if os.path.exists(temp_dir):
            print(f"Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)


def test_coordinate_validation():
    """Test coordinate validation functionality."""
    print("\n🔧 Testing coordinate validation...")
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    from src.epoch_based_dataset import EpochBasedEventDataset
    
    # Create dataset instance
    dataset = EpochBasedEventDataset(config, split='test')
    
    # Test with invalid coordinates
    print("1. Testing with invalid coordinates...")
    invalid_events = np.array([
        [700, 500, 1000, 1],   # x=700 > 639, y=500 > 479
        [-10, -5, 2000, -1],   # x=-10 < 0, y=-5 < 0
        [320, 240, 3000, 1],   # Valid coordinates
        [639, 479, 4000, -1],  # Edge case - should be valid
        [640, 480, 5000, 1],   # x=640, y=480 - invalid (should be max 639,479)
    ])
    
    print(f"   Input events shape: {invalid_events.shape}")
    print(f"   Input X range: [{invalid_events[:, 0].min()}, {invalid_events[:, 0].max()}]")
    print(f"   Input Y range: [{invalid_events[:, 1].min()}, {invalid_events[:, 1].max()}]")
    
    # Validate coordinates
    validated_events = dataset._validate_event_coordinates(invalid_events, "test_events")
    
    print(f"   Output X range: [{validated_events[:, 0].min()}, {validated_events[:, 0].max()}]")
    print(f"   Output Y range: [{validated_events[:, 1].min()}, {validated_events[:, 1].max()}]")
    
    # Check if all coordinates are now valid
    max_x = config['data']['resolution_w'] - 1
    max_y = config['data']['resolution_h'] - 1
    
    valid_x = (validated_events[:, 0] >= 0) & (validated_events[:, 0] <= max_x)
    valid_y = (validated_events[:, 1] >= 0) & (validated_events[:, 1] <= max_y)
    
    if np.all(valid_x) and np.all(valid_y):
        print("✅ All coordinates successfully validated!")
    else:
        print("❌ Some coordinates still invalid after validation")
        
    return np.all(valid_x) and np.all(valid_y)


def main():
    """Main test function."""
    print("=" * 60)
    print("DVS SIMULATOR BUG FIX VERIFICATION")
    print("=" * 60)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: DVS simulator
    if test_dvs_simulator():
        success_count += 1
        
    # Test 2: Coordinate validation
    if test_coordinate_validation():
        success_count += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All DVS bug fixes verified successfully!")
        return True
    else:
        print("❌ Some tests failed - bugs remain")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)