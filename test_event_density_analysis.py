#!/usr/bin/env python3
"""
事件密度对比分析

对比分析:
1. DSEC原始事件密度 
2. DVS仿真器生成的事件密度
3. 不同参数设置的影响

目标: 找到合理的1K-10K events/ms范围
"""

import sys
import os
import yaml
import traceback
import numpy as np
import h5py
from pathlib import Path

# Add src path for imports
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

def analyze_dsec_event_density():
    """分析DSEC数据集的原始事件密度."""
    print("📊 Analyzing DSEC Original Event Density")
    print("=" * 50)
    
    try:
        # Load config
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        dsec_path = config['data']['dsec_path']
        print(f"DSEC path: {dsec_path}")
        
        # Find a few event files to sample
        event_files = []
        for root, dirs, files in os.walk(dsec_path):
            for file in files:
                if file == 'events.h5':
                    event_files.append(os.path.join(root, file))
                if len(event_files) >= 3:  # Sample 3 files
                    break
        
        if not event_files:
            print("❌ No DSEC event files found!")
            return None
            
        print(f"Found {len(event_files)} event files to analyze")
        
        densities = []
        for i, event_file in enumerate(event_files):
            print(f"\n📁 Analyzing file {i+1}: {os.path.basename(os.path.dirname(event_file))}")
            
            try:
                with h5py.File(event_file, 'r') as f:
                    # Get event data
                    events = f['events']
                    t = events['t'][:]  # Timestamps in microseconds
                    
                    # Sample first 1M events for analysis
                    sample_size = min(1000000, len(t))
                    t_sample = t[:sample_size]
                    
                    if len(t_sample) < 1000:
                        print(f"   ⚠️  Too few events ({len(t_sample)}), skipping")
                        continue
                    
                    # Calculate time span and event density
                    time_span_us = t_sample[-1] - t_sample[0]
                    time_span_ms = time_span_us / 1000.0
                    
                    if time_span_ms <= 0:
                        print(f"   ⚠️  Invalid time span ({time_span_ms}ms), skipping")
                        continue
                    
                    event_density = len(t_sample) / time_span_ms
                    densities.append(event_density)
                    
                    print(f"   Events: {len(t_sample):,}")
                    print(f"   Time span: {time_span_ms:.1f}ms")
                    print(f"   Event density: {event_density:.1f} events/ms")
                    
            except Exception as e:
                print(f"   ❌ Error reading {event_file}: {e}")
                continue
        
        if densities:
            avg_density = np.mean(densities)
            std_density = np.std(densities)
            print(f"\n📈 DSEC Event Density Summary:")
            print(f"   Average: {avg_density:.1f} ± {std_density:.1f} events/ms")
            print(f"   Range: {min(densities):.1f} - {max(densities):.1f} events/ms")
            return avg_density
        else:
            print("❌ No valid event densities calculated")
            return None
            
    except Exception as e:
        print(f"❌ DSEC analysis failed: {e}")
        traceback.print_exc()
        return None

def test_dvs_event_density():
    """测试DVS仿真器的事件密度."""
    print("\n🔬 Testing DVS Simulator Event Density")
    print("=" * 50)
    
    try:
        # Import the integration module
        from dvs_flare_integration import create_flare_event_generator
        
        # Load configuration
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Ensure DVS is selected
        config['data']['event_simulator']['type'] = 'dvs_voltmeter'
        config['debug_mode'] = False  # No debug output for faster testing
        
        # Display current parameters
        dvs_params = config['data']['event_simulator']['dvs_voltmeter']['parameters']['dvs346_k']
        print(f"DVS346 parameters: {dvs_params}")
        
        print("Creating DVS generator...")
        generator = create_flare_event_generator(config)
        print(f"Generator created: {type(generator).__name__}")
        
        # Generate multiple samples for better statistics
        densities = []
        for i in range(3):
            print(f"\n🔄 Sample {i+1}/3:")
            events, timing_info = generator.generate_flare_events()
            
            duration_sec = timing_info.get('duration_sec', 0.1)
            duration_ms = duration_sec * 1000
            
            if len(events) > 0 and duration_ms > 0:
                event_density = len(events) / duration_ms
                densities.append(event_density)
                
                print(f"   Generated: {len(events):,} events")
                print(f"   Duration: {duration_ms:.1f}ms") 
                print(f"   Density: {event_density:.1f} events/ms")
            else:
                print(f"   ⚠️  Invalid sample: {len(events)} events, {duration_ms}ms")
        
        if densities:
            avg_density = np.mean(densities)
            std_density = np.std(densities)
            print(f"\n📈 DVS Event Density Summary:")
            print(f"   Average: {avg_density:.1f} ± {std_density:.1f} events/ms")
            print(f"   Range: {min(densities):.1f} - {max(densities):.1f} events/ms")
            return avg_density
        else:
            print("❌ No valid DVS densities calculated")
            return None
            
    except Exception as e:
        print(f"❌ DVS simulation failed: {e}")
        traceback.print_exc()
        return None

def main():
    """Run event density analysis."""
    print("Event Density Comparative Analysis")
    print("=" * 70)
    
    # Analyze DSEC baseline
    dsec_density = analyze_dsec_event_density()
    
    # Test DVS simulation
    dvs_density = test_dvs_event_density()
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    if dsec_density and dvs_density:
        ratio = dvs_density / dsec_density
        print(f"📊 DSEC Original:  {dsec_density:.1f} events/ms")
        print(f"🔬 DVS Simulated:  {dvs_density:.1f} events/ms")
        print(f"📈 Ratio (DVS/DSEC): {ratio:.2f}x")
        print()
        
        # Assessment
        target_min = 1000  # 1K events/ms
        target_max = 10000  # 10K events/ms
        
        if dvs_density < target_min:
            print(f"⚠️  DVS density TOO LOW: {dvs_density:.1f} < {target_min} events/ms")
            print("   Recommendation: Increase k1 (sensitivity) or decrease k2 (threshold)")
        elif dvs_density > target_max:
            print(f"⚠️  DVS density TOO HIGH: {dvs_density:.1f} > {target_max} events/ms")
            print("   Recommendation: Decrease k1 (sensitivity) or increase k2 (threshold)")
        else:
            print(f"✅ DVS density IN RANGE: {target_min} ≤ {dvs_density:.1f} ≤ {target_max} events/ms")
            
        # DSEC comparison
        if dsec_density:
            if dvs_density < dsec_density * 0.1:
                print(f"⚠️  DVS much lower than DSEC baseline ({ratio:.2f}x)")
            elif dvs_density > dsec_density * 10:
                print(f"⚠️  DVS much higher than DSEC baseline ({ratio:.2f}x)")
            else:
                print(f"✅ DVS comparable to DSEC baseline ({ratio:.2f}x)")
    else:
        print("❌ Unable to complete comparison - missing data")
    
    print("\n💡 Next Steps:")
    print("- If DVS too low: Increase k1 or decrease k2")
    print("- If DVS too high: Decrease k1 or increase k2") 
    print("- Target range: 1K-10K events/ms for flare+flicker scenarios")

if __name__ == "__main__":
    main()