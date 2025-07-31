#!/usr/bin/env python3
"""
Debug V2CE Events Structure
"""
import numpy as np
import sys
import os

v2ce_path = "/mnt/e/2025/event_flick_flare/main/simulator/V2CE-Toolbox-master"
sys.path.insert(0, v2ce_path)

def debug_ldati_output():
    """Debug what LDATI function actually returns"""
    print("Debugging LDATI output structure...")
    
    try:
        from scripts.LDATI import sample_voxel_statistical
        from functools import partial
        import torch
        
        # Create dummy voxel data
        dummy_voxel = torch.randn(2, 2, 10, 260, 346).cuda()
        
        # Initialize LDATI
        ldati = partial(
            sample_voxel_statistical, 
            fps=30, 
            bidirectional=False, 
            additional_events_strategy='slope'
        )
        
        # Test LDATI output
        events_list = ldati(dummy_voxel)
        
        print(f"LDATI returned {len(events_list)} items")
        
        for i, events in enumerate(events_list):
            print(f"Item {i}:")
            print(f"  Type: {type(events)}")
            print(f"  Shape: {events.shape if hasattr(events, 'shape') else 'N/A'}")
            print(f"  Data type: {events.dtype if hasattr(events, 'dtype') else 'N/A'}")
            
            if hasattr(events, 'dtype') and events.dtype.names:
                print(f"  Fields: {events.dtype.names}")
                if len(events) > 0:
                    print(f"  Sample: {events[0]}")
            
            print(f"  Length: {len(events)}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_ldati_output()