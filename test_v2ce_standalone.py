#!/usr/bin/env python3
"""
Standalone V2CE Test Script
Test V2CE toolbox directly with simple input
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add V2CE toolbox to path
v2ce_path = "/mnt/e/2025/event_flick_flare/main/simulator/V2CE-Toolbox-master"
sys.path.insert(0, v2ce_path)

def create_test_sequence():
    """Create a simple test sequence with moving circle"""
    output_dir = "./test_v2ce_simple"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 20 frames with moving white circle on black background
    frames = []
    for i in range(20):
        # Create black image (346x260 - V2CE standard resolution)
        img = np.zeros((260, 346), dtype=np.uint8)
        
        # Add moving white circle
        center_x = 50 + i * 10  # Move right
        center_y = 130  # Center vertically
        cv2.circle(img, (center_x, center_y), 20, 255, -1)
        
        # Save frame
        frame_path = os.path.join(output_dir, f"{i:06d}.png")
        cv2.imwrite(frame_path, img)
        frames.append(frame_path)
    
    return output_dir, frames

def test_v2ce_inference():
    """Test V2CE inference with simple sequence"""
    print("=" * 50)
    print("V2CE Standalone Test")
    print("=" * 50)
    
    try:
        # Change to V2CE directory
        os.chdir(v2ce_path)
        
        # Create test sequence
        print("1. Creating test sequence...")
        test_dir, frame_paths = create_test_sequence()
        print(f"   Created {len(frame_paths)} frames in {test_dir}")
        
        # Import V2CE functions
        print("2. Loading V2CE model...")
        from v2ce import get_trained_mode, video_to_voxels
        from scripts.LDATI import sample_voxel_statistical
        from functools import partial
        
        model_path = "./weights/v2ce_3d.pt"
        model = get_trained_mode(model_path)
        print("   ✅ V2CE model loaded")
        
        # Run V2CE inference
        print("3. Running V2CE inference...")
        pred_voxel = video_to_voxels(
            model=model,
            image_paths=frame_paths,
            vidcap=None,
            infer_type='center',
            seq_len=16,
            width=346,
            height=260,
            batch_size=1
        )
        print(f"   ✅ Voxel prediction complete: {pred_voxel.shape}")
        
        # Convert voxel to events
        print("4. Converting voxel to events...")
        L, _, _, H, W = pred_voxel.shape
        stage2_input = pred_voxel.reshape(L, 2, 10, H, W)
        
        # Import torch for GPU processing
        import torch
        stage2_input = torch.from_numpy(stage2_input).cuda()
        
        # Initialize LDATI function
        ldati = partial(
            sample_voxel_statistical, 
            fps=30, 
            bidirectional=False, 
            additional_events_strategy='slope'
        )
        
        # Generate events
        event_stream_per_frame = []
        stage2_batch_size = 24
        for i in range(0, stage2_input.shape[0], stage2_batch_size):
            batch_events = ldati(stage2_input[i:i+stage2_batch_size])
            event_stream_per_frame.extend(batch_events)
        
        # Merge events
        event_stream = []
        for i in range(L):
            if i < len(event_stream_per_frame):
                frame_events = event_stream_per_frame[i]
                # Add time offset
                frame_time_offset = int(i * 1 / 30 * 1e6)  # 30 FPS
                frame_events['timestamp'] += frame_time_offset
                event_stream.append(frame_events)
        
        # Combine all events
        if event_stream:
            combined_events = np.concatenate(event_stream)
            
            print(f"   ✅ Events generated!")
            print(f"   Total events: {len(combined_events)}")
            print(f"   Event fields: {combined_events.dtype.names}")
            
            # Analyze events
            if len(combined_events) > 0:
                print(f"   Time range: {combined_events['timestamp'].min()} - {combined_events['timestamp'].max()} μs")
                print(f"   X range: {combined_events['x'].min()} - {combined_events['x'].max()}")
                print(f"   Y range: {combined_events['y'].min()} - {combined_events['y'].max()}")
                
                # Polarity analysis
                pos_events = np.sum(combined_events['polarity'] > 0)
                neg_events = len(combined_events) - pos_events
                print(f"   Polarity: {pos_events} ON ({pos_events/len(combined_events)*100:.1f}%), {neg_events} OFF")
                
                return True, combined_events
        else:
            print("   ❌ No events generated")
            return False, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, events = test_v2ce_inference()
    if success:
        print(f"\n✅ V2CE standalone test PASSED")
        print(f"Generated {len(events)} events successfully")
    else:
        print(f"\n❌ V2CE standalone test FAILED")