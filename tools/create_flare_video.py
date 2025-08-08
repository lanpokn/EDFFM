#!/usr/bin/env python3
"""
Simple tool to create video from flare sequence frames for PPT presentation.

Usage:
    python tools/create_flare_video.py
    
Output:
    - flare_sequence.mp4 in the same directory as the frames
"""

import cv2
import os
import glob
from pathlib import Path

def create_flare_video(frames_dir, output_name="flare_sequence.mp4", fps=10):
    """Create video from flare sequence frames.
    
    Args:
        frames_dir: Directory containing frame_*.png files
        output_name: Output video filename
        fps: Frames per second for the video
    """
    # Find all frame files
    frame_pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))
    
    if not frame_files:
        print(f"❌ No frame files found in {frames_dir}")
        return None
    
    print(f"📁 Found {len(frame_files)} frames in {frames_dir}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    if first_frame is None:
        print(f"❌ Cannot read first frame: {frame_files[0]}")
        return None
    
    height, width, channels = first_frame.shape
    print(f"📐 Frame dimensions: {width}x{height}")
    
    # Setup video writer
    output_path = os.path.join(frames_dir, output_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"❌ Cannot create video writer for {output_path}")
        return None
    
    # Write frames to video
    print(f"🎬 Creating video at {fps} FPS...")
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        if frame is not None:
            video_writer.write(frame)
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(frame_files)} frames")
        else:
            print(f"⚠️  Warning: Cannot read frame {frame_file}")
    
    # Release resources
    video_writer.release()
    cv2.destroyAllWindows()
    
    # Check if video was created successfully
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Video created successfully!")
        print(f"   📄 Output: {output_path}")
        print(f"   📊 Size: {file_size:.2f} MB")
        print(f"   ⏱️  Duration: {len(frame_files) / fps:.1f} seconds")
        return output_path
    else:
        print(f"❌ Video creation failed")
        return None

def main():
    """Main function to create flare video."""
    # Default path to flare sequence frames
    default_frames_dir = "output/debug_epoch_000/flare_sequence_frames"
    
    # Check if default directory exists
    if os.path.exists(default_frames_dir):
        frames_dir = default_frames_dir
        print(f"🔍 Using default frames directory: {frames_dir}")
    else:
        print(f"❌ Default frames directory not found: {default_frames_dir}")
        print("💡 Please run debug mode first: python main.py --config configs/config.yaml --debug")
        return
    
    # Create video
    video_path = create_flare_video(frames_dir, fps=10)
    
    if video_path:
        print(f"\n🎯 Ready for PPT! Video saved to:")
        print(f"   {os.path.abspath(video_path)}")
        print(f"\n💡 You can also try different speeds:")
        print(f"   - Slower: python tools/create_flare_video.py --fps 5")
        print(f"   - Faster: python tools/create_flare_video.py --fps 20")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create video from flare sequence frames")
    parser.add_argument("--frames-dir", default="output/debug_epoch_000/flare_sequence_frames",
                        help="Directory containing frame_*.png files")
    parser.add_argument("--output", default="flare_sequence.mp4",
                        help="Output video filename")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second (default: 10)")
    
    args = parser.parse_args()
    
    # Always use the specified parameters
    video_path = create_flare_video(args.frames_dir, args.output, args.fps)