"""
Debug Script for Flare Synthesis Pipeline
Saves intermediate results for visual inspection and debugging
"""

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.flare_synthesis import FlareFlickeringSynthesizer

def create_debug_directory():
    """Create debug output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = f"debug_output_{timestamp}"
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir

def save_original_flare(flare_rgb, debug_dir, flare_name):
    """Save original flare image"""
    output_path = os.path.join(debug_dir, "01_original_flare.png")
    flare_bgr = cv2.cvtColor((flare_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, flare_bgr)
    print(f"  Saved original flare: {output_path}")
    
    # Also save metadata
    metadata_path = os.path.join(debug_dir, "flare_info.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Original flare: {flare_name}\n")
        f.write(f"Shape: {flare_rgb.shape}\n")
        f.write(f"Min/Max RGB: {flare_rgb.min():.3f}/{flare_rgb.max():.3f}\n")

def save_intensity_map(intensity, debug_dir):
    """Save light intensity map"""
    output_path = os.path.join(debug_dir, "02_light_intensity.png")
    
    # Normalize intensity for visualization
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    intensity_vis = (intensity_norm * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, intensity_vis)
    print(f"  Saved intensity map: {output_path}")
    
    # Save intensity statistics
    stats_path = os.path.join(debug_dir, "intensity_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"Light Intensity Statistics:\n")
        f.write(f"Min: {intensity.min():.6f}\n")
        f.write(f"Max: {intensity.max():.6f}\n")
        f.write(f"Mean: {intensity.mean():.6f}\n")
        f.write(f"Std: {intensity.std():.6f}\n")

def save_flicker_curves(frequencies, duration, fps, debug_dir):
    """Save flicker curve visualizations"""
    curve_types = ["sine", "square", "triangle", "exponential"]
    
    plt.figure(figsize=(15, 10))
    
    for i, curve_type in enumerate(curve_types):
        plt.subplot(2, 2, i+1)
        
        for freq in frequencies:
            # Generate flicker curve
            num_frames = int(duration * fps)
            t = np.linspace(0, duration, num_frames)
            omega = 2 * np.pi * freq
            
            if curve_type == "sine":
                curve = 0.5 * (1 + np.sin(omega * t))
            elif curve_type == "square":
                curve = 0.5 * (1 + np.sign(np.sin(omega * t)))
            elif curve_type == "triangle":
                phase = (omega * t) % (2 * np.pi)
                curve = np.where(phase < np.pi, phase / np.pi, 2 - phase / np.pi)
            elif curve_type == "exponential":
                sine_base = np.sin(omega * t)
                curve = 0.5 * (1 + sine_base * np.exp(-np.abs(sine_base) * 2))
            
            plt.plot(t, curve, label=f'{freq} Hz', linewidth=2)
        
        plt.title(f'{curve_type.title()} Flicker Pattern')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Intensity Multiplier')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
    
    plt.tight_layout()
    output_path = os.path.join(debug_dir, "03_flicker_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved flicker curves: {output_path}")

def save_video_frames(video_frames, frequency, curve_type, debug_dir):
    """Save video frames and create visualization"""
    frame_dir = os.path.join(debug_dir, "04_video_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    # Save individual frames (every 10th frame to save space)
    for i in range(0, len(video_frames), 10):
        frame = video_frames[i]
        frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)
    
    print(f"  Saved {len(video_frames)//10} sample frames to: {frame_dir}/")
    
    # Create intensity timeline visualization
    intensities = []
    for frame in video_frames:
        # Convert to grayscale and get mean intensity
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        intensities.append(gray.mean())
    
    plt.figure(figsize=(12, 6))
    frame_indices = np.arange(len(intensities))
    time_seconds = frame_indices / 100.0  # Assuming 100 fps
    
    plt.plot(time_seconds, intensities, 'b-', linewidth=2, label='Mean Frame Intensity')
    plt.title(f'Video Intensity Timeline ({frequency} Hz {curve_type})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Mean Pixel Intensity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    timeline_path = os.path.join(debug_dir, "05_intensity_timeline.png")
    plt.savefig(timeline_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved intensity timeline: {timeline_path}")

def save_frame_rate_analysis(debug_dir):
    """Analyze and document frame rate calculation"""
    analysis_path = os.path.join(debug_dir, "frame_rate_analysis.txt")
    
    with open(analysis_path, 'w') as f:
        f.write("Frame Rate Analysis for Flare Synthesis\n")
        f.write("="*50 + "\n\n")
        
        f.write("Method Used:\n")
        f.write("- Base FPS from config: 100 Hz\n")
        f.write("- Duration: 1.0 seconds\n")
        f.write("- Total frames = duration * fps = 1.0 * 100 = 100 frames\n\n")
        
        f.write("Flicker Frequency Relationship:\n")
        f.write("- Flicker frequency (e.g., 10 Hz) determines how many complete cycles occur in 1 second\n")
        f.write("- With 100 fps, each flicker cycle (10 Hz) spans 100/10 = 10 frames\n")
        f.write("- This satisfies Nyquist theorem: sampling rate (100 Hz) > 2 * signal frequency (20 Hz for 10 Hz flicker)\n\n")
        
        f.write("Sampling Theory Application:\n")
        f.write("- For proper reconstruction, we need fps >= 2 * max_flicker_frequency\n")
        f.write("- Max flicker frequency: 20 Hz\n")
        f.write("- Minimum required fps: 40 Hz\n")
        f.write("- Current fps (100 Hz) provides 2.5x oversampling for safety\n\n")
        
        f.write("DVS Event Generation:\n")
        f.write("- DVS cameras respond to intensity changes, not absolute intensity\n")
        f.write("- Higher frame rates capture more detailed intensity transitions\n")
        f.write("- This results in more realistic event generation patterns\n")
    
    print(f"  Saved frame rate analysis: {analysis_path}")

def main():
    """Main debug pipeline"""
    print("Flare Synthesis Pipeline Debug")
    print("="*50)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create debug directory
    debug_dir = create_debug_directory()
    print(f"Debug output directory: {debug_dir}")
    
    # Initialize synthesizer
    synthesizer = FlareFlickeringSynthesizer(config)
    
    # Step 1: Load a random flare image
    print("\n1. Loading flare image...")
    flare_rgb = synthesizer.load_random_flare_image()
    flare_name = "random_flare"
    save_original_flare(flare_rgb, debug_dir, flare_name)
    
    # Step 2: Convert to light intensity
    print("\n2. Converting RGB to light intensity...")
    intensity = synthesizer.rgb_to_light_intensity(flare_rgb)
    save_intensity_map(intensity, debug_dir)
    
    # Step 3: Generate and visualize flicker curves
    print("\n3. Generating flicker curves...")
    frequencies = config['data']['flare_synthesis']['flicker_frequencies']
    duration = config['data']['flare_synthesis']['duration_sec']
    fps = config['data']['flare_synthesis']['base_fps']
    save_flicker_curves(frequencies, duration, fps, debug_dir)
    
    # Step 4: Generate video sequence
    print("\n4. Generating video sequence...")
    test_frequency = 10  # Hz
    test_curve = "sine"
    video_frames = synthesizer.generate_flickering_video_frames(
        flare_rgb, test_frequency, test_curve
    )
    save_video_frames(video_frames, test_frequency, test_curve, debug_dir)
    
    # Step 5: Frame rate analysis
    print("\n5. Analyzing frame rate calculation...")
    save_frame_rate_analysis(debug_dir)
    
    # Summary
    print(f"\n✅ Debug pipeline completed!")
    print(f"📁 All results saved to: {debug_dir}/")
    print(f"📊 Generated {len(video_frames)} frames for {test_frequency} Hz {test_curve} flicker")
    print(f"⏱️  Frame rate relationship: {fps} fps captures {fps/test_frequency:.1f} frames per flicker cycle")

if __name__ == "__main__":
    main()