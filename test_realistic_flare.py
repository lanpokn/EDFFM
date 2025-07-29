"""
Test Realistic Flare Synthesis with Power Grid Frequencies
This script tests the updated flare synthesis with realistic artificial light frequencies
"""

import os
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.flare_synthesis import FlareFlickeringSynthesizer

def create_test_directory():
    """Create test output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = f"realistic_flare_test_{timestamp}"
    os.makedirs(test_dir, exist_ok=True)
    return test_dir

def test_frequency_distribution(synthesizer, num_samples=50):
    """Test the distribution of realistic frequencies"""
    frequencies = []
    for _ in range(num_samples):
        freq = synthesizer.get_realistic_flicker_frequency()
        frequencies.append(freq)
    
    return frequencies

def save_frequency_analysis(frequencies, test_dir):
    """Save frequency distribution analysis"""
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.subplot(2, 2, 1)
    plt.hist(frequencies, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Realistic Flicker Frequency Distribution')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Count')
    plt.axvline(100, color='red', linestyle='--', label='50Hz Grid (100Hz flicker)')
    plt.axvline(120, color='orange', linestyle='--', label='60Hz Grid (120Hz flicker)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics
    plt.subplot(2, 2, 2)
    stats_text = f"""Frequency Statistics:
Min: {min(frequencies):.1f} Hz
Max: {max(frequencies):.1f} Hz
Mean: {np.mean(frequencies):.1f} Hz
Std: {np.std(frequencies):.1f} Hz

Grid Standards:
50Hz Countries: ~100Hz flicker
60Hz Countries: ~120Hz flicker
Japan East: ~100Hz flicker  
Japan West: ~120Hz flicker"""
    
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    plt.axis('off')
    plt.title('Analysis Summary')
    
    # Frame rate analysis
    plt.subplot(2, 2, 3)
    fps_values = []
    for freq in frequencies:
        fps = freq * 8  # min_samples_per_cycle = 8
        fps_values.append(min(fps, 2000))  # max_fps = 2000
    
    plt.scatter(frequencies, fps_values, alpha=0.6)
    plt.title('Frequency vs Frame Rate')
    plt.xlabel('Flicker Frequency (Hz)')
    plt.ylabel('Required FPS')
    plt.grid(True, alpha=0.3)
    
    # Duration analysis
    plt.subplot(2, 2, 4)
    frame_counts = [fps * 0.5 for fps in fps_values]  # 0.5 second duration
    plt.scatter(frequencies, frame_counts, alpha=0.6, color='green')
    plt.title('Frequency vs Frame Count (0.5s duration)')
    plt.xlabel('Flicker Frequency (Hz)')
    plt.ylabel('Total Frames')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(test_dir, "frequency_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved frequency analysis: {output_path}")
    return frequencies

def test_different_scenarios(synthesizer, test_dir):
    """Test different realistic scenarios"""
    scenarios = [
        ("European 50Hz Grid", None),  # Will use realistic frequency
        ("North American 60Hz Grid", None),
        ("Japan East (Tokyo)", None),
        ("Japan West (Osaka)", None),
    ]
    
    results = []
    
    for i, (scenario_name, fixed_freq) in enumerate(scenarios):
        print(f"\nTesting {scenario_name}...")
        
        # Load flare and generate video
        flare_rgb = synthesizer.load_random_flare_image()
        video_frames, metadata = synthesizer.generate_flickering_video_frames(
            flare_rgb, frequency=fixed_freq
        )
        
        # Save sample frames
        scenario_dir = os.path.join(test_dir, f"scenario_{i+1}_{scenario_name.replace(' ', '_')}")
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Save first, middle, and last frames
        key_frames = [0, len(video_frames)//2, len(video_frames)-1]
        for frame_idx in key_frames:
            frame = video_frames[frame_idx]
            frame_path = os.path.join(scenario_dir, f"frame_{frame_idx:03d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
        
        # Save metadata
        metadata_path = os.path.join(scenario_dir, "metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Scenario: {scenario_name}\n")
            f.write(f"Frequency: {metadata['frequency_hz']:.1f} Hz\n")
            f.write(f"Curve Type: {metadata['curve_type']}\n")
            f.write(f"FPS: {metadata['fps']} fps\n")
            f.write(f"Duration: {metadata['duration_sec']} sec\n")
            f.write(f"Total Frames: {metadata['total_frames']}\n")
            f.write(f"Samples per Cycle: {metadata['samples_per_cycle']:.1f}\n")
        
        results.append({
            'scenario': scenario_name,
            'metadata': metadata,
            'frame_count': len(video_frames)
        })
        
        print(f"  Generated {len(video_frames)} frames at {metadata['frequency_hz']:.1f} Hz")
    
    return results

def create_comparison_chart(results, test_dir):
    """Create comparison chart of different scenarios"""
    plt.figure(figsize=(15, 10))
    
    scenarios = [r['scenario'] for r in results]
    frequencies = [r['metadata']['frequency_hz'] for r in results]
    fps_values = [r['metadata']['fps'] for r in results]
    frame_counts = [r['frame_count'] for r in results]
    
    # Frequency comparison
    plt.subplot(2, 2, 1)
    bars = plt.bar(range(len(scenarios)), frequencies, color=['blue', 'orange', 'green', 'red'])
    plt.title('Flicker Frequencies by Region')
    plt.ylabel('Frequency (Hz)')
    plt.xticks(range(len(scenarios)), [s.split()[0] for s in scenarios], rotation=45)
    for bar, freq in zip(bars, frequencies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{freq:.1f}Hz', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # FPS comparison  
    plt.subplot(2, 2, 2)
    bars = plt.bar(range(len(scenarios)), fps_values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Required Frame Rates')
    plt.ylabel('FPS')
    plt.xticks(range(len(scenarios)), [s.split()[0] for s in scenarios], rotation=45)
    for bar, fps in zip(bars, fps_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{fps}fps', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Frame count comparison
    plt.subplot(2, 2, 3)
    bars = plt.bar(range(len(scenarios)), frame_counts, color=['blue', 'orange', 'green', 'red'])
    plt.title('Total Frames (0.5s duration)')
    plt.ylabel('Frame Count')
    plt.xticks(range(len(scenarios)), [s.split()[0] for s in scenarios], rotation=45)
    for bar, count in zip(bars, frame_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}', ha='center', va='bottom')
    plt.grid(True, alpha=0.3)
    
    # Summary table
    plt.subplot(2, 2, 4)
    table_data = []
    for r in results:
        table_data.append([
            r['scenario'].split()[0],
            f"{r['metadata']['frequency_hz']:.1f} Hz",
            f"{r['metadata']['fps']} fps",
            f"{r['frame_count']} frames"
        ])
    
    table = plt.table(cellText=table_data,
                     colLabels=['Region', 'Freq', 'FPS', 'Frames'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    plt.axis('off')
    plt.title('Summary Comparison')
    
    plt.tight_layout()
    output_path = os.path.join(test_dir, "scenario_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved scenario comparison: {output_path}")

def main():
    """Main test function"""
    print("Realistic Flare Synthesis Test")
    print("="*50)
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create test directory
    test_dir = create_test_directory()
    print(f"Test output directory: {test_dir}")
    
    # Initialize synthesizer
    synthesizer = FlareFlickeringSynthesizer(config)
    
    # Test frequency distribution
    print("\n1. Testing frequency distribution...")
    frequencies = test_frequency_distribution(synthesizer, num_samples=100)
    save_frequency_analysis(frequencies, test_dir)
    
    # Test different scenarios
    print("\n2. Testing different regional scenarios...")
    results = test_different_scenarios(synthesizer, test_dir)
    
    # Create comparison
    print("\n3. Creating comparison charts...")
    create_comparison_chart(results, test_dir)
    
    # Performance summary
    print(f"\n✅ Realistic flare synthesis test completed!")
    print(f"📁 Results saved to: {test_dir}/")
    print(f"📊 Frequency range: {min(frequencies):.1f} - {max(frequencies):.1f} Hz")
    print(f"📈 Mean frequency: {np.mean(frequencies):.1f} ± {np.std(frequencies):.1f} Hz")
    print(f"🎬 Frame counts: {min([r['frame_count'] for r in results])} - {max([r['frame_count'] for r in results])}")
    
    # Check if any frame count is too high
    max_frames = max([r['frame_count'] for r in results])
    if max_frames > 1000:
        print(f"⚠️  Warning: Maximum frame count ({max_frames}) may be too high for efficient processing")
    else:
        print(f"✅ Frame counts are within reasonable range for efficient processing")

if __name__ == "__main__":
    main()