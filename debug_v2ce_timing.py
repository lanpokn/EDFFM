#!/usr/bin/env python3
"""
Debug V2CE timing and FPS relationship
"""
import yaml
import numpy as np

# Load config
with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Get our flare video parameters
flare_config = config['data']['flare_synthesis']
duration_sec = 0.05  # 50ms test
frequency_hz = 100   # typical flare frequency

# Calculate our flare video specs
flare_fps = flare_config['max_fps']  # 1600 FPS max
min_samples = flare_config['min_samples_per_cycle']  # 24 samples per cycle
actual_fps = min(frequency_hz * min_samples, flare_fps)  # Real fps of our video

print(f"Our Flare Video Analysis:")
print(f"  Duration: {duration_sec*1000:.0f} ms")
print(f"  Frequency: {frequency_hz} Hz")
print(f"  Required FPS: {frequency_hz * min_samples} (for {min_samples} samples/cycle)")
print(f"  Actual FPS: {actual_fps} (limited by max_fps={flare_fps})")
print(f"  Total frames: {int(duration_sec * actual_fps)}")
print(f"  Frame interval: {1/actual_fps*1000:.2f} ms")

print(f"\nV2CE FPS Analysis:")
print(f"V2CE expects fps parameter to match the input video framerate")

# What should V2CE fps be?
print(f"\nCorrect V2CE fps should be: {actual_fps}")
print(f"This means:")
print(f"  V2CE frame interval: {1/actual_fps*1e6:.1f} μs")
print(f"  Total time span: {int(duration_sec * actual_fps) * (1/actual_fps) * 1000:.1f} ms")
print(f"  Expected duration: {duration_sec*1000:.0f} ms")

# Let's verify with a simple calculation
print(f"\nVerification:")
v2ce_fps_should_be = actual_fps
v2ce_time_per_frame = 1 / v2ce_fps_should_be * 1e6  # microseconds
total_frames = int(duration_sec * actual_fps)
total_v2ce_duration = total_frames * v2ce_time_per_frame / 1000  # ms

print(f"  If V2CE fps = {v2ce_fps_should_be}:")
print(f"    Frame interval = {v2ce_time_per_frame:.1f} μs")
print(f"    {total_frames} frames × {v2ce_time_per_frame:.1f} μs = {total_v2ce_duration:.1f} ms")
print(f"    Should match expected {duration_sec*1000:.0f} ms? {abs(total_v2ce_duration - duration_sec*1000) < 5}")

print(f"\n🎯 Conclusion:")
print(f"V2CE fps parameter should be {actual_fps} to match our flare video's actual framerate")
print(f"Current issue: we're using fps=30-800, but should use fps≈{actual_fps}")