"""
Flare Synthesis Module for EventMamba-FX

This module converts RGB flare images from Flare7Kpp dataset to dynamic flickering 
light intensity sequences, then generates video frames for DVS simulation.

Key features:
- RGB to light intensity conversion using luminance formula
- Multiple flickering patterns (sine, square, triangle, exponential)
- Configurable frequency and duration
- Output compatible with DVS simulator
"""

import numpy as np
import cv2
import os
import random
import time
from typing import Tuple, List, Dict, Optional
import glob
from PIL import Image
import torchvision.transforms as transforms


class FlareFlickeringSynthesizer:
    """Synthesizes flickering flare videos from static RGB flare images."""
    
    def __init__(self, config: Dict):
        """Initialize the synthesizer with configuration parameters.
        
        Args:
            config: Configuration dictionary with flare synthesis parameters
        """
        self.config = config
        self.flare7k_path = config['data']['flare7k_path']
        self.synthesis_config = config['data']['flare_synthesis']
        
        # 获取DSEC事件分辨率 (关键修复!)
        self.target_resolution = (
            config['data']['resolution_w'],  # 640
            config['data']['resolution_h']   # 480
        )
        
        # 初始化炫光多样性变换 (基于Flare7K风格)
        self._init_flare_transforms()
        
        # Cache flare image paths for faster loading
        self._cache_flare_paths()
    
    def _init_flare_transforms(self):
        """初始化炫光多样性变换 (Flare7K风格)."""
        # 获取目标分辨率
        target_w, target_h = self.target_resolution
        
        # 计算相对变换参数 (基于目标分辨率)
        translate_ratio = 0.2  # 最大平移20%
        translate_w = translate_ratio
        translate_h = translate_ratio
        
        self.flare_transform = transforms.Compose([
            # 随机仿射变换 (旋转、缩放、平移、剪切)
            transforms.RandomAffine(
                degrees=(0, 360),              # 全方向旋转
                scale=(0.8, 1.5),              # 0.8-1.5倍缩放
                translate=(translate_w, translate_h),  # 相对平移
                shear=(-20, 20)                # ±20度剪切
            ),
            # 裁剪到目标分辨率 (关键: 与DSEC分辨率对齐!)
            transforms.CenterCrop((target_h, target_w)),  # 注意PIL格式是(H,W)
            # 随机翻转增加多样性
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 垂直翻转概率稍低
        ])
        
        print(f"Initialized flare transforms for resolution: {target_w}x{target_h}")
        
    def _cache_flare_paths(self):
        """Cache all available flare image paths from both Compound_Flare directories."""
        self.compound_flare_paths = []
        
        # Check both Flare-R and Flare7K directories
        compound_dirs = [
            os.path.join(self.flare7k_path, "Flare-R", "Compound_Flare"),
            os.path.join(self.flare7k_path, "Flare7K", "Compound_Flare") if os.path.exists(
                os.path.join(self.flare7k_path, "Flare7K", "Compound_Flare")) else None
        ]
        
        for compound_dir in compound_dirs:
            if compound_dir and os.path.exists(compound_dir):
                patterns = [
                    os.path.join(compound_dir, "*.png"),
                    os.path.join(compound_dir, "*.jpg"),
                    os.path.join(compound_dir, "*.jpeg")
                ]
                for pattern in patterns:
                    self.compound_flare_paths.extend(glob.glob(pattern))
        
        print(f"Found {len(self.compound_flare_paths)} flare images in Compound_Flare directories")
    
    def get_realistic_flicker_frequency(self) -> float:
        """Get a realistic flicker frequency based on real-world power grid standards.
        
        Returns:
            Flicker frequency in Hz with random variation
        """
        # Base frequencies from different power grids
        base_frequencies = [
            self.synthesis_config['realistic_frequencies']['power_50hz'],  # 100 Hz
            self.synthesis_config['realistic_frequencies']['power_60hz'],  # 120 Hz  
            self.synthesis_config['realistic_frequencies']['japan_east'],  # 100 Hz
            self.synthesis_config['realistic_frequencies']['japan_west'],  # 120 Hz
        ]
        
        # Randomly select a base frequency
        base_freq = random.choice(base_frequencies)
        
        # Add random variation (±5Hz) to simulate:
        # - Power grid instability
        # - Dimmer effects
        # - Aging equipment
        # - Electronic ballast variations
        variation = self.synthesis_config['frequency_variation']
        random_variation = random.uniform(-variation, variation)
        
        final_frequency = base_freq + random_variation
        
        # Ensure positive frequency
        return max(final_frequency, 10.0)  # Minimum 10Hz for safety
    
    def calculate_dynamic_fps(self, frequency: float) -> int:
        """Calculate optimal frame rate based on flicker frequency.
        
        Args:
            frequency: Flicker frequency in Hz
            
        Returns:
            Optimal frames per second for capturing the flicker
        """
        # Apply Nyquist theorem: fps >= 2 * frequency
        # Add safety margin with min_samples_per_cycle
        min_samples = self.synthesis_config['min_samples_per_cycle']
        required_fps = frequency * min_samples
        
        # Apply maximum limit and优化: 减少帧数以降低计算开销
        max_fps = min(self.synthesis_config['max_fps'], 500)  # 降低最大帧率
        optimal_fps = min(required_fps, max_fps)
        
        # 确保最小帧率足够捕获闪烁
        return max(int(optimal_fps), 60)  # 最低60fps保证质量
        
    def rgb_to_light_intensity(self, rgb_image: np.ndarray) -> np.ndarray:
        """Convert RGB flare image to light intensity using luminance formula.
        
        Based on synthesis.py line 13: I = R*0.2126 + G*0.7152 + B*0.0722
        This follows the ITU-R BT.709 standard for luminance calculation.
        
        Args:
            rgb_image: RGB image array with values in [0, 1] range
            
        Returns:
            Light intensity array (single channel) in [0, 1] range
        """
        # Ensure input is float and in [0,1] range
        if rgb_image.dtype == np.uint8:
            rgb_image = rgb_image.astype(np.float32) / 255.0
        
        # Apply luminance formula (ITU-R BT.709)
        intensity = (rgb_image[:, :, 0] * 0.2126 + 
                    rgb_image[:, :, 1] * 0.7152 + 
                    rgb_image[:, :, 2] * 0.0722)
        
        return intensity
    
    def generate_flicker_curve(self, frequency: float, duration: float, fps: float, 
                             curve_type: str = "sine") -> np.ndarray:
        """Generate a flickering intensity curve over time.
        
        Args:
            frequency: Flicker frequency in Hz
            duration: Total duration in seconds
            fps: Frames per second
            curve_type: Type of curve ("sine", "square", "triangle", "exponential")
            
        Returns:
            Array of intensity multipliers over time (length = duration * fps)
        """
        num_frames = int(duration * fps)
        t = np.linspace(0, duration, num_frames)
        omega = 2 * np.pi * frequency
        
        if curve_type == "sine":
            # Smooth sinusoidal flickering
            curve = 0.5 * (1 + np.sin(omega * t))
        elif curve_type == "square":
            # Sharp on/off flickering
            curve = 0.5 * (1 + np.sign(np.sin(omega * t)))
        elif curve_type == "triangle":
            # Linear ramp flickering
            phase = (omega * t) % (2 * np.pi)
            curve = np.where(phase < np.pi, phase / np.pi, 2 - phase / np.pi)
        elif curve_type == "exponential":
            # Exponential decay/rise flickering
            sine_base = np.sin(omega * t)
            curve = 0.5 * (1 + sine_base * np.exp(-np.abs(sine_base) * 2))
        else:
            raise ValueError(f"Unknown curve type: {curve_type}")
        
        # Ensure curve is in [0, 1] range
        curve = np.clip(curve, 0.0, 1.0)
        
        return curve
    
    def load_random_flare_image(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load and transform a random flare image with diversity augmentation.
        
        Args:
            target_size: Optional (width, height) - 如果提供，覆盖默认的DSEC分辨率
            
        Returns:
            RGB flare image array in [0, 1] range, 已对齐到DSEC分辨率
        """
        if not self.compound_flare_paths:
            raise ValueError("No flare images found in Compound_Flare directories")
        
        # Select random flare image
        flare_path = random.choice(self.compound_flare_paths)
        
        try:
            # 使用PIL加载 (支持更多格式，与transforms兼容)
            flare_pil = Image.open(flare_path).convert('RGB')
            
            # 应用多样性变换 (旋转、缩放、平移、翻转等)
            flare_transformed = self.flare_transform(flare_pil)
            
            # 转换为numpy array
            flare_rgb = np.array(flare_transformed)
            
            # 验证最终分辨率
            actual_h, actual_w = flare_rgb.shape[:2]
            expected_w, expected_h = target_size if target_size else self.target_resolution
            
            if actual_w != expected_w or actual_h != expected_h:
                print(f"Warning: Flare resolution mismatch. Expected: {expected_w}x{expected_h}, Got: {actual_w}x{actual_h}")
                # 强制resize到正确分辨率
                flare_rgb = cv2.resize(flare_rgb, (expected_w, expected_h))
            
            # Normalize to [0, 1]
            flare_rgb = flare_rgb.astype(np.float32) / 255.0
            
            return flare_rgb
            
        except Exception as e:
            print(f"Error loading/transforming flare image {flare_path}: {e}")
            # 回退到简单加载
            return self._load_flare_image_fallback(flare_path, target_size)
    
    def _load_flare_image_fallback(self, flare_path: str, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        """回退的简单炫光图像加载方法."""
        flare_rgb = cv2.imread(flare_path)
        if flare_rgb is None:
            raise ValueError(f"Failed to load flare image: {flare_path}")
        
        # Convert BGR to RGB
        flare_rgb = cv2.cvtColor(flare_rgb, cv2.COLOR_BGR2RGB)
        
        # Resize to target resolution
        final_size = target_size if target_size else self.target_resolution
        flare_rgb = cv2.resize(flare_rgb, final_size)
        
        # Normalize to [0, 1]
        flare_rgb = flare_rgb.astype(np.float32) / 255.0
        
        return flare_rgb
    
    def generate_flickering_video_frames(self, flare_rgb: np.ndarray, 
                                       frequency: Optional[float] = None, 
                                       curve_type: Optional[str] = None,
                                       position: Optional[Tuple[int, int]] = None) -> List[np.ndarray]:
        """Generate a sequence of flickering video frames from a static flare image.
        
        Args:
            flare_rgb: Static RGB flare image
            frequency: Flicker frequency in Hz (if None, uses realistic frequency)
            curve_type: Flicker curve type (if None, random selection)
            position: Optional (x, y) position to place flare, random if None
            
        Returns:
            List of RGB video frames showing flickering flare
        """
        # Use realistic frequency if not provided
        if frequency is None:
            frequency = self.get_realistic_flicker_frequency()
        
        # Random curve type if not provided
        if curve_type is None:
            curve_type = random.choice(self.synthesis_config['flicker_curves'])
        
        duration = self.synthesis_config['duration_sec']
        fps = self.calculate_dynamic_fps(frequency)  # Dynamic FPS based on frequency
        
        # Convert RGB to light intensity
        flare_intensity = self.rgb_to_light_intensity(flare_rgb)
        
        # Generate flicker curve
        flicker_curve = self.generate_flicker_curve(frequency, duration, fps, curve_type)
        
        # Generate video frames
        frames = []
        height, width = flare_intensity.shape
        
        for frame_idx, intensity_multiplier in enumerate(flicker_curve):
            # Apply flicker to intensity
            flickered_intensity = flare_intensity * intensity_multiplier
            
            # Convert back to RGB (assuming grayscale flare)
            # For more realistic results, we could maintain the original color ratios
            frame_rgb = np.stack([flickered_intensity] * 3, axis=-1)
            
            # Apply intensity scaling if configured
            scale_range = self.synthesis_config.get('intensity_scale', [1.0, 1.0])
            scale_factor = random.uniform(scale_range[0], scale_range[1])
            frame_rgb = np.clip(frame_rgb * scale_factor, 0.0, 1.0)
            
            # Convert to uint8 for video output
            frame_uint8 = (frame_rgb * 255).astype(np.uint8)
            frames.append(frame_uint8)
        
        # Return frames with metadata for debugging
        metadata = {
            'frequency_hz': frequency,
            'curve_type': curve_type,
            'fps': fps,
            'duration_sec': duration,
            'total_frames': len(frames),
            'samples_per_cycle': fps / frequency
        }
        
        return frames, metadata
    
    def create_flare_event_sequence(self, target_resolution: Optional[Tuple[int, int]] = None,
                                  flare_position: Optional[Tuple[int, int]] = None,
                                  frequency: Optional[float] = None,
                                  curve_type: Optional[str] = None) -> Tuple[List[np.ndarray], Dict]:
        """Create a complete flickering flare video sequence.
        
        Args:
            target_resolution: Optional (width, height) - 默认使用DSEC分辨率
            flare_position: Optional (x, y) position, random if None  
            frequency: Optional frequency, random if None
            curve_type: Optional curve type, random if None
            
        Returns:
            Tuple of (video_frames, metadata_dict)
        """
        start_time = time.time()
        
        # 使用DSEC分辨率作为默认 (关键修复!)
        if target_resolution is None:
            target_resolution = self.target_resolution
        
        # Load random flare image with diversity transforms
        flare_rgb = self.load_random_flare_image(target_size=target_resolution)
        
        # Generate flickering video (uses realistic frequency if not specified)
        video_frames, video_metadata = self.generate_flickering_video_frames(
            flare_rgb, frequency, curve_type, flare_position
        )
        
        # Create metadata
        metadata = video_metadata.copy()
        metadata.update({
            'resolution': target_resolution,
            'actual_flare_shape': flare_rgb.shape[:2],  # (H, W)
            'generation_time_sec': time.time() - start_time
        })
        
        return video_frames, metadata
    
    def save_video_sequence(self, video_frames: List[np.ndarray], 
                           output_dir: str, sequence_name: str,
                           create_info_txt: bool = True) -> str:
        """Save video sequence as individual frames for DVS simulator.
        
        Args:
            video_frames: List of RGB frames
            output_dir: Output directory path
            sequence_name: Name for this sequence
            create_info_txt: Whether to create info.txt file for DVS simulator
            
        Returns:
            Path to the created sequence directory
        """
        sequence_dir = os.path.join(output_dir, sequence_name)
        os.makedirs(sequence_dir, exist_ok=True)
        
        frame_paths = []
        timestamps = []
        
        # Calculate timestamps based on FPS
        fps = self.synthesis_config['base_fps']
        frame_duration_us = int(1e6 / fps)  # microseconds per frame
        
        for i, frame in enumerate(video_frames):
            # Save frame
            frame_filename = f"{i:06d}.png"
            frame_path = os.path.join(sequence_dir, frame_filename)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Record paths and timestamps
            frame_paths.append(f"./{sequence_name}/{frame_filename}")
            timestamps.append(i * frame_duration_us)
        
        # Create info.txt for DVS simulator if requested
        if create_info_txt:
            info_path = os.path.join(output_dir, "info.txt")
            with open(info_path, 'w') as f:
                for frame_path, timestamp in zip(frame_paths, timestamps):
                    f.write(f"{frame_path} {timestamp:012d}\n")
        
        return sequence_dir


def test_flare_synthesis():
    """Test function for flare synthesis module."""
    # Create test config
    test_config = {
        'data': {
            'flare7k_path': "/mnt/e/2025/physical_deflare/Datasets/Flare7Kpp/Flare7Kpp",
            'flare_synthesis': {
                'duration_sec': 1.0,
                'base_fps': 100,
                'flicker_frequencies': [5, 10, 15, 20],
                'flicker_curves': ["sine", "square", "triangle", "exponential"],
                'position_random': True,
                'intensity_scale': [0.5, 2.0]
            }
        }
    }
    
    # Initialize synthesizer
    synthesizer = FlareFlickeringSynthesizer(test_config)
    
    # Generate test sequence
    video_frames, metadata = synthesizer.create_flare_event_sequence(
        target_resolution=(240, 180)  # DVS simulator resolution
    )
    
    print(f"Generated flare sequence:")
    print(f"  Frequency: {metadata['frequency_hz']} Hz")
    print(f"  Curve: {metadata['curve_type']}")
    print(f"  Frames: {metadata['num_frames']}")
    print(f"  Generation time: {metadata['generation_time_sec']:.3f}s")
    
    return video_frames, metadata


if __name__ == "__main__":
    # Run test
    frames, meta = test_flare_synthesis()
    print("Flare synthesis test completed successfully!")