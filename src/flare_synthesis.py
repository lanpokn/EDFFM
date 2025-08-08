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

# ✅ 修复：torchvision现在完全可用，移除fallback机制
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
        """初始化分离的炫光变换管道 (解决黑框问题)."""
        # 获取目标分辨率
        target_w, target_h = self.target_resolution
        
        # 计算相对变换参数 (基于目标分辨率)
        translate_ratio = 0.2  # 最大平移20%
        translate_w = translate_ratio
        translate_h = translate_ratio
        
        # 🚨 分离变换：位置变换 (在大图上进行，保留完整炫光)
        self.positioning_transform = transforms.Compose([
            # 随机仿射变换 (旋转、缩放、平移、剪切)
            transforms.RandomAffine(
                degrees=(0, 360),              # 全方向旋转
                scale=(0.8, 1.5),              # 0.8-1.5倍缩放
                translate=(translate_w, translate_h),  # 相对平移
                shear=(-20, 20)                # ±20度剪切
            ),
            # 随机翻转增加多样性
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 垂直翻转概率稍低
        ])
        
        # 🚨 裁剪变换：最后阶段裁剪到目标分辨率 (运动+闪烁后)
        self.final_crop_transform = transforms.CenterCrop((target_h, target_w))
        
        print(f"✅ Initialized split flare transforms: positioning + final crop to {target_w}x{target_h}")
        
    def _cache_flare_paths(self):
        """Cache all available flare image paths from both Compound_Flare directories."""
        self.compound_flare_paths = []
        
        # ✅ 修正：正确的两个炫光目录路径
        compound_dirs = [
            # 1. Flare-R/Compound_Flare/
            os.path.join(self.flare7k_path, "Flare-R", "Compound_Flare"),
            # 2. Flare7K/Scattering_Flare/Compound_Flare/ (之前缺失)
            os.path.join(self.flare7k_path, "Flare7K", "Scattering_Flare", "Compound_Flare")
        ]
        
        for compound_dir in compound_dirs:
            if os.path.exists(compound_dir):
                patterns = [
                    os.path.join(compound_dir, "*.png"),
                    os.path.join(compound_dir, "*.jpg"),
                    os.path.join(compound_dir, "*.jpeg")
                ]
                files_found = 0
                for pattern in patterns:
                    found_files = glob.glob(pattern)
                    self.compound_flare_paths.extend(found_files)
                    files_found += len(found_files)
                
                print(f"✅ Loaded {files_found} flare images from: {os.path.basename(os.path.dirname(compound_dir))}/Compound_Flare/")
            else:
                print(f"⚠️  Directory not found: {compound_dir}")
        
        print(f"📊 Total: {len(self.compound_flare_paths)} flare images from all Compound_Flare directories")
    
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
        
        # Apply maximum limit (remove hardcoded 500fps constraint)
        max_fps = self.synthesis_config['max_fps']  # Use configured max_fps directly
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
        """Generate a realistic flickering intensity curve for artificial light sources.
        
        Args:
            frequency: Flicker frequency in Hz
            duration: Total duration in seconds
            fps: Frames per second
            curve_type: Type of curve ("sine", "square", "triangle", "exponential")
            
        Returns:
            Array of intensity multipliers over time with realistic baseline (length = duration * fps)
        """
        num_frames = int(duration * fps)
        
        # 🚨 新增：30%概率生成非闪烁炫光 (只移动不闪烁)
        non_flicker_probability = 0.3
        if random.random() < non_flicker_probability:
            # 非闪烁模式：使用恒定强度
            constant_intensity = random.uniform(0.7, 1.0)  # 较高的恒定强度
            curve = np.full(num_frames, constant_intensity)
            print(f"  Generated non-flickering curve: constant intensity={constant_intensity:.2f} (movement-only)")
            return curve
        
        # 原有闪烁逻辑
        t = np.linspace(0, duration, num_frames)
        omega = 2 * np.pi * frequency
        
        # 🚨 修复：生成随机最低强度基线 (模拟真实人造光源)
        min_baseline_range = self.synthesis_config.get('min_intensity_baseline', [0.0, 0.7])
        min_intensity = random.uniform(min_baseline_range[0], min_baseline_range[1])
        max_intensity = self.synthesis_config.get('max_intensity', 1.0)
        intensity_range = max_intensity - min_intensity
        
        # 🚨 修复：使用直观的线性变化，避免在最低值附近停滞
        # 简单的三角波：直线上升→直线下降，变化均匀直观
        phase = (omega * t) % (2 * np.pi)  # [0, 2π)
        raw_curve = np.where(phase < np.pi, phase / np.pi, 2 - phase / np.pi)  # [0, 1] 线性变化
        curve = min_intensity + intensity_range * raw_curve
        
        # 记录使用的是简化线性变化
        curve_type = "linear_triangle"
        
        # Ensure curve is in [min_intensity, max_intensity] range
        curve = np.clip(curve, min_intensity, max_intensity)
        
        print(f"  Generated linear_triangle flicker curve: baseline={min_intensity:.2f}, range=[{min_intensity:.2f}, {max_intensity:.2f}]")
        
        return curve
    
    def _generate_realistic_movement_path(self, duration_sec: float, num_frames: int, 
                                        resolution: Tuple[int, int]) -> np.ndarray:
        """Generate realistic movement path for flare based on automotive scenarios.
        
        参考自动驾驶场景中的典型运动速度：
        - 车灯（对向车辆）: 50-80 km/h → 14-22 m/s
        - 路灯（侧向移动）: 30-60 km/h → 8-17 m/s  
        - 摄像头安装高度: ~1.5m，焦距: ~28mm，像素密度: ~1 pixel/cm
        - 换算: 1 m/s ≈ 10-20 pixels/s (取决于距离和焦距)
        
        Args:
            duration_sec: Sequence duration in seconds
            num_frames: Number of frames in sequence
            resolution: (width, height) of target resolution
            
        Returns:
            Array of (x, y) positions for each frame, shape: (num_frames, 2)
        """
        width, height = resolution
        
        # 🚨 修改：使用更大的随机移动范围，不依赖时长
        # 适应低分辨率场景，确保运动可见性
        min_distance_pixels = 0.0    # 可以完全不移动
        max_distance_pixels = 180.0  # 最大移动180像素 (原60像素的3倍，~28%画面宽度)
        
        # 直接随机选择移动距离 (像素)
        total_distance_pixels = random.uniform(min_distance_pixels, max_distance_pixels)
        
        # 随机选择运动方向 (角度，0-360度)
        movement_angle = random.uniform(0, 2 * np.pi)
        
        # 计算运动向量
        dx_total = total_distance_pixels * np.cos(movement_angle)
        dy_total = total_distance_pixels * np.sin(movement_angle)
        
        # 选择起始位置 (确保整个轨迹都在画面内)
        # 考虑炫光尺寸，留出边界
        margin = 50  # 50像素边界
        min_x = margin - min(0, dx_total)
        max_x = width - margin - max(0, dx_total)
        min_y = margin - min(0, dy_total)  
        max_y = height - margin - max(0, dy_total)
        
        # 确保起始位置合理
        if max_x <= min_x:
            # 水平移动距离太大，使用画面中心
            start_x = width // 2
            dx_total = 0  # 禁用水平移动
        else:
            start_x = random.uniform(min_x, max_x)
            
        if max_y <= min_y:
            # 垂直移动距离太大，使用画面中心  
            start_y = height // 2
            dy_total = 0  # 禁用垂直移动
        else:
            start_y = random.uniform(min_y, max_y)
        
        # 生成平滑的运动轨迹 (线性插值)
        t_values = np.linspace(0, 1, num_frames)
        x_positions = start_x + dx_total * t_values
        y_positions = start_y + dy_total * t_values
        
        # 组合成轨迹数组
        movement_path = np.column_stack((x_positions, y_positions))
        
        # 计算等效速度 (仅用于显示)
        equivalent_speed = total_distance_pixels / duration_sec if duration_sec > 0 else 0
        
        print(f"  Generated movement: {total_distance_pixels:.1f} pixels in {duration_sec:.3f}s "
              f"(≈{equivalent_speed:.1f} pixels/s), angle={np.degrees(movement_angle):.1f}°")
        
        return movement_path
    
    def load_random_flare_image(self, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Load and transform a random flare image using split transform pipeline.
        
        🚨 新方法：分离变换管道，避免黑框问题
        1. 加载原始大图
        2. 应用位置变换 (旋转、缩放、平移等) - 保持大尺寸
        3. 返回变换后的大图，供运动+闪烁处理
        4. 最后由调用方进行裁剪
        
        Args:
            target_size: Optional (width, height) - 如果提供，覆盖默认的DSEC分辨率
            
        Returns:
            RGB flare image array in [0, 1] range, 已变换但未裁剪 (可能比目标尺寸大)
        """
        if not self.compound_flare_paths:
            raise ValueError("No flare images found in Compound_Flare directories")
        
        # Select random flare image
        flare_path = random.choice(self.compound_flare_paths)
        
        try:
            # 使用PIL加载 (支持更多格式，与transforms兼容)
            flare_pil = Image.open(flare_path).convert('RGB')
            
            # 🚨 仅应用位置变换，不裁剪 (保留完整炫光)
            flare_positioned = self.positioning_transform(flare_pil)
            flare_rgb = np.array(flare_positioned)
            
            # Normalize to [0, 1]
            flare_rgb = flare_rgb.astype(np.float32) / 255.0
            
            print(f"  Loaded positioned flare: {flare_rgb.shape[:2]} (before final crop)")
            
            return flare_rgb
            
        except Exception as e:
            print(f"Error loading/positioning flare image {flare_path}: {e}")
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
        """Generate flickering and moving video frames using natural cropping workflow.
        
        🚨 新方法：自然裁剪工作流，消除黑框
        1. 在变换后的大图上应用闪烁
        2. 在大图上应用运动轨迹
        3. 最后裁剪到目标分辨率 (自然边界)
        
        Args:
            flare_rgb: Positioned RGB flare image (可能比目标尺寸大)
            frequency: Flicker frequency in Hz (if None, uses realistic frequency)
            curve_type: Flicker curve type (if None, random selection)
            position: Optional (x, y) position to place flare, random if None
            
        Returns:
            List of RGB video frames showing flickering and moving flare
        """
        # Use realistic frequency if not provided
        if frequency is None:
            frequency = self.get_realistic_flicker_frequency()
        
        # Random curve type if not provided
        if curve_type is None:
            curve_type = random.choice(self.synthesis_config['flicker_curves'])
        
        # 🚨 FIX: Get duration from duration_range instead of deprecated duration_sec
        duration_range = self.synthesis_config.get('duration_range', [0.05, 0.15])
        if isinstance(duration_range, list) and len(duration_range) == 2:
            # If range has been set to fixed value by epoch_iteration_dataset, use it
            if duration_range[0] == duration_range[1]:
                duration = duration_range[0]
            else:
                # Random duration from range
                duration = random.uniform(duration_range[0], duration_range[1])
        else:
            # Fallback to default
            duration = 0.1
            
        fps = self.calculate_dynamic_fps(frequency)  # Dynamic FPS based on frequency
        
        # 获取炫光图像的实际尺寸 (变换后的大图)
        flare_h, flare_w = flare_rgb.shape[:2]
        target_w, target_h = self.target_resolution
        
        print(f"  Working with positioned flare: {flare_h}x{flare_w}, target: {target_h}x{target_w}")
        
        # Convert RGB to light intensity
        flare_intensity = self.rgb_to_light_intensity(flare_rgb)
        
        # Generate flicker curve
        flicker_curve = self.generate_flicker_curve(frequency, duration, fps, curve_type)
        
        # 🚨 简化：直接在变换后的图像上生成运动，不需要额外画布
        # 确保运动范围不超出最终裁剪边界
        effective_w = min(flare_w, target_w + 120)  # 给运动留一些空间
        effective_h = min(flare_h, target_h + 120)
        
        movement_path = self._generate_realistic_movement_path(
            duration, len(flicker_curve), (effective_w, effective_h)
        )
        
        # 🚨 修复：将随机强度缩放移到循环外，避免破坏规律频闪
        scale_range = self.synthesis_config.get('intensity_scale', [1.0, 1.0])
        global_scale_factor = random.uniform(scale_range[0], scale_range[1])  # 整个序列统一缩放
        
        frames = []
        
        for frame_idx, intensity_multiplier in enumerate(flicker_curve):
            # 1. Apply flicker to the positioned flare image
            flickered_intensity = flare_intensity * intensity_multiplier
            
            # 保持原始RGB颜色比例
            original_luminance = self.rgb_to_light_intensity(flare_rgb)
            safe_luminance = np.where(original_luminance > 1e-8, original_luminance, 1e-8)
            intensity_ratio = flickered_intensity / safe_luminance
            
            # 按原始颜色比例调制RGB
            frame_rgb = flare_rgb * np.expand_dims(intensity_ratio, axis=-1)
            frame_rgb = np.clip(frame_rgb * global_scale_factor, 0.0, 1.0)
            
            # 2. 🚨 简化运动：直接平移变换后的炫光图像
            # 获取当前帧的运动偏移
            current_pos = movement_path[frame_idx]
            start_pos = movement_path[0]
            offset_x = int(current_pos[0] - start_pos[0])
            offset_y = int(current_pos[1] - start_pos[1]) 
            
            # 应用平移 (简单的numpy数组移动)
            moved_frame = np.zeros_like(frame_rgb)
            
            # 计算有效的复制区域
            src_start_x = max(0, -offset_x)
            src_end_x = min(flare_w, flare_w - offset_x)
            src_start_y = max(0, -offset_y)
            src_end_y = min(flare_h, flare_h - offset_y)
            
            dst_start_x = max(0, offset_x)
            dst_end_x = dst_start_x + (src_end_x - src_start_x)
            dst_start_y = max(0, offset_y)
            dst_end_y = dst_start_y + (src_end_y - src_start_y)
            
            # 复制移动后的图像
            if src_end_x > src_start_x and src_end_y > src_start_y:
                moved_frame[dst_start_y:dst_end_y, dst_start_x:dst_end_x] = \
                    frame_rgb[src_start_y:src_end_y, src_start_x:src_end_x]
            
            # 3. 🚨 关键：自然裁剪到目标分辨率
            moved_frame_uint8 = (moved_frame * 255).astype(np.uint8)
            moved_frame_pil = Image.fromarray(moved_frame_uint8)
            
            # 应用最终裁剪变换
            final_frame_pil = self.final_crop_transform(moved_frame_pil)
            final_frame = np.array(final_frame_pil)
            
            frames.append(final_frame)
        
        # Return frames with metadata for debugging
        metadata = {
            'frequency_hz': frequency,
            'curve_type': curve_type,
            'fps': fps,
            'duration_sec': duration,
            'total_frames': len(frames),
            'samples_per_cycle': fps / frequency,
            'movement_distance_pixels': np.linalg.norm(movement_path[-1] - movement_path[0]),
            'movement_speed_pixels_per_sec': np.linalg.norm(movement_path[-1] - movement_path[0]) / duration,
            'positioned_flare_size': (flare_h, flare_w),
            'effective_work_area': (effective_h, effective_w)
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