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
import sys
from typing import Tuple, List, Dict, Optional
import glob
from PIL import Image

# ✅ 修复：torchvision现在完全可用，移除fallback机制
import torchvision.transforms as transforms

# 🚨 新增：反射炫光生成器导入
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'R_flare_generator'))
try:
    from GLSL_flare_ultra_fast_gpu import FlareGeneratorUltraFastGPU
    GLSL_REFLECTION_AVAILABLE = True
    print("✅ GLSL reflection flare generator imported successfully")
except ImportError as e:
    print(f"⚠️  GLSL reflection generator not available: {e}")
    GLSL_REFLECTION_AVAILABLE = False


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
        
        # 🚨 新增：初始化反射炫光生成器和噪声纹理
        self.glsl_generator = None
        self.noise_textures = []
        self._init_reflection_flare_generator()
    
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
        """
        🔄 修改: 缓存炫光和光源图像路径，并建立一一对应的配对关系
        """
        self.compound_flare_paths = []  # 旧的列表保留，用于向下兼容或随机选择
        self.flare_light_source_pairs = []  # 🆕 新增: (炫光路径, 光源路径) 元组的列表
        
        # ✅ 修正：正确的两个炫光目录路径
        compound_dirs = [
            # 1. Flare-R/Compound_Flare/
            os.path.join(self.flare7k_path, "Flare-R", "Compound_Flare"),
            # 2. Flare7K/Scattering_Flare/Compound_Flare/ (之前缺失)
            os.path.join(self.flare7k_path, "Flare7K", "Scattering_Flare", "Compound_Flare")
        ]
        
        print("🔍 Caching flare and light source image pairs...")
        for compound_dir in compound_dirs:
            if not os.path.exists(compound_dir):
                print(f"⚠️  Directory not found: {compound_dir}")
                continue

            # 假设 Light_Source 文件夹与 Compound_Flare 在同一父目录下
            light_source_dir = os.path.join(os.path.dirname(compound_dir), "Light_Source")
            
            patterns = [
                os.path.join(compound_dir, "*.png"),
                os.path.join(compound_dir, "*.jpg"),
                os.path.join(compound_dir, "*.jpeg")
            ]
            flare_files = []
            for pattern in patterns:
                flare_files.extend(glob.glob(pattern))
            
            # 排序确保一致性
            flare_files = sorted(flare_files)
            self.compound_flare_paths.extend(flare_files)
            
            files_found = len(flare_files)
            paired_count = 0
            
            if os.path.exists(light_source_dir):
                for flare_path in flare_files:
                    basename = os.path.basename(flare_path)
                    light_source_path = os.path.join(light_source_dir, basename)
                    
                    if os.path.exists(light_source_path):
                        self.flare_light_source_pairs.append((flare_path, light_source_path))
                        paired_count += 1
                
                print(f"  ✅ Paired {paired_count} images from: {os.path.basename(os.path.dirname(compound_dir))}")
            else:
                print(f"  ⚠️  Light source directory not found for {compound_dir}, skipping pairing.")
            
            print(f"  📁 Loaded {files_found} flare images from: {os.path.basename(os.path.dirname(compound_dir))}/Compound_Flare/")
        
        print(f"📊 Total: {len(self.compound_flare_paths)} flare images found.")
        print(f"🔗 Total: {len(self.flare_light_source_pairs)} flare/light-source pairs created.")
    
    def _init_reflection_flare_generator(self):
        """初始化GLSL反射炫光生成器和噪声纹理"""
        if not GLSL_REFLECTION_AVAILABLE:
            print("⚠️  GLSL reflection generator not available, reflection flare disabled")
            return
            
        try:
            # 初始化GLSL生成器
            self.glsl_generator = FlareGeneratorUltraFastGPU(
                output_size=self.target_resolution
            )
            
            # 加载噪声纹理列表
            noise_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   'R_flare_generator', 'noise_textures')
            if os.path.exists(noise_dir):
                self.noise_textures = [
                    os.path.join(noise_dir, f) for f in os.listdir(noise_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                print(f"🎨 Loaded {len(self.noise_textures)} noise textures for reflection flare")
            else:
                print(f"⚠️  Noise texture directory not found: {noise_dir}")
                
            print("✅ GLSL reflection flare generator initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize GLSL reflection generator: {e}")
            self.glsl_generator = None
    
    def _detect_light_source_from_frame(self, frame_rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Tuple[float, float, float]]:
        """
        从散射炫光帧中检测光源位置和颜色
        使用最亮的10个像素的平均位置和颜色作为光源特征
        
        Args:
            frame_rgb: RGB帧 [H, W, 3], 值范围 [0, 255]
            
        Returns:
            Tuple of (光源位置(x,y) or None, 光源颜色(r,g,b))
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None, (1.0, 1.0, 1.0)
            
        # 确保数据格式正确
        if frame_rgb.dtype != np.uint8:
            frame_work = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            frame_work = frame_rgb
            
        # 计算亮度
        frame_float = frame_work.astype(np.float32) / 255.0
        luminance = (frame_float[:, :, 0] * 0.2126 + 
                    frame_float[:, :, 1] * 0.7152 + 
                    frame_float[:, :, 2] * 0.0722)
        
        # 找到最亮的10个像素位置
        flat_luminance = luminance.flatten()
        flat_indices = np.argsort(flat_luminance)[-10:]  # 最亮的10个像素索引
        
        # 过滤掉太暗的像素（亮度阈值0.1）
        bright_indices = []
        for idx in flat_indices:
            if flat_luminance[idx] > 0.1:
                bright_indices.append(idx)
                
        if len(bright_indices) == 0:
            return None, (1.0, 1.0, 1.0)
        
        # 转换索引为2D坐标
        h, w = luminance.shape
        bright_coords = []
        bright_colors = []
        
        for idx in bright_indices:
            y = idx // w
            x = idx % w
            bright_coords.append((x, y))
            bright_colors.append(frame_float[y, x, :])  # RGB颜色
            
        # 计算平均位置
        avg_x = np.mean([coord[0] for coord in bright_coords])
        avg_y = np.mean([coord[1] for coord in bright_coords])
        light_pos = (int(avg_x), int(avg_y))
        
        # 计算平均颜色
        avg_color = np.mean(bright_colors, axis=0)
        light_color = tuple(float(c) for c in avg_color)
        
        return light_pos, light_color
    
    def _detect_light_source_improved(self, frame_rgb: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Tuple[float, float, float]]:
        """
        改进的光源检测：最亮50个像素的平均位置 + 整个炫光图片的颜色平均
        
        Args:
            frame_rgb: RGB帧 [H, W, 3], 值范围 [0, 255]
            
        Returns:
            Tuple of (光源位置(x,y) or None, 光源颜色(r,g,b))
        """
        if frame_rgb is None or frame_rgb.size == 0:
            return None, (1.0, 1.0, 1.0)
            
        # 确保数据格式正确
        if frame_rgb.dtype != np.uint8:
            frame_work = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            frame_work = frame_rgb
            
        # 转换为float进行计算
        frame_float = frame_work.astype(np.float32) / 255.0
        
        # 计算亮度
        luminance = (frame_float[:, :, 0] * 0.2126 + 
                    frame_float[:, :, 1] * 0.7152 + 
                    frame_float[:, :, 2] * 0.0722)
        
        # 🚀 向量化优化：找到最亮50个像素并直接计算平均位置
        flat_luminance = luminance.flatten()
        flat_indices = np.argsort(flat_luminance)[-50:]  # 最亮的50个像素索引
        
        # 向量化过滤：只保留亮度>0.1的像素
        bright_mask = flat_luminance[flat_indices] > 0.1
        bright_indices = flat_indices[bright_mask]
                
        if len(bright_indices) == 0:
            return None, (1.0, 1.0, 1.0)
        
        # 向量化坐标转换：直接用NumPy计算2D坐标
        h, w = luminance.shape
        y_coords = bright_indices // w  # 所有y坐标
        x_coords = bright_indices % w   # 所有x坐标
            
        # 向量化平均位置计算
        avg_x = int(np.mean(x_coords))
        avg_y = int(np.mean(y_coords))
        light_pos = (avg_x, avg_y)
        
        # 🚀 新方法：整个炫光图片的颜色平均（更稳定的颜色）
        # 只考虑有炫光的区域（亮度>0.05的像素）
        flare_mask = luminance > 0.05
        if np.any(flare_mask):
            flare_pixels = frame_float[flare_mask]  # 选择有炫光的像素
            avg_color = np.mean(flare_pixels, axis=0)  # 对RGB分别求平均
            light_color = tuple(float(c) for c in avg_color)
        else:
            # 无炫光区域，使用默认白色
            light_color = (1.0, 1.0, 1.0)
        
        return light_pos, light_color
    
    def prepare_sequence_parameters(self) -> Dict:
        """
        🆕 新增: 生成一个包含所有随机参数的"剧本"字典。
        这是确保对齐的核心！
        """
        # 1. 随机持续时间、频率等
        duration_range = self.synthesis_config.get('duration_range', [0.05, 0.15])
        if isinstance(duration_range, list) and len(duration_range) == 2:
            # 如果range已被epoch_iteration_dataset设置为固定值，使用它
            if duration_range[0] == duration_range[1]:
                duration = duration_range[0]
            else:
                # 从范围中随机选择
                duration = random.uniform(duration_range[0], duration_range[1])
        else:
            # 回退到默认值
            duration = 0.1
            
        frequency = self.get_realistic_flicker_frequency()
        fps = self.calculate_dynamic_fps(frequency)
        num_frames = int(duration * fps)

        # 2. 生成频闪曲线
        curve_type = random.choice(self.synthesis_config['flicker_curves'])
        flicker_curve = self.generate_flicker_curve(frequency, duration, fps, curve_type)

        # 3. 生成运动路径
        # 注意：这里的resolution是为了确定运动范围，可以稍微放大以避免裁剪问题
        movement_resolution = (self.target_resolution[0] + 120, self.target_resolution[1] + 120)
        movement_path = self._generate_realistic_movement_path(duration, len(flicker_curve), movement_resolution)
        
        # 4. 生成变换种子和参数 (重要！)
        # 为了让 torchvision.transforms 可复现，我们需要固定种子
        transform_seed = random.randint(0, 2**32 - 1)
        
        # 5. 生成GLSL反射炫光参数 (即使光源视频不用，也要预先生成以保持随机状态一致)
        reflection_params = {}
        if self.glsl_generator is not None and len(self.noise_textures) > 0:
            reflection_params = {
                'noise_texture': random.choice(self.noise_textures),
                'flare_size': random.uniform(0.15, 0.25),
                'time_seed': random.random() * 50
            }

        # 6. 将所有参数打包成一个"剧本"
        script = {
            "duration": duration,
            "frequency": frequency,
            "fps": fps,
            "curve_type": curve_type,
            "flicker_curve": flicker_curve,
            "movement_path": movement_path,
            "transform_seed": transform_seed,
            "reflection_params": reflection_params,
            "global_scale_factor": random.uniform(*self.synthesis_config.get('intensity_scale', [1.0, 1.0])),
            "num_frames": num_frames
        }
        
        print(f"  📋 Generated sequence script: {duration*1000:.1f}ms, {frequency:.1f}Hz, {fps}fps, {len(flicker_curve)} frames")
        
        return script
    
    def _generate_reflection_flare(self, light_pos: Tuple[int, int], 
                                 light_color: Tuple[float, float, float],
                                 intensity_multiplier: float) -> Optional[np.ndarray]:
        """
        使用GLSL生成反射炫光
        
        Args:
            light_pos: 光源位置 (x, y)
            light_color: 光源颜色 (r, g, b)
            intensity_multiplier: 强度系数A（与散射炫光相同的频闪系数）
            
        Returns:
            反射炫光图像 [H, W, 3] uint8格式, 或None
        """
        if self.glsl_generator is None or not self.noise_textures:
            return None
            
        try:
            # 随机选择噪声纹理
            noise_texture_path = random.choice(self.noise_textures)
            
            # GLSL反射炫光参数
            flare_size = random.uniform(0.15, 0.25)  # 反射炫光尺寸
            time_seed = random.random() * 50  # 随机时间种子
            
            # 生成反射炫光
            reflection_pil = self.glsl_generator.generate(
                light_pos=light_pos,
                noise_image_path=noise_texture_path,
                time=time_seed,
                flare_size=flare_size,
                light_color=light_color,
                generate_main_glow=False,   # 不生成主光源
                generate_reflections=True  # 只生成反射
            )
            
            # 转换为numpy并应用强度系数
            reflection_array = np.array(reflection_pil).astype(np.float32)
            reflection_scaled = reflection_array * intensity_multiplier
            reflection_final = np.clip(reflection_scaled, 0, 255).astype(np.uint8)
            
            return reflection_final
            
        except Exception as e:
            # 静默处理错误，避免中断主流程
            return None
    
    def _generate_reflection_flare_continuous(self, light_pos: Tuple[int, int], 
                                            light_color: Tuple[float, float, float],
                                            intensity_multiplier: float,
                                            noise_texture_path: str,
                                            flare_size: float,
                                            time_seed: float) -> Optional[np.ndarray]:
        """
        使用GLSL生成连续性反射炫光（修复版 - 固定种子）
        
        Args:
            light_pos: 光源位置 (x, y) - 唯一变化的参数！
            light_color: 光源颜色 (r, g, b)
            intensity_multiplier: 强度系数A（与散射炫光相同的频闪系数）
            noise_texture_path: 固定的噪声纹理路径
            flare_size: 固定的炫光尺寸
            time_seed: 固定的随机种子（不是时间！）
            
        Returns:
            反射炫光图像 [H, W, 3] uint8格式, 或None
        """
        if self.glsl_generator is None:
            return None
            
        try:
            # 生成反射炫光（关键：只有light_pos变化，其他参数全部固定）
            reflection_pil = self.glsl_generator.generate(
                light_pos=light_pos,                # 唯一变化参数 - 跟随光源移动
                noise_image_path=noise_texture_path, # 序列级固定
                time=time_seed,                     # 序列级固定种子（不变！）
                flare_size=flare_size,              # 序列级固定
                light_color=light_color,            # 检测的光源颜色
                generate_main_glow=False,           # 不生成主光源
                generate_reflections=True           # 只生成反射
            )
            
            # 转换为numpy并应用强度系数
            reflection_array = np.array(reflection_pil).astype(np.float32)
            reflection_scaled = reflection_array * intensity_multiplier
            reflection_final = np.clip(reflection_scaled, 0, 255).astype(np.uint8)
            
            return reflection_final
            
        except Exception as e:
            # 静默处理错误，避免中断主流程
            return None
    
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
            # print(f"  Generated non-flickering curve: constant intensity={constant_intensity:.2f} (movement-only)")
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
        
        # print(f"  Generated linear_triangle flicker curve: baseline={min_intensity:.2f}, range=[{min_intensity:.2f}, {max_intensity:.2f}]")
        
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
        
        # print(f"  Generated movement: {total_distance_pixels:.1f} pixels in {duration_sec:.3f}s "
        #       f"(≈{equivalent_speed:.1f} pixels/s), angle={np.degrees(movement_angle):.1f}°")
        
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
            
            # print(f"  Loaded positioned flare: {flare_rgb.shape[:2]} (before final crop)")
            
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
    
    def generate_flickering_video_frames(self, 
                                       base_image_rgb: np.ndarray, 
                                       sequence_script: Dict,
                                       apply_reflection: bool = True) -> Tuple[List[np.ndarray], Dict]:
        """
        🔄 重构: 此函数现在是确定性的，严格按照 sequence_script 执行。
        
        Args:
            base_image_rgb: 基础图像 (炫光或光源)
            sequence_script: 包含所有随机参数的"剧本"
            apply_reflection: 是否应用GLSL反射炫光
            
        Returns:
            Tuple of (video_frames, metadata)
        """
        # 从"剧本"中解包所有参数
        flicker_curve = sequence_script['flicker_curve']
        movement_path = sequence_script['movement_path']
        transform_seed = sequence_script['transform_seed']
        reflection_params = sequence_script['reflection_params']
        global_scale_factor = sequence_script['global_scale_factor']
        duration = sequence_script['duration']
        frequency = sequence_script['frequency']
        fps = sequence_script['fps']
        
        # 关键对齐步骤：为图像变换设置固定种子！
        import torch
        torch.manual_seed(transform_seed)
        random.seed(transform_seed)  # 确保transforms内部的随机性也一致

        # 应用位置变换
        base_image_pil = Image.fromarray((base_image_rgb * 255).astype(np.uint8))
        positioned_pil = self.positioning_transform(base_image_pil)
        positioned_rgb = np.array(positioned_pil).astype(np.float32) / 255.0
        
        # 获取图像的实际尺寸 (变换后的大图)
        positioned_h, positioned_w = positioned_rgb.shape[:2]
        target_w, target_h = self.target_resolution
        
        # print(f"  Working with positioned image: {positioned_h}x{positioned_w}, target: {target_h}x{target_w}")
        
        # Convert RGB to light intensity
        base_intensity = self.rgb_to_light_intensity(positioned_rgb)
        
        # 从剧本中获取反射炫光参数
        if apply_reflection and self.glsl_generator and reflection_params:
            sequence_noise_texture = reflection_params['noise_texture']
            sequence_flare_size = reflection_params['flare_size']
            sequence_time_seed = reflection_params['time_seed']
            print(f"  Reflection sequence params: noise={os.path.basename(sequence_noise_texture)}, "
                  f"size={sequence_flare_size:.3f}, seed={sequence_time_seed:.1f}")
        else:
            sequence_noise_texture = None
            sequence_flare_size = 0.2
            sequence_time_seed = 0.0
        
        frames = []
        
        for frame_idx, intensity_multiplier in enumerate(flicker_curve):
            # 1. Apply flicker to the positioned base image
            flickered_intensity = base_intensity * intensity_multiplier
            
            # 保持原始RGB颜色比例
            original_luminance = self.rgb_to_light_intensity(positioned_rgb)
            safe_luminance = np.where(original_luminance > 1e-8, original_luminance, 1e-8)
            intensity_ratio = flickered_intensity / safe_luminance
            
            # 按原始颜色比例调制RGB
            frame_rgb = positioned_rgb * np.expand_dims(intensity_ratio, axis=-1)
            frame_rgb = np.clip(frame_rgb * global_scale_factor, 0.0, 1.0)
            
            # 2. 🚨 简化运动：直接平移变换后的图像
            # 获取当前帧的运动偏移
            current_pos = movement_path[frame_idx]
            start_pos = movement_path[0]
            offset_x = int(current_pos[0] - start_pos[0])
            offset_y = int(current_pos[1] - start_pos[1]) 
            
            # 应用平移 (简单的numpy数组移动)
            moved_frame = np.zeros_like(frame_rgb)
            
            # 计算有效的复制区域
            src_start_x = max(0, -offset_x)
            src_end_x = min(positioned_w, positioned_w - offset_x)
            src_start_y = max(0, -offset_y)
            src_end_y = min(positioned_h, positioned_h - offset_y)
            
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
            
            # 4. 🚨 根据apply_reflection标志控制反射炫光叠加
            if apply_reflection and sequence_noise_texture is not None:
                try:
                    # 🚀 改进光源检测：更精确的位置和颜色
                    light_pos, light_color = self._detect_light_source_improved(final_frame)
                    
                    if light_pos is not None:
                        # 🚀 连续性修复：使用固定种子，只让光源位置变化
                        # time参数是种子，应该保持固定！只有light_pos变化才能实现平滑移动
                        
                        # 生成反射炫光（使用改进检测 + 序列级固定参数）
                        reflection_frame = self._generate_reflection_flare_continuous(
                            light_pos, light_color, intensity_multiplier,
                            sequence_noise_texture, sequence_flare_size, sequence_time_seed
                        )
                        
                        if reflection_frame is not None:
                            # 叠加反射炫光到散射炫光
                            final_frame_float = final_frame.astype(np.float32)
                            reflection_float = reflection_frame.astype(np.float32)
                            combined_frame = final_frame_float + reflection_float
                            final_frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
                            
                            # Debug信息（只打印前5帧）
                            if frame_idx < 5 and apply_reflection:
                                print(f"    Frame {frame_idx}: Added reflection flare at {light_pos} (top50), "
                                      f"color={[f'{c:.2f}' for c in light_color]} (avg), "
                                      f"intensity={intensity_multiplier:.3f}, seed={sequence_time_seed:.1f}")
                    
                except Exception as e:
                    # 静默处理错误，确保主流程不中断
                    pass
            
            frames.append(final_frame)
        
        # 基于剧本返回元数据
        metadata = {
            'frequency_hz': frequency,
            'curve_type': sequence_script.get('curve_type', 'unknown'),
            'fps': fps,
            'duration_sec': duration,
            'total_frames': len(frames),
            'samples_per_cycle': fps / frequency,
            'movement_distance_pixels': np.linalg.norm(movement_path[-1] - movement_path[0]),
            'movement_speed_pixels_per_sec': np.linalg.norm(movement_path[-1] - movement_path[0]) / duration,
            'positioned_image_size': (positioned_h, positioned_w),
            'reflection_flare_applied': apply_reflection and self.glsl_generator is not None,
            'noise_textures_count': len(self.noise_textures)
        }
        
        return frames, metadata
    
    def create_synced_flare_and_light_source_sequences(self) -> Tuple[Optional[List], Optional[List], Dict]:
        """
        🆕 新增: 协调生成一对完美同步的炫光和光源视频帧序列
        """
        if not self.flare_light_source_pairs:
            print("❌ No flare/light source pairs found. Cannot generate synced sequences.")
            return None, None, {}

        # 1. 随机选择一对配对好的图片
        flare_path, light_source_path = random.choice(self.flare_light_source_pairs)
        
        try:
            # 加载图片对
            flare_image_rgb = np.array(Image.open(flare_path).convert('RGB')).astype(np.float32) / 255.0
            light_source_image_rgb = np.array(Image.open(light_source_path).convert('RGB')).astype(np.float32) / 255.0
            
            print(f"  🎭 Selected image pair:")
            print(f"    Flare: {os.path.basename(flare_path)}")  
            print(f"    Light source: {os.path.basename(light_source_path)}")
            
        except Exception as e:
            print(f"❌ Error loading image pair: {e}")
            return None, None, {}

        # 2. 生成一份共享的"剧本"
        sequence_script = self.prepare_sequence_parameters()
        
        # 3. 根据同一份"剧本"分别渲染炫光和光源视频
        try:
            # 渲染炫光视频 (包含反射)
            flare_frames, flare_metadata = self.generate_flickering_video_frames(
                base_image_rgb=flare_image_rgb,
                sequence_script=sequence_script,
                apply_reflection=True
            )
            
            print(f"  ✅ Generated {len(flare_frames)} flare frames (with reflection)")
            
            # 渲染光源视频 (不含反射)
            light_source_frames, light_source_metadata = self.generate_flickering_video_frames(
                base_image_rgb=light_source_image_rgb,
                sequence_script=sequence_script,
                apply_reflection=False
            )
            
            print(f"  ✅ Generated {len(light_source_frames)} light source frames (no reflection)")
            
            # 验证帧数一致性
            if len(flare_frames) != len(light_source_frames):
                print(f"⚠️  Warning: Frame count mismatch! Flare: {len(flare_frames)}, Light source: {len(light_source_frames)}")
            
            # 合并元数据，添加配对信息
            combined_metadata = flare_metadata.copy()
            combined_metadata.update({
                'flare_image_path': flare_path,
                'light_source_image_path': light_source_path,
                'sync_confirmed': len(flare_frames) == len(light_source_frames),
                'generation_method': 'synced_script_based'
            })

            return flare_frames, light_source_frames, combined_metadata
            
        except Exception as e:
            print(f"❌ Error during video generation: {e}")
            return None, None, {}
    
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