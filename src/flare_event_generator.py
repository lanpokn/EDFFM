"""
Independent Flare Event Generator for EventMamba-FX
===================================================

Step 1: Generate pure flare events using DVS simulator.
Output: Standard DVS H5 format in output/data/flare_events/

Key Features:
- Pure flare event generation without background mixing
- Standard DVS H5 format: /events/x, /events/y, /events/t, /events/p
- No feature extraction or post-processing
- Debug visualization support
"""

import os
import time
import random
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional

from src.dvs_flare_integration import create_flare_event_generator
from src.event_visualization_utils import EventVisualizer


class FlareEventGenerator:
    """
    独立的炫光事件生成器 - Step 1
    生成炫光事件和光源事件，输出标准DVS格式H5文件
    """
    
    def __init__(self, config: Dict):
        """
        初始化炫光事件生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 输出路径设置
        self.output_dir = os.path.join('output', 'data', 'flare_events')
        self.light_source_output_dir = os.path.join('output', 'data', 'light_source_events')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.light_source_output_dir, exist_ok=True)
        
        # Debug模式设置
        self.debug_mode = config.get('debug_mode', False)
        if self.debug_mode:
            self.debug_dir = os.path.join('output', 'debug', 'flare_generation')
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 初始化可视化器
            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.visualizer = EventVisualizer(self.debug_dir, resolution)
            print(f"🎯 FlareEventGenerator Debug Mode: {self.debug_dir}")
        
        # 初始化DVS炫光生成器
        self.dvs_generator = create_flare_event_generator(config)
        
        # 初始化光源DVS生成器 (复用现有架构)
        self.light_source_dvs_generator = self._create_light_source_dvs_generator(config)
        
        # 生成参数
        flare_config = config['data']['flare_synthesis']
        self.duration_range = flare_config['duration_range']
        
        print(f"🚀 FlareEventGenerator initialized:")
        print(f"  Flare output directory: {self.output_dir}")
        print(f"  Light source output directory: {self.light_source_output_dir}")
        print(f"  Duration range: {self.duration_range[0]*1000:.0f}-{self.duration_range[1]*1000:.0f}ms")
        print(f"  Debug mode: {self.debug_mode}")
    
    def generate_single_flare_sequence(self, sequence_id: int) -> Tuple[str, str]:
        """
        生成单个炫光事件序列和对应的光源事件序列
        
        Args:
            sequence_id: 序列ID（用于文件命名）
            
        Returns:
            Tuple[炫光事件H5文件路径, 光源事件H5文件路径]
        """
        start_time = time.time()
        
        # 随机持续时间
        import random
        duration_sec = random.uniform(self.duration_range[0], self.duration_range[1])
        
        # 临时修改配置以固定持续时间
        original_range = self.config['data']['flare_synthesis']['duration_range']
        self.config['data']['flare_synthesis']['duration_range'] = [duration_sec, duration_sec]
        
        try:
            # 设置序列级别的随机种子以确保炫光和光源使用相同的图片
            sequence_seed = random.randint(0, 1000000) + sequence_id
            random.seed(sequence_seed)
            np.random.seed(sequence_seed)
            
            # 生成炫光事件
            flare_events, metadata, flare_frames = self.dvs_generator.generate_flare_events(cleanup=True)
            
            # 保存种子到metadata中
            metadata['random_seed'] = sequence_seed
            
            if len(flare_events) == 0:
                print(f"⚠️  Warning: No flare events generated for sequence {sequence_id}")
                return None, None
            
            # 创建输出文件名
            timestamp = int(time.time() * 1000)
            flare_filename = f"flare_sequence_{timestamp}_{sequence_id:05d}.h5"
            light_source_filename = f"light_source_sequence_{timestamp}_{sequence_id:05d}.h5"
            
            flare_output_path = os.path.join(self.output_dir, flare_filename)
            light_source_output_path = os.path.join(self.light_source_output_dir, light_source_filename)
            
            # 保存炫光事件为标准DVS格式
            self._save_events_dvs_format(flare_events, flare_output_path, metadata)
            
            # 生成对应的光源事件 (使用相同的随机种子和参数)
            light_source_events, light_source_metadata, light_source_frames = self._generate_light_source_events_with_same_params(metadata)
            
            if light_source_events is not None and len(light_source_events) > 0:
                # 保存光源事件为标准DVS格式
                self._save_events_dvs_format(light_source_events, light_source_output_path, light_source_metadata)
                
                generation_time = time.time() - start_time
                
                print(f"✅ Generated sequence {sequence_id}:")
                print(f"  Flare events: {len(flare_events):,}")
                print(f"  Light source events: {len(light_source_events):,}")
                print(f"  Duration: {duration_sec*1000:.1f}ms")
                print(f"  Time: {generation_time:.2f}s")
                print(f"  Flare file: {flare_filename}")
                print(f"  Light source file: {light_source_filename}")
                
                # Debug可视化
                if self.debug_mode:
                    self._save_debug_visualization(flare_events, flare_frames, sequence_id, metadata, 'flare')
                    self._save_debug_visualization(light_source_events, light_source_frames, sequence_id, light_source_metadata, 'light_source')
                
                return flare_output_path, light_source_output_path
            else:
                print(f"⚠️  Warning: No light source events generated for sequence {sequence_id}")
                return flare_output_path, None
            
        except Exception as e:
            print(f"❌ Error generating sequence {sequence_id}: {e}")
            return None, None
            
        finally:
            # 恢复原始配置
            self.config['data']['flare_synthesis']['duration_range'] = original_range
    
    def _create_light_source_dvs_generator(self, config: Dict):
        """创建光源DVS生成器 - 复用现有架构，只修改图片路径"""
        import copy
        from src.flare_synthesis import FlareFlickeringSynthesizer
        import glob
        
        # 创建Light_Source版本的FlareFlickeringSynthesizer
        class LightSourceSynthesizer(FlareFlickeringSynthesizer):
            def _cache_flare_paths(self):
                """重写：从Light_Source文件夹加载图像，但不加载GLSL反射炫光"""
                self.compound_flare_paths = []
                
                # 关键：禁用GLSL反射炫光（光源事件不需要反射）
                self.glsl_generator = None
                self.noise_texture = None
                
                # Light_Source目录路径
                light_source_dirs = [
                    os.path.join(self.flare7k_path, "Flare-R", "Light_Source"),
                    os.path.join(self.flare7k_path, "Flare7K", "Scattering_Flare", "Light_Source")
                ]
                
                for light_source_dir in light_source_dirs:
                    if os.path.exists(light_source_dir):
                        patterns = [
                            os.path.join(light_source_dir, "*.png"),
                            os.path.join(light_source_dir, "*.jpg"),
                            os.path.join(light_source_dir, "*.jpeg")
                        ]
                        files_found = 0
                        for pattern in patterns:
                            found_files = glob.glob(pattern)
                            self.compound_flare_paths.extend(found_files)
                            files_found += len(found_files)
                        
                        print(f"✅ Loaded {files_found} light source images from: {os.path.basename(os.path.dirname(light_source_dir))}/Light_Source/")
                    else:
                        print(f"⚠️  Light source directory not found: {light_source_dir}")
                
                print(f"📊 Total: {len(self.compound_flare_paths)} light source images from all Light_Source directories")
        
        # 创建光源配置副本
        light_source_config = copy.deepcopy(config)
        
        # 创建光源DVS生成器
        if config['data']['event_simulator']['type'].lower() == 'dvs_voltmeter':
            from src.dvs_flare_integration import DVSFlareEventGenerator
            
            class LightSourceDVSGenerator(DVSFlareEventGenerator):
                def __init__(self, config):
                    super().__init__(config)
                    # 替换炫光合成器为光源合成器
                    self.flare_synthesizer = LightSourceSynthesizer(config)
            
            return LightSourceDVSGenerator(light_source_config)
        else:
            raise ValueError("目前只支持DVS-Voltmeter仿真器生成光源事件")
    
    def _generate_light_source_events_with_same_params(self, flare_metadata: Dict) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[List[np.ndarray]]]:
        """
        使用与炫光相同的随机种子和参数生成光源事件
        
        Args:
            flare_metadata: 炫光生成的元数据
            
        Returns:
            Tuple[光源事件数组, 光源元数据, 光源图像序列] 或 (None, None, None)
        """
        try:
            # 设置相同的随机种子以确保参数一致性
            if 'random_seed' in flare_metadata:
                import random
                random.seed(flare_metadata['random_seed'])
                np.random.seed(flare_metadata['random_seed'])
            
            # 直接调用光源DVS生成器
            light_source_events, light_source_metadata, light_source_frames = self.light_source_dvs_generator.generate_flare_events(cleanup=True)
            
            if light_source_events is not None and len(light_source_events) > 0:
                print(f"    Light source events generated: {len(light_source_events):,}")
            else:
                print(f"    Light source events: 0 (光源可能在场景外或变化太小)")
                # 创建空的事件数组，保持格式一致性
                light_source_events = np.empty((0, 4), dtype=np.float64)
                if light_source_metadata is None:
                    light_source_metadata = flare_metadata.copy()
                    light_source_metadata['num_events'] = 0
            
            return light_source_events, light_source_metadata, light_source_frames
            
        except Exception as e:
            error_msg = str(e)
            if "need at least one array to concatenate" in error_msg:
                print(f"    Light source events: 0 (光源变化太小，DVS未检测到事件)")
                # 创建空的事件数组
                empty_events = np.empty((0, 4), dtype=np.float64)
                empty_metadata = flare_metadata.copy()
                empty_metadata['num_events'] = 0
                return empty_events, empty_metadata, []
            else:
                print(f"❌ Error generating light source events: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None
    
    def _save_events_dvs_format(self, events: np.ndarray, output_path: str, metadata: Dict):
        """
        保存事件为标准DVS格式H5文件
        
        Args:
            events: 事件数组 [N, 4] 格式 [t, x, y, p] (DVS格式)
            output_path: 输出文件路径
            metadata: 元数据
        """
        # 🚨 炫光时间随机偏移：0-20ms，确保总长度不超过100ms
        if len(events) > 0:
            import random
            events_normalized = events.copy()
            t_min = events_normalized[:, 0].min()
            events_normalized[:, 0] = events_normalized[:, 0] - t_min  # 先归零
            
            # 随机起始时间：0-20ms (0-20000μs)
            random_start_us = random.uniform(0, 20000)
            events_normalized[:, 0] = events_normalized[:, 0] + random_start_us
            
            print(f"    Flare timing: starts at {random_start_us/1000:.1f}ms (duration: {metadata.get('duration_sec', 0)*1000:.1f}ms)")
        else:
            events_normalized = events
        
        with h5py.File(output_path, 'w') as f:
            # 创建标准DVS格式组织结构
            events_group = f.create_group('events')
            
            # DVS格式：事件数组格式为 [t, x, y, p]
            events_group.create_dataset('t', data=events_normalized[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events_normalized[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events_normalized[:, 2].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events_normalized[:, 3].astype(np.int8), 
                                      compression='gzip', compression_opts=9)
            
            # 保存元数据
            events_group.attrs['num_events'] = len(events)
            events_group.attrs['duration_sec'] = metadata.get('duration_sec', 0)
            events_group.attrs['frequency_hz'] = metadata.get('frequency_hz', 0)
            events_group.attrs['resolution_height'] = self.config['data']['resolution_h']
            events_group.attrs['resolution_width'] = self.config['data']['resolution_w']
            events_group.attrs['simulator'] = 'dvs_voltmeter'
            events_group.attrs['generation_time'] = time.time()
    
    def _save_debug_visualization(self, events: np.ndarray, frames: List[np.ndarray], 
                                sequence_id: int, metadata: Dict, event_type: str = 'flare'):
        """
        保存debug可视化
        
        Args:
            events: 事件 [N, 4] DVS格式
            frames: 图像序列
            sequence_id: 序列ID
            metadata: 元数据
            event_type: 'flare' 或 'light_source'
        """
        if event_type == 'flare':
            debug_seq_dir = os.path.join(self.debug_dir, f"flare_sequence_{sequence_id:03d}")
        else:
            debug_seq_dir = os.path.join(self.debug_dir, f"light_source_sequence_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        # 转换事件格式为可视化格式 [x, y, t, p]
        if len(events) > 0:
            vis_events = np.zeros_like(events)
            vis_events[:, 0] = events[:, 1]  # x
            vis_events[:, 1] = events[:, 2]  # y
            vis_events[:, 2] = events[:, 0]  # t
            vis_events[:, 3] = events[:, 3]  # p
            
            # 时间归一化到从0开始
            if len(vis_events) > 0:
                t_min = vis_events[:, 2].min()
                vis_events[:, 2] = vis_events[:, 2] - t_min
            
            # 创建事件可视化
            self._create_event_visualization(vis_events, debug_seq_dir, metadata, event_type)
        
        # 保存原始图像序列
        if frames:
            self._save_frames(frames, debug_seq_dir, event_type)
        
        # 保存元数据
        self._save_sequence_metadata(debug_seq_dir, events, metadata, event_type)
    
    def _create_event_visualization(self, events: np.ndarray, output_dir: str, metadata: Dict, event_type: str = 'flare'):
        """创建事件的多分辨率可视化 - 基于原始帧率和帧数"""
        if len(events) == 0:
            return
            
        # 从metadata获取原始帧参数
        original_fps = metadata.get('fps', 100)  # 原始帧率
        duration_sec = metadata.get('duration_sec', 0.1)  # 持续时间
        total_frames = metadata.get('total_frames', int(original_fps * duration_sec))  # 总帧数
        
        print(f"    Debug vis: {total_frames} frames, {original_fps}fps, {duration_sec*1000:.1f}ms")
        
        # 🚀 优化：只生成0.5x事件可视化，大幅减少处理时间
        resolution_scales = [0.5]  # 只保留0.5x，移除1,2,4x避免过慢
        base_frame_interval_us = 1e6 / original_fps  # 原始帧间间隔(微秒)
        
        for scale in resolution_scales:
            scale_dir = os.path.join(output_dir, f"events_temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
            
            # 计算实际积累时间窗口
            # 1x = 原始帧间隔, 2x = 1/2间隔, 0.5x = 2倍间隔
            accumulation_window_us = base_frame_interval_us / scale
            
            # 生成帧数：
            # 1x应该是source_frames-1 (因为是帧间积累)
            # 其他尺度按比例调整
            if scale == 1.0:
                vis_frames = max(1, total_frames - 1)
            else:
                vis_frames = max(1, int((total_frames - 1) / scale))
            
            print(f"      {scale}x: {vis_frames} frames, window={accumulation_window_us/1000:.1f}ms")
            
            # 时间范围
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            time_step = (t_max - t_min) / vis_frames if vis_frames > 1 else 0
            
            # 生成可视化帧
            resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])
            
            for frame_idx in range(vis_frames):
                # 基于原始帧节奏的时间窗口
                frame_center = t_min + frame_idx * time_step
                frame_start = frame_center
                frame_end = frame_start + accumulation_window_us
                
                # 过滤事件
                mask = (events[:, 2] >= frame_start) & (events[:, 2] < frame_end)
                frame_events = events[mask]
                
                # 创建可视化
                frame = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
                
                if len(frame_events) > 0:
                    for event in frame_events:
                        x, y, t, p = event
                        x, y = int(x), int(y)
                        
                        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
                            # 根据事件类型使用不同颜色
                            if event_type == 'flare':
                                color = (0, 255, 255) if p > 0 else (255, 255, 0)  # 炫光: ON=黄, OFF=青
                            else:  # light_source
                                color = (0, 0, 255) if p > 0 else (255, 0, 0)  # 光源: ON=红, OFF=蓝
                            frame[y, x] = color
                
                # 保存帧
                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _save_frames(self, frames: List[np.ndarray], output_dir: str, event_type: str = 'flare'):
        """保存图像序列"""
        frames_dir = os.path.join(output_dir, f"source_{event_type}_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        import cv2
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
    
    def _save_sequence_metadata(self, output_dir: str, events: np.ndarray, metadata: Dict, event_type: str = 'flare'):
        """保存序列元数据"""
        metadata_path = os.path.join(output_dir, "metadata.txt")
        
        with open(metadata_path, 'w') as f:
            title = "Flare Event Generation Metadata" if event_type == 'flare' else "Light Source Event Generation Metadata"
            f.write(f"{title}\n")
            f.write("=" * len(title) + "\n\n")
            
            f.write(f"Events: {len(events):,}\n")
            f.write(f"Duration: {metadata.get('duration_sec', 0)*1000:.1f}ms\n")
            f.write(f"Frequency: {metadata.get('frequency_hz', 0):.1f}Hz\n")
            f.write(f"FPS: {metadata.get('fps', 0):.0f}\n")
            f.write(f"Samples per cycle: {metadata.get('samples_per_cycle', 0):.1f}\n")
            
            if len(events) > 0:
                t_min, t_max = events[:, 0].min(), events[:, 0].max()
                pos_events = np.sum(events[:, 3] > 0)
                neg_events = np.sum(events[:, 3] <= 0)
                
                f.write(f"Time range: {t_min:.0f} - {t_max:.0f} μs\n")
                f.write(f"Event rate: {len(events) / (metadata.get('duration_sec', 1)):.1f} events/s\n")
                f.write(f"Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), ")
                f.write(f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)\n")
    
    def generate_batch(self, num_sequences: int) -> Tuple[List[str], List[str]]:
        """
        批量生成炫光事件序列和光源事件序列
        
        Args:
            num_sequences: 要生成的序列数量
            
        Returns:
            Tuple[炫光事件文件路径列表, 光源事件文件路径列表]
        """
        print(f"\n🚀 Generating {num_sequences} flare and light source event sequences...")
        
        flare_generated_files = []
        light_source_generated_files = []
        start_time = time.time()
        
        for i in range(num_sequences):
            print(f"\n--- Generating sequence {i+1}/{num_sequences} ---")
            
            flare_path, light_source_path = self.generate_single_flare_sequence(i)
            if flare_path:
                flare_generated_files.append(flare_path)
            if light_source_path:
                light_source_generated_files.append(light_source_path)
        
        total_time = time.time() - start_time
        flare_success_rate = len(flare_generated_files) / num_sequences * 100
        light_source_success_rate = len(light_source_generated_files) / num_sequences * 100
        
        print(f"\n✅ Event generation complete:")
        print(f"  Flare sequences: {len(flare_generated_files)}/{num_sequences} ({flare_success_rate:.1f}%)")
        print(f"  Light source sequences: {len(light_source_generated_files)}/{num_sequences} ({light_source_success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/num_sequences:.1f}s per sequence")
        print(f"  Flare output: {self.output_dir}")
        print(f"  Light source output: {self.light_source_output_dir}")
        
        return flare_generated_files, light_source_generated_files


def test_flare_generator():
    """测试炫光事件生成器"""
    import yaml
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 启用debug模式
    config['debug_mode'] = True
    
    # 创建生成器
    generator = FlareEventGenerator(config)
    
    # 生成测试序列
    flare_files, light_source_files = generator.generate_batch(3)
    
    print(f"Test complete! Generated {len(flare_files)} flare files and {len(light_source_files)} light source files.")
    return flare_files, light_source_files


if __name__ == "__main__":
    test_flare_generator()