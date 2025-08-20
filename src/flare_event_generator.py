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
    只生成纯炫光事件，输出标准DVS格式H5文件
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
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        # 生成参数
        flare_config = config['data']['flare_synthesis']
        self.duration_range = flare_config['duration_range']
        
        print(f"🚀 FlareEventGenerator initialized:")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Duration range: {self.duration_range[0]*1000:.0f}-{self.duration_range[1]*1000:.0f}ms")
        print(f"  Debug mode: {self.debug_mode}")
    
    def generate_single_flare_sequence(self, sequence_id: int) -> str:
        """
        生成单个炫光事件序列
        
        Args:
            sequence_id: 序列ID（用于文件命名）
            
        Returns:
            生成的H5文件路径
        """
        start_time = time.time()
        
        # 随机持续时间
        duration_sec = random.uniform(self.duration_range[0], self.duration_range[1])
        
        # 临时修改配置以固定持续时间
        original_range = self.config['data']['flare_synthesis']['duration_range']
        self.config['data']['flare_synthesis']['duration_range'] = [duration_sec, duration_sec]
        
        try:
            # 生成炫光事件
            flare_events, metadata, flare_frames = self.dvs_generator.generate_flare_events(cleanup=True)
            
            if len(flare_events) == 0:
                print(f"⚠️  Warning: No flare events generated for sequence {sequence_id}")
                return None
            
            # 创建输出文件名
            timestamp = int(time.time() * 1000)
            filename = f"flare_sequence_{timestamp}_{sequence_id:05d}.h5"
            output_path = os.path.join(self.output_dir, filename)
            
            # 保存为标准DVS格式
            self._save_events_dvs_format(flare_events, output_path, metadata)
            
            generation_time = time.time() - start_time
            
            print(f"✅ Generated flare sequence {sequence_id}:")
            print(f"  Events: {len(flare_events):,}")
            print(f"  Duration: {duration_sec*1000:.1f}ms")
            print(f"  Time: {generation_time:.2f}s")
            print(f"  File: {filename}")
            
            # Debug可视化
            if self.debug_mode and sequence_id < 3:  # 只为前3个序列生成debug
                self._save_debug_visualization(flare_events, flare_frames, sequence_id, metadata)
            
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating flare sequence {sequence_id}: {e}")
            return None
            
        finally:
            # 恢复原始配置
            self.config['data']['flare_synthesis']['duration_range'] = original_range
    
    def _save_events_dvs_format(self, events: np.ndarray, output_path: str, metadata: Dict):
        """
        保存事件为标准DVS格式H5文件
        
        Args:
            events: 事件数组 [N, 4] 格式 [t, x, y, p] (DVS格式)
            output_path: 输出文件路径
            metadata: 元数据
        """
        with h5py.File(output_path, 'w') as f:
            # 创建标准DVS格式组织结构
            events_group = f.create_group('events')
            
            # DVS格式：事件数组格式为 [t, x, y, p]
            events_group.create_dataset('t', data=events[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events[:, 2].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events[:, 3].astype(np.int8), 
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
                                sequence_id: int, metadata: Dict):
        """
        保存debug可视化
        
        Args:
            events: 炫光事件 [N, 4] DVS格式
            frames: 炫光图像序列
            sequence_id: 序列ID
            metadata: 元数据
        """
        debug_seq_dir = os.path.join(self.debug_dir, f"flare_sequence_{sequence_id:03d}")
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
            self._create_flare_event_visualization(vis_events, debug_seq_dir, metadata)
        
        # 保存原始炫光图像序列
        if frames:
            self._save_flare_frames(frames, debug_seq_dir)
        
        # 保存元数据
        self._save_sequence_metadata(debug_seq_dir, events, metadata)
    
    def _create_flare_event_visualization(self, events: np.ndarray, output_dir: str, metadata: Dict):
        """创建炫光事件的多分辨率可视化"""
        if len(events) == 0:
            return
            
        # 多分辨率策略
        resolution_scales = [0.5, 1, 2, 4]
        
        for scale in resolution_scales:
            scale_dir = os.path.join(output_dir, f"events_temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
            
            # 时间参数
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            duration_ms = (t_max - t_min) / 1000.0
            
            base_window_ms = 10.0
            window_duration_ms = base_window_ms / scale
            window_duration_us = window_duration_ms * 1000
            
            num_frames = max(10, int(duration_ms / window_duration_ms))
            frame_step = (t_max - t_min) / num_frames if num_frames > 1 else 0
            
            # 生成可视化帧
            resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])
            
            for frame_idx in range(min(num_frames, 30)):  # 限制30帧
                frame_start = t_min + frame_idx * frame_step
                frame_end = frame_start + window_duration_us
                
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
                            # 炫光事件用黄色/橙色
                            color = (0, 255, 255) if p > 0 else (0, 128, 255)  # BGR格式
                            frame[y, x] = color
                
                # 保存帧
                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _save_flare_frames(self, frames: List[np.ndarray], output_dir: str):
        """保存炫光图像序列"""
        frames_dir = os.path.join(output_dir, "source_frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        import cv2
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(frame_path, frame_bgr)
    
    def _save_sequence_metadata(self, output_dir: str, events: np.ndarray, metadata: Dict):
        """保存序列元数据"""
        metadata_path = os.path.join(output_dir, "metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Flare Event Generation Metadata\n")
            f.write("===============================\n\n")
            
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
    
    def generate_batch(self, num_sequences: int) -> List[str]:
        """
        批量生成炫光事件序列
        
        Args:
            num_sequences: 要生成的序列数量
            
        Returns:
            生成的H5文件路径列表
        """
        print(f"\n🚀 Generating {num_sequences} flare event sequences...")
        
        generated_files = []
        start_time = time.time()
        
        for i in range(num_sequences):
            print(f"\n--- Generating sequence {i+1}/{num_sequences} ---")
            
            file_path = self.generate_single_flare_sequence(i)
            if file_path:
                generated_files.append(file_path)
        
        total_time = time.time() - start_time
        success_rate = len(generated_files) / num_sequences * 100
        
        print(f"\n✅ Flare event generation complete:")
        print(f"  Generated: {len(generated_files)}/{num_sequences} sequences ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/num_sequences:.1f}s per sequence")
        print(f"  Output: {self.output_dir}")
        
        return generated_files


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
    files = generator.generate_batch(3)
    
    print(f"Test complete! Generated {len(files)} files.")
    return files


if __name__ == "__main__":
    test_flare_generator()