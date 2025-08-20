"""
Event Composer for EventMamba-FX
=================================

Step 2: Compose background events + flare events → merged events.
Output: Standard DVS H5 format for bg_events and merge_events.

Key Features:
- Read pre-generated flare events from Step 1
- Load DSEC background events
- Temporal merging with configurable strategies
- Standard DVS H5 format output
- Debug visualization for all three event types
"""

import os
import time
import random
import glob
import numpy as np
import h5py
from typing import Dict, List, Tuple, Optional

from src.dsec_efficient import DSECEventDatasetEfficient
from src.event_visualization_utils import EventVisualizer


class EventComposer:
    """
    事件合成器 - Step 2
    读取炫光事件 + 背景事件 → 合成完整事件序列
    """
    
    def __init__(self, config: Dict):
        """
        初始化事件合成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 输入路径：炫光事件
        self.flare_events_dir = os.path.join('output', 'data', 'flare_events')
        
        # 输出路径
        self.bg_events_dir = os.path.join('output', 'data', 'bg_events')
        self.merge_events_dir = os.path.join('output', 'data', 'merge_events')
        os.makedirs(self.bg_events_dir, exist_ok=True)
        os.makedirs(self.merge_events_dir, exist_ok=True)
        
        # Debug模式设置
        self.debug_mode = config.get('debug_mode', False)
        if self.debug_mode:
            self.debug_dir = os.path.join('output', 'debug', 'event_composition')
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # 初始化可视化器
            resolution = (config['data']['resolution_w'], config['data']['resolution_h'])
            self.visualizer = EventVisualizer(self.debug_dir, resolution)
            print(f"🎯 EventComposer Debug Mode: {self.debug_dir}")
        
        # 初始化DSEC背景数据集
        self.dsec_dataset = DSECEventDatasetEfficient(
            dsec_path=config['data']['dsec_path'],
            flare_path="",  # 不使用
            time_window_us=config['data']['time_window_us']
        )
        
        # 背景事件持续时间配置
        bg_range = config['data']['randomized_training']['background_duration_range']
        self.bg_min_duration_ms = bg_range[0] * 1000
        self.bg_max_duration_ms = bg_range[1] * 1000
        
        print(f"🚀 EventComposer initialized:")
        print(f"  Flare events input: {self.flare_events_dir}")
        print(f"  Background events output: {self.bg_events_dir}")
        print(f"  Merged events output: {self.merge_events_dir}")
        print(f"  DSEC dataset: {len(self.dsec_dataset)} time windows")
        print(f"  Background duration: {self.bg_min_duration_ms:.0f}-{self.bg_max_duration_ms:.0f}ms")
        print(f"  Debug mode: {self.debug_mode}")
    
    def load_flare_events(self, flare_file_path: str) -> np.ndarray:
        """
        加载炫光事件文件
        
        Args:
            flare_file_path: 炫光事件H5文件路径
            
        Returns:
            炫光事件数组 [N, 4] 格式 [t, x, y, p] (DVS格式)
        """
        with h5py.File(flare_file_path, 'r') as f:
            events_group = f['events']
            
            t = events_group['t'][:]
            x = events_group['x'][:]
            y = events_group['y'][:]
            p = events_group['p'][:]
            
            # 组合为 [t, x, y, p] 格式
            events = np.column_stack([t, x, y, p])
            
            return events.astype(np.float64)
    
    def generate_background_events(self) -> np.ndarray:
        """
        生成背景事件
        
        Returns:
            背景事件数组 [N, 4] 格式 [x, y, t, p] (项目格式)
        """
        # 随机持续时间
        duration_ms = random.uniform(self.bg_min_duration_ms, self.bg_max_duration_ms)
        duration_us = int(duration_ms * 1000)
        
        # 随机选择DSEC样本
        idx = random.randint(0, len(self.dsec_dataset) - 1)
        background_events = self.dsec_dataset[idx]  # 返回 [x, y, t, p] 格式
        
        # 裁剪到指定持续时间
        if len(background_events) > 0:
            t_min = background_events[:, 2].min()
            t_max = background_events[:, 2].max()
            current_duration = t_max - t_min
            
            if current_duration > duration_us:
                # 随机时间窗口
                max_start_offset = current_duration - duration_us
                start_offset = random.uniform(0, max_start_offset)
                start_time = t_min + start_offset
                end_time = start_time + duration_us
                
                # 过滤事件
                mask = (background_events[:, 2] >= start_time) & (background_events[:, 2] < end_time)
                background_events = background_events[mask]
            
            # 时间归一化到从0开始
            if len(background_events) > 0:
                t_min_bg = background_events[:, 2].min()
                background_events[:, 2] = background_events[:, 2] - t_min_bg
        
        return background_events if len(background_events) > 0 else np.empty((0, 4))
    
    def convert_flare_to_project_format(self, flare_events: np.ndarray) -> np.ndarray:
        """
        转换炫光事件从DVS格式到项目格式
        
        Args:
            flare_events: [N, 4] DVS格式 [t, x, y, p]
            
        Returns:
            [N, 4] 项目格式 [x, y, t, p]
        """
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        project_events = np.zeros_like(flare_events)
        project_events[:, 0] = flare_events[:, 1]  # x
        project_events[:, 1] = flare_events[:, 2]  # y
        project_events[:, 2] = flare_events[:, 0]  # t
        project_events[:, 3] = flare_events[:, 3]  # p
        
        # 时间归一化到从0开始
        if len(project_events) > 0:
            t_min = project_events[:, 2].min()
            project_events[:, 2] = project_events[:, 2] - t_min
        
        # 确保极性格式一致（DSEC使用1/-1）
        project_events[:, 3] = np.where(project_events[:, 3] > 0, 1, -1)
        
        return project_events
    
    def convert_to_dvs_format(self, events: np.ndarray) -> np.ndarray:
        """
        转换事件从项目格式到DVS格式
        
        Args:
            events: [N, 4] 项目格式 [x, y, t, p]
            
        Returns:
            [N, 4] DVS格式 [t, x, y, p]
        """
        if len(events) == 0:
            return np.empty((0, 4))
        
        dvs_events = np.zeros_like(events)
        dvs_events[:, 0] = events[:, 2]  # t
        dvs_events[:, 1] = events[:, 0]  # x
        dvs_events[:, 2] = events[:, 1]  # y
        dvs_events[:, 3] = events[:, 3]  # p
        
        return dvs_events
    
    def merge_events(self, background_events: np.ndarray, flare_events: np.ndarray) -> np.ndarray:
        """
        合并背景事件和炫光事件
        
        Args:
            background_events: [N, 4] 项目格式 [x, y, t, p]
            flare_events: [N, 4] 项目格式 [x, y, t, p]
            
        Returns:
            合并的事件数组 [N, 4] 项目格式 [x, y, t, p]
        """
        # 处理空情况
        if len(background_events) == 0 and len(flare_events) == 0:
            return np.empty((0, 4))
        elif len(background_events) == 0:
            return flare_events
        elif len(flare_events) == 0:
            return background_events
        
        # 合并事件
        all_events = np.vstack([background_events, flare_events])
        
        # 按时间戳排序
        sort_indices = np.argsort(all_events[:, 2])
        merged_events = all_events[sort_indices]
        
        return merged_events
    
    def save_events_dvs_format(self, events: np.ndarray, output_path: str, metadata: Optional[Dict] = None):
        """
        保存事件为标准DVS格式H5文件
        
        Args:
            events: 事件数组，自动检测格式并转换为DVS格式
            output_path: 输出文件路径
            metadata: 可选元数据
        """
        if len(events) == 0:
            print(f"⚠️  Warning: No events to save for {output_path}")
            return
        
        # 确保转换为DVS格式 [t, x, y, p]
        if events.shape[1] == 4:
            # 检测格式：如果时间戳在第一列，认为是DVS格式；否则转换
            t_col_0 = events[:, 0]  # 假设是DVS格式的时间戳
            t_col_2 = events[:, 2]  # 假设是项目格式的时间戳
            
            # 启发式判断：时间戳应该是递增的较大数值
            if np.mean(t_col_0) > np.mean(t_col_2) and np.std(t_col_0) > np.std(t_col_2):
                # 第0列像时间戳，认为已是DVS格式
                dvs_events = events
            else:
                # 第2列像时间戳，认为是项目格式，需要转换
                dvs_events = self.convert_to_dvs_format(events)
        else:
            raise ValueError(f"Unexpected event array shape: {events.shape}")
        
        with h5py.File(output_path, 'w') as f:
            # 创建标准DVS格式组织结构
            events_group = f.create_group('events')
            
            # 保存数据
            events_group.create_dataset('t', data=dvs_events[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=dvs_events[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=dvs_events[:, 2].astype(np.uint16), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=dvs_events[:, 3].astype(np.int8), 
                                      compression='gzip', compression_opts=9)
            
            # 保存元数据
            events_group.attrs['num_events'] = len(dvs_events)
            events_group.attrs['resolution_height'] = self.config['data']['resolution_h']
            events_group.attrs['resolution_width'] = self.config['data']['resolution_w']
            events_group.attrs['composition_time'] = time.time()
            
            if metadata:
                for key, value in metadata.items():
                    events_group.attrs[key] = value
    
    def compose_single_sequence(self, flare_file_path: str, sequence_id: int) -> Tuple[str, str]:
        """
        合成单个事件序列
        
        Args:
            flare_file_path: 炫光事件文件路径
            sequence_id: 序列ID
            
        Returns:
            (bg_events_file, merge_events_file) 输出文件路径
        """
        start_time = time.time()
        
        print(f"  Processing flare file: {os.path.basename(flare_file_path)}")
        
        # 1. 加载炫光事件 (DVS格式)
        flare_events_dvs = self.load_flare_events(flare_file_path)
        flare_events_project = self.convert_flare_to_project_format(flare_events_dvs)
        
        # 2. 生成背景事件 (项目格式)
        background_events_project = self.generate_background_events()
        
        # 3. 合并事件 (项目格式)
        merged_events_project = self.merge_events(background_events_project, flare_events_project)
        
        # 4. 创建输出文件名
        timestamp = int(time.time() * 1000)
        base_name = f"composed_sequence_{timestamp}_{sequence_id:05d}"
        
        bg_output_path = os.path.join(self.bg_events_dir, f"{base_name}_bg.h5")
        merge_output_path = os.path.join(self.merge_events_dir, f"{base_name}_merge.h5")
        
        # 5. 保存背景事件
        bg_metadata = {
            'event_type': 'background',
            'source': 'dsec',
            'duration_ms': (background_events_project[:, 2].max() - background_events_project[:, 2].min()) / 1000.0 if len(background_events_project) > 0 else 0
        }
        self.save_events_dvs_format(background_events_project, bg_output_path, bg_metadata)
        
        # 6. 保存合并事件
        merge_metadata = {
            'event_type': 'merged',
            'background_events': len(background_events_project),
            'flare_events': len(flare_events_project),
            'total_events': len(merged_events_project),
            'source_flare_file': os.path.basename(flare_file_path)
        }
        self.save_events_dvs_format(merged_events_project, merge_output_path, merge_metadata)
        
        composition_time = time.time() - start_time
        
        print(f"    Background events: {len(background_events_project):,}")
        print(f"    Flare events: {len(flare_events_project):,}")
        print(f"    Merged events: {len(merged_events_project):,}")
        print(f"    Time: {composition_time:.2f}s")
        
        # 7. Debug可视化
        if self.debug_mode and sequence_id < 3:
            self._save_debug_visualization(
                background_events_project, flare_events_project, merged_events_project, 
                sequence_id, flare_file_path
            )
        
        return bg_output_path, merge_output_path
    
    def _save_debug_visualization(self, bg_events: np.ndarray, flare_events: np.ndarray, 
                                merged_events: np.ndarray, sequence_id: int, flare_source: str):
        """保存debug可视化"""
        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        # 创建三种事件的可视化
        self._create_event_visualization(bg_events, debug_seq_dir, "background", "Background Events (DSEC)")
        self._create_event_visualization(flare_events, debug_seq_dir, "flare", "Flare Events (DVS)")
        self._create_event_visualization(merged_events, debug_seq_dir, "merged", "Merged Events")
        
        # 保存元数据
        self._save_composition_metadata(debug_seq_dir, bg_events, flare_events, merged_events, flare_source)
    
    def _create_event_visualization(self, events: np.ndarray, output_dir: str, 
                                  event_type: str, title: str):
        """创建事件可视化"""
        if len(events) == 0:
            return
        
        type_dir = os.path.join(output_dir, f"{event_type}_events")
        os.makedirs(type_dir, exist_ok=True)
        
        # 多分辨率策略
        resolution_scales = [0.5, 1, 2, 4]
        resolution = (self.config['data']['resolution_w'], self.config['data']['resolution_h'])
        
        for scale in resolution_scales:
            scale_dir = os.path.join(type_dir, f"temporal_{scale}x")
            os.makedirs(scale_dir, exist_ok=True)
            
            # 时间参数 (事件格式为 [x, y, t, p])
            t_min, t_max = events[:, 2].min(), events[:, 2].max()
            duration_ms = (t_max - t_min) / 1000.0
            
            base_window_ms = 10.0
            window_duration_ms = base_window_ms / scale
            window_duration_us = window_duration_ms * 1000
            
            num_frames = max(10, int(duration_ms / window_duration_ms))
            frame_step = (t_max - t_min) / num_frames if num_frames > 1 else 0
            
            for frame_idx in range(min(num_frames, 30)):
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
                            # 根据事件类型选择颜色
                            if event_type == "background":
                                color = (0, 0, 255) if p > 0 else (255, 0, 0)  # 红/蓝
                            elif event_type == "flare":
                                color = (0, 255, 255) if p > 0 else (0, 128, 255)  # 黄/橙
                            else:  # merged
                                # 这里无法区分来源，使用默认颜色
                                color = (255, 255, 255) if p > 0 else (128, 128, 128)  # 白/灰
                            
                            frame[y, x] = color
                
                # 保存帧
                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _save_composition_metadata(self, output_dir: str, bg_events: np.ndarray, 
                                 flare_events: np.ndarray, merged_events: np.ndarray, 
                                 flare_source: str):
        """保存合成元数据"""
        metadata_path = os.path.join(output_dir, "composition_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Event Composition Metadata\n")
            f.write("==========================\n\n")
            
            f.write(f"Source flare file: {os.path.basename(flare_source)}\n\n")
            
            # 背景事件统计
            if len(bg_events) > 0:
                bg_t_min, bg_t_max = bg_events[:, 2].min(), bg_events[:, 2].max()
                bg_duration_ms = (bg_t_max - bg_t_min) / 1000.0
                bg_pos = np.sum(bg_events[:, 3] > 0)
                bg_neg = np.sum(bg_events[:, 3] <= 0)
                
                f.write(f"Background Events:\n")
                f.write(f"  Count: {len(bg_events):,}\n")
                f.write(f"  Duration: {bg_duration_ms:.1f}ms\n")
                f.write(f"  Event rate: {len(bg_events) / (bg_duration_ms / 1000):.1f} events/s\n")
                f.write(f"  Polarity: {bg_pos} ON ({bg_pos/len(bg_events)*100:.1f}%), ")
                f.write(f"{bg_neg} OFF ({bg_neg/len(bg_events)*100:.1f}%)\n\n")
            
            # 炫光事件统计
            if len(flare_events) > 0:
                fl_t_min, fl_t_max = flare_events[:, 2].min(), flare_events[:, 2].max()
                fl_duration_ms = (fl_t_max - fl_t_min) / 1000.0
                fl_pos = np.sum(flare_events[:, 3] > 0)
                fl_neg = np.sum(flare_events[:, 3] <= 0)
                
                f.write(f"Flare Events:\n")
                f.write(f"  Count: {len(flare_events):,}\n")
                f.write(f"  Duration: {fl_duration_ms:.1f}ms\n")
                f.write(f"  Event rate: {len(flare_events) / (fl_duration_ms / 1000):.1f} events/s\n")
                f.write(f"  Polarity: {fl_pos} ON ({fl_pos/len(flare_events)*100:.1f}%), ")
                f.write(f"{fl_neg} OFF ({fl_neg/len(flare_events)*100:.1f}%)\n\n")
            
            # 合并事件统计
            if len(merged_events) > 0:
                mg_t_min, mg_t_max = merged_events[:, 2].min(), merged_events[:, 2].max()
                mg_duration_ms = (mg_t_max - mg_t_min) / 1000.0
                mg_pos = np.sum(merged_events[:, 3] > 0)
                mg_neg = np.sum(merged_events[:, 3] <= 0)
                
                f.write(f"Merged Events:\n")
                f.write(f"  Total count: {len(merged_events):,}\n")
                f.write(f"  Duration: {mg_duration_ms:.1f}ms\n")
                f.write(f"  Event rate: {len(merged_events) / (mg_duration_ms / 1000):.1f} events/s\n")
                f.write(f"  Polarity: {mg_pos} ON ({mg_pos/len(merged_events)*100:.1f}%), ")
                f.write(f"{mg_neg} OFF ({mg_neg/len(merged_events)*100:.1f}%)\n")
                f.write(f"  Composition: {len(bg_events)} background + {len(flare_events)} flare\n")
    
    def compose_batch(self, max_sequences: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        批量合成事件序列
        
        Args:
            max_sequences: 最大处理序列数，None表示处理所有
            
        Returns:
            (bg_files, merge_files) 生成的文件路径列表
        """
        # 查找所有炫光事件文件
        flare_pattern = os.path.join(self.flare_events_dir, "*.h5")
        flare_files = sorted(glob.glob(flare_pattern))
        
        if not flare_files:
            print(f"❌ No flare event files found in {self.flare_events_dir}")
            print("   Please run Step 1 (Flare Event Generation) first.")
            return [], []
        
        if max_sequences is not None:
            flare_files = flare_files[:max_sequences]
        
        print(f"\n🚀 Composing events for {len(flare_files)} sequences...")
        
        bg_files = []
        merge_files = []
        start_time = time.time()
        
        for i, flare_file in enumerate(flare_files):
            print(f"\n--- Composing sequence {i+1}/{len(flare_files)} ---")
            
            try:
                bg_file, merge_file = self.compose_single_sequence(flare_file, i)
                bg_files.append(bg_file)
                merge_files.append(merge_file)
            except Exception as e:
                print(f"❌ Error composing sequence {i}: {e}")
                continue
        
        total_time = time.time() - start_time
        success_rate = len(bg_files) / len(flare_files) * 100
        
        print(f"\n✅ Event composition complete:")
        print(f"  Processed: {len(bg_files)}/{len(flare_files)} sequences ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/len(flare_files):.1f}s per sequence")
        print(f"  Background events: {self.bg_events_dir}")
        print(f"  Merged events: {self.merge_events_dir}")
        
        return bg_files, merge_files


def test_event_composer():
    """测试事件合成器"""
    import yaml
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 启用debug模式
    config['debug_mode'] = True
    
    # 创建合成器
    composer = EventComposer(config)
    
    # 合成测试序列
    bg_files, merge_files = composer.compose_batch(max_sequences=3)
    
    print(f"Test complete! Generated {len(bg_files)} background files and {len(merge_files)} merge files.")
    return bg_files, merge_files


if __name__ == "__main__":
    test_event_composer()