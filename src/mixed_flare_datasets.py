"""
Mixed Flare Dataset Module for EventMamba-FX

This module creates datasets that combine real DSEC background events with 
synthetic flickering flare events for training the denoising model.

Key features:
- Load 1-second DSEC background events using memory-efficient loader
- Generate synthetic flare events using DVS-Flare integration
- Combine events with proper temporal alignment
- Create labels (0=background, 1=flare) for supervised training
- Memory-efficient streaming for large-scale training
"""

import os
import sys
import time
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional, Union

# Import existing efficient loaders
from src.dsec_efficient import DSECEventDatasetEfficient
from src.dvs_flare_integration import DVSFlareEventGenerator


class MixedFlareDataset(Dataset):
    """Dataset that combines DSEC background events with synthetic flare events."""
    
    def __init__(self, config: Dict, split: str = 'train'):
        """Initialize the mixed flare dataset.
        
        Args:
            config: Configuration dictionary
            split: Dataset split ('train', 'val', 'test')
        """
        self.config = config
        self.split = split
        
        # Initialize DSEC background loader
        self.dsec_dataset = DSECEventDatasetEfficient(
            dsec_path=config['data']['dsec_path'],
            flare_path="",  # Not used for background events
            time_window_us=config['data']['time_window_us'],
            sequence_length=config['data']['sequence_length']
        )
        
        # Initialize flare event generator
        self.flare_generator = DVSFlareEventGenerator(config)
        
        # Cache for generated flare events (to avoid regenerating identical sequences)
        self.flare_cache = {}
        self.max_cache_size = 20  # Smaller cache to save memory
        
        # Always use DVS simulation for proper flare events (removed fast mode)
        # This ensures temporal consistency and realistic event generation
        
        # Training parameters
        self.sequence_length = config['data']['sequence_length']
        self.flare_mix_probability = 0.5  # 50% chance of adding flare to each sample
        
        print(f"Initialized MixedFlareDataset ({split}): {len(self.dsec_dataset)} background samples")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dsec_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a randomized mixed sample with advanced generalization features.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (events_tensor, labels_tensor)
            events_tensor: [N, 4] tensor with [x, y, t, p] format  
            labels_tensor: [N] tensor with 0=background, 1=flare
        """
        try:
            # 1. 随机选择场景类型 (实现训练集偏向)
            scenario = self._select_random_scenario()
            
            # 2. 生成随机化的事件数据
            combined_events, labels = self._generate_randomized_events(idx, scenario)
            
            # 3. 安全检查和最终处理
            combined_events, labels = self._apply_final_processing(combined_events, labels)
            
            # 4. 转换为tensor
            events_tensor = torch.tensor(combined_events, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            return events_tensor, labels_tensor
        
        except Exception as e:
            print(f"Error in __getitem__({idx}): {e}")
            # 安全回退: 返回简单的背景事件
            return self._safe_fallback_sample()
    
    def _select_random_scenario(self) -> str:
        """随机选择训练场景类型 (实现训练集偏向)."""
        rand_val = random.random()
        config = self.config['data']['randomized_training']
        
        if rand_val < config['background_contains_flare_prob']:
            return "background_with_flare"  # 75% - 最常见场景
        elif rand_val < config['background_contains_flare_prob'] + config['flare_only_prob']:
            return "flare_only"  # 10% - 只有炫光
        else:
            return "background_only"  # 15% - 只有背景
    
    def _generate_randomized_events(self, idx: int, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
        """生成随机化的事件数据."""
        config = self.config['data']['randomized_training']
        
        # 初始化
        background_events = np.empty((0, 4))
        flare_events = np.empty((0, 4))
        
        # 1. 生成背景事件 (如果需要)
        if scenario in ["background_with_flare", "background_only"]:
            background_events = self._generate_random_background_events(idx, config)
        
        # 2. 生成炫光事件 (如果需要)  
        if scenario in ["background_with_flare", "flare_only"]:
            flare_events = self._generate_random_flare_events(config)
        
        # 3. 应用随机偏移并合并
        combined_events, labels = self._apply_random_offsets_and_merge(
            background_events, flare_events, config
        )
        
        return combined_events, labels
    
    def _generate_random_background_events(self, idx: int, config: Dict) -> np.ndarray:
        """生成随机长度的背景事件."""
        # 随机选择背景事件长度
        bg_duration = random.uniform(*config['background_duration_range'])
        bg_duration_us = int(bg_duration * 1e6)
        
        # 从DSEC数据集加载指定长度的背景事件
        background_data = self.dsec_dataset[idx]
        if isinstance(background_data, tuple):
            background_events = background_data[0]
        else:
            background_events = background_data
            
        if isinstance(background_events, torch.Tensor):
            background_events = background_events.numpy()
        
        # 如果事件数据非空，裁剪到指定长度
        if len(background_events) > 0:
            t_min = background_events[:, 2].min()
            t_max = background_events[:, 2].max()
            current_duration = t_max - t_min
            
            if current_duration > bg_duration_us:
                # 随机选择一个时间段
                max_start_offset = current_duration - bg_duration_us
                start_offset = random.uniform(0, max_start_offset)
                start_time = t_min + start_offset
                end_time = start_time + bg_duration_us
                
                # 筛选时间范围内的事件
                mask = (background_events[:, 2] >= start_time) & (background_events[:, 2] < end_time)
                background_events = background_events[mask]
        
        return background_events
    
    def _generate_random_flare_events(self, config: Dict) -> np.ndarray:
        """生成随机长度的炫光事件."""
        # 随机选择炫光事件长度
        flare_duration = random.uniform(*config['flare_duration_range'])
        
        # 临时修改配置中的duration_sec来生成指定长度的炫光
        original_duration = self.config['data']['flare_synthesis'].get('duration_sec', 0.3)
        self.config['data']['flare_synthesis']['duration_sec'] = flare_duration
        
        try:
            # 生成炫光事件 (带安全检查)
            flare_events = self._get_flare_events()
            
            # 安全检查: 限制最大帧数
            if len(flare_events) > config['max_flare_frames']:
                print(f"Warning: Flare events ({len(flare_events)}) exceed limit, sampling...")
                # 随机采样到限制内
                indices = np.random.choice(len(flare_events), config['max_flare_frames'], replace=False)
                indices = np.sort(indices)  # 保持时间顺序
                flare_events = flare_events[indices]
                
        finally:
            # 恢复原始配置
            self.config['data']['flare_synthesis']['duration_sec'] = original_duration
        
        return flare_events
    
    def _apply_random_offsets_and_merge(self, background_events: np.ndarray, 
                                      flare_events: np.ndarray, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """应用随机偏移并合并事件."""
        # 1. 应用随机偏移
        if len(background_events) > 0:
            bg_offset = random.uniform(*config['background_offset_range']) * 1e6  # 转换为微秒
            background_events = background_events.copy()
            background_events[:, 2] += bg_offset
        
        if len(flare_events) > 0:
            flare_offset = random.uniform(*config['flare_offset_range']) * 1e6
            # 炫光事件需要格式转换和偏移
            flare_formatted = self._format_flare_events_simple(flare_events, flare_offset)
        else:
            flare_formatted = np.empty((0, 4))
        
        # 2. 合并事件
        if len(background_events) == 0 and len(flare_formatted) == 0:
            return np.empty((0, 4)), np.empty(0, dtype=np.int64)
        elif len(background_events) == 0:
            # 只有炫光事件
            labels = np.ones(len(flare_formatted), dtype=np.int64)
            return flare_formatted, labels
        elif len(flare_formatted) == 0:
            # 只有背景事件
            labels = np.zeros(len(background_events), dtype=np.int64)
            return background_events, labels
        else:
            # 两者都有，线性合并
            bg_labels = np.zeros(len(background_events), dtype=np.int64)
            flare_labels = np.ones(len(flare_formatted), dtype=np.int64)
            
            combined_events, combined_labels = self._merge_sorted_events(
                background_events, bg_labels, flare_formatted, flare_labels
            )
            return combined_events, combined_labels
    
    def _format_flare_events_simple(self, flare_events: np.ndarray, time_offset: float) -> np.ndarray:
        """简化的炫光事件格式转换，只应用时间偏移."""
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        # Convert DVS format [t, x, y, p] to EventMamba format [x, y, t, p]
        formatted_events = np.zeros_like(flare_events, dtype=np.float64)
        formatted_events[:, 0] = flare_events[:, 1]  # x
        formatted_events[:, 1] = flare_events[:, 2]  # y 
        formatted_events[:, 2] = flare_events[:, 0] + time_offset  # t + offset
        formatted_events[:, 3] = flare_events[:, 3]  # p
        
        # Convert polarity if needed
        unique_polarities = np.unique(formatted_events[:, 3])
        if np.any(unique_polarities > 1):
            formatted_events[:, 3] = np.where(formatted_events[:, 3] > 0, 1, -1)
        
        return formatted_events
    
    def _apply_final_processing(self, combined_events: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用最终处理：随机输入长度和安全检查."""
        if len(combined_events) == 0:
            return combined_events, labels
        
        config = self.config['data']['randomized_training']
        
        # 1. 随机选择最终输入长度
        final_duration = random.uniform(*config['final_duration_range'])
        final_duration_us = final_duration * 1e6
        
        # 2. 安全检查：不超过最大时长
        max_duration_us = config['max_total_duration'] * 1e6
        final_duration_us = min(final_duration_us, max_duration_us)
        
        # 3. 裁剪到最终长度
        t_min = combined_events[:, 2].min()
        t_max = combined_events[:, 2].max()
        current_duration = t_max - t_min
        
        if current_duration > final_duration_us:
            # 随机选择一个时间段
            max_start_offset = current_duration - final_duration_us
            start_offset = random.uniform(0, max_start_offset)
            start_time = t_min + start_offset
            end_time = start_time + final_duration_us
            
            # 筛选时间范围内的事件
            mask = (combined_events[:, 2] >= start_time) & (combined_events[:, 2] < end_time)
            combined_events = combined_events[mask]
            labels = labels[mask]
        
        # 4. 限制事件数量（防止内存问题）
        max_events = self.sequence_length * 3  # 允许比sequence_length稍大
        if len(combined_events) > max_events:
            # 均匀采样
            indices = np.linspace(0, len(combined_events)-1, max_events, dtype=int)
            combined_events = combined_events[indices]
            labels = labels[indices]
        
        return combined_events, labels
    
    def _safe_fallback_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """安全回退样本 (纯背景事件)."""
        try:
            # 尝试加载简单的背景事件
            idx = random.randint(0, len(self.dsec_dataset) - 1)
            background_data = self.dsec_dataset[idx]
            
            if isinstance(background_data, tuple):
                background_events = background_data[0]
            else:
                background_events = background_data
                
            if isinstance(background_events, torch.Tensor):
                background_events = background_events.numpy()
            
            # 限制长度
            if len(background_events) > self.sequence_length:
                background_events = background_events[:self.sequence_length]
            
            labels = np.zeros(len(background_events), dtype=np.int64)
            
            events_tensor = torch.tensor(background_events, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            
            return events_tensor, labels_tensor
            
        except:
            # 最终安全回退：空样本
            empty_events = torch.zeros((1, 4), dtype=torch.float32)
            empty_labels = torch.zeros(1, dtype=torch.long)
            return empty_events, empty_labels
    
    def _get_flare_events(self) -> np.ndarray:
        """Get flare events, using cache when possible.
        
        Returns:
            Flare events array in [timestamp_us, x, y, polarity] format
        """
        # 恢复DVS-Voltmeter模拟器调用以保持炫光真实性
        # 使用缓存来减少重复计算开销
        return self._get_dvs_flare_events()
    
    def _generate_synthetic_flare_events(self) -> np.ndarray:
        """Generate synthetic flare events with realistic temporal patterns.
        
        Returns:
            Synthetic flare events in [timestamp_us, x, y, polarity] format
        """
        # Generate realistic flare parameters using power grid frequencies
        realistic_freqs = self.config['data']['flare_synthesis']['realistic_frequencies']
        base_frequencies = [
            realistic_freqs['power_50hz'],
            realistic_freqs['power_60hz'],
            realistic_freqs['japan_east'], 
            realistic_freqs['japan_west']
        ]
        base_freq = random.choice(base_frequencies)
        variation = self.config['data']['flare_synthesis']['frequency_variation']
        frequency = base_freq + random.uniform(-variation, variation)
        
        # Generate realistic spatial distribution (flare-like pattern)
        center_x = random.randint(50, 190)  # DVS resolution 240x180
        center_y = random.randint(30, 150)
        flare_radius = random.randint(20, 60)
        
        # Generate temporally consistent events based on flickering pattern
        duration_us = self.config['data']['time_window_us']  # 0.5s duration
        fps = min(2000, frequency * 8)  # Dynamic fps based on frequency
        frame_duration_us = int(1e6 / fps)
        num_frames = int(duration_us / frame_duration_us)
        
        # Generate flicker intensity curve
        omega = 2 * np.pi * frequency / 1e6  # Convert to per-microsecond
        
        events = []
        for frame_idx in range(num_frames):
            t_frame = frame_idx * frame_duration_us
            
            # Calculate flicker intensity at this time
            intensity = 0.5 * (1 + np.sin(omega * t_frame))
            
            # Number of events proportional to intensity (more events when brighter)
            base_events_per_frame = random.randint(5, 25)
            events_per_frame = int(base_events_per_frame * intensity)
            
            # Generate events within this frame
            for _ in range(events_per_frame):
                # Time within frame (small random offset for realism)
                t = t_frame + random.randint(0, frame_duration_us - 1)
                
                # Spatial position around flare center
                angle = random.uniform(0, 2 * np.pi)
                radius = np.random.exponential(flare_radius / 3)
                x = int(center_x + radius * np.cos(angle))
                y = int(center_y + radius * np.sin(angle))
                
                # Clip to DVS resolution
                x = np.clip(x, 0, 239)
                y = np.clip(y, 0, 179)
                
                # Polarity based on intensity change (more realistic)
                p = 1 if intensity > 0.5 else 0  # Bright = ON, dim = OFF
                
                events.append([t, x, y, p])
        
        # Convert to numpy array (already temporally sorted)
        events = np.array(events) if events else np.empty((0, 4))
        
        return events
    
    def _get_dvs_flare_events(self) -> np.ndarray:
        """Get flare events using DVS simulation (slow but realistic method).
        
        Returns:
            DVS-simulated flare events in [timestamp_us, x, y, polarity] format
        """
        try:
            # Create cache key based on random parameters (using realistic frequencies)
            realistic_freqs = self.config['data']['flare_synthesis']['realistic_frequencies']
            base_frequencies = [
                realistic_freqs['power_50hz'],
                realistic_freqs['power_60hz'],
                realistic_freqs['japan_east'], 
                realistic_freqs['japan_west']
            ]
            base_freq = random.choice(base_frequencies)
            variation = self.config['data']['flare_synthesis']['frequency_variation']
            frequency = base_freq + random.uniform(-variation, variation)
            
            curve_type = random.choice(self.config['data']['flare_synthesis']['flicker_curves'])
            cache_key = f"{frequency:.1f}_{curve_type}"
            
            # Check cache first
            if cache_key in self.flare_cache:
                cached_events, _ = self.flare_cache[cache_key]
                if len(cached_events) > 0:  # 安全检查
                    return cached_events.copy()  # Return copy to avoid modification
            
            # Generate new flare events using DVS simulation
            flare_events, metadata = self.flare_generator.generate_flare_events(cleanup=True)
            
            # 安全检查: 确保生成的事件不为空
            if len(flare_events) == 0:
                print(f"Warning: DVS simulation generated no events for {cache_key}")
                # 如果DVS失败，回退到快速合成方法
                return self._generate_synthetic_flare_events()
            
            # Add to cache if not full
            if len(self.flare_cache) < self.max_cache_size:
                self.flare_cache[cache_key] = (flare_events.copy(), metadata)
            
            return flare_events
            
        except Exception as e:
            print(f"Error in DVS flare generation: {e}")
            print("Falling back to synthetic flare generation...")
            # 出错时回退到快速合成方法
            return self._generate_synthetic_flare_events()
    
    def _combine_events(self, background_events: np.ndarray, 
                       flare_events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combine background and flare events with proper temporal alignment.
        
        Args:
            background_events: Background events [N1, 4] - [x, y, t, p]
            flare_events: Flare events [N2, 4] - [t, x, y, p] (DVS simulator format)
            
        Returns:
            Tuple of (combined_events, labels)
            combined_events: [N1+N2, 4] - [x, y, t, p] format
            labels: [N1+N2] - 0=background, 1=flare
        """
        # Ensure both arrays have events
        if len(background_events) == 0:
            if len(flare_events) == 0:
                return np.empty((0, 4)), np.empty(0, dtype=np.int64)
            # Only flare events
            flare_formatted = self._format_flare_events(flare_events, background_events)
            labels = np.ones(len(flare_formatted), dtype=np.int64)
            return flare_formatted, labels
        
        if len(flare_events) == 0:
            # Only background events
            labels = np.zeros(len(background_events), dtype=np.int64)
            return background_events, labels
        
        # Align flare events temporally with background events
        flare_formatted = self._format_flare_events(flare_events, background_events)
        
        # Create labels
        bg_labels = np.zeros(len(background_events), dtype=np.int64)
        flare_labels = np.ones(len(flare_formatted), dtype=np.int64)
        
        # Linear merge of two sorted event streams (O(n+m) instead of O(nlogn) sorting)
        combined_events, combined_labels = self._merge_sorted_events(
            background_events, bg_labels, flare_formatted, flare_labels
        )
        
        return combined_events, combined_labels
    
    def _merge_sorted_events(self, events1: np.ndarray, labels1: np.ndarray,
                           events2: np.ndarray, labels2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two sorted event streams using linear merge (like merge sort).
        
        Args:
            events1: First sorted event stream [N1, 4] - [x, y, t, p]
            labels1: Labels for first stream [N1]
            events2: Second sorted event stream [N2, 4] - [x, y, t, p] 
            labels2: Labels for second stream [N2]
            
        Returns:
            Tuple of (merged_events, merged_labels)
        """
        if len(events1) == 0:
            return events2, labels2
        if len(events2) == 0:
            return events1, labels1
            
        # Initialize pointers and result arrays
        i, j = 0, 0
        total_events = len(events1) + len(events2)
        merged_events = np.zeros((total_events, 4), dtype=events1.dtype)
        merged_labels = np.zeros(total_events, dtype=labels1.dtype)
        k = 0
        
        # Linear merge using dual pointers
        while i < len(events1) and j < len(events2):
            # Compare timestamps (column 2)
            if events1[i, 2] <= events2[j, 2]:
                merged_events[k] = events1[i]
                merged_labels[k] = labels1[i]
                i += 1
            else:
                merged_events[k] = events2[j] 
                merged_labels[k] = labels2[j]
                j += 1
            k += 1
        
        # Add remaining events from first stream
        while i < len(events1):
            merged_events[k] = events1[i]
            merged_labels[k] = labels1[i]
            i += 1
            k += 1
            
        # Add remaining events from second stream
        while j < len(events2):
            merged_events[k] = events2[j]
            merged_labels[k] = labels2[j]
            j += 1
            k += 1
            
        return merged_events, merged_labels
    
    def _format_flare_events(self, flare_events: np.ndarray, 
                           background_events: np.ndarray) -> np.ndarray:
        """Format flare events to match background event format and add smart temporal offset.
        
        Args:
            flare_events: [N, 4] - [timestamp_us, x, y, polarity] (DVS format)
            background_events: [M, 4] - [x, y, t, p] (EventMamba format)
            
        Returns:
            Formatted flare events [N, 4] - [x, y, t, p] (EventMamba format)
        """
        if len(flare_events) == 0:
            return np.empty((0, 4))
        
        # Convert DVS format [t, x, y, p] to EventMamba format [x, y, t, p]
        formatted_events = np.zeros_like(flare_events, dtype=np.float64)
        formatted_events[:, 0] = flare_events[:, 1]  # x
        formatted_events[:, 1] = flare_events[:, 2]  # y
        formatted_events[:, 2] = flare_events[:, 0]  # t
        formatted_events[:, 3] = flare_events[:, 3]  # p
        
        # 实现智能时间偏移策略
        if len(background_events) > 0:
            bg_t_min = background_events[:, 2].min()
            bg_t_max = background_events[:, 2].max()
            bg_duration = bg_t_max - bg_t_min
            
            # 获取配置的时间偏移范围
            offset_range = self.config['data']['flare_synthesis']['time_offset_range']
            offset_start = offset_range[0] * 1e6  # 转换为微秒
            offset_end = offset_range[1] * 1e6
            
            # 随机选择炫光出现的时间点 (在0.2s-0.7s范围内)
            available_offset_range = min(offset_end - offset_start, bg_duration - (formatted_events[:, 2].max() - formatted_events[:, 2].min()))
            if available_offset_range > 0:
                random_offset = random.uniform(0, available_offset_range)
                target_start_time = bg_t_min + offset_start + random_offset
                
                # 将炫光事件的时间戳调整到目标位置
                flare_t_min = formatted_events[:, 2].min()
                formatted_events[:, 2] = formatted_events[:, 2] - flare_t_min + target_start_time
                
                # 确保炫光事件不超出背景事件的时间范围
                formatted_events[:, 2] = np.clip(formatted_events[:, 2], bg_t_min, bg_t_max)
            else:
                # 如果背景时间太短，退回到简单的比例缩放
                flare_t_min = formatted_events[:, 2].min()
                flare_t_max = formatted_events[:, 2].max()
                flare_duration = flare_t_max - flare_t_min
                
                if flare_duration > 0:
                    scale_factor = min(1.0, bg_duration / flare_duration)
                    formatted_events[:, 2] = (formatted_events[:, 2] - flare_t_min) * scale_factor + bg_t_min
        
        # Convert polarity from DVS format (0/255) to EventMamba format (-1/1) if needed
        unique_polarities = np.unique(formatted_events[:, 3])
        if np.any(unique_polarities > 1):  # DVS format detected
            formatted_events[:, 3] = np.where(formatted_events[:, 3] > 0, 1, -1)
        
        return formatted_events
    
    def get_timing_stats(self) -> Dict[str, float]:
        """Get timing statistics for dataset operations.
        
        Returns:
            Dictionary with timing information
        """
        # Test a few samples to get timing stats
        num_samples = min(10, len(self))
        times = []
        
        for i in range(num_samples):
            start_time = time.time()
            _ = self[random.randint(0, len(self) - 1)]
            times.append(time.time() - start_time)
        
        return {
            'mean_sample_time_sec': np.mean(times),
            'std_sample_time_sec': np.std(times),
            'min_sample_time_sec': np.min(times),
            'max_sample_time_sec': np.max(times),
        }


def create_mixed_flare_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for mixed flare datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = MixedFlareDataset(config, split='train')
    val_dataset = MixedFlareDataset(config, split='val')
    test_dataset = MixedFlareDataset(config, split='test')
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def test_mixed_flare_dataset(config_path: str = "configs/config.yaml"):
    """Test the mixed flare dataset functionality."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create test dataset
    print("Creating mixed flare dataset...")
    dataset = MixedFlareDataset(config, split='train')
    
    # Test a few samples
    print(f"\nTesting {len(dataset)} samples...")
    
    sample_indices = [0, len(dataset)//2, len(dataset)-1] if len(dataset) > 2 else [0]
    
    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1} (index {idx}):")
        start_time = time.time()
        
        events, labels = dataset[idx]
        
        sample_time = time.time() - start_time
        
        print(f"  Events shape: {events.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Background events: {torch.sum(labels == 0).item()}")
        print(f"  Flare events: {torch.sum(labels == 1).item()}")
        print(f"  Sample time: {sample_time:.3f}s")
        
        # Check event format
        if len(events) > 0:
            print(f"  Time range: {events[:, 2].min():.0f} - {events[:, 2].max():.0f} μs")
            print(f"  Spatial range: x=[{events[:, 0].min():.0f}, {events[:, 0].max():.0f}], "
                  f"y=[{events[:, 1].min():.0f}, {events[:, 1].max():.0f}]")
    
    # Get timing statistics
    print(f"\nGetting timing statistics...")
    timing_stats = dataset.get_timing_stats()
    for key, value in timing_stats.items():
        print(f"  {key}: {value:.3f}s")
    
    print(f"\nMixed flare dataset test completed successfully!")
    
    return dataset


if __name__ == "__main__":
    # Run test
    dataset = test_mixed_flare_dataset()
    print("Mixed flare dataset module test completed!")