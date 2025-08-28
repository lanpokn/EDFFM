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
        self.composition_config = config.get('composition', {})
        
        # 输入路径：炫光事件和光源事件
        self.flare_events_dir = os.path.join('output', 'data', 'flare_events')
        self.light_source_events_dir = os.path.join('output', 'data', 'light_source_events')
        
        # 合成方法设置
        self.merge_method = self.composition_config.get('merge_method', 'simple')
        self.generate_both_methods = self.composition_config.get('generate_both_methods', False)
        
        # 输出路径字典结构
        self.output_dirs = {}
        
        # 始终为 'simple' 方法创建目录
        self.output_dirs['simple'] = {
            'stage1': os.path.join('output', 'data', 'simple_method', 'background_with_light_events'),
            'stage2': os.path.join('output', 'data', 'simple_method', 'background_with_flare_events')
        }
        
        # 仅在需要时为 'physics' 方法创建目录
        if self.merge_method == 'physics' or self.generate_both_methods:
            self.output_dirs['physics'] = {
                'stage1': os.path.join('output', 'data', 'physics_method', 'background_with_light_events'),
                'stage2': os.path.join('output', 'data', 'physics_method', 'background_with_flare_events')
            }
        
        # 循环创建所有需要的目录
        for method_name, paths in self.output_dirs.items():
            os.makedirs(paths['stage1'], exist_ok=True)
            os.makedirs(paths['stage2'], exist_ok=True)
        
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
        
        # 背景事件持续时间：固定100ms匹配炫光最大长度
        self.bg_duration_ms = 100.0  # 固定100ms
        
        print(f"🚀 EventComposer initialized (Dual-Stage Composition Mode):")
        print(f"  ✅ CORRECTED LOGIC - Three separate compositions:")
        print(f"    - Stage 1: Background + Light Source → Clean scene")
        print(f"    - Stage 2: Background + Flare → Flare-contaminated scene")
        print(f"  Merge method: {self.merge_method}")
        print(f"  Generate both methods: {self.generate_both_methods}")
        print(f"  Inputs:")
        print(f"    - Flare events: {self.flare_events_dir}")
        print(f"    - Light Source events: {self.light_source_events_dir}")
        print(f"    - Background events: DSEC Dataset (randomly sampled)")
        print(f"  Outputs:")
        for method_name, paths in self.output_dirs.items():
            print(f"    - Method '{method_name}':")
            print(f"      - Stage 1 (BG+Light): {paths['stage1']}")
            print(f"      - Stage 2 (BG+Flare): {paths['stage2']}")
        print(f"  DSEC dataset size: {len(self.dsec_dataset)} time windows")
        print(f"  Background duration: {self.bg_duration_ms:.0f}ms (fixed)")
        print(f"  Debug mode: {self.debug_mode}")
    
    def load_flare_events(self, flare_file_path: str) -> np.ndarray:
        """
        通用H5事件加载器，用于加载炫光和光源事件
        
        Args:
            flare_file_path: 事件H5文件路径
            
        Returns:
            事件数组 [N, 4] 格式 [t, x, y, p] (DVS格式)
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
        生成背景事件 - 固定100ms长度
        
        Returns:
            背景事件数组 [N, 4] 格式 [x, y, t, p] (项目格式)，
            确保返回的是一个标准的、非结构化的 float64 ndarray。
        """
        # 固定100ms时长
        duration_ms = self.bg_duration_ms
        duration_us = int(duration_ms * 1000)
        
        # 随机选择DSEC样本
        idx = random.randint(0, len(self.dsec_dataset) - 1)
        
        # ==================== MODIFICATION: START ====================
        # 原始调用，可能返回结构化数组或其他问题数据
        raw_events = self.dsec_dataset[idx]  # 返回 [x, y, t, p] 格式
        
        # 如果没有事件，直接返回空的标准数组
        if len(raw_events) == 0:
            return np.empty((0, 4), dtype=np.float64)

        # **核心修复**：强制重建为标准ndarray，确保类型纯净
        # 即使原始数据看起来是正常的ndarray，我们也要重新构建以确保没有隐藏的类型问题
        
        # 逐列提取并重新堆叠，确保每列都是纯数值类型
        x = np.asarray(raw_events[:, 0], dtype=np.float64)
        y = np.asarray(raw_events[:, 1], dtype=np.float64)
        t = np.asarray(raw_events[:, 2], dtype=np.float64)
        p = np.asarray(raw_events[:, 3], dtype=np.float64)
        
        # 重新构建为完全标准的ndarray，消除任何潜在的类型污染
        background_events = np.column_stack([x, y, t, p]).astype(np.float64)
        # ===================== MODIFICATION: END =====================
        
        # 裁剪到指定持续时间 (现在作用于干净的`background_events`数组)
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
        
        return background_events if len(background_events) > 0 else np.empty((0, 4), dtype=np.float64)
    
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
    
    def _merge_events_physics(self, events1: np.ndarray, events2: np.ndarray, 
                              weight1: float, weight2: float) -> np.ndarray:
        """
        Merges two event streams based on a physically-grounded probabilistic model.
        
        Args:
            events1: First event stream (e.g., background).
            events2: Second event stream (e.g., light source or flare).
            weight1: The intensity weight to accumulate for each event in events1.
            weight2: The intensity weight to accumulate for each event in events2.
            
        Returns:
            The merged event array.
        """
        
        # --- 1. 获取通用参数 ---
        params = self.composition_config.get('physics_params', {})
        jitter_us = params.get('temporal_jitter_us', 50)
        epsilon_raw = params.get('epsilon', 1e-9)
        # **修复**: 强制转换epsilon为浮点数，防止YAML解析为字符串
        epsilon = float(epsilon_raw)
        W, H = self.config['data']['resolution_w'], self.config['data']['resolution_h']

        # --- 2. 动态估计光强图 ---
        Y_est1 = np.zeros((H, W), dtype=np.float32)
        x1, y1 = None, None
        if len(events1) > 0:
            # **清理**: 不再需要 np.array(events1, ...)，因为events1已经是干净的了
            x1 = np.clip(events1[:, 0].astype(np.int32), 0, W-1)
            y1 = np.clip(events1[:, 1].astype(np.int32), 0, H-1)
            # **加固**: 使用 np.add.at 是最高效、最安全的方式
            np.add.at(Y_est1, (y1, x1), weight1)

        Y_est2 = np.zeros((H, W), dtype=np.float32)
        x2, y2 = None, None
        if len(events2) > 0:
            # **清理**: 不再需要 np.array(events2, ...)
            x2 = np.clip(events2[:, 0].astype(np.int32), 0, W-1)
            y2 = np.clip(events2[:, 1].astype(np.int32), 0, H-1)
            # **加固**: 同样使用 np.add.at
            np.add.at(Y_est2, (y2, x2), weight2)

        # --- 3. 计算权重图 A(x,y) for events2 ---
        # A(x,y) 代表了 events2 在该像素的"主导权"或保留概率
        A = Y_est2 / (Y_est1 + Y_est2 + epsilon)
        
        # 保存权重图用于debug
        self._last_weight_map = A
        
        # --- 4. 概率门控 ---
        if len(events1) > 0 and x1 is not None and y1 is not None:
            # 使用已经验证过的坐标
            prob_keep1 = 1.0 - A[y1, x1] # 保留概率是 1 - A
            mask1 = np.random.rand(len(events1)) < prob_keep1
            # **清理**: 直接使用 events1
            kept_events1 = events1[mask1]
        else:
            kept_events1 = np.empty((0, 4), dtype=np.float64) # 确保空数组类型一致
            
        if len(events2) > 0 and x2 is not None and y2 is not None:
            # 使用已经验证过的坐标
            prob_keep2 = A[y2, x2] # 保留概率是 A
            mask2 = np.random.rand(len(events2)) < prob_keep2
            # **清理**: 直接使用 events2
            kept_events2 = events2[mask2]
        else:
            kept_events2 = np.empty((0, 4), dtype=np.float64) # 确保空数组类型一致

        # --- 5. 合并、时间扰动和排序 ---
        if len(kept_events1) == 0 and len(kept_events2) == 0:
            return np.empty((0, 4), dtype=np.float64)
        elif len(kept_events1) == 0:
            merged_events = kept_events2
        elif len(kept_events2) == 0:
            merged_events = kept_events1
        else:
            merged_events = np.vstack([kept_events1, kept_events2])
        
        # 时间扰动 (仅在扰动范围大于0时应用)
        if jitter_us > 0 and len(merged_events) > 0:
            time_jitter = np.random.uniform(-jitter_us, jitter_us, len(merged_events))
            merged_events[:, 2] += time_jitter
            
        # 按时间排序
        if len(merged_events) > 0:
            sort_indices = np.argsort(merged_events[:, 2])
            merged_events = merged_events[sort_indices]
            
        return merged_events
    
    def _merge_events_simple(self, background_events: np.ndarray, flare_events: np.ndarray) -> np.ndarray:
        """
        简单的事件合并方法 - 原有逻辑
        
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

    def merge_events(self, events1: np.ndarray, events2: np.ndarray,
                     method: str = "simple", 
                     weight1: float = 1.0, weight2: float = 1.0) -> np.ndarray:
        """
        合并两个事件流 - 支持多种合成方法
        
        Args:
            events1: [N, 4] 项目格式 [x, y, t, p] - 第一个事件流 (如背景)
            events2: [N, 4] 项目格式 [x, y, t, p] - 第二个事件流 (如炫光/光源)  
            method: 合成方法 "simple" 或 "physics"
            weight1: 第一个事件流的权重 (physics方法使用)
            weight2: 第二个事件流的权重 (physics方法使用)
            
        Returns:
            合并的事件数组 [N, 4] 项目格式 [x, y, t, p]
        """
        if method == "simple":
            return self._merge_events_simple(events1, events2)
        elif method == "physics":
            return self._merge_events_physics(events1, events2, weight1, weight2)
        else:
            raise ValueError(f"Unknown merge method: {method}")
    
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
    
    def compose_single_sequence(self, flare_file_path: str, light_source_file_path: str, sequence_id: int) -> Tuple[str, str]:
        """
        合成单个事件序列 - 支持双方法并行生成
        
        Args:
            flare_file_path: 炫光事件文件路径
            light_source_file_path: 光源事件文件路径
            sequence_id: 序列ID
            
        Returns:
            (bg_light_file, full_scene_file) 主方法的输出文件路径
        """
        start_time = time.time()
        
        print(f"  Processing flare file: {os.path.basename(flare_file_path)}")
        print(f"  Processing light source file: {os.path.basename(light_source_file_path)}")
        
        # 1. 加载所有事件数据
        flare_events_dvs = self.load_flare_events(flare_file_path)
        flare_events_project = self.convert_flare_to_project_format(flare_events_dvs)
        
        light_source_events_dvs = self.load_flare_events(light_source_file_path)
        light_source_events_project = self.convert_flare_to_project_format(light_source_events_dvs)
        
        background_events_project = self.generate_background_events()
        
        print(f"    Background events: {len(background_events_project):,}")
        print(f"    Light source events: {len(light_source_events_project):,}")
        print(f"    Flare events: {len(flare_events_project):,}")
        
        # 定义一个内部函数来处理单个方法的完整流程，以实现代码复用
        def _run_composition_for_method(method_name: str):
            print(f"    Running composition for method: '{method_name}'")
            
            params = self.composition_config.get('physics_params', {})
            
            # --- Stage 1: BG + Light Source ---
            bg_weight = params.get('background_event_weight', 0.2)
            light_weight = params.get('light_source_event_weight', 1.0)
            s1_merged = self.merge_events(background_events_project, 
                                          light_source_events_project, 
                                          method=method_name,
                                          weight1=bg_weight, weight2=light_weight)
            
            # 保存 Stage 1 权重图 (如果适用)
            if method_name == 'physics' and self.debug_mode:
                self._save_weight_map_visualization(sequence_id, "stage1_bg_light")

            # --- Stage 2: BG + Flare (正确的三元合成逻辑) ---
            # ✅ 修复逻辑错误：Stage 2 应该是 Background + Flare，而不是 (BG+Light) + Flare
            # 这样才能提供干净的"背景+光源"和"背景+炫光"两种场景供对比
            
            bg_weight = params.get('background_event_weight', 0.2)
            flare_weight = params.get('flare_intensity_multiplier', 1.0)
            s2_merged = self.merge_events(background_events_project, 
                                          flare_events_project,
                                          method=method_name,
                                          weight1=bg_weight, weight2=flare_weight)

            # 保存 Stage 2 权重图 (如果适用)
            if method_name == 'physics' and self.debug_mode:
                self._save_weight_map_visualization(sequence_id, "stage2_full_scene")

            # --- 保存文件 ---
            base_name = f"composed_{int(time.time() * 1000)}_{sequence_id:05d}"
            s1_path = os.path.join(self.output_dirs[method_name]['stage1'], f"{base_name}_bg_light.h5")
            s2_path = os.path.join(self.output_dirs[method_name]['stage2'], f"{base_name}_bg_flare.h5")
            
            # 保存 Stage 1 事件
            bg_light_metadata = {
                'event_type': 'background_with_light',
                'method': method_name,
                'background_events': len(background_events_project),
                'light_source_events': len(light_source_events_project),
                'stage1_merged_events': len(s1_merged),
                'source_light_file': os.path.basename(light_source_file_path)
            }
            self.save_events_dvs_format(s1_merged, s1_path, bg_light_metadata)
            
            # 保存 Stage 2 事件
            full_scene_metadata = {
                'event_type': 'background_with_flare',
                'method': method_name,
                'background_events': len(background_events_project),
                'flare_events': len(flare_events_project),
                'total_events': len(s2_merged),
                'source_flare_file': os.path.basename(flare_file_path)
            }
            self.save_events_dvs_format(s2_merged, s2_path, full_scene_metadata)
            
            print(f"      Stage 1 ({method_name}): {len(s1_merged):,} events")
            print(f"      Stage 2 ({method_name}): {len(s2_merged):,} events")
            
            return s1_merged, s2_merged, s1_path, s2_path

        # --- 根据配置决定运行哪个(些)流程 ---
        final_paths = ()
        if self.generate_both_methods:
            _, _, s1_p, s2_p = _run_composition_for_method('simple')
            _, _, s1_p_phys, s2_p_phys = _run_composition_for_method('physics')
            # 返回主方法的结果路径
            if self.merge_method == 'physics':
                final_paths = (s1_p_phys, s2_p_phys)
            else:
                final_paths = (s1_p, s2_p)
        else:
            _, _, s1_p, s2_p = _run_composition_for_method(self.merge_method)
            final_paths = (s1_p, s2_p)
            
        composition_time = time.time() - start_time
        print(f"    Total composition time: {composition_time:.2f}s")
        
        # Debug可视化
        if self.debug_mode:
            debug_events = {
                "01_background_raw": background_events_project,
                "02_light_source_raw": light_source_events_project,
                "03_flare_raw": flare_events_project,
            }
            debug_metadata = {
                'flare_file': os.path.basename(flare_file_path),
                'light_source_file': os.path.basename(light_source_file_path),
            }
            self._save_debug_visualization(debug_events, sequence_id, debug_metadata)
        
        return final_paths
    
    def _save_weight_map_visualization(self, sequence_id: int, stage_name: str):
        """Saves the last computed weight map A(x,y) as a heatmap."""
        
        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        if hasattr(self, '_last_weight_map') and self._last_weight_map is not None:
            import cv2
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            
            # 获取权重图
            A = self._last_weight_map
            H, W = A.shape
            
            # 创建热力图可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 原始权重图
            im1 = ax1.imshow(A, cmap='viridis', vmin=0, vmax=1)
            ax1.set_title(f'Weight Map A(x,y) - {stage_name}')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1, label='Probability')
            
            # 权重图的直方图分布
            ax2.hist(A.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_title('Weight Distribution')
            ax2.set_xlabel('Weight Value')
            ax2.set_ylabel('Pixel Count')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_weight = np.mean(A)
            std_weight = np.std(A)
            max_weight = np.max(A)
            min_weight = np.min(A)
            ax2.axvline(mean_weight, color='red', linestyle='--', label=f'Mean: {mean_weight:.3f}')
            ax2.legend()
            
            plt.tight_layout()
            
            # 保存可视化
            vis_path = os.path.join(debug_seq_dir, f"weight_map_{stage_name}.png")
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 同时保存为OpenCV热力图
            heatmap_normalized = (A * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_VIRIDIS)
            heatmap_path = os.path.join(debug_seq_dir, f"weight_heatmap_{stage_name}.png")
            cv2.imwrite(heatmap_path, heatmap_colored)
            
            # 保存权重图的统计信息
            stats_path = os.path.join(debug_seq_dir, f"weight_stats_{stage_name}.txt")
            with open(stats_path, 'w') as f:
                f.write(f"Weight Map Statistics - {stage_name}\\n")
                f.write(f"=====================================\\n")
                f.write(f"Resolution: {W}x{H}\\n")
                f.write(f"Mean weight: {mean_weight:.6f}\\n")
                f.write(f"Std weight: {std_weight:.6f}\\n")
                f.write(f"Min weight: {min_weight:.6f}\\n")
                f.write(f"Max weight: {max_weight:.6f}\\n")
                f.write(f"Non-zero pixels: {np.count_nonzero(A)} ({np.count_nonzero(A)/(W*H)*100:.2f}%)\\n")
            
            print(f"      Weight map saved: {vis_path}")
            self._last_weight_map = None  # 清理
    
    def _save_debug_visualization(self, events_dict: Dict[str, np.ndarray], 
                                sequence_id: int, metadata: Dict):
        """保存debug可视化"""
        debug_seq_dir = os.path.join(self.debug_dir, f"composition_{sequence_id:03d}")
        os.makedirs(debug_seq_dir, exist_ok=True)
        
        # 为每一个事件流生成可视化
        for event_name, events in events_dict.items():
            if len(events) > 0:
                title = self._get_event_title(event_name)
                self._create_event_visualization(events, debug_seq_dir, event_name, title)
        
        # 保存元数据
        self._save_enhanced_composition_metadata(debug_seq_dir, events_dict, metadata)
    
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
                            # 统一使用红/蓝颜色 (极性区分)
                            color = (0, 0, 255) if p > 0 else (255, 0, 0)  # ON=红, OFF=蓝
                            frame[y, x] = color
                
                # 保存帧
                import cv2
                frame_path = os.path.join(scale_dir, f"frame_{frame_idx:03d}.png")
                cv2.imwrite(frame_path, frame)
    
    def _get_event_title(self, event_name: str) -> str:
        """获取事件类型的标题"""
        title_map = {
            "01_background_raw": "Background Events (DSEC)",
            "02_light_source_raw": "Light Source Events (DVS)", 
            "03_flare_raw": "Flare Events (DVS)",
            "04_background_with_light": "Stage 1: Background + Light Source",
            "05_full_scene": "Stage 2: Full Scene (BG+Light+Flare)"
        }
        return title_map.get(event_name, event_name.replace("_", " ").title())
    
    def _save_enhanced_composition_metadata(self, output_dir: str, events_dict: Dict[str, np.ndarray], metadata: Dict):
        """保存增强合成元数据"""
        metadata_path = os.path.join(output_dir, "composition_metadata.txt")
        
        with open(metadata_path, 'w') as f:
            f.write("Event Composition Metadata (Three-Source Mode)\n")
            f.write("===============================================\n\n")
            
            f.write(f"Source Files:\n")
            f.write(f"  Flare: {metadata.get('flare_file', 'N/A')}\n")
            f.write(f"  Light Source: {metadata.get('light_source_file', 'N/A')}\n\n")
            
            # 为每种事件类型生成统计
            for event_name, events in events_dict.items():
                if len(events) > 0:
                    title = self._get_event_title(event_name)
                    t_min, t_max = events[:, 2].min(), events[:, 2].max()
                    duration_ms = (t_max - t_min) / 1000.0
                    pos_events = np.sum(events[:, 3] > 0)
                    neg_events = np.sum(events[:, 3] <= 0)
                    
                    f.write(f"{title}:\n")
                    f.write(f"  Count: {len(events):,}\n")
                    f.write(f"  Duration: {duration_ms:.1f}ms\n")
                    if duration_ms > 0:
                        f.write(f"  Event rate: {len(events) / (duration_ms / 1000):.1f} events/s\n")
                    f.write(f"  Polarity: {pos_events} ON ({pos_events/len(events)*100:.1f}%), ")
                    f.write(f"{neg_events} OFF ({neg_events/len(events)*100:.1f}%)\n\n")
    
    def compose_batch(self, max_sequences: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        批量合成事件序列。匹配flare和light_source文件，并为每对匹配随机采样一个背景。
        
        Args:
            max_sequences: 最大处理序列数，None表示处理所有
            
        Returns:
            (bg_light_files, full_scene_files) 生成的文件路径列表
        """
        # 查找 flare 和 light_source 事件文件
        flare_files = {os.path.basename(p): p for p in glob.glob(os.path.join(self.flare_events_dir, "*.h5"))}
        light_source_files = {os.path.basename(p): p for p in glob.glob(os.path.join(self.light_source_events_dir, "*.h5"))}
        
        if not flare_files or not light_source_files:
            print(f"❌ Flare or light source event files not found.")
            print(f"   Flare dir: {self.flare_events_dir}")
            print(f"   Light source dir: {self.light_source_events_dir}")
            return [], []
        
        # 找到两个目录中文件名共同的部分，作为匹配的序列
        # 提取文件名的公共部分（去掉前缀）
        flare_bases = {f.replace('flare_', ''): f for f in flare_files.keys()}
        light_source_bases = {f.replace('light_source_', ''): f for f in light_source_files.keys()}
        
        # 找到公共的base名字
        common_bases = sorted(list(set(flare_bases.keys()) & set(light_source_bases.keys())))
        
        if not common_bases:
            print("❌ No matching flare and light source files found.")
            return [], []
        
        if max_sequences is not None:
            common_bases = common_bases[:max_sequences]
        
        print(f"\n🚀 Found {len(common_bases)} matched flare/light-source sequences. Composing...")
        
        bg_light_files_out = []
        full_scene_files_out = []
        start_time = time.time()
        
        for i, base_name in enumerate(common_bases):
            print(f"\n--- Composing sequence {i+1}/{len(common_bases)} ({base_name}) ---")
            
            flare_filename = flare_bases[base_name]
            light_source_filename = light_source_bases[base_name]
            
            flare_path = flare_files[flare_filename]
            light_path = light_source_files[light_source_filename]
            
            try:
                # 调用更新后的 single sequence 方法
                bg_light_file, full_scene_file = self.compose_single_sequence(flare_path, light_path, i)
                bg_light_files_out.append(bg_light_file)
                full_scene_files_out.append(full_scene_file)
            except Exception as e:
                print(f"❌ Error composing sequence for {base_name}: {e}")
                continue
        
        total_time = time.time() - start_time
        success_rate = len(bg_light_files_out) / len(common_bases) * 100
        
        print(f"\n✅ Event composition complete:")
        print(f"  Processed: {len(bg_light_files_out)}/{len(common_bases)} sequences ({success_rate:.1f}%)")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average: {total_time/len(common_bases):.1f}s per sequence")
        # 输出所有方法的目录信息
        for method_name, paths in self.output_dirs.items():
            print(f"  {method_name} method outputs:")
            print(f"    - Stage 1 (bg+light): {paths['stage1']}")
            print(f"    - Stage 2 (full scene): {paths['stage2']}")
        
        return bg_light_files_out, full_scene_files_out


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