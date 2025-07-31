#!/usr/bin/env python3
"""
事件可视化工具模块

用于在debug模式下可视化和分析：
1. DSEC原始背景事件
2. DVS仿真炫光事件
3. 合并后的完整事件序列
4. 事件密度对比分析
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Dict, Optional, List
import cv2


class EventVisualizer:
    """事件可视化和分析工具类."""
    
    def __init__(self, output_dir: str, resolution: Tuple[int, int] = (640, 480)):
        """初始化可视化器.
        
        Args:
            output_dir: 输出目录
            resolution: 图像分辨率 (width, height)
        """
        self.output_dir = output_dir
        self.resolution = resolution  # (width, height)
        self.width, self.height = resolution
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 可视化配置
        self.colors = {
            'background_pos': (1, 0, 0, 0.6),    # 红色，半透明
            'background_neg': (0, 0, 1, 0.6),    # 蓝色，半透明
            'flare_pos': (1, 1, 0, 0.8),         # 黄色，更不透明
            'flare_neg': (1, 0.5, 0, 0.8),       # 橙色，更不透明
        }
    
    def analyze_and_visualize_pipeline(self, 
                                     background_events: np.ndarray,
                                     flare_events: np.ndarray, 
                                     merged_events: np.ndarray,
                                     labels: np.ndarray,
                                     sample_idx: int) -> Dict[str, float]:
        """分析并可视化完整的事件处理管线.
        
        Args:
            background_events: DSEC背景事件 [N1, 4]
            flare_events: DVS炫光事件 [N2, 4]  
            merged_events: 合并后事件 [N_total, 4]
            labels: 事件标签 [N_total] (0=背景, 1=炫光)
            sample_idx: 样本索引
            
        Returns:
            Dict包含各种统计信息
        """
        print(f"\n📊 Analyzing Event Pipeline for Sample {sample_idx}")
        print("=" * 60)
        
        # 1. 分析事件密度
        density_stats = self._analyze_event_densities(
            background_events, flare_events, merged_events, labels
        )
        
        # 2. 创建空间分布可视化
        self._visualize_spatial_distribution(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        # 3. 创建时间分布可视化
        self._visualize_temporal_distribution(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        # 4. 创建多分辨率事件可视化 (类似DVS的时间窗口分析)
        self._create_multi_resolution_event_visualizations(
            background_events, flare_events, merged_events, labels, sample_idx
        )
        
        # 5. 创建统计报告
        self._generate_statistics_report(
            background_events, flare_events, merged_events, labels, 
            density_stats, sample_idx
        )
        
        return density_stats
    
    def _analyze_event_densities(self, background_events: np.ndarray,
                                flare_events: np.ndarray,
                                merged_events: np.ndarray,
                                labels: np.ndarray) -> Dict[str, float]:
        """分析事件密度."""
        
        def calculate_density(events: np.ndarray) -> Tuple[float, float]:
            """计算事件密度 (events/ms)."""
            if len(events) == 0:
                return 0.0, 0.0
            
            time_span_us = events[:, 2].max() - events[:, 2].min()
            time_span_ms = time_span_us / 1000.0
            
            if time_span_ms <= 0:
                return 0.0, time_span_ms
                
            density = len(events) / time_span_ms
            return density, time_span_ms
        
        # 计算各类事件密度
        bg_density, bg_duration = calculate_density(background_events)
        flare_density, flare_duration = calculate_density(flare_events)
        merged_density, merged_duration = calculate_density(merged_events)
        
        # 从合并事件中分离背景和炫光（基于标签）
        if len(labels) > 0:
            bg_mask = labels == 0
            flare_mask = labels == 1
            
            merged_bg_events = merged_events[bg_mask] if bg_mask.any() else np.empty((0, 4))
            merged_flare_events = merged_events[flare_mask] if flare_mask.any() else np.empty((0, 4))
            
            merged_bg_density, _ = calculate_density(merged_bg_events)
            merged_flare_density, _ = calculate_density(merged_flare_events)
        else:
            merged_bg_density = 0.0
            merged_flare_density = 0.0
        
        stats = {
            'background_events': len(background_events),
            'flare_events': len(flare_events),
            'merged_events': len(merged_events),
            'background_density': bg_density,
            'flare_density': flare_density,
            'merged_density': merged_density,
            'merged_background_density': merged_bg_density,
            'merged_flare_density': merged_flare_density,
            'background_duration_ms': bg_duration,
            'flare_duration_ms': flare_duration,
            'merged_duration_ms': merged_duration,
        }
        
        return stats
    
    def _visualize_spatial_distribution(self, background_events: np.ndarray,
                                       flare_events: np.ndarray,
                                       merged_events: np.ndarray,
                                       labels: np.ndarray,
                                       sample_idx: int):
        """可视化空间分布."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Event Spatial Distribution - Sample {sample_idx}', fontsize=16)
        
        # 1. DSEC背景事件
        ax = axes[0, 0]
        if len(background_events) > 0:
            pos_mask = background_events[:, 3] == 1
            neg_mask = background_events[:, 3] == 0
            
            if pos_mask.any():
                ax.scatter(background_events[pos_mask, 0], background_events[pos_mask, 1], 
                          c='red', s=0.1, alpha=0.6, label=f'ON ({pos_mask.sum()})')
            if neg_mask.any():
                ax.scatter(background_events[neg_mask, 0], background_events[neg_mask, 1],
                          c='blue', s=0.1, alpha=0.6, label=f'OFF ({neg_mask.sum()})')
        
        ax.set_title(f'DSEC Background Events ({len(background_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. DVS炫光事件
        ax = axes[0, 1]
        if len(flare_events) > 0:
            pos_mask = flare_events[:, 3] == 1
            neg_mask = flare_events[:, 3] == 0
            
            if pos_mask.any():
                ax.scatter(flare_events[pos_mask, 0], flare_events[pos_mask, 1],
                          c='yellow', s=0.5, alpha=0.8, label=f'ON ({pos_mask.sum()})')
            if neg_mask.any():
                ax.scatter(flare_events[neg_mask, 0], flare_events[neg_mask, 1],
                          c='orange', s=0.5, alpha=0.8, label=f'OFF ({neg_mask.sum()})')
        
        ax.set_title(f'DVS Flare Events ({len(flare_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 合并后事件（按原始类型着色）
        ax = axes[1, 0]
        if len(merged_events) > 0 and len(labels) > 0:
            bg_mask = labels == 0
            flare_mask = labels == 1
            
            # 背景事件
            if bg_mask.any():
                bg_events = merged_events[bg_mask]
                pos_bg = bg_events[bg_events[:, 3] == 1]
                neg_bg = bg_events[bg_events[:, 3] == 0]
                
                if len(pos_bg) > 0:
                    ax.scatter(pos_bg[:, 0], pos_bg[:, 1], c='red', s=0.1, alpha=0.6, 
                              label=f'BG ON ({len(pos_bg)})')
                if len(neg_bg) > 0:
                    ax.scatter(neg_bg[:, 0], neg_bg[:, 1], c='blue', s=0.1, alpha=0.6,
                              label=f'BG OFF ({len(neg_bg)})')
            
            # 炫光事件
            if flare_mask.any():
                flare_events_merged = merged_events[flare_mask]
                pos_flare = flare_events_merged[flare_events_merged[:, 3] == 1]
                neg_flare = flare_events_merged[flare_events_merged[:, 3] == 0]
                
                if len(pos_flare) > 0:
                    ax.scatter(pos_flare[:, 0], pos_flare[:, 1], c='yellow', s=0.5, alpha=0.8,
                              label=f'Flare ON ({len(pos_flare)})')
                if len(neg_flare) > 0:
                    ax.scatter(neg_flare[:, 0], neg_flare[:, 1], c='orange', s=0.5, alpha=0.8,
                              label=f'Flare OFF ({len(neg_flare)})')
        
        ax.set_title(f'Merged Events by Type ({len(merged_events)})')
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 热力图
        ax = axes[1, 1]
        if len(merged_events) > 0:
            # 创建二维直方图
            hist, xedges, yedges = np.histogram2d(
                merged_events[:, 0], merged_events[:, 1],
                bins=[64, 48], range=[[0, self.width], [0, self.height]]
            )
            
            im = ax.imshow(hist.T, origin='lower', aspect='auto', cmap='hot',
                          extent=[0, self.width, 0, self.height])
            ax.set_title('Event Density Heatmap')
            plt.colorbar(im, ax=ax, label='Event Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'spatial_distribution_sample_{sample_idx}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_temporal_distribution(self, background_events: np.ndarray,
                                        flare_events: np.ndarray,
                                        merged_events: np.ndarray,
                                        labels: np.ndarray,
                                        sample_idx: int):
        """可视化时间分布."""
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle(f'Event Temporal Distribution - Sample {sample_idx}', fontsize=16)
        
        # 1. 事件计数随时间变化
        ax = axes[0]
        
        # 计算时间窗口
        all_times = []
        if len(background_events) > 0:
            all_times.extend(background_events[:, 2])
        if len(flare_events) > 0:
            all_times.extend(flare_events[:, 2])
        if len(merged_events) > 0:
            all_times.extend(merged_events[:, 2])
            
        if all_times:
            t_min, t_max = min(all_times), max(all_times)
            time_bins = np.linspace(t_min, t_max, 100)
            
            # 背景事件直方图
            if len(background_events) > 0:
                ax.hist(background_events[:, 2], bins=time_bins, alpha=0.6, 
                       label=f'Background ({len(background_events)})', color='blue')
            
            # 炫光事件直方图  
            if len(flare_events) > 0:
                ax.hist(flare_events[:, 2], bins=time_bins, alpha=0.8,
                       label=f'Flare ({len(flare_events)})', color='orange')
            
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Event Count')
        ax.set_title('Event Count Distribution Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 事件间隔分析
        ax = axes[1]
        
        def plot_interval_histogram(events, label, color):
            if len(events) > 1:
                intervals = np.diff(np.sort(events[:, 2]))
                intervals_ms = intervals / 1000.0  # 转换为毫秒
                
                # 过滤异常值
                intervals_ms = intervals_ms[intervals_ms < np.percentile(intervals_ms, 95)]
                
                ax.hist(intervals_ms, bins=50, alpha=0.7, label=f'{label} (median: {np.median(intervals_ms):.2f}ms)',
                       color=color, density=True)
        
        plot_interval_histogram(background_events, 'Background', 'blue')
        plot_interval_histogram(flare_events, 'Flare', 'orange')
        
        ax.set_xlabel('Inter-event Interval (ms)')
        ax.set_ylabel('Density')  
        ax.set_title('Inter-event Interval Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 3. 累积事件计数
        ax = axes[2]
        
        if len(merged_events) > 0 and len(labels) > 0:
            # 按时间排序
            sorted_indices = np.argsort(merged_events[:, 2])
            sorted_times = merged_events[sorted_indices, 2]
            sorted_labels = labels[sorted_indices]
            
            # 累积计数
            bg_cumsum = np.cumsum(sorted_labels == 0)
            flare_cumsum = np.cumsum(sorted_labels == 1)
            total_cumsum = np.arange(1, len(sorted_times) + 1)
            
            ax.plot(sorted_times, bg_cumsum, label='Background Events', color='blue', linewidth=2)
            ax.plot(sorted_times, flare_cumsum, label='Flare Events', color='orange', linewidth=2)
            ax.plot(sorted_times, total_cumsum, label='Total Events', color='black', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Cumulative Event Count')
        ax.set_title('Cumulative Event Count Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'temporal_distribution_sample_{sample_idx}.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _generate_statistics_report(self, background_events: np.ndarray,
                                   flare_events: np.ndarray,
                                   merged_events: np.ndarray,
                                   labels: np.ndarray,
                                   density_stats: Dict[str, float],
                                   sample_idx: int):
        """生成统计报告."""
        
        report_path = os.path.join(self.output_dir, f'statistics_report_sample_{sample_idx}.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Event Pipeline Analysis Report - Sample {sample_idx}\n")
            f.write("=" * 60 + "\n\n")
            
            # 事件计数
            f.write("Event Counts:\n")
            f.write(f"  DSEC Background Events: {len(background_events):,}\n")
            f.write(f"  DVS Flare Events: {len(flare_events):,}\n")
            f.write(f"  Merged Total Events: {len(merged_events):,}\n")
            
            if len(labels) > 0:
                bg_count = (labels == 0).sum()
                flare_count = (labels == 1).sum()
                f.write(f"  Merged Background Events: {bg_count:,}\n")
                f.write(f"  Merged Flare Events: {flare_count:,}\n")
            
            f.write("\n")
            
            # 事件密度
            f.write("Event Densities (events/ms):\n")
            f.write(f"  DSEC Background: {density_stats['background_density']:.1f}\n")
            f.write(f"  DVS Flare: {density_stats['flare_density']:.1f}\n")
            f.write(f"  Merged Total: {density_stats['merged_density']:.1f}\n")
            f.write(f"  Merged Background: {density_stats['merged_background_density']:.1f}\n")
            f.write(f"  Merged Flare: {density_stats['merged_flare_density']:.1f}\n")
            f.write("\n")
            
            # 时长信息
            f.write("Time Durations (ms):\n")
            f.write(f"  Background Duration: {density_stats['background_duration_ms']:.1f}\n")
            f.write(f"  Flare Duration: {density_stats['flare_duration_ms']:.1f}\n")
            f.write(f"  Merged Duration: {density_stats['merged_duration_ms']:.1f}\n")
            f.write("\n")
            
            # 极性分布
            f.write("Polarity Distribution:\n")
            
            def analyze_polarity(events, name):
                if len(events) > 0:
                    pos_count = (events[:, 3] == 1).sum()
                    neg_count = (events[:, 3] == 0).sum()
                    pos_ratio = pos_count / len(events) * 100
                    f.write(f"  {name}: ON={pos_count:,} ({pos_ratio:.1f}%), OFF={neg_count:,} ({100-pos_ratio:.1f}%)\n")
                else:
                    f.write(f"  {name}: No events\n")
            
            analyze_polarity(background_events, "DSEC Background")
            analyze_polarity(flare_events, "DVS Flare")
            analyze_polarity(merged_events, "Merged Total")
            
            f.write("\n")
            
            # 空间分布
            f.write("Spatial Distribution:\n")
            
            def analyze_spatial(events, name):
                if len(events) > 0:
                    x_range = f"{events[:, 0].min():.0f}-{events[:, 0].max():.0f}"
                    y_range = f"{events[:, 1].min():.0f}-{events[:, 1].max():.0f}"
                    x_center = events[:, 0].mean()
                    y_center = events[:, 1].mean()
                    f.write(f"  {name}: X=[{x_range}] Y=[{y_range}] Center=({x_center:.1f},{y_center:.1f})\n")
                else:
                    f.write(f"  {name}: No events\n")
            
            analyze_spatial(background_events, "DSEC Background")
            analyze_spatial(flare_events, "DVS Flare")
            analyze_spatial(merged_events, "Merged Total")
            
        print(f"📄 Statistics report saved: {report_path}")
        
        # 打印关键统计到控制台
        print(f"📊 Key Statistics for Sample {sample_idx}:")
        print(f"   DSEC Background: {len(background_events):,} events, {density_stats['background_density']:.1f} events/ms")
        print(f"   DVS Flare: {len(flare_events):,} events, {density_stats['flare_density']:.1f} events/ms")  
        print(f"   Merged Total: {len(merged_events):,} events, {density_stats['merged_density']:.1f} events/ms")
        
        # 评估密度合理性（仅检查过低，不限制上限）
        target_min = 500  # 最低合理密度
        if density_stats['merged_density'] < target_min:
            print(f"   ⚠️  Merged density too low: {density_stats['merged_density']:.1f} < {target_min}")
        else:
            print(f"   ✅ Merged density acceptable: {density_stats['merged_density']:.1f} events/ms")
    
    def _create_multi_resolution_event_visualizations(self, 
                                                    background_events: np.ndarray,
                                                    flare_events: np.ndarray, 
                                                    merged_events: np.ndarray,
                                                    labels: np.ndarray,
                                                    sample_idx: int):
        """创建多分辨率事件可视化 (类似DVS的多时间窗口分析).
        
        为DSEC背景事件和合并事件创建与DVS炫光事件类似的多分辨率时间窗口可视化，
        便于分析事件分布和算法正确性。
        """
        print(f"🔍 Creating multi-resolution event visualizations for sample {sample_idx}...")
        
        # 创建输出目录
        sample_viz_dir = os.path.join(self.output_dir, f"multi_resolution_sample_{sample_idx}")
        os.makedirs(sample_viz_dir, exist_ok=True)
        
        # 多分辨率策略 (与DVS一致)
        resolution_strategies = [0.5, 1, 2, 4]  # 时间窗口缩放因子
        
        # 为每种事件类型创建可视化
        event_types = {
            'dsec_background': background_events,
            'merged_total': merged_events
        }
        
        # 如果有炫光事件，也包含进来作为对比
        if len(flare_events) > 0:
            event_types['dvs_flare'] = flare_events
        
        for event_type, events in event_types.items():
            if len(events) == 0:
                print(f"   Skipping {event_type}: no events")
                continue
                
            # 为每个事件类型创建子目录
            type_dir = os.path.join(sample_viz_dir, event_type)
            os.makedirs(type_dir, exist_ok=True)
            
            print(f"   Processing {event_type}: {len(events)} events")
            
            # 计算时间范围
            if len(events) > 0:
                t_min = events[:, 2].min()
                t_max = events[:, 2].max()
                total_duration_us = t_max - t_min
                
                if total_duration_us <= 1000:  # < 1ms，创建基于事件数量的图像序列
                    print(f"   {event_type} has short duration ({total_duration_us:.0f}μs), creating event-count based image sequence")
                    self._create_event_sequence_visualization(events, type_dir, event_type)
                    continue
                
                # 为每个分辨率策略创建可视化
                for resolution_factor in resolution_strategies:
                    resolution_dir = os.path.join(type_dir, f"{resolution_factor}x_resolution")
                    os.makedirs(resolution_dir, exist_ok=True)
                    
                    # 计算时间窗口大小
                    window_duration_us = total_duration_us / resolution_factor
                    num_windows = int(np.ceil(total_duration_us / window_duration_us))
                    
                    print(f"     {resolution_factor}x: {window_duration_us:.0f}μs windows, {num_windows} total")
                    
                    # 为每个时间窗口创建可视化
                    for window_idx in range(num_windows):
                        window_start = t_min + window_idx * window_duration_us
                        window_end = min(window_start + window_duration_us, t_max)
                        
                        # 筛选此时间窗口内的事件
                        mask = (events[:, 2] >= window_start) & (events[:, 2] < window_end)
                        window_events = events[mask]
                        
                        if len(window_events) == 0:
                            continue
                            
                        # 创建DVS风格的事件图像（黑色背景）
                        self._create_dvs_style_event_image(
                            window_events, resolution_dir, 
                            f"window_{window_idx:03d}_{resolution_factor}x", 
                            event_type, window_start, window_end
                        )
                
                print(f"   ✅ Created visualizations for {event_type}")
        
        print(f"✅ Multi-resolution visualizations saved to: {sample_viz_dir}")
    
    def _create_single_event_visualization(self, events: np.ndarray, output_dir: str, 
                                         filename: str, event_type: str,
                                         t_start: float, t_end: float):
        """创建单个时间窗口的事件可视化."""
        import matplotlib.pyplot as plt
        
        # 创建图像
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        if len(events) == 0:
            ax.text(0.5, 0.5, 'No events in this window', 
                   ha='center', va='center', transform=ax.transAxes)
        else:
            # 提取坐标和极性
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            p = events[:, 3]
            
            # 根据极性分色 (支持1/-1和1/0两种格式)
            # 检测极性格式
            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                # DSEC格式: 1/-1
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                # DVS格式: 1/0
                pos_mask = p > 0
                neg_mask = p == 0
            
            # 绘制正极性事件 (红色)
            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c='red', s=1, alpha=0.7, label=f'ON ({np.sum(pos_mask)})')
            
            # 绘制负极性事件 (蓝色)  
            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c='blue', s=1, alpha=0.7, label=f'OFF ({np.sum(neg_mask)})')
        
        # 设置图像属性
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title(f'{event_type.replace("_", " ").title()}\n'
                    f'Time: {(t_start/1000):.1f}-{(t_end/1000):.1f}ms ({len(events)} events)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_event_sequence_visualization(self, events: np.ndarray, output_dir: str, event_type: str):
        """为短时长事件创建基于事件数量分割的图像序列（类似DVS风格）."""
        import matplotlib.pyplot as plt
        
        if len(events) == 0:
            return
            
        # 创建图像序列目录
        sequence_dir = os.path.join(output_dir, "event_sequence")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # 根据事件数量分割（每张图显示8-16个事件）
        events_per_frame = max(8, len(events) // 10)  # 至少8个事件，最多10帧
        num_frames = max(1, len(events) // events_per_frame)
        
        print(f"     Creating {num_frames} frames with ~{events_per_frame} events each")
        
        for frame_idx in range(num_frames):
            start_idx = frame_idx * events_per_frame
            end_idx = min(start_idx + events_per_frame, len(events))
            frame_events = events[start_idx:end_idx]
            
            if len(frame_events) == 0:
                continue
                
            # 创建DVS风格的黑色背景图像
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
            ax.set_facecolor('black')
            
            # 提取坐标和极性
            x = frame_events[:, 0]
            y = frame_events[:, 1]
            p = frame_events[:, 3]
            
            # 根据极性分色 - 使用亮色在黑背景上显示
            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                # DSEC格式: 1/-1
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                # DVS格式: 1/0
                pos_mask = p > 0
                neg_mask = p == 0
            
            # 绘制正极性事件 (亮红色)
            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c='red', s=8, alpha=0.9, 
                          label=f'ON ({np.sum(pos_mask)})', edgecolors='white', linewidth=0.2)
            
            # 绘制负极性事件 (亮蓝色)  
            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c='cyan', s=8, alpha=0.9, 
                          label=f'OFF ({np.sum(neg_mask)})', edgecolors='white', linewidth=0.2)
            
            # 设置DVS风格的图像属性
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            ax.invert_yaxis()  # 翻转Y轴匹配图像坐标
            ax.set_xlabel('X coordinate', color='white')
            ax.set_ylabel('Y coordinate', color='white')
            ax.set_title(f'{event_type.replace("_", " ").title()} - Frame {frame_idx+1}/{num_frames}\\n'
                        f'Events: {len(frame_events)} ({start_idx}-{end_idx})', color='white')
            ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.2, color='gray')
            ax.tick_params(colors='white')
            
            # 保存图像
            output_path = os.path.join(sequence_dir, f"frame_{frame_idx:03d}.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close()
            
        print(f"     ✅ Saved {num_frames} event sequence frames to {sequence_dir}")
    
    def _create_dvs_style_event_image(self, events: np.ndarray, output_dir: str, 
                                     filename: str, event_type: str,
                                     t_start: float, t_end: float):
        """创建DVS风格的单个事件图像（黑色背景，亮色事件点）."""
        import matplotlib.pyplot as plt
        
        # 创建DVS风格的黑色背景图像
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), facecolor='black')
        ax.set_facecolor('black')
        
        if len(events) == 0:
            ax.text(0.5, 0.5, 'No events in this window', 
                   ha='center', va='center', transform=ax.transAxes, color='white')
        else:
            # 提取坐标和极性
            x = events[:, 0]
            y = events[:, 1]
            t = events[:, 2]
            p = events[:, 3]
            
            # 根据极性分色 - 使用亮色在黑背景上显示
            unique_polarities = np.unique(p)
            if np.any(unique_polarities < 0):
                # DSEC格式: 1/-1
                pos_mask = p > 0
                neg_mask = p < 0
            else:
                # DVS格式: 1/0
                pos_mask = p > 0
                neg_mask = p == 0
            
            # 绘制正极性事件 (亮红色)
            if np.any(pos_mask):
                ax.scatter(x[pos_mask], y[pos_mask], c='red', s=6, alpha=0.8, 
                          label=f'ON ({np.sum(pos_mask)})', edgecolors='white', linewidth=0.1)
            
            # 绘制负极性事件 (亮蓝色)  
            if np.any(neg_mask):
                ax.scatter(x[neg_mask], y[neg_mask], c='cyan', s=6, alpha=0.8, 
                          label=f'OFF ({np.sum(neg_mask)})', edgecolors='white', linewidth=0.1)
        
        # 设置DVS风格的图像属性
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.invert_yaxis()  # 翻转Y轴匹配图像坐标
        ax.set_xlabel('X coordinate', color='white')
        ax.set_ylabel('Y coordinate', color='white')
        ax.set_title(f'{event_type.replace("_", " ").title()}\\n'
                    f'Time: {(t_start/1000):.1f}-{(t_end/1000):.1f}ms ({len(events)} events)', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.tick_params(colors='white')
        
        # 保存图像
        output_path = os.path.join(output_dir, f"{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return output_path