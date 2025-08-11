#!/usr/bin/env python3
"""
创建小规模测试推理数据
直接使用DSEC背景数据加手工合成简单炫光事件
"""
import os
import h5py
import hdf5plugin
import numpy as np
import random

def create_simple_flare_test_data():
    """创建简单的包含炫光事件的测试数据"""
    
    print("🔧 创建简单炫光测试数据...")
    
    # 1. 从DSEC背景数据中读取一小段
    dsec_file = 'data/bg_events/zurich_city_00_a.h5'
    print(f"📥 从DSEC背景数据读取: {dsec_file}")
    
    with h5py.File(dsec_file, 'r') as f:
        # 读取前50,000个事件作为背景
        n_bg_events = 50000
        bg_x = f['events/x'][:n_bg_events]
        bg_y = f['events/y'][:n_bg_events]
        bg_t = f['events/t'][:n_bg_events]
        bg_p = f['events/p'][:n_bg_events]
        
        print(f"  - 背景事件: {n_bg_events:,}")
        print(f"  - 时间跨度: {(bg_t[-1] - bg_t[0])/1000:.1f}ms")
    
    # 2. 时间戳归一化
    bg_t_norm = bg_t - bg_t[0]
    
    # 3. 生成简单的炫光事件（集中在某个区域的高密度事件）
    print("✨ 生成人工炫光事件...")
    
    n_flare_events = 5000  # 10% 炫光事件
    
    # 炫光区域：图像中心区域的高密度事件
    center_x, center_y = 320, 240
    flare_radius = 50
    
    flare_events = []
    time_span = bg_t_norm[-1] - bg_t_norm[0]
    
    for i in range(n_flare_events):
        # 随机位置在炫光区域内
        angle = random.uniform(0, 2*np.pi)
        radius = random.uniform(0, flare_radius)
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # 随机时间
        t = random.uniform(bg_t_norm[0], bg_t_norm[-1])
        
        # 随机极性
        p = random.choice([1, -1])
        
        flare_events.append([x, y, t, p])
    
    flare_events = np.array(flare_events)
    print(f"  - 炫光事件: {len(flare_events):,}")
    print(f"  - 炫光区域: 中心({center_x}, {center_y}), 半径{flare_radius}px")
    
    # 4. 合并背景和炫光事件
    bg_events = np.column_stack([bg_x, bg_y, bg_t_norm, np.where(bg_p > 0, 1, -1)])
    all_events = np.vstack([bg_events, flare_events])
    
    # 按时间排序
    sort_indices = np.argsort(all_events[:, 2])
    all_events = all_events[sort_indices]
    
    # 创建标签：前n_bg_events个是背景(0)，后面是炫光(1)
    labels = np.zeros(len(all_events), dtype=np.bool_)
    labels[n_bg_events:] = 1  # 炫光事件标记为1
    labels = labels[sort_indices]  # 按时间排序后的标签
    
    print(f"✅ 合并完成:")
    print(f"  - 总事件数: {len(all_events):,}")
    print(f"  - 背景事件: {np.sum(~labels):,} ({np.mean(~labels)*100:.1f}%)")
    print(f"  - 炫光事件: {np.sum(labels):,} ({np.mean(labels)*100:.1f}%)")
    
    # 5. 保存为DSEC格式
    output_path = 'data/inference/test_synthetic_flare.h5'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 创建events组
        events_group = f.create_group('events')
        
        # 按DSEC格式保存
        events_group.create_dataset('x', data=all_events[:, 0].astype(np.uint16), 
                                   compression=hdf5plugin.FILTERS['blosc'])
        events_group.create_dataset('y', data=all_events[:, 1].astype(np.uint16),
                                   compression=hdf5plugin.FILTERS['blosc']) 
        events_group.create_dataset('t', data=all_events[:, 2].astype(np.int64),
                                   compression=hdf5plugin.FILTERS['blosc'])
        events_group.create_dataset('p', data=np.where(all_events[:, 3] > 0, True, False),
                                   compression=hdf5plugin.FILTERS['blosc'])
        
        # 保存ground truth标签供对比
        f.create_dataset('ground_truth_labels', data=labels)
    
    print(f"💾 测试数据已保存: {output_path}")
    print(f"  - 时间跨度: {all_events[-1, 2]/1000:.1f}ms")
    print(f"  - 包含ground truth标签用于验证")
    
    return output_path, labels

if __name__ == '__main__':
    create_simple_flare_test_data()