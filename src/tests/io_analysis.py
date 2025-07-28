import torch
import numpy as np
import time
import yaml
import os
import sys

# 添加父目录到路径以便导入src模块
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.model import EventDenoisingMamba
from src.datasets import EventDenoisingDataset
from torch.utils.data import DataLoader

def analyze_io_vs_compute():
    """分析I/O开销 vs 计算开销"""
    print("🔍 I/O vs Compute Performance Analysis")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load config and model
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = EventDenoisingMamba(config).to(device)
    try:
        model.load_state_dict(torch.load(config['evaluation']['checkpoint_path'], map_location=device))
    except:
        pass
    model.eval()
    
    # Test 1: 测量文件读取时间
    print("\n📁 Test 1: File I/O Performance")
    base_path = os.path.join(os.path.dirname(__file__), '../..')
    test_file = os.path.join(base_path, "data/mixed_events/train_data.txt")
    
    start_time = time.time()
    with open(test_file, 'r') as f:
        lines = f.readlines()
    file_read_time = time.time() - start_time
    
    print(f"   File read time: {file_read_time:.4f}s")
    print(f"   Lines read: {len(lines):,}")
    print(f"   Read speed: {len(lines)/file_read_time:.0f} lines/sec")
    
    # Test 2: 测量数据解析时间
    print("\n🔄 Test 2: Data Parsing Performance")
    start_time = time.time()
    parsed_data = []
    for line in lines[:10000]:  # 只解析前1万行
        parts = line.strip().split()
        parsed_data.append([float(parts[0]), float(parts[1]), float(parts[2]), 
                           float(parts[3]), float(parts[4])])
    parsing_time = time.time() - start_time
    
    print(f"   Parsing time (10K events): {parsing_time:.4f}s")
    print(f"   Parsing speed: {10000/parsing_time:.0f} events/sec")
    
    # Test 3: 测量DataLoader开销
    print("\n📊 Test 3: DataLoader Performance")
    dataset = EventDenoisingDataset(test_file, config)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    start_time = time.time()
    batch_count = 0
    total_events_loaded = 0
    
    for events, labels in dataloader:
        batch_count += 1
        total_events_loaded += events.shape[0] * events.shape[1]
        if batch_count >= 100:  # 只测试前100个batch
            break
    
    dataloader_time = time.time() - start_time
    print(f"   DataLoader time (100 batches): {dataloader_time:.4f}s")
    print(f"   Events loaded: {total_events_loaded:,}")
    print(f"   DataLoader speed: {total_events_loaded/dataloader_time:.0f} events/sec")
    
    # Test 4: 纯推理性能（预加载数据）
    print("\n🚀 Test 4: Pure Inference Performance (Pre-loaded Data)")
    
    # 预加载数据到内存
    sample_events = []
    sample_labels = []
    
    dataloader_fresh = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    for i, (events, labels) in enumerate(dataloader_fresh):
        if i >= 50:  # 预加载50个batch
            break
        sample_events.append(events.to(device))
        sample_labels.append(labels.to(device))
    
    print(f"   Pre-loaded {len(sample_events)} batches to GPU memory")
    
    # 纯推理测试
    total_pure_events = 0
    start_time = time.time()
    
    with torch.no_grad():
        for events, labels in zip(sample_events, sample_labels):
            if device == "cuda":
                torch.cuda.synchronize()
            
            predictions = model(events)
            total_pure_events += events.shape[0] * events.shape[1]
            
            if device == "cuda":
                torch.cuda.synchronize()
    
    pure_inference_time = time.time() - start_time
    pure_throughput = total_pure_events / pure_inference_time
    
    print(f"   Pure inference time: {pure_inference_time:.4f}s")
    print(f"   Events processed: {total_pure_events:,}")
    print(f"   Pure inference speed: {pure_throughput:.0f} events/sec")
    
    # Test 5: 包含I/O的完整测试
    print("\n🔄 Test 5: Complete Pipeline (I/O + Compute)")
    
    dataloader_complete = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    
    total_complete_events = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, (events, labels) in enumerate(dataloader_complete):
            if i >= 50:  # 测试50个batch
                break
                
            events = events.to(device)
            labels = labels.to(device)
            
            predictions = model(events)
            total_complete_events += events.shape[0] * events.shape[1]
    
    complete_time = time.time() - start_time
    complete_throughput = total_complete_events / complete_time
    
    print(f"   Complete pipeline time: {complete_time:.4f}s")
    print(f"   Events processed: {total_complete_events:,}")
    print(f"   Complete pipeline speed: {complete_throughput:.0f} events/sec")
    
    # 分析结果
    print("\n" + "=" * 50)
    print("📈 PERFORMANCE BREAKDOWN ANALYSIS")
    print("=" * 50)
    
    io_overhead = (pure_throughput - complete_throughput) / pure_throughput * 100
    
    print(f"🏆 Performance Comparison:")
    print(f"   Pure Inference:     {pure_throughput:,.0f} events/sec")
    print(f"   Complete Pipeline:  {complete_throughput:,.0f} events/sec")
    print(f"   I/O Overhead:       {io_overhead:.1f}%")
    
    # 芯片级性能预估
    print(f"\n🔬 On-Chip Performance Estimation:")
    speedup_factor = pure_throughput / complete_throughput
    print(f"   Speedup without I/O: {speedup_factor:.2f}x")
    
    # 1万和100万事件的时间对比
    events_10k = 10000
    events_1m = 1000000
    
    print(f"\n⏱️  Time Comparison for Different Scales:")
    print(f"{'Scale':<15} {'With I/O':<15} {'Pure Compute':<15} {'Speedup':<10}")
    print("-" * 60)
    
    time_10k_io = events_10k / complete_throughput
    time_10k_pure = events_10k / pure_throughput
    time_1m_io = events_1m / complete_throughput
    time_1m_pure = events_1m / pure_throughput
    
    print(f"{'10K events':<15} {time_10k_io:.3f}s         {time_10k_pure:.3f}s         {time_10k_io/time_10k_pure:.2f}x")
    print(f"{'1M events':<15} {time_1m_io:.2f}s          {time_1m_pure:.2f}s          {time_1m_io/time_1m_pure:.2f}x")
    
    print(f"\n💡 Key Insights:")
    print(f"   • I/O overhead: {io_overhead:.1f}% of total time")
    print(f"   • On-chip processing would be {speedup_factor:.2f}x faster")
    print(f"   • 1M events on-chip: ~{time_1m_pure:.2f}s vs {time_1m_io:.2f}s with I/O")
    
    return {
        'pure_throughput': pure_throughput,
        'complete_throughput': complete_throughput,
        'io_overhead_percent': io_overhead,
        'speedup_factor': speedup_factor
    }

if __name__ == "__main__":
    results = analyze_io_vs_compute()