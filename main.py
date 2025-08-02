import argparse
import yaml
import torch
import os
import numpy as np
import time

from src.epoch_iteration_dataset import create_epoch_iteration_dataloaders
from src.model import EventDenoisingMamba # 确认导入的是修正后的模型
from src.trainer import Trainer
from src.evaluate import Evaluator

def analyze_events_detailed(events: np.ndarray, name: str) -> dict:
    """详细分析事件的范围和统计信息"""
    if len(events) == 0:
        print(f"  📊 {name}: EMPTY (0 events)")
        return {'count': 0, 'x_range': None, 'y_range': None, 't_range': None}
    
    x_min, x_max = events[:, 0].min(), events[:, 0].max()
    y_min, y_max = events[:, 1].min(), events[:, 1].max() 
    t_min, t_max = events[:, 2].min(), events[:, 2].max()
    duration_ms = (t_max - t_min) / 1000.0
    
    pos_events = np.sum(events[:, 3] > 0)
    neg_events = np.sum(events[:, 3] < 0)
    
    print(f"  📊 {name}: {len(events)} events")
    print(f"    - X range: [{x_min:.1f}, {x_max:.1f}] (span: {x_max-x_min:.1f})")
    print(f"    - Y range: [{y_min:.1f}, {y_max:.1f}] (span: {y_max-y_min:.1f})")
    print(f"    - T range: [{t_min:.0f}, {t_max:.0f}] μs (duration: {duration_ms:.1f}ms)")
    print(f"    - Polarity: {pos_events} positive, {neg_events} negative")
    
    return {
        'count': len(events), 'x_range': (x_min, x_max), 'y_range': (y_min, y_max),
        't_range': (t_min, t_max), 'duration_ms': duration_ms,
        'pos_events': pos_events, 'neg_events': neg_events
    }

def analyze_epoch_iteration_details(train_loader):
    """分析Epoch-Iteration架构的详细信息"""
    print(f"\n🔍 详细分析Epoch-Iteration数据流")
    print("=" * 60)
    
    # 获取数据集实例
    if hasattr(train_loader, 'dataset'):
        dataset = train_loader.dataset
        
        # 手动生成epoch数据来获取详细信息
        print(f"\n🔄 生成新的Epoch数据...")
        epoch_start = time.time()
        
        # 生成背景事件
        print(f"\n📖 Step 1: 生成背景事件...")
        background_events = dataset._generate_background_events()
        bg_stats = analyze_events_detailed(background_events, "Background Events")
        
        # 生成炫光事件
        print(f"\n✨ Step 2: 生成炫光事件...")
        try:
            flare_events_raw = dataset._generate_flare_events()
            if len(flare_events_raw) > 0:
                flare_events = dataset._format_flare_events(flare_events_raw)
            else:
                flare_events = np.empty((0, 4))
            flare_stats = analyze_events_detailed(flare_events, "Flare Events")
        except Exception as e:
            print(f"    ❌ 炫光生成失败: {e}")
            flare_events = np.empty((0, 4))
            flare_stats = {'count': 0}
        
        # 合并事件
        print(f"\n🔗 Step 3: 合并事件...")
        merged_events, merged_labels = dataset._merge_and_sort_events(background_events, flare_events)
        merged_stats = analyze_events_detailed(merged_events, "Merged Events")
        
        # 特征提取
        print(f"\n🧠 Step 4: 特征提取...")
        feature_start = time.time()
        if len(merged_events) > 0:
            long_features = dataset.feature_extractor.process_sequence(merged_events)
            dataset.long_feature_sequence = long_features
            dataset.long_labels = merged_labels
            dataset.num_iterations = max(1, len(long_features) - dataset.sequence_length + 1)
        else:
            dataset.long_feature_sequence = np.zeros((1, 11), dtype=np.float32)
            dataset.long_labels = np.zeros(1, dtype=np.int64)
            dataset.num_iterations = 1
        
        feature_time = time.time() - feature_start
        epoch_time = time.time() - epoch_start
        
        print(f"  ✅ 特征提取完成: {feature_time:.3f}s")
        print(f"  📊 特征序列形状: {dataset.long_feature_sequence.shape}")
        print(f"  📊 可用迭代数: {dataset.num_iterations}")
        print(f"  📊 总Epoch时间: {epoch_time:.3f}s")
        
        # 分析iterations
        print(f"\n🎯 Step 5: 分析Iteration连续性...")
        analyze_iterations(dataset, num_iterations=min(10, dataset.num_iterations))
        
        return True
    else:
        print("  ❌ 无法获取数据集实例进行详细分析")
        return False

def analyze_iterations(dataset, num_iterations=10):
    """分析iteration的连续性"""
    print(f"\n📊 分析前{num_iterations}个Iterations:")
    print("-" * 50)
    
    for i in range(num_iterations):
        try:
            features, labels = dataset[i]
            
            # 转换为numpy
            if isinstance(features, torch.Tensor):
                features_np = features.numpy()
                labels_np = labels.numpy()
            else:
                features_np = features
                labels_np = labels
            
            bg_count = np.sum(labels_np == 0)
            flare_count = np.sum(labels_np == 1)
            
            # 检查标签段数（连续性指标）
            label_changes = np.diff(labels_np.astype(int))
            num_segments = np.sum(np.abs(label_changes)) + 1
            
            print(f"  Iter {i:2d}: {len(features_np):3d} events "
                  f"(BG:{bg_count:3d}, FL:{flare_count:3d}) "
                  f"| {num_segments} segments")
            
            # 检查连续iteration之间的连续性
            if i > 0:
                prev_features, prev_labels = dataset[i-1]
                if isinstance(prev_features, torch.Tensor):
                    prev_features = prev_features.numpy()
                    prev_labels = prev_labels.numpy()
                
                # 检查滑动窗口重叠部分
                if len(prev_features) == dataset.sequence_length and len(features_np) == dataset.sequence_length:
                    overlap_prev = prev_features[1:]  # 前一个的后半部分
                    overlap_curr = features_np[:-1]   # 当前的前半部分
                    
                    if overlap_prev.shape == overlap_curr.shape:
                        max_diff = np.max(np.abs(overlap_prev - overlap_curr))
                        if max_diff < 1e-6:
                            continuity = "✅连续"
                        else:
                            continuity = f"❌不连续({max_diff:.6f})"
                        print(f"        连续性检查: {continuity}")
        
        except Exception as e:
            print(f"  Iter {i:2d}: ❌ 分析失败: {e}")
    
    print(f"\n⏱️ 滑动窗口连续性总结:")
    print(f"  - 每个iteration应该有{dataset.sequence_length}个事件")
    print(f"  - 相邻iteration应该有{dataset.sequence_length-1}个重叠事件")
    print(f"  - 标签段数越少表示事件越连续")

def main(config):
    """
    Main function to run the training and evaluation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Debug mode setup
    if config.get('debug_mode', False):
        output_dir = os.path.join("output", "debug_visualizations")
        os.makedirs(output_dir, exist_ok=True)
        config['debug_output_dir'] = output_dir
        print(f"🚨 DEBUG MODE: Saving visualizations to {output_dir}")
        print(f"🚨 DEBUG MODE: Will run limited iterations for debugging")

    # 1. 创建数据集加载器 (统一使用Epoch-Iteration架构)
    print("🔄 Using Epoch-Iteration architecture (先完整序列特征提取，再滑动窗口)")
    train_loader, val_loader, test_loader = create_epoch_iteration_dataloaders(config)
    
    # 🔍 详细分析Epoch-Iteration数据流
    print("\n" + "="*80)
    print("🔍 EPOCH-ITERATION 详细数据流分析")
    print("="*80)
    analyze_epoch_iteration_details(train_loader)
    print("="*80)

    # 2. 初始化模型 (关键修正)
    # 现在我们将特征提取器和Mamba模型的配置分开传递
    model = EventDenoisingMamba(config).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. 根据模式选择执行
    if config['run']['mode'] == 'analyze':
        print("\n🎯 Analysis完成! 退出程序 (不进行训练)")
        print("详细的Epoch-Iteration分析报告已显示在上方。")
        return
    elif config['run']['mode'] == 'train':
        trainer = Trainer(model, train_loader, val_loader, config, device)
        trainer.train()
    elif config['run']['mode'] == 'evaluate':
        evaluator = Evaluator(model, test_loader, config, device)
        model.load_state_dict(torch.load(config['evaluation']['checkpoint_path']))
        print(f"Loaded checkpoint from: {config['evaluation']['checkpoint_path']}")
        evaluator.evaluate()
    else:
        raise ValueError(f"Unknown mode: {config['run']['mode']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the EventMamba-FX model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to the YAML configuration file.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to save flare image sequences and event visualizations.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Enable debug mode if --debug flag is set
    if args.debug:
        config['debug_mode'] = True
        # Limit iterations for debug mode
        config['training']['max_epochs'] = 1
        config['training']['max_samples_debug'] = 8  # Only process a few samples
        # Debug event visualization parameters (multiple temporal resolutions)
        config['debug_event_subdivisions'] = [0.5, 1, 2, 4]  # Multiple subdivision strategies

    main(config)