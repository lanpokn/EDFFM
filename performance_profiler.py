#!/usr/bin/env python3
"""
Performance Profiler for EventMamba-FX Training Pipeline
Measures time spent in different components: I/O, simulation, training, etc.
"""

import time
import yaml
import torch
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any
import psutil
import os

# Import project modules
from src.mixed_flare_dataloaders import create_mixed_flare_dataloaders
from src.model import EventDenoisingMamba
from src.trainer import Trainer

@contextmanager
def timer(name: str, profiler: Dict[str, float]):
    """Context manager to time code blocks."""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        profiler[name] = profiler.get(name, 0.0) + elapsed

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def profile_single_batch(config: Dict[str, Any]) -> Dict[str, float]:
    """Profile a single batch through the entire pipeline."""
    profiler = {}
    memory_tracker = {}
    
    print("=== Starting Performance Profiling ===")
    memory_tracker['start'] = get_memory_usage()
    
    # 1. Data Loading Setup
    with timer("dataloader_creation", profiler):
        train_loader, val_loader, test_loader = create_mixed_flare_dataloaders(config)
    
    memory_tracker['after_dataloader_creation'] = get_memory_usage()
    
    # 2. Model Creation
    with timer("model_creation", profiler):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EventDenoisingMamba(config).to(device)
    
    memory_tracker['after_model_creation'] = get_memory_usage()
    
    # 3. Single Batch Data Loading (I/O + Simulation)
    batch_times = []
    io_simulation_times = []
    
    print("\n=== Profiling 2 batches for average timing ===")
    for batch_idx, (features, labels) in enumerate(train_loader):
        if batch_idx >= 2:  # Only profile 2 batches
            break
            
        batch_start = time.time()
        
        # Move to device (I/O overhead)
        with timer(f"batch_{batch_idx}_device_transfer", profiler):
            features = features.to(device)
            labels = labels.to(device)
        
        batch_total = time.time() - batch_start
        batch_times.append(batch_total)
        io_simulation_times.append(batch_total)
        
        # 4. Model Forward Pass
        with timer(f"batch_{batch_idx}_forward", profiler):
            with torch.no_grad():
                outputs = model(features)
        
        # 5. Loss Calculation
        with timer(f"batch_{batch_idx}_loss", profiler):
            criterion = torch.nn.BCELoss()
            labels_float = labels.float().unsqueeze(-1)  # Convert to float and add dimension
            loss = criterion(outputs, labels_float)
        
        memory_tracker[f'after_batch_{batch_idx}'] = get_memory_usage()
        
        print(f"  Batch {batch_idx+1}: {batch_total:.2f}s (I/O+Sim), Forward: {profiler[f'batch_{batch_idx}_forward']:.3f}s, Loss: {profiler[f'batch_{batch_idx}_loss']:.3f}s")
    
    # Calculate averages
    avg_batch_time = np.mean(batch_times)
    avg_forward_time = np.mean([profiler[f'batch_{idx}_forward'] for idx in range(2)])
    avg_loss_time = np.mean([profiler[f'batch_{idx}_loss'] for idx in range(2)])
    avg_device_transfer = np.mean([profiler[f'batch_{idx}_device_transfer'] for idx in range(2)])
    
    # Summary statistics
    summary = {
        'dataloader_creation_time': profiler['dataloader_creation'],
        'model_creation_time': profiler['model_creation'], 
        'avg_batch_io_simulation_time': avg_batch_time,
        'avg_forward_pass_time': avg_forward_time,
        'avg_loss_calculation_time': avg_loss_time,
        'avg_device_transfer_time': avg_device_transfer,
        'memory_usage': memory_tracker
    }
    
    return summary

def analyze_component_breakdown(summary: Dict[str, Any]) -> Dict[str, float]:
    """Analyze the percentage breakdown of different components."""
    
    # Core components timing
    io_sim_time = summary['avg_batch_io_simulation_time']
    forward_time = summary['avg_forward_pass_time'] 
    loss_time = summary['avg_loss_calculation_time']
    device_time = summary['avg_device_transfer_time']
    
    # Estimate actual I/O+Simulation time (subtract GPU operations)
    pure_io_sim_time = io_sim_time - device_time
    
    # Total time per training step
    total_step_time = pure_io_sim_time + forward_time + loss_time + device_time
    
    # Calculate percentages
    breakdown = {
        'I/O + Data Generation (%)': (pure_io_sim_time / total_step_time) * 100,
        'Model Forward Pass (%)': (forward_time / total_step_time) * 100,
        'Loss Calculation (%)': (loss_time / total_step_time) * 100, 
        'Device Transfer (%)': (device_time / total_step_time) * 100,
    }
    
    # Add absolute times for reference
    breakdown.update({
        'I/O + Data Generation (sec)': pure_io_sim_time,
        'Model Forward Pass (sec)': forward_time,
        'Loss Calculation (sec)': loss_time,
        'Device Transfer (sec)': device_time,
        'Total per Step (sec)': total_step_time
    })
    
    return breakdown

def main():
    """Main profiling function."""
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Force small sample size for profiling
    config['data']['max_samples_debug'] = 4  # Just enough for 2 batches with batch_size=2
    
    print("EventMamba-FX Performance Profiler")
    print("==================================")
    print(f"Configuration: {config['data']['max_samples_debug']} samples, batch_size={config['training']['batch_size']}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Initial Memory: {get_memory_usage():.1f} MB")
    
    try:
        # Profile the pipeline
        summary = profile_single_batch(config)
        
        # Analyze breakdown
        breakdown = analyze_component_breakdown(summary)
        
        # Display results
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS RESULTS")
        print("="*50)
        
        print(f"\n📊 Component Time Breakdown:")
        for component, value in breakdown.items():
            if '(%)' in component:
                print(f"  {component:<30}: {value:6.1f}%")
            elif '(sec)' in component:
                print(f"  {component:<30}: {value:6.3f}s")
        
        print(f"\n💾 Memory Usage Progression:")
        for stage, memory in summary['memory_usage'].items():
            print(f"  {stage:<30}: {memory:6.1f} MB")
        
        # Identify bottlenecks
        max_component = max([(k, v) for k, v in breakdown.items() if '(%)' in k], key=lambda x: x[1])
        print(f"\n🔍 Primary Bottleneck: {max_component[0]} ({max_component[1]:.1f}%)")
        
        # Recommendations
        print(f"\n💡 Optimization Recommendations:")
        if breakdown['I/O + Data Generation (%)'] > 50:
            print("  • Data I/O and simulation dominate - consider data caching or pre-generation")
        if breakdown['Model Forward Pass (%)'] > 30:
            print("  • Model computation is significant - consider model optimization")
        if breakdown['Device Transfer (%)'] > 10:
            print("  • Device transfer overhead is notable - consider data batching optimization")
        
        # Estimate full training time
        total_samples = 364  # From earlier logs
        steps_per_epoch = total_samples // config['training']['batch_size']
        time_per_epoch = steps_per_epoch * breakdown['Total per Step (sec)']
        
        print(f"\n⏱️  Full Training Estimates:")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Time per epoch: {time_per_epoch/60:.1f} minutes")
        print(f"  Time for 1 epoch: {time_per_epoch/3600:.2f} hours")
        
    except Exception as e:
        print(f"\n❌ Profiling failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()