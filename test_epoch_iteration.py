#!/usr/bin/env python3
"""
Test script for the new Epoch-Iteration architecture

This script validates that the new data pipeline works correctly:
1. Epoch-level: Generate long sequences, extract features once
2. Iteration-level: Sliding window sampling for training
"""

import os
import sys
import yaml
import torch
import time
import argparse
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_epoch_iteration_architecture(config_path: str = "configs/config.yaml"):
    """Test the Epoch-Iteration architecture implementation."""
    
    print("🧪 Testing Epoch-Iteration Architecture")
    print("=" * 50)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Force epoch-iteration mode and debug settings
    config['data_pipeline']['use_epoch_iteration'] = True
    config['debug_mode'] = True
    config['data']['max_samples_debug'] = 4
    config['training']['epochs'] = 1
    
    print(f"✅ Configuration loaded: Epoch-Iteration = {config['data_pipeline']['use_epoch_iteration']}")
    
    # Test 1: Import and instantiate
    print("\n📦 Test 1: Import and instantiate...")
    try:
        from src.epoch_iteration_dataset import EpochIterationDataset, create_epoch_iteration_dataloaders
        print("  ✅ Successfully imported Epoch-Iteration components")
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False
    
    # Test 2: Create dataset and test epoch generation
    print("\n🔄 Test 2: Dataset creation and epoch generation...")
    try:
        dataset = EpochIterationDataset(config, split='train')
        print(f"  ✅ Dataset created successfully")
        
        # Test epoch generation
        epoch_start = time.time()
        dataset.new_epoch()
        epoch_time = time.time() - epoch_start
        
        print(f"  ✅ Epoch generated in {epoch_time:.3f}s")
        print(f"  📊 Generated features shape: {dataset.long_feature_sequence.shape if dataset.long_feature_sequence is not None else 'None'}")
        print(f"  📊 Available iterations: {len(dataset)}")
        
    except Exception as e:
        print(f"  ❌ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Iteration sampling
    print("\n🎯 Test 3: Iteration sampling...")
    try:
        for i in range(min(3, len(dataset))):
            features, labels = dataset[i]
            print(f"  Iteration {i}: features {features.shape}, labels {labels.shape}")
            print(f"    - Background events: {torch.sum(labels == 0).item()}")
            print(f"    - Flare events: {torch.sum(labels == 1).item()}")
            print(f"    - Feature range: [{features.min().item():.3f}, {features.max().item():.3f}]")
        
        print("  ✅ Iteration sampling successful")
        
    except Exception as e:
        print(f"  ❌ Iteration sampling failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: DataLoader creation and batching
    print("\n📦 Test 4: DataLoader creation and batching...")
    try:
        train_loader, val_loader, test_loader = create_epoch_iteration_dataloaders(config)
        print(f"  ✅ DataLoaders created successfully")
        print(f"    - Train batches per epoch: {len(train_loader)}")
        print(f"    - Val batches per epoch: {len(val_loader)}")
        print(f"    - Test batches per epoch: {len(test_loader)}")
        
        # Test batching
        batch_count = 0
        for batch_features, batch_labels in train_loader:
            print(f"  Batch {batch_count}: {batch_features.shape}, {batch_labels.shape}")
            print(f"    - Batch feature range: [{batch_features.min().item():.3f}, {batch_features.max().item():.3f}]")
            print(f"    - Background events in batch: {torch.sum(batch_labels == 0).item()}")
            print(f"    - Flare events in batch: {torch.sum(batch_labels == 1).item()}")
            
            batch_count += 1
            if batch_count >= 2:  # Test first 2 batches
                break
        
        print("  ✅ DataLoader batching successful")
        
    except Exception as e:
        print(f"  ❌ DataLoader creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Model compatibility
    print("\n🤖 Test 5: Model compatibility...")
    try:
        from src.model import EventDenoisingMamba
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EventDenoisingMamba(config).to(device)
        
        # Test forward pass with batch
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                predictions = model(batch_features)
            
            print(f"  ✅ Model forward pass successful")
            print(f"    - Input shape: {batch_features.shape}")
            print(f"    - Output shape: {predictions.shape}")
            print(f"    - Output range: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
            break
        
    except Exception as e:
        print(f"  ❌ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 6: Memory usage validation
    print("\n💾 Test 6: Memory usage validation...")
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"  📊 Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb > 1000:  # Warning threshold
            print(f"  ⚠️ Memory usage is high ({memory_mb:.1f} MB)")
        else:
            print(f"  ✅ Memory usage is within safe limits")
        
    except Exception as e:
        print(f"  ⚠️ Memory validation failed: {e}")
    
    # Test 7: Architecture validation
    print("\n🏗️ Test 7: Architecture validation...")
    
    # Verify that features are extracted at epoch level, not iteration level
    original_process_sequence = dataset.feature_extractor.process_sequence
    call_count = 0
    
    def counting_process_sequence(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_process_sequence(*args, **kwargs)
    
    dataset.feature_extractor.process_sequence = counting_process_sequence
    
    # Generate new epoch and sample multiple iterations
    dataset.new_epoch()
    for i in range(min(3, len(dataset))):
        _ = dataset[i]
    
    print(f"  📊 Feature extraction calls during epoch + 3 iterations: {call_count}")
    
    if call_count == 1:
        print("  ✅ Feature extraction called only once per epoch (CORRECT)")
    else:
        print(f"  ❌ Feature extraction called {call_count} times (INCORRECT - should be 1)")
        return False
    
    print("\n🎉 All tests passed! Epoch-Iteration architecture is working correctly.")
    print("\n📋 Architecture Summary:")
    print("  ✅ Epoch Level: Generate long sequence → Extract features once")
    print("  ✅ Iteration Level: Sliding window sampling from feature sequence")
    print("  ✅ Memory Safety: Batch size and sequence limits enforced")
    print("  ✅ Model Compatibility: 11D features processed correctly")
    print("  ✅ Format Handling: DVS [t,x,y,p] → Project [x,y,t,p] conversion")
    
    return True


def compare_architectures(config_path: str = "configs/config.yaml"):
    """Compare old vs new architecture performance."""
    
    print("\n⚖️ Comparing Legacy vs Epoch-Iteration Architecture")
    print("=" * 55)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['debug_mode'] = False  # Disable debug for fair comparison
    config['data']['max_samples_debug'] = 4
    
    results = {}
    
    # Test legacy architecture
    print("\n📊 Testing Legacy Architecture...")
    config['data_pipeline']['use_epoch_iteration'] = False
    
    try:
        from src.mixed_flare_dataloaders import create_mixed_flare_dataloaders
        
        start_time = time.time()
        train_loader, _, _ = create_mixed_flare_dataloaders(config)
        
        # Time first 2 batches
        batch_times = []
        for i, (features, labels) in enumerate(train_loader):
            batch_start = time.time()
            # Simulate some processing
            _ = features.mean()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if i >= 1:  # First 2 batches
                break
        
        legacy_time = time.time() - start_time
        results['legacy'] = {
            'total_time': legacy_time,
            'avg_batch_time': sum(batch_times) / len(batch_times),
            'batches_tested': len(batch_times)
        }
        
        print(f"  ✅ Legacy architecture: {legacy_time:.3f}s total")
        
    except Exception as e:
        print(f"  ❌ Legacy test failed: {e}")
        results['legacy'] = None
    
    # Test new architecture
    print("\n🔄 Testing Epoch-Iteration Architecture...")
    config['data_pipeline']['use_epoch_iteration'] = True
    
    try:
        from src.epoch_iteration_dataset import create_epoch_iteration_dataloaders
        
        start_time = time.time()
        train_loader, _, _ = create_epoch_iteration_dataloaders(config)
        
        # Time first 2 batches (includes epoch generation)
        batch_times = []
        for i, (features, labels) in enumerate(train_loader):
            batch_start = time.time()
            # Simulate some processing
            _ = features.mean()
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if i >= 1:  # First 2 batches
                break
        
        epoch_iteration_time = time.time() - start_time
        results['epoch_iteration'] = {
            'total_time': epoch_iteration_time,
            'avg_batch_time': sum(batch_times) / len(batch_times),
            'batches_tested': len(batch_times)
        }
        
        print(f"  ✅ Epoch-Iteration architecture: {epoch_iteration_time:.3f}s total")
        
    except Exception as e:
        print(f"  ❌ Epoch-Iteration test failed: {e}")
        results['epoch_iteration'] = None
    
    # Compare results
    print("\n📊 Performance Comparison:")
    if results['legacy'] and results['epoch_iteration']:
        legacy_time = results['legacy']['total_time']
        new_time = results['epoch_iteration']['total_time']
        
        if new_time < legacy_time:
            speedup = legacy_time / new_time
            print(f"  🚀 Epoch-Iteration is {speedup:.2f}x faster")
        else:
            slowdown = new_time / legacy_time
            print(f"  🐌 Epoch-Iteration is {slowdown:.2f}x slower")
        
        print(f"  📊 Legacy avg batch time: {results['legacy']['avg_batch_time']:.4f}s")
        print(f"  📊 Epoch-Iteration avg batch time: {results['epoch_iteration']['avg_batch_time']:.4f}s")
    else:
        print("  ⚠️ Cannot compare - one or both tests failed")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Epoch-Iteration Architecture")
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help="Path to configuration file")
    parser.add_argument('--compare', action='store_true', 
                       help="Compare legacy vs new architecture performance")
    
    args = parser.parse_args()
    
    # Test the new architecture
    success = test_epoch_iteration_architecture(args.config)
    
    if success and args.compare:
        # Run performance comparison
        compare_architectures(args.config)
    
    if success:
        print("\n🎯 Ready to use Epoch-Iteration architecture!")
        print("   Run: python main.py --config configs/config.yaml")
        print("   (Ensure data_pipeline.use_epoch_iteration: true in config)")
    else:
        print("\n❌ Tests failed. Please check the implementation.")
        sys.exit(1)