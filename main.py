import argparse
import yaml
import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.unified_dataset import UnifiedSequenceDataset, create_unified_dataloaders
from src.model import EventDenoisingMamba
from src.trainer import Trainer
from src.evaluate import Evaluator

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ### BEGIN BUGFIX 2 & 3: WORKFLOW REFACTOR ###

    pipeline_mode = config['data_pipeline']['mode']
    run_mode = config['run']['mode']

    # --- 工作流 1: 数据预生成 ---
    if pipeline_mode == 'generate':
        print("🚀 Starting DATA PRE-GENERATION mode. This will not start training.")
        
        # 生成训练数据
        print("\n--- Generating training data ---")
        train_gen_dataset = UnifiedSequenceDataset(config, split='train')
        for i in tqdm(range(len(train_gen_dataset)), desc="Generating Train Seqs"):
            train_gen_dataset[i]
        
        # 生成验证数据
        print("\n--- Generating validation data ---")
        val_gen_dataset = UnifiedSequenceDataset(config, split='val')
        for i in tqdm(range(len(val_gen_dataset)), desc="Generating Val Seqs"):
            val_gen_dataset[i]

        # (可选) 生成测试数据
        if config.get('evaluation', {}).get('num_long_sequences_per_epoch', 0) > 0:
            print("\n--- Generating test data ---")
            test_gen_dataset = UnifiedSequenceDataset(config, split='test')
            for i in tqdm(range(len(test_gen_dataset)), desc="Generating Test Seqs"):
                test_gen_dataset[i]
        
        print("\n✅ Data pre-generation complete.")
        print("   Please change 'mode' in your config's 'data_pipeline' to 'load' to start training or evaluation.")
        return # 关键：生成完数据后直接退出程序

    # --- 必须是 'load' 模式才能继续 ---
    if pipeline_mode != 'load':
        raise ValueError(f"Invalid data_pipeline mode '{pipeline_mode}'. Must be 'generate' or 'load'.")

    # --- 初始化模型 ---
    model = EventDenoisingMamba(config).to(device)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 工作流 2: 模型训练 ---
    if run_mode == 'train':
        print("\n🚀 Starting TRAINING mode...")
        train_loader, val_loader = create_unified_dataloaders(config)
        trainer = Trainer(model, train_loader, val_loader, config, device)
        trainer.train()

    # --- 工作流 3: 模型评估 ---
    elif run_mode == 'evaluate':
        print("\n🚀 Starting EVALUATION mode...")
        print("🛠️ Creating test dataloader for evaluation...")
        
        try:
            test_dataset = UnifiedSequenceDataset(config, split='test')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
            print(f"✅ Test dataloader created with {len(test_dataset)} sequences.")
        except FileNotFoundError as e:
            print(f"❌ ERROR: {e}")
            print("❌ Please ensure you have pre-generated the 'test' data split using the 'generate' mode.")
            return

        checkpoint_path = config['evaluation']['checkpoint_path']
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint for evaluation not found at: {checkpoint_path}")
        
        # 注意: 如果只评估，加载完整的state_dict是不安全的。
        # 这里我们假设checkpoint是可信的。
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model state from: {checkpoint_path}")
        
        # 假设 Evaluator 类也实现了正确的状态重置和不完整块处理
        evaluator = Evaluator(model, test_loader, config, device)
        evaluator.evaluate()

    else:
        raise ValueError(f"Unknown run mode: {run_mode}")
    
    # ### END BUGFIX ###

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
        print("Debug mode enabled.")
        # Limit iterations for debug mode
        config['training']['max_epochs'] = 1
        # Reduce sequences for quick debug validation
        config['training']['num_long_sequences_per_epoch'] = 8
        # Debug event visualization parameters (multiple temporal resolutions)
        config['debug_event_subdivisions'] = [0.5, 1, 2, 4]  # Multiple subdivision strategies

    main(config)