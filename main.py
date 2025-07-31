import argparse
import yaml
import torch
import os

from src.mixed_flare_dataloaders import create_mixed_flare_dataloaders
from src.model import EventDenoisingMamba # 确认导入的是修正后的模型
from src.trainer import Trainer
from src.evaluate import Evaluator

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

    # 1. 创建数据集加载器 (混合flare数据)
    train_loader, val_loader, test_loader = create_mixed_flare_dataloaders(config)

    # 2. 初始化模型 (关键修正)
    # 现在我们将特征提取器和Mamba模型的配置分开传递
    model = EventDenoisingMamba(config).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # 3. 根据模式选择执行
    if config['run']['mode'] == 'train':
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