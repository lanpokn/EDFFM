import argparse
import yaml
import torch

from src.model import EventDenoisingMamba # 确认导入的是修正后的模型
from src.epoch_based_trainer import create_epoch_based_trainer
from src.evaluate import Evaluator

def main(config):
    """
    Main function to run the training and evaluation pipeline.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 初始化模型 (现在直接接收11维特征)
    model = EventDenoisingMamba(config).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    print(f"Model expects {config['model']['input_feature_dim']}-dimensional features")

    # 2. 根据模式选择执行
    if config['run']['mode'] == 'train':
        # 使用新的epoch-based训练器
        trainer = create_epoch_based_trainer(model, config, device)
        trainer.train()
    elif config['run']['mode'] == 'evaluate':
        # TODO: 需要实现epoch-based评估器
        print("Evaluation mode not yet implemented for epoch-based architecture")
        print("Please set mode to 'train' in config.yaml")
    else:
        raise ValueError(f"Unknown mode: {config['run']['mode']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the EventMamba-FX model.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to the YAML configuration file.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)