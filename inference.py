#!/usr/bin/env python3
"""
EventMamba-FX Clean Inference Script
Simple, production-ready inference for event camera denoising.
"""
import argparse
import yaml
import torch
import os

from src.model import EventDenoisingMamba
from src.inference import Predictor


def main():
    parser = argparse.ArgumentParser(
        description="Denoise event camera H5 files using EventMamba-FX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to YAML configuration file")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument('--input', type=str, required=True, 
                        help="Path to input H5 event file")
    parser.add_argument('--output', type=str, required=True, 
                        help="Path for output denoised H5 file")
    parser.add_argument('--threshold', type=float, default=0.5, 
                        help="Prediction threshold for flare event removal")
    parser.add_argument('--block-size', type=int, default=5_000_000, 
                        help="Events per processing block (adjust for RAM)")
    parser.add_argument('--time-limit', type=float, default=None,
                        help="Time limit in seconds for processing")

    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Override inference settings
    config['inference'] = {
        'denoise_threshold': args.threshold,
        'block_size_events': args.block_size
    }

    # Initialize model
    model = EventDenoisingMamba(config)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded: {args.checkpoint}")

    # Run inference
    predictor = Predictor(model, config, device)
    
    time_limit_us = args.time_limit * 1_000_000 if args.time_limit else None
    predictor.denoise_file(args.input, args.output, time_limit_us)
    
    print(f"✅ Denoising complete: {args.output}")


if __name__ == '__main__':
    main()