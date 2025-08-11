#!/usr/bin/env python3
"""
EventMamba-FX Robust Inference Script
Denoise large H5 event files using a trained EventMamba-FX model with streaming approach to prevent OOM.
"""
import argparse
import yaml
import torch
import os

from src.model import EventDenoisingMamba
from src.predictor import Predictor  # 确保导入新的 Predictor

def inference(args):
    # 1. Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Add/override inference-specific settings
    #    This allows command-line control over memory usage
    config['inference'] = {
        'denoise_threshold': args.threshold,
        'block_size_events': args.block_size
    }

    # 3. Initialize the model
    model = EventDenoisingMamba(config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # 4. Load the trained model weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model weights from: {args.checkpoint}")

    # 5. Initialize and run the robust predictor
    predictor = Predictor(model, config, device)
    
    # Apply time limit if specified (1 second = 1,000,000 microseconds)
    time_limit_us = args.time_limit * 1_000_000 if args.time_limit else None
    predictor.denoise_file(args.input, args.output, time_limit_us)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Robustly denoise a large H5 event file using a trained EventMamba-FX model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Nicer help messages
    )
    
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML configuration file from training.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint (.pth).")
    parser.add_argument('--input', type=str, required=True, help="Path to the large input H5 event file.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the new, denoised H5 file.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Prediction threshold to remove flare events.")
    parser.add_argument('--block-size', type=int, default=5_000_000, 
                        help="Number of events to load into RAM at once. Decrease this value if you run out of RAM.")
    parser.add_argument('--time-limit', type=float, default=0.1,
                        help="Time limit in seconds for processing events from the start of the file.")

    args = parser.parse_args()
    
    inference(args)