#!/usr/bin/env python3
"""
Run inference and create 1000 visualization frames
"""
import os
import argparse
from src.inference_visualizer import InferenceEventVisualizer

def main():
    parser = argparse.ArgumentParser(description="Run inference on first 1 second and create 1000 visualizations")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Config file")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help="Model checkpoint")
    parser.add_argument('--input', type=str, default='data/inference/zurich_city_12_a.h5', help="Input H5 file")
    parser.add_argument('--output-dir', type=str, default='data/inference/visualization_output', help="Output directory")
    
    args = parser.parse_args()
    
    # Paths
    original_file = args.input
    clean_file = os.path.join(args.output_dir, 'zurich_city_12_a_clean_1s.h5')
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    
    print("🚀 Running EventMamba-FX Inference + Visualization Pipeline")
    print("=" * 60)
    print(f"Input file: {original_file}")
    print(f"Clean file: {clean_file}")  
    print(f"Visualization dir: {viz_dir}")
    print("=" * 60)
    
    # Step 1: Run inference on first 1 second
    print("\n📋 Step 1: Running inference on first 1 second...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    import subprocess
    inference_cmd = [
        'python', 'inference.py',
        '--config', args.config,
        '--checkpoint', args.checkpoint, 
        '--input', original_file,
        '--output', clean_file,
        '--time-limit', '1.0',
        '--block-size', '500000'  # Smaller block size for 1 second
    ]
    
    result = subprocess.run(inference_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Inference failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        return
    else:
        print("✅ Inference completed successfully!")
        
    # Step 2: Create visualizations
    print("\n📊 Step 2: Creating 1000 visualization frames...")
    visualizer = InferenceEventVisualizer(resolution=(640, 480))
    visualizer.visualize_inference_comparison(
        original_h5_path=original_file,
        clean_h5_path=clean_file,
        output_dir=viz_dir,
        time_limit_us=1_000_000  # 1 second
    )
    
    print(f"\n🎉 Pipeline complete!")
    print(f"📁 Results saved in: {args.output_dir}")
    print(f"🎬 1000 visualization frames created in: {viz_dir}")

if __name__ == '__main__':
    main()