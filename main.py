import argparse
import yaml
from tqdm import tqdm

from src.unified_dataset import UnifiedSequenceDataset

def main(config):
    print("🚀 Starting EventMamba-FX Data Generator")
    
    # 只支持generate模式
    pipeline_mode = config['data_pipeline']['mode']
    if pipeline_mode != 'generate':
        raise ValueError(f"This generator only supports 'generate' mode, got '{pipeline_mode}'. Please set data_pipeline.mode: 'generate' in your config.")

    print("🚀 Starting DATA GENERATION mode...")
    
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
    test_sequences = config.get('generation', {}).get('num_test_sequences', 0)
    if test_sequences > 0:
        print("\n--- Generating test data ---")
        test_gen_dataset = UnifiedSequenceDataset(config, split='test')
        for i in tqdm(range(len(test_gen_dataset)), desc="Generating Test Seqs"):
            test_gen_dataset[i]
    
    print("\n✅ Data generation complete!")
    print("   Generated H5 files are saved in:", config['data_pipeline']['h5_archive_path'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EventMamba-FX Data Generator - Generate simulation datasets with flare events.")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to the YAML configuration file.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode to save flare image sequences and event visualizations.")
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Enable debug mode if --debug flag is set
    if args.debug:
        config['debug_mode'] = True
        print("Debug mode enabled.")
        # Use debug settings from config or set defaults
        generation_config = config.setdefault('generation', {})
        generation_config['num_train_sequences'] = generation_config.get('debug_sequences', 8)
        generation_config['num_val_sequences'] = generation_config.get('debug_sequences', 8) // 2
        # Debug event visualization parameters
        config['debug_event_subdivisions'] = generation_config.get('debug_subdivisions', [0.5, 1, 2, 4])

    main(config)