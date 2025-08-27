import argparse
import yaml
import os
from tqdm import tqdm

from src.event_composer import EventComposer

def run_step1_flare_generation(config):
    """Step 1: 生成纯炫光事件数据"""
    print("🚀 Step 1: Flare Event Generation")
    print("=" * 50)
    
    # 只在需要时导入FlareEventGenerator
    from src.flare_event_generator import FlareEventGenerator
    
    # 获取生成数量
    generation_config = config.get('generation', {})
    num_train = generation_config.get('num_train_sequences', 10)
    num_val = generation_config.get('num_val_sequences', 5)
    
    total_sequences = num_train + num_val
    print(f"Will generate {total_sequences} flare sequences ({num_train} train + {num_val} val)")
    
    # 创建炫光事件生成器
    flare_generator = FlareEventGenerator(config)
    
    # 生成炫光事件
    generated_files = flare_generator.generate_batch(total_sequences)
    
    print(f"\n✅ Step 1 Complete: Generated {len(generated_files)} flare event files")
    # print(f"   Output directory: {flare_generator.output_dir}")
    
    return generated_files

def run_step2_event_composition(config):
    """Step 2: 合成背景+炫光事件"""
    print("\n🚀 Step 2: Event Composition")
    print("=" * 50)
    
    # 检查Step1输出是否存在
    flare_events_dir = os.path.join('output', 'data', 'flare_events')
    if not os.path.exists(flare_events_dir) or not os.listdir(flare_events_dir):
        print(f"❌ Error: No flare events found in {flare_events_dir}")
        print("   Please run Step 1 first with: python main.py --step 1")
        return [], []
    
    # 创建事件合成器
    event_composer = EventComposer(config)
    
    # 合成事件
    bg_files, merge_files = event_composer.compose_batch()
    
    print(f"\n✅ Step 2 Complete: Generated {len(bg_files)} background + {len(merge_files)} merged event files")
    
    # 输出所有方法的目录信息
    for method_name, paths in event_composer.output_dirs.items():
        print(f"   {method_name} method:")
        print(f"     - Stage 1 (BG+Light): {paths['stage1']}")
        print(f"     - Stage 2 (Full Scene): {paths['stage2']}")
    
    return bg_files, merge_files

def run_both_steps(config):
    """运行完整的两步流程"""
    print("🚀 EventMamba-FX Two-Step Event Generator")
    print("=" * 60)
    
    # Step 1: 生成炫光事件
    flare_files = run_step1_flare_generation(config)
    
    if not flare_files:
        print("❌ Step 1 failed, stopping pipeline")
        return
    
    # Step 2: 合成事件
    bg_files, merge_files = run_step2_event_composition(config)
    
    print(f"\n🎉 Complete Pipeline Success!")
    print(f"   Flare events: {len(flare_files)} files")
    print(f"   Stage 1 (BG+Light): {len(bg_files)} files") 
    print(f"   Stage 2 (Full Scene): {len(merge_files)} files")
    print(f"   Total processing complete.")

def main(config, step=None):
    """主函数 - 支持分步执行"""
    
    if step == 1:
        # 只运行Step 1
        run_step1_flare_generation(config)
    elif step == 2:
        # 只运行Step 2
        run_step2_event_composition(config)
    else:
        # 运行完整流程
        run_both_steps(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="EventMamba-FX Two-Step Event Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline (Step 1 + Step 2)
  python main.py --step 1           # Only generate flare events  
  python main.py --step 2           # Only compose events (requires Step 1 first)
  python main.py --debug            # Run with debug visualizations
  python main.py --step 1 --debug   # Generate flare events with debug
        """
    )
    
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help="Path to the YAML configuration file.")
    parser.add_argument('--step', type=int, choices=[1, 2], 
                       help="Run specific step: 1=flare generation, 2=event composition")
    parser.add_argument('--debug', action='store_true', 
                       help="Enable debug mode with visualizations.")
    
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Configure debug mode
    if args.debug:
        config['debug_mode'] = True
        print("🔍 Debug mode enabled - visualizations will be saved")
        
        # Reduce sequences for debug mode
        generation_config = config.setdefault('generation', {})
        generation_config['num_train_sequences'] = generation_config.get('debug_sequences', 3)
        generation_config['num_val_sequences'] = generation_config.get('debug_sequences', 2)
        print(f"   Debug sequences: {generation_config['num_train_sequences']} train + {generation_config['num_val_sequences']} val")

    # Run the specified step(s)
    main(config, step=args.step)