#!/usr/bin/env python3
"""
快速测试Step2修复
仅处理1个序列，验证physics方法是否正常工作
"""

import yaml
import os
import sys

def test_step2_physics_fix():
    """测试Step2 Physics方法修复"""
    print("🧪 Testing Step 2 Physics Fix (Single Sequence)")
    print("=" * 60)
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 启用debug模式
    config['debug_mode'] = True
    
    # 确保使用physics方法
    config['composition']['merge_method'] = 'physics'
    config['composition']['generate_both_methods'] = True
    
    print(f"🎯 Configuration:")
    print(f"   Merge method: {config['composition']['merge_method']}")
    print(f"   Generate both methods: {config['composition']['generate_both_methods']}")
    print(f"   Debug mode: {config['debug_mode']}")
    
    # 检查Step1输出是否存在
    flare_events_dir = os.path.join('output', 'data', 'flare_events')
    light_source_events_dir = os.path.join('output', 'data', 'light_source_events')
    
    if not os.path.exists(flare_events_dir) or not os.listdir(flare_events_dir):
        print(f"❌ Error: No flare events found in {flare_events_dir}")
        return False
        
    if not os.path.exists(light_source_events_dir) or not os.listdir(light_source_events_dir):
        print(f"❌ Error: No light source events found in {light_source_events_dir}")
        return False
    
    print(f"✅ Found Step 1 outputs:")
    print(f"   Flare events: {len(os.listdir(flare_events_dir))} files")
    print(f"   Light source events: {len(os.listdir(light_source_events_dir))} files")
    
    # 创建事件合成器
    try:
        from src.event_composer import EventComposer
        event_composer = EventComposer(config)
        
        print(f"\n🚀 Running Step 2 composition (max 1 sequence for testing)...")
        
        # 只处理1个序列进行快速测试
        bg_files, merge_files = event_composer.compose_batch(max_sequences=1)
        
        if bg_files and merge_files:
            print(f"\n✅ Step 2 Physics Fix Test SUCCESS!")
            print(f"   Generated background+light files: {len(bg_files)}")
            print(f"   Generated full scene files: {len(merge_files)}")
            
            # 显示输出目录
            for method_name, paths in event_composer.output_dirs.items():
                print(f"   {method_name} method:")
                print(f"     - Stage 1: {paths['stage1']}")
                print(f"     - Stage 2: {paths['stage2']}")
            
            return True
        else:
            print(f"❌ Step 2 failed - no files generated")
            return False
            
    except Exception as e:
        print(f"❌ Step 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_step2_physics_fix()
    if success:
        print(f"\n🎉 Physics method bug fix verification PASSED!")
        sys.exit(0)
    else:
        print(f"\n💥 Physics method bug fix verification FAILED!")
        sys.exit(1)