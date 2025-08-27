#!/usr/bin/env python3
"""
Physics Event Composer Test Script
==================================

测试新的物理混合模型事件合成器
"""

import os
import yaml
import numpy as np
from src.event_composer import EventComposer

def test_physics_composition():
    """测试物理混合模型"""
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("🧪 Testing Physics Event Composer")
    print("=================================")
    
    # 修改配置启用physics方法和debug模式
    config['composition'] = {
        'merge_method': 'physics',           # 使用physics方法
        'generate_both_methods': True,       # 同时生成两种方法对比
        'physics_params': {
            'background_event_weight': 0.3,
            'light_source_event_weight': 1.0,
            'flare_intensity_multiplier': 1.2,
            'temporal_jitter_us': 30,
            'epsilon': 1e-9
        }
    }
    config['debug_mode'] = True
    
    # 创建合成器
    composer = EventComposer(config)
    
    print(f"\\n📋 Configuration:")
    print(f"  Merge method: {composer.merge_method}")
    print(f"  Generate both methods: {composer.generate_both_methods}")
    print(f"  Physics params: {composer.composition_config.get('physics_params', {})}")
    
    # 检查输入文件
    if not os.path.exists(composer.flare_events_dir):
        print(f"❌ Flare events directory not found: {composer.flare_events_dir}")
        print("   Please run Step 1 first: python main.py --step 1 --debug")
        return False
    
    if not os.path.exists(composer.light_source_events_dir):
        print(f"❌ Light source events directory not found: {composer.light_source_events_dir}")
        print("   Please run Step 1 first: python main.py --step 1 --debug")
        return False
    
    # 运行合成测试
    try:
        print(f"\\n🚀 Running physics composition test...")
        bg_files, merge_files = composer.compose_batch(max_sequences=2)
        
        print(f"\\n✅ Test Results:")
        print(f"  Generated background+light files: {len(bg_files)}")
        print(f"  Generated full scene files: {len(merge_files)}")
        
        # 检查输出目录
        print(f"\\n📁 Output Directories:")
        for method_name, paths in composer.output_dirs.items():
            stage1_count = len([f for f in os.listdir(paths['stage1']) if f.endswith('.h5')])
            stage2_count = len([f for f in os.listdir(paths['stage2']) if f.endswith('.h5')])
            print(f"  {method_name} method:")
            print(f"    - Stage 1: {stage1_count} files in {paths['stage1']}")
            print(f"    - Stage 2: {stage2_count} files in {paths['stage2']}")
        
        # 检查debug可视化
        debug_dir = composer.debug_dir
        if os.path.exists(debug_dir):
            debug_subdirs = [d for d in os.listdir(debug_dir) if os.path.isdir(os.path.join(debug_dir, d))]
            print(f"\\n🎨 Debug Visualizations:")
            print(f"  Debug directory: {debug_dir}")
            print(f"  Generated {len(debug_subdirs)} debug subdirectories")
            
            # 检查权重图
            for subdir in debug_subdirs:
                subdir_path = os.path.join(debug_dir, subdir)
                weight_maps = [f for f in os.listdir(subdir_path) if 'weight_map' in f]
                if weight_maps:
                    print(f"    - {subdir}: {len(weight_maps)} weight maps")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_physics_composition()
    
    if success:
        print(f"\\n🎉 Physics composition test completed successfully!")
        print(f"\\n📋 Next steps:")
        print(f"   1. Check output directories for physics vs simple method comparison")
        print(f"   2. Review debug visualizations including weight maps A(x,y)")
        print(f"   3. Compare event statistics between methods")
        
        print(f"\\n🔍 Verification commands:")
        print(f"   ls -la output/data/simple_method/")
        print(f"   ls -la output/data/physics_method/")
        print(f"   ls -la output/debug/event_composition/")
        
    else:
        print(f"\\n💥 Test failed - please check the error messages above")
    
    return success

if __name__ == "__main__":
    main()