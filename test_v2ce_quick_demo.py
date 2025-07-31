#!/usr/bin/env python3
"""
V2CE快速演示 - 多分辨率可视化
"""
import yaml
import os
from src.dvs_flare_integration import create_flare_event_generator

def quick_v2ce_demo():
    """快速V2CE演示"""
    
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['event_simulator']['type'] = 'v2ce'
    config['debug_mode'] = True
    config['debug_output_dir'] = './output/v2ce_quick_demo'
    config['data']['flare_synthesis']['duration_sec'] = 0.03  # 30ms
    
    print("🚀 V2CE快速演示")
    print("=" * 40)
    
    generator = create_flare_event_generator(config)
    events, timing = generator.generate_flare_events()
    
    if len(events) > 0:
        print(f"\n✅ 生成结果:")
        print(f"   事件数量: {len(events):,}")
        print(f"   时间跨度: {(events[-1,0] - events[0,0])/1000:.1f}ms")
        
        # 检查多分辨率文件
        base_dir = "./output/v2ce_quick_demo/flare_seq_v2ce_000/v2ce_event_visualizations"
        
        if os.path.exists(base_dir):
            print(f"\n🎨 多分辨率可视化:")
            for resolution in ['0.5x', '1x', '2x', '4x']:
                res_dir = os.path.join(base_dir, f"temporal_{resolution}")
                if os.path.exists(res_dir):
                    file_count = len([f for f in os.listdir(res_dir) if f.endswith('.png')])
                    print(f"   {resolution}: {file_count} 文件")
            
            print(f"\n📁 输出位置: {base_dir}")
            print(f"   temporal_0.5x/ : 低时间分辨率 (2帧合并)")
            print(f"   temporal_1x/   : 标准分辨率 (1帧1窗口)")  
            print(f"   temporal_2x/   : 高时间分辨率 (1帧2细分)")
            print(f"   temporal_4x/   : 超高时间分辨率 (1帧4细分)")

if __name__ == "__main__":
    quick_v2ce_demo()