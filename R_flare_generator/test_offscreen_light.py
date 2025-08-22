#!/usr/bin/env python3
"""
测试GLSL反射炫光对屏幕外光源的处理
"""
import os
from GLSL_flare_ultra_fast_gpu import FlareGeneratorUltraFastGPU

def test_offscreen_light():
    """测试光源在屏幕外时的反射炫光"""
    print("🧪 测试GLSL对屏幕外光源的处理...")
    
    # 创建输出目录
    output_dir = "offscreen_light_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化生成器
    resolution = (640, 480)
    generator = FlareGeneratorUltraFastGPU(output_size=resolution)
    
    # 固定参数
    flare_size = 0.2
    light_color = (1.0, 1.0, 1.0)
    time_seed = 15.0
    
    # 噪声纹理
    noise_dir = "noise_textures"
    noise_files = [f for f in os.listdir(noise_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    noise_path = os.path.join(noise_dir, noise_files[0])
    
    # 测试不同位置的光源
    test_positions = [
        (320, 240, "center"),           # 屏幕中心
        (640, 240, "right_edge"),       # 右边缘
        (700, 240, "right_offscreen"),  # 右侧屏幕外
        (800, 240, "far_right"),        # 远离屏幕右侧
        (-60, 240, "left_offscreen"),   # 左侧屏幕外
        (320, -60, "top_offscreen"),    # 上方屏幕外
        (320, 540, "bottom_offscreen"), # 下方屏幕外
    ]
    
    print(f"📍 屏幕分辨率: {resolution}")
    print(f"🎨 噪声纹理: {os.path.basename(noise_path)}")
    print(f"🔍 测试{len(test_positions)}个位置...")
    
    for i, (x, y, desc) in enumerate(test_positions):
        print(f"  [{i+1}/{len(test_positions)}] 测试位置 ({x}, {y}) - {desc}")
        
        try:
            # 生成反射炫光
            img = generator.generate(
                light_pos=(x, y),
                noise_image_path=noise_path,
                time=time_seed,
                flare_size=flare_size,
                light_color=light_color,
                generate_main_glow=False,
                generate_reflections=True
            )
            
            # 保存图片
            output_path = os.path.join(output_dir, f"{i+1:02d}_{desc}_pos{x}x{y}.png")
            img.save(output_path)
            print(f"    ✅ 生成成功: {os.path.basename(output_path)}")
            
        except Exception as e:
            print(f"    ❌ 生成失败: {e}")
    
    print(f"\n✅ 屏幕外光源测试完成!")
    print(f"📁 输出目录: {output_dir}")
    print(f"🔍 观察要点:")
    print(f"   - 屏幕中心是否有明显反射炫光")
    print(f"   - 边缘位置反射是否正常")
    print(f"   - 屏幕外光源是否仍能产生可见反射")
    print(f"   - 远离屏幕的光源影响如何")

if __name__ == "__main__":
    test_offscreen_light()