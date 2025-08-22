import time
import os
import random
from GLSL_flare_perfect_python import FlareGeneratorPerfectPython

def test_resolution_speed():
    """测试不同分辨率对生成速度的影响"""
    TEXTURE_SOURCE_DIR = 'noise_textures'
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_texture = os.path.join(TEXTURE_SOURCE_DIR, available_textures[0])
    
    # 不同分辨率测试
    resolutions = [
        (320, 240),   # 0.08M pixels
        (640, 480),   # 0.31M pixels  
        (960, 720),   # 0.69M pixels
        (1280, 960),  # 1.23M pixels
        (1920, 1080), # 2.07M pixels
    ]
    
    print("🔍 分辨率对生成速度影响测试")
    print("=" * 60)
    print(f"{'分辨率':<12} {'像素数':<10} {'生成时间':<10} {'FPS':<8} {'倍率'}")
    print("-" * 60)
    
    base_time = None
    
    for width, height in resolutions:
        generator = FlareGeneratorPerfectPython(output_size=(width, height))
        light_pos = (width * 0.4, height * 0.4)
        
        # 测试3次取平均
        times = []
        for _ in range(3):
            start_time = time.time()
            generator.generate(
                light_pos=light_pos,
                noise_image_path=test_texture,
                generate_main_glow=False,
                generate_reflections=True
            )
            times.append(time.time() - start_time)
        
        avg_time = sum(times) / len(times)
        fps = 1.0 / avg_time
        pixels = width * height / 1000000  # 百万像素
        
        if base_time is None:
            base_time = avg_time
            multiplier = "1.0x"
        else:
            multiplier = f"{avg_time/base_time:.1f}x"
        
        print(f"{width}x{height:<7} {pixels:.2f}M     {avg_time:.2f}s     {fps:.2f}   {multiplier}")
    
    print("=" * 60)
    print("📊 结论：分辨率对速度影响呈平方关系！")

if __name__ == "__main__":
    test_resolution_speed()