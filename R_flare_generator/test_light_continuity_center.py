import os
import math
import time
from GLSL_flare_perfect_gpu import FlareGeneratorPerfectGPU

def generate_center_continuity_test():
    """生成50张光源从中心开始的连续变化测试图片"""
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_center_continuity_test'
    OUTPUT_RESOLUTION = (640, 480)
    
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    generator = FlareGeneratorPerfectGPU(output_size=OUTPUT_RESOLUTION)
    
    print("🎯 光源中心起始连续性测试 - 50张图片")
    print("=" * 60)
    print("📋 测试配置:")
    print(f"  - 分辨率: {OUTPUT_RESOLUTION}")
    print(f"  - 光源轨迹: 从中心开始的螺旋路径")
    print(f"  - 固定参数: 同一噪声纹理, seed=15.0, size=0.25")
    print(f"  - 变化参数: 光源位置 (50步连续变化)")
    
    # 固定所有其他参数
    fixed_texture = os.path.join(TEXTURE_SOURCE_DIR, available_textures[10])  # 选择一个好的纹理
    fixed_seed = 15.0   # 增大seed获得更强效果
    fixed_size = 0.25   # 增大炫光尺寸
    fixed_color = (1.0, 1.0, 1.0)
    
    print(f"  - 固定纹理: {os.path.basename(fixed_texture)}")
    print(f"  - 固定seed: {fixed_seed}")
    print(f"  - 固定size: {fixed_size}")
    
    # 设计光源位置轨迹 - 从中心开始的螺旋路径
    center_x = OUTPUT_RESOLUTION[0] // 2
    center_y = OUTPUT_RESOLUTION[1] // 2
    max_radius = min(OUTPUT_RESOLUTION[0], OUTPUT_RESOLUTION[1]) // 3
    
    print(f"  - 中心点: ({center_x}, {center_y})")
    print(f"  - 最大半径: {max_radius}px")
    print(f"  - 轨迹: 螺旋扩展 (中心→边缘)")
    print("=" * 60)
    
    total_start_time = time.time()
    
    # 先生成一张中心位置的测试图检查效果
    print("🔍 预测试: 生成中心位置图片检查炫光效果...")
    test_img = generator.generate(
        light_pos=(center_x, center_y),
        noise_image_path=fixed_texture,
        time=fixed_seed,
        flare_size=fixed_size,
        light_color=fixed_color,
        generate_main_glow=False,
        generate_reflections=True
    )
    test_img.save(os.path.join(OUTPUT_DIR, "00_center_test.png"))
    print("  -> 保存预测试图片: 00_center_test.png")
    print("  -> 请检查此图是否有明显炫光效果")
    
    for i in range(50):
        # 螺旋路径: 半径从0逐渐增加到max_radius
        progress = i / 49.0  # 0到1的进度
        radius = max_radius * progress  # 半径从0增长到max_radius
        angle = progress * 4 * math.pi  # 转2圈 (720度)
        
        light_x = center_x + radius * math.cos(angle)
        light_y = center_y + radius * math.sin(angle)
        light_pos = (light_x, light_y)
        
        start_time = time.time()
        
        # 生成图片
        img = generator.generate(
            light_pos=light_pos,
            noise_image_path=fixed_texture,
            time=fixed_seed,
            flare_size=fixed_size,
            light_color=fixed_color,
            generate_main_glow=False,   # 只生成反射炫光
            generate_reflections=True
        )
        
        end_time = time.time()
        
        # 文件命名包含轨迹信息
        output_name = f"spiral_{i+1:03d}_r{radius:.0f}_pos{light_x:.0f}x{light_y:.0f}_a{math.degrees(angle):.0f}.png"
        img.save(os.path.join(OUTPUT_DIR, output_name))
        
        print(f"[{i+1:2d}/50] 半径{radius:3.0f} 位置({light_x:.0f},{light_y:.0f}) 角度{math.degrees(angle):4.0f}° "
              f"-> {output_name} ({end_time-start_time:.2f}s)")
        
        # 每10张报告进度
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            remaining = (elapsed / (i + 1)) * (50 - i - 1)
            print(f"    进度: {i+1}/50 完成，预计剩余 {remaining:.0f}s")
    
    total_time = time.time() - total_start_time
    avg_time = total_time / 50
    
    print("\n📊 中心起始连续性测试完成")
    print("=" * 60)
    print(f"总时间: {total_time:.1f}秒")
    print(f"平均每张: {avg_time:.2f}秒")
    print(f"等效帧率: {1/avg_time:.2f} FPS")
    print(f"输出路径: {OUTPUT_DIR}/")
    
    print(f"\n🔍 连续性检查指南:")
    print("1. 首先查看 '00_center_test.png' 确认中心有明显炫光")
    print("2. 按文件名顺序查看50张图片:")
    print("   - 炫光从中心开始是否清晰可见？")
    print("   - 随着光源螺旋移动，反射是否连续变化？")
    print("   - 光源移至边缘时，炫光是否仍然可见？")
    print("3. 重点关注:")
    print("   - 中心区域的强反射效果")
    print("   - 螺旋路径的平滑过渡")
    print("   - 不同半径处的炫光强度变化")
    
    # 生成轨迹信息文件
    trajectory_file = os.path.join(OUTPUT_DIR, "spiral_trajectory.txt")
    with open(trajectory_file, 'w') as f:
        f.write("# 螺旋轨迹连续性测试 - 轨迹记录\n")
        f.write("# Frame, Radius, Light_X, Light_Y, Angle_Degrees, Filename\n")
        for i in range(50):
            progress = i / 49.0
            radius = max_radius * progress
            angle = progress * 4 * math.pi
            light_x = center_x + radius * math.cos(angle)
            light_y = center_y + radius * math.sin(angle)
            filename = f"spiral_{i+1:03d}_r{radius:.0f}_pos{light_x:.0f}x{light_y:.0f}_a{math.degrees(angle):.0f}.png"
            f.write(f"{i+1:3d}, {radius:6.1f}, {light_x:6.1f}, {light_y:6.1f}, {math.degrees(angle):6.1f}, {filename}\n")
    
    print(f"\n📝 轨迹记录已保存: {trajectory_file}")
    print("🎉 中心起始连续性测试完成！")
    print("\n💡 提示: 如果中心测试图没有明显炫光，可能需要:")
    print("    - 调整seed参数 (尝试1.0, 5.0, 20.0, 50.0)")
    print("    - 增大flare_size (尝试0.3, 0.4, 0.5)")
    print("    - 更换噪声纹理")

if __name__ == "__main__":
    generate_center_continuity_test()