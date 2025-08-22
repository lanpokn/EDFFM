import time
import os
import random
from GLSL_flare_perfect_python import FlareGeneratorPerfectPython
from GLSL_flare_perfect_gpu import FlareGeneratorPerfectGPU

def performance_comparison():
    """对比CPU NumPy vs GPU PyTorch版本性能"""
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_RESOLUTION = (640, 480)
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    test_texture = os.path.join(TEXTURE_SOURCE_DIR, available_textures[0])
    
    print("⚔️  CPU vs GPU 性能对比测试")
    print("=" * 70)
    
    # 测试参数
    test_configs = [
        ("反射炫光", {"generate_main_glow": False, "generate_reflections": True}),
        ("主光辉", {"generate_main_glow": True, "generate_reflections": False}),
        ("完整效果", {"generate_main_glow": True, "generate_reflections": True}),
    ]
    
    # 初始化生成器
    print("🔧 初始化生成器...")
    cpu_generator = FlareGeneratorPerfectPython(output_size=OUTPUT_RESOLUTION)
    gpu_generator = FlareGeneratorPerfectGPU(output_size=OUTPUT_RESOLUTION)
    
    results = []
    
    for config_name, config_params in test_configs:
        print(f"\n🧪 测试配置: {config_name}")
        print("-" * 40)
        
        light_pos = (OUTPUT_RESOLUTION[0] * 0.4, OUTPUT_RESOLUTION[1] * 0.4)
        
        # CPU版本测试
        print("  CPU NumPy版本...")
        cpu_times = []
        for i in range(3):
            start_time = time.time()
            cpu_generator.generate(
                light_pos=light_pos,
                noise_image_path=test_texture,
                **config_params
            )
            cpu_times.append(time.time() - start_time)
        cpu_avg = sum(cpu_times) / len(cpu_times)
        cpu_fps = 1.0 / cpu_avg
        
        # GPU版本测试
        print("  GPU PyTorch版本...")
        gpu_times = []
        for i in range(3):
            start_time = time.time()
            gpu_generator.generate(
                light_pos=light_pos,
                noise_image_path=test_texture,
                **config_params
            )
            gpu_times.append(time.time() - start_time)
        gpu_avg = sum(gpu_times) / len(gpu_times)
        gpu_fps = 1.0 / gpu_avg
        
        # 计算提升倍数
        speedup = cpu_avg / gpu_avg
        
        results.append({
            'config': config_name,
            'cpu_time': cpu_avg,
            'gpu_time': gpu_avg, 
            'cpu_fps': cpu_fps,
            'gpu_fps': gpu_fps,
            'speedup': speedup
        })
        
        print(f"    CPU: {cpu_avg:.3f}s ({cpu_fps:.2f} FPS)")
        print(f"    GPU: {gpu_avg:.3f}s ({gpu_fps:.2f} FPS)")
        print(f"    GPU加速: {speedup:.2f}x")
    
    # 汇总报告
    print("\n📊 性能汇总报告")
    print("=" * 70)
    print(f"{'配置':<12} {'CPU时间':<10} {'GPU时间':<10} {'CPU FPS':<8} {'GPU FPS':<8} {'加速比'}")
    print("-" * 70)
    
    total_cpu_time = 0
    total_gpu_time = 0
    
    for result in results:
        print(f"{result['config']:<12} {result['cpu_time']:.3f}s    {result['gpu_time']:.3f}s    "
              f"{result['cpu_fps']:.2f}     {result['gpu_fps']:.2f}     {result['speedup']:.2f}x")
        total_cpu_time += result['cpu_time']
        total_gpu_time += result['gpu_time']
    
    avg_speedup = total_cpu_time / total_gpu_time
    
    print("-" * 70)
    print(f"平均加速比: {avg_speedup:.2f}x")
    print(f"GPU总体性能提升: {((avg_speedup-1)*100):.1f}%")
    
    # 分辨率影响对比
    print(f"\n🎯 分辨率对GPU性能影响测试")
    print("=" * 50)
    print(f"{'分辨率':<12} {'GPU时间':<10} {'FPS':<8} {'相对320x240'}")
    print("-" * 50)
    
    resolutions = [(320, 240), (640, 480), (1280, 960)]
    base_gpu_time = None
    
    for width, height in resolutions:
        temp_generator = FlareGeneratorPerfectGPU(output_size=(width, height))
        light_pos = (width * 0.4, height * 0.4)
        
        start_time = time.time()
        temp_generator.generate(
            light_pos=light_pos,
            noise_image_path=test_texture,
            generate_main_glow=False,
            generate_reflections=True
        )
        gpu_time = time.time() - start_time
        fps = 1.0 / gpu_time
        
        if base_gpu_time is None:
            base_gpu_time = gpu_time
            relative = "1.0x"
        else:
            relative = f"{gpu_time/base_gpu_time:.1f}x"
        
        print(f"{width}x{height:<7} {gpu_time:.3f}s    {fps:.2f}   {relative}")
    
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    performance_comparison()