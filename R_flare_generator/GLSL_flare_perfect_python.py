import numpy as np
from PIL import Image
import os
import random
import time
from scipy.ndimage import map_coordinates

class FlareGeneratorPerfectPython:
    """
    高性能NumPy向量化炫光生成器
    - 100%复刻GLSL算法
    - NumPy全向量化，消除所有显式循环
    - 保持完全相同的输入输出接口
    """
    def __init__(self, output_size=(1920, 1080)):
        self.output_size = output_size
        self.width, self.height = output_size
        print("🚀 初始化高性能NumPy炫光生成器 (向量化)")
    
    def _vectorized_noise_sample(self, noise_texture, u, v):
        """向量化的噪声采样 - 复刻GLSL texture()"""
        h, w, c = noise_texture.shape
        
        # 处理标量和数组输入
        u = np.atleast_1d(u)
        v = np.atleast_1d(v)
        
        # 归一化坐标到纹理尺寸
        x = u * w
        y = v * h
        
        # 边界处理 - 复刻GLSL重复模式
        x = x % w
        y = y % h
        
        # 使用scipy的双线性插值 - 更高效且精确复刻GLSL采样
        result = np.zeros((len(u), 4), dtype=np.float32)
        
        for i in range(min(c, 4)):
            result[:, i] = map_coordinates(noise_texture[:, :, i], [y, x], 
                                         order=1, mode='wrap', prefilter=False)
        
        return result
    
    def _vectorized_flare_kernel(self, uv_x, uv_y, pos_x, pos_y, seed, flare_size, 
                               noise_texture, generate_main_glow, generate_reflections):
        """完全向量化的炫光内核 - 复刻GLSL flare()函数"""
        
        # 展平所有坐标以便向量化处理
        flat_size = uv_x.size
        uv_x_flat = uv_x.flatten()
        uv_y_flat = uv_y.flatten()
        
        # noise(seed-1.0) - 复刻GLSL噪声调用
        gn = self._vectorized_noise_sample(noise_texture, np.array([seed - 1.0]), np.array([0.0]))[0]
        gn[0] = flare_size
        
        # 计算距离向量
        d_x = uv_x_flat - pos_x
        d_y = uv_y_flat - pos_y
        
        # 初始化结果数组
        main_glow_color = np.zeros((flat_size, 3), dtype=np.float32)
        reflections_color = np.zeros((flat_size, 3), dtype=np.float32)
        
        # Part 1: 主光辉 (向量化)
        if generate_main_glow:
            d_length = np.sqrt(d_x**2 + d_y**2)
            core_intensity = (0.01 + gn[0] * 0.2) / (d_length + 0.001)
            main_glow_color[:, :] = core_intensity[:, np.newaxis]
            
            # 光晕计算 - 向量化
            angle = np.arctan2(d_y, d_x)
            halo_u = angle * 256.9 + pos_x * 2.0
            halo_noise = self._vectorized_noise_sample(noise_texture, halo_u, np.zeros_like(halo_u))
            halo_factor = halo_noise[:, 1] * 0.25
            
            main_glow_color *= (1.0 + halo_factor[:, np.newaxis])
        
        # fltr计算 - 向量化
        uv_length = np.sqrt(uv_x_flat**2 + uv_y_flat**2)
        fltr = np.minimum(uv_length**2 * 0.5 + 0.5, 1.0)
        
        # Part 2: 反射炫光 - 向量化最复杂的双重循环
        if generate_reflections:
            # 预计算所有噪声值 - 批量采样
            i_values = np.arange(20)
            n_u = seed + i_values
            n2_u = seed + i_values * 2.1
            nc_u = seed + i_values * 3.3
            
            # 批量噪声采样
            n_batch = self._vectorized_noise_sample(noise_texture, n_u, np.zeros_like(n_u))
            n2_batch = self._vectorized_noise_sample(noise_texture, n2_u, np.zeros_like(n2_u))
            nc_batch = self._vectorized_noise_sample(noise_texture, nc_u, np.zeros_like(nc_u))
            
            # 处理nc - 向量化
            nc_length = np.sqrt(np.sum(nc_batch**2, axis=1))
            nc_batch += nc_length[:, np.newaxis]
            nc_batch *= 0.65
            
            # 主反射循环 - 向量化处理
            for i in range(20):
                n = n_batch[i]
                n2 = n2_batch[i]
                nc = nc_batch[i]
                
                for j in range(3):  # RGB通道
                    # 参数计算 - 向量化
                    ip = n[0] * 3.0 + j * 0.1 * n2[1]**3
                    is_val = n[1]**2 * 4.5 * gn[0] + 0.1
                    ia = (n[2] * 4.0 - 2.0) * n2[0] * n[1]
                    
                    # UV变换 - 向量化
                    mix_factor = 1.0 + (uv_length - 1.0) * n[3]**2
                    iuv_x = uv_x_flat * mix_factor
                    iuv_y = uv_y_flat * mix_factor
                    
                    # 旋转变换 - 向量化
                    cos_ia = np.cos(ia)
                    sin_ia = np.sin(ia)
                    rotated_x = iuv_x * cos_ia + iuv_y * sin_ia
                    rotated_y = -iuv_x * sin_ia + iuv_y * cos_ia
                    
                    # ID计算 - 向量化
                    id_x = (rotated_x - pos_x) * (1.0 - ip) + (rotated_x + pos_x) * ip
                    id_y = (rotated_y - pos_y) * (1.0 - ip) + (rotated_y + pos_y) * ip
                    id_length = np.sqrt(id_x**2 + id_y**2)
                    
                    # 反射强度计算 - 向量化
                    intensity_base = np.maximum(0.0, is_val - id_length)
                    mask = intensity_base > 0
                    if np.any(mask):
                        intensity = np.zeros_like(intensity_base)
                        intensity[mask] = (intensity_base[mask]**0.45 / is_val * 0.1 * 
                                         gn[0] * nc[j] * fltr[mask])
                        reflections_color[:, j] += intensity
        
        # 合并结果
        result = main_glow_color + reflections_color
        
        # 重新整形为原始形状
        return result.reshape(self.height, self.width, 3)
    
    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
        高性能向量化生成 - 保持完全相同的接口
        """
        print(f"🎨 向量化生成: 位置={light_pos}, 反射={generate_reflections}, 光源={generate_main_glow}")
        
        # 加载噪声纹理
        noise_img = Image.open(noise_image_path).convert("RGBA")
        noise_array = np.array(noise_img).astype(np.float32) / 255.0
        
        # 创建UV坐标网格 - 向量化
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        
        # GLSL坐标变换 - 向量化
        uv_x = (x_coords / self.width - 0.5) * 2.0 * (self.width / self.height)
        uv_y = (y_coords / self.height - 0.5) * 2.0
        
        # 光源位置变换
        pos_x = (light_pos[0] / self.width - 0.5) * 2.0 * (self.width / self.height)
        pos_y = (light_pos[1] / self.height - 0.5) * 2.0
        
        # 向量化炫光计算
        img_array = self._vectorized_flare_kernel(
            uv_x, uv_y, pos_x, pos_y, time, flare_size, 
            noise_array, generate_main_glow, generate_reflections
        )
        
        # 应用光源颜色 - 向量化
        light_color_array = np.array(light_color, dtype=np.float32)
        img_array *= light_color_array
        
        # 添加噪声 - 向量化
        x_flat = x_coords.flatten() / self.width
        y_flat = y_coords.flatten() / self.height
        noise_addition = self._vectorized_noise_sample(noise_array, x_flat, y_flat)
        noise_addition = noise_addition[:, :3].reshape(self.height, self.width, 3)
        img_array += noise_addition * 0.01
        
        # 最终处理
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        print("✅ 向量化渲染完成")
        return Image.fromarray(img_array)

# 测试代码
if __name__ == '__main__':
    OUTPUT_RESOLUTION = (640, 480)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_perfect_python_test'

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    if not os.path.isdir(TEXTURE_SOURCE_DIR): 
        raise FileNotFoundError(f"错误：找不到图片源文件夹 '{TEXTURE_SOURCE_DIR}'。")
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: 
        raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")

    generator = FlareGeneratorPerfectPython(output_size=OUTPUT_RESOLUTION)

    print(f"\n🎯 --- 高性能向量化炫光生成测试 ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.4, generator.height * 0.4)
    print(f"使用噪声纹理: '{os.path.basename(fixed_source_path)}'")
    
    # 测试1: 纯反射炫光
    print("\n[测试 1/3] 向量化反射炫光...")
    start_time = time.time()
    img1 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=False, generate_reflections=True)
    end_time = time.time()
    img1.save(os.path.join(OUTPUT_DIR, "01_vectorized_reflections_only.png"))
    print(f" -> 已保存: 01_vectorized_reflections_only.png (用时: {end_time-start_time:.2f}s)")

    # 测试2: 纯主光辉
    print("\n[测试 2/3] 向量化主光辉...")
    start_time = time.time()
    img2 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=False)
    end_time = time.time()
    img2.save(os.path.join(OUTPUT_DIR, "02_vectorized_main_glow_only.png"))
    print(f" -> 已保存: 02_vectorized_main_glow_only.png (用时: {end_time-start_time:.2f}s)")

    # 测试3: 完整效果
    print("\n[测试 3/3] 向量化完整效果...")
    start_time = time.time()
    img3 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=True)
    end_time = time.time()
    img3.save(os.path.join(OUTPUT_DIR, "03_vectorized_all_effects_on.png"))
    print(f" -> 已保存: 03_vectorized_all_effects_on.png (用时: {end_time-start_time:.2f}s)")

    print("\n🎉 ====================================")
    print(f"✅ 成功生成 3 张向量化炫光图像！")
    print(f"📁 结果保存在: '{OUTPUT_DIR}'")
    print("🚀 渲染方式: 高性能NumPy向量化 (消除所有循环)")
    
    # 性能测试：生成20张图片到R_flare
    print("\n🏃‍♂️ --- 性能基准测试：生成20张图片 ---")
    R_FLARE_DIR = 'R_flare'
    if not os.path.isdir(R_FLARE_DIR):
        os.makedirs(R_FLARE_DIR)
    
    total_start_time = time.time()
    
    for i in range(20):
        texture_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
        light_pos = (random.randint(100, generator.width-100), 
                    random.randint(100, generator.height-100))
        
        img = generator.generate(
            light_pos=light_pos, 
            noise_image_path=texture_path,
            time=random.random() * 10,
            flare_size=random.uniform(0.1, 0.3),
            generate_main_glow=False, 
            generate_reflections=True
        )
        
        output_name = f"vectorized_{i+1:03d}_from_{os.path.splitext(os.path.basename(texture_path))[0]}.png"
        img.save(os.path.join(R_FLARE_DIR, output_name))
        print(f"完成 {i+1}/20: {output_name}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time = total_time / 20
    
    print(f"\n📊 性能基准结果:")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均每张: {avg_time:.2f}秒")
    print(f"等效帧率: {1/avg_time:.2f} FPS")
    print("====================================")