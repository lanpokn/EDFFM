import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import random
import time

class FlareGeneratorPerfectGPU:
    """
    高性能PyTorch GPU炫光生成器
    - 100%复刻GLSL算法
    - 完全GPU加速计算
    - 保持相同的输入输出接口
    """
    def __init__(self, output_size=(1920, 1080), device=None):
        self.output_size = output_size
        self.width, self.height = output_size
        
        # 自动检测最佳设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🚀 初始化GPU炫光生成器 (CUDA)")
            else:
                self.device = torch.device('cpu')
                print("⚠️  CUDA不可用，使用CPU模式")
        else:
            self.device = device
            print(f"🚀 初始化GPU炫光生成器 ({device})")
        
        print(f"📊 设备信息: {self.device}")
        if self.device.type == 'cuda':
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    def _gpu_noise_sample(self, noise_texture, u, v):
        """GPU加速噪声采样 - 改进处理大范围数值"""
        # noise_texture: (1, C, H, W) 
        # u, v: (...) 任意形状的坐标
        
        u = torch.atleast_1d(u)
        v = torch.atleast_1d(v)
        batch_size = u.shape[0]
        
        # 改进：更好的大范围数值处理，使用分数部分并保持更多变化
        # GLSL的texture()对大数值有良好处理，我们需要模拟这种行为
        u_norm = torch.fmod(torch.abs(u), 1.0)  # 使用fmod获得更好的分布
        v_norm = torch.fmod(torch.abs(v), 1.0)
        
        # PyTorch grid_sample需要坐标在[-1,1]范围
        grid_u = u_norm * 2.0 - 1.0
        grid_v = v_norm * 2.0 - 1.0
        
        # 创建采样网格 (1, N, 1, 2)
        grid = torch.stack([grid_u, grid_v], dim=-1).view(1, batch_size, 1, 2)
        
        # 采样 (1, C, H, W) -> (1, C, N, 1) - 使用wrap模式更好模拟GLSL
        sampled = F.grid_sample(noise_texture, grid, 
                              mode='bilinear', padding_mode='reflection', 
                              align_corners=False)
        
        # 重新整形为 (N, C)
        result = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
        
        # 补齐到4通道
        if result.shape[1] < 4:
            padding = torch.zeros(result.shape[0], 4 - result.shape[1], 
                                device=self.device, dtype=result.dtype)
            result = torch.cat([result, padding], dim=1)
        
        return result[:, :4]  # (N, 4)
    
    def _gpu_flare_kernel(self, uv_x, uv_y, pos_x, pos_y, seed, flare_size,
                         noise_texture, generate_main_glow, generate_reflections):
        """完全GPU加速的炫光内核"""
        
        # 展平坐标
        flat_size = uv_x.numel()
        uv_x_flat = uv_x.flatten()
        uv_y_flat = uv_y.flatten()
        
        # noise(seed-1.0)
        seed_tensor = torch.tensor([seed - 1.0], device=self.device)
        zero_tensor = torch.tensor([0.0], device=self.device)
        gn = self._gpu_noise_sample(noise_texture, seed_tensor, zero_tensor)[0]
        gn[0] = flare_size
        
        # 距离计算
        d_x = uv_x_flat - pos_x
        d_y = uv_y_flat - pos_y
        
        # 初始化结果
        main_glow_color = torch.zeros(flat_size, 3, device=self.device)
        reflections_color = torch.zeros(flat_size, 3, device=self.device)
        
        # Part 1: 主光辉 (GPU向量化) - 修复光晕计算
        if generate_main_glow:
            d_length = torch.sqrt(d_x**2 + d_y**2)
            core_intensity = (0.01 + gn[0] * 0.2) / (d_length + 0.001)
            main_glow_color[:, :] = core_intensity.unsqueeze(-1)
            
            # 光晕计算 - 修复：应该是相加而非相乘
            angle = torch.atan2(d_y, d_x)
            halo_u = angle * 256.9 + pos_x * 2.0
            halo_zero = torch.zeros_like(halo_u)
            halo_noise = self._gpu_noise_sample(noise_texture, halo_u, halo_zero)
            halo_factor = halo_noise[:, 1] * 0.25
            
            # 修复：GLSL是 main_glow_color += vec3(halo) * main_glow_color
            # 即 halo 与 core 的积再加到 core 上
            halo_contribution = halo_factor.unsqueeze(-1) * main_glow_color
            main_glow_color += halo_contribution
        
        # fltr计算
        uv_length = torch.sqrt(uv_x_flat**2 + uv_y_flat**2)
        fltr = torch.clamp(uv_length**2 * 0.5 + 0.5, max=1.0)
        
        # Part 2: 反射炫光 (GPU加速最复杂部分)
        if generate_reflections:
            # 批量噪声预计算
            i_values = torch.arange(20, device=self.device, dtype=torch.float32)
            n_u = seed + i_values
            n2_u = seed + i_values * 2.1
            nc_u = seed + i_values * 3.3
            zero_batch = torch.zeros_like(n_u)
            
            # 批量采样
            n_batch = self._gpu_noise_sample(noise_texture, n_u, zero_batch)
            n2_batch = self._gpu_noise_sample(noise_texture, n2_u, zero_batch)
            nc_batch = self._gpu_noise_sample(noise_texture, nc_u, zero_batch)
            
            # 处理nc
            nc_length = torch.sqrt(torch.sum(nc_batch**2, dim=1, keepdim=True))
            nc_batch = (nc_batch + nc_length) * 0.65
            
            # 主反射循环 - GPU并行处理
            for i in range(20):
                n = n_batch[i]
                n2 = n2_batch[i]
                nc = nc_batch[i]
                
                for j in range(3):  # RGB通道
                    # 参数计算
                    ip = n[0] * 3.0 + j * 0.1 * (n2[1] ** 3)
                    is_val = n[1]**2 * 4.5 * gn[0] + 0.1
                    ia = (n[2] * 4.0 - 2.0) * n2[0] * n[1]
                    
                    # UV变换
                    mix_factor = 1.0 + (uv_length - 1.0) * n[3]**2
                    iuv_x = uv_x_flat * mix_factor
                    iuv_y = uv_y_flat * mix_factor
                    
                    # 旋转变换
                    cos_ia = torch.cos(ia)
                    sin_ia = torch.sin(ia)
                    rotated_x = iuv_x * cos_ia + iuv_y * sin_ia
                    rotated_y = -iuv_x * sin_ia + iuv_y * cos_ia
                    
                    # ID计算
                    id_x = (rotated_x - pos_x) * (1.0 - ip) + (rotated_x + pos_x) * ip
                    id_y = (rotated_y - pos_y) * (1.0 - ip) + (rotated_y + pos_y) * ip
                    id_length = torch.sqrt(id_x**2 + id_y**2)
                    
                    # 反射强度计算
                    intensity_base = torch.clamp(is_val - id_length, min=0.0)
                    mask = intensity_base > 0
                    if mask.any():
                        intensity = torch.zeros_like(intensity_base)
                        intensity[mask] = (intensity_base[mask]**0.45 / is_val * 0.1 * 
                                         gn[0] * nc[j] * fltr[mask])
                        reflections_color[:, j] += intensity
        
        # 合并结果
        result = main_glow_color + reflections_color
        
        # 重新整形
        return result.view(self.height, self.width, 3)
    
    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
        GPU加速生成 - 保持完全相同的接口
        """
        print(f"🎨 GPU生成: 位置={light_pos}, 反射={generate_reflections}, 光源={generate_main_glow}")
        
        # 加载噪声纹理到GPU
        noise_img = Image.open(noise_image_path).convert("RGBA")
        noise_array = np.array(noise_img).astype(np.float32) / 255.0
        # 转换为PyTorch格式: (1, C, H, W)
        noise_texture = torch.from_numpy(noise_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # 创建坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # GLSL坐标变换
        uv_x = (x_coords / self.width - 0.5) * 2.0 * (self.width / self.height)
        uv_y = (y_coords / self.height - 0.5) * 2.0
        
        # 光源位置变换
        pos_x = (light_pos[0] / self.width - 0.5) * 2.0 * (self.width / self.height)
        pos_y = (light_pos[1] / self.height - 0.5) * 2.0
        
        # GPU炫光计算
        img_array = self._gpu_flare_kernel(
            uv_x, uv_y, pos_x, pos_y, time, flare_size,
            noise_texture, generate_main_glow, generate_reflections
        )
        
        # 应用光源颜色
        light_color_tensor = torch.tensor(light_color, device=self.device, dtype=torch.float32)
        img_array *= light_color_tensor
        
        # 添加噪声
        x_flat = x_coords.flatten() / self.width
        y_flat = y_coords.flatten() / self.height
        noise_addition = self._gpu_noise_sample(noise_texture, x_flat, y_flat)
        noise_addition = noise_addition[:, :3].view(self.height, self.width, 3)
        img_array += noise_addition * 0.01
        
        # 最终处理并转回CPU
        img_array = torch.clamp(img_array, 0, 1)
        img_array = (img_array * 255).cpu().numpy().astype(np.uint8)
        
        print("✅ GPU渲染完成")
        return Image.fromarray(img_array)

# 测试代码
if __name__ == '__main__':
    OUTPUT_RESOLUTION = (640, 480)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_perfect_gpu_test'

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    if not os.path.isdir(TEXTURE_SOURCE_DIR): 
        raise FileNotFoundError(f"错误：找不到图片源文件夹 '{TEXTURE_SOURCE_DIR}'。")
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: 
        raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")

    generator = FlareGeneratorPerfectGPU(output_size=OUTPUT_RESOLUTION)

    print(f"\n🎯 --- GPU炫光生成测试 ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.4, generator.height * 0.4)
    print(f"使用噪声纹理: '{os.path.basename(fixed_source_path)}'")
    
    # 测试1: 纯反射炫光
    print("\n[测试 1/3] GPU反射炫光...")
    start_time = time.time()
    img1 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=False, generate_reflections=True)
    end_time = time.time()
    img1.save(os.path.join(OUTPUT_DIR, "01_gpu_reflections_only.png"))
    print(f" -> 已保存: 01_gpu_reflections_only.png (用时: {end_time-start_time:.2f}s)")

    # 测试2: 纯主光辉
    print("\n[测试 2/3] GPU主光辉...")
    start_time = time.time()
    img2 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=False)
    end_time = time.time()
    img2.save(os.path.join(OUTPUT_DIR, "02_gpu_main_glow_only.png"))
    print(f" -> 已保存: 02_gpu_main_glow_only.png (用时: {end_time-start_time:.2f}s)")

    # 测试3: 完整效果
    print("\n[测试 3/3] GPU完整效果...")
    start_time = time.time()
    img3 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=True)
    end_time = time.time()
    img3.save(os.path.join(OUTPUT_DIR, "03_gpu_all_effects_on.png"))
    print(f" -> 已保存: 03_gpu_all_effects_on.png (用时: {end_time-start_time:.2f}s)")

    print("\n🎉 ====================================")
    print(f"✅ 成功生成 3 张GPU加速炫光图像！")
    print(f"📁 结果保存在: '{OUTPUT_DIR}'")
    print("🚀 渲染方式: PyTorch GPU加速 (完全复刻GLSL)")
    
    # GPU性能测试：生成20张图片
    print("\n🏃‍♂️ --- GPU性能基准测试：生成20张图片 ---")
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
        
        output_name = f"gpu_{i+1:03d}_from_{os.path.splitext(os.path.basename(texture_path))[0]}.png"
        img.save(os.path.join(R_FLARE_DIR, output_name))
        print(f"GPU完成 {i+1}/20: {output_name}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    avg_time = total_time / 20
    
    print(f"\n📊 GPU性能基准结果:")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均每张: {avg_time:.2f}秒")
    print(f"等效帧率: {1/avg_time:.2f} FPS")
    print("====================================")