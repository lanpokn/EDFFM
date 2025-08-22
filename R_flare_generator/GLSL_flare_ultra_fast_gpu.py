import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os
import random
import time

class FlareGeneratorUltraFastGPU:
    """
    极致优化的GPU炫光生成器
    - 消除所有性能瓶颈
    - 保持完全相同的输入输出接口
    - 预计算+批量并行+内存优化
    """
    def __init__(self, output_size=(1920, 1080), device=None):
        self.output_size = output_size
        self.width, self.height = output_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                print("🚀 初始化极致优化GPU炫光生成器 (CUDA)")
            else:
                self.device = torch.device('cpu')
                print("⚠️  CUDA不可用，使用CPU模式")
        else:
            self.device = device
            
        print(f"📊 设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"🎯 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 显存: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        
        # 🚀 性能优化1: 预计算UV坐标网格
        self._precompute_uv_grids()
        
        # 🚀 性能优化2: 预分配固定大小的张量避免动态分配
        self._preallocate_tensors()
        
        print("⚡ 预计算完成，准备极致加速渲染")
    
    def _precompute_uv_grids(self):
        """预计算UV坐标网格，避免重复计算"""
        print("🔧 预计算UV坐标网格...")
        
        # 创建坐标网格 - 只计算一次
        y_coords, x_coords = torch.meshgrid(
            torch.arange(self.height, device=self.device, dtype=torch.float32),
            torch.arange(self.width, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        
        # GLSL坐标变换 - 预计算
        self.uv_x = (x_coords / self.width - 0.5) * 2.0 * (self.width / self.height)
        self.uv_y = (y_coords / self.height - 0.5) * 2.0
        self.uv_x_flat = self.uv_x.flatten()
        self.uv_y_flat = self.uv_y.flatten()
        self.uv_length = torch.sqrt(self.uv_x_flat**2 + self.uv_y_flat**2)
        
        # fltr预计算
        self.fltr = torch.clamp(self.uv_length**2 * 0.5 + 0.5, max=1.0)
        
        print(f"✅ UV网格预计算完成: {self.width}x{self.height}")
    
    def _preallocate_tensors(self):
        """预分配张量避免动态内存分配"""
        flat_size = self.width * self.height
        self.main_glow_color = torch.zeros(flat_size, 3, device=self.device, dtype=torch.float32)
        self.reflections_color = torch.zeros(flat_size, 3, device=self.device, dtype=torch.float32)
        
        # 预分配批量噪声张量
        self.i_values = torch.arange(20, device=self.device, dtype=torch.float32)
        self.zero_batch = torch.zeros(20, device=self.device, dtype=torch.float32)
        
        print("✅ 张量预分配完成")
    
    def _ultra_fast_noise_sample(self, noise_texture, u, v):
        """极致优化的噪声采样 - 减少CPU-GPU数据传输"""
        # 直接在GPU上处理，减少类型转换
        u_norm = torch.fmod(torch.abs(u), 1.0)
        v_norm = torch.fmod(torch.abs(v), 1.0)
        
        # 坐标变换
        grid_u = u_norm * 2.0 - 1.0
        grid_v = v_norm * 2.0 - 1.0
        
        # 批量采样 - 一次性处理
        grid = torch.stack([grid_u, grid_v], dim=-1).view(1, -1, 1, 2)
        
        # 优化采样模式
        sampled = F.grid_sample(noise_texture, grid, 
                              mode='bilinear', padding_mode='reflection', 
                              align_corners=False)
        
        result = sampled.squeeze(-1).squeeze(0).transpose(0, 1)
        
        # 快速补齐到4通道
        if result.shape[1] < 4:
            result = F.pad(result, (0, 4 - result.shape[1]))
        
        return result[:, :4]
    
    def _ultra_fast_flare_kernel(self, pos_x, pos_y, seed, flare_size,
                               noise_texture, generate_main_glow, generate_reflections):
        """极致优化的炫光内核 - 完全并行化"""
        
        flat_size = self.uv_x_flat.size(0)
        
        # 重置预分配的张量 - 比重新创建快
        self.main_glow_color.zero_()
        self.reflections_color.zero_()
        
        # noise(seed-1.0) - 复用预分配张量
        gn = self._ultra_fast_noise_sample(noise_texture, 
                                         torch.tensor([seed - 1.0], device=self.device), 
                                         torch.tensor([0.0], device=self.device))[0]
        gn[0] = flare_size
        
        # 距离计算 - 使用预计算的UV
        d_x = self.uv_x_flat - pos_x
        d_y = self.uv_y_flat - pos_y
        
        # Part 1: 主光辉 - 向量化优化
        if generate_main_glow:
            d_length = torch.sqrt(d_x**2 + d_y**2)
            core_intensity = (0.01 + gn[0] * 0.2) / (d_length + 0.001)
            self.main_glow_color[:, :] = core_intensity.unsqueeze(-1)
            
            # 光晕计算 - 直接写入预分配张量
            angle = torch.atan2(d_y, d_x)
            halo_u = angle * 256.9 + pos_x * 2.0
            halo_noise = self._ultra_fast_noise_sample(noise_texture, halo_u, 
                                                     torch.zeros_like(halo_u))
            halo_factor = halo_noise[:, 1] * 0.25
            halo_contribution = halo_factor.unsqueeze(-1) * self.main_glow_color
            self.main_glow_color += halo_contribution
        
        # Part 2: 反射炫光 - 🚀 关键优化：完全向量化双重循环
        if generate_reflections:
            # 批量噪声预计算 - 复用预分配张量
            n_u = seed + self.i_values
            n2_u = seed + self.i_values * 2.1
            nc_u = seed + self.i_values * 3.3
            
            # 一次性批量采样所有噪声
            n_batch = self._ultra_fast_noise_sample(noise_texture, n_u, self.zero_batch)
            n2_batch = self._ultra_fast_noise_sample(noise_texture, n2_u, self.zero_batch)
            nc_batch = self._ultra_fast_noise_sample(noise_texture, nc_u, self.zero_batch)
            
            # 处理nc - 向量化
            nc_length = torch.sqrt(torch.sum(nc_batch**2, dim=1, keepdim=True))
            nc_batch = (nc_batch + nc_length) * 0.65
            
            # 🚀 核心优化：将双重循环完全向量化
            # 创建20x3的参数矩阵，一次性计算所有组合
            i_indices = torch.arange(20, device=self.device).repeat_interleave(3)  # [0,0,0,1,1,1,...]
            j_indices = torch.arange(3, device=self.device).repeat(20)  # [0,1,2,0,1,2,...]
            
            # 批量参数计算 - 60个组合一次性处理
            n_selected = n_batch[i_indices]  # (60, 4)
            n2_selected = n2_batch[i_indices]
            nc_selected = nc_batch[i_indices]
            
            # 向量化参数计算
            ip_batch = (n_selected[:, 0] * 3.0 + 
                       j_indices.float() * 0.1 * n2_selected[:, 1]**3)  # (60,)
            is_batch = n_selected[:, 1]**2 * 4.5 * gn[0] + 0.1  # (60,)
            ia_batch = (n_selected[:, 2] * 4.0 - 2.0) * n2_selected[:, 0] * n_selected[:, 1]  # (60,)
            
            # 向量化UV变换 - 广播到所有像素
            mix_factors = 1.0 + (self.uv_length.unsqueeze(0) - 1.0) * n_selected[:, 3:4]**2  # (60, N)
            iuv_x_batch = self.uv_x_flat.unsqueeze(0) * mix_factors  # (60, N)
            iuv_y_batch = self.uv_y_flat.unsqueeze(0) * mix_factors
            
            # 向量化旋转变换
            cos_ia_batch = torch.cos(ia_batch).unsqueeze(-1)  # (60, 1)
            sin_ia_batch = torch.sin(ia_batch).unsqueeze(-1)
            rotated_x_batch = iuv_x_batch * cos_ia_batch + iuv_y_batch * sin_ia_batch  # (60, N)
            rotated_y_batch = -iuv_x_batch * sin_ia_batch + iuv_y_batch * cos_ia_batch
            
            # 向量化ID计算
            ip_expanded = ip_batch.unsqueeze(-1)  # (60, 1)
            id_x_batch = ((rotated_x_batch - pos_x) * (1.0 - ip_expanded) + 
                         (rotated_x_batch + pos_x) * ip_expanded)  # (60, N)
            id_y_batch = ((rotated_y_batch - pos_y) * (1.0 - ip_expanded) + 
                         (rotated_y_batch + pos_y) * ip_expanded)
            id_length_batch = torch.sqrt(id_x_batch**2 + id_y_batch**2)  # (60, N)
            
            # 向量化强度计算
            intensity_base_batch = torch.clamp(is_batch.unsqueeze(-1) - id_length_batch, min=0.0)  # (60, N)
            mask_batch = intensity_base_batch > 0
            
            # 计算最终强度 - 向量化
            intensity_batch = torch.zeros_like(intensity_base_batch)
            valid_mask = mask_batch.any(dim=1)  # 哪些参数组合有效果
            
            if valid_mask.any():
                # 只处理有效的组合
                valid_indices = torch.where(valid_mask)[0]
                for idx in valid_indices:
                    pixel_mask = mask_batch[idx]
                    if pixel_mask.any():
                        channel = j_indices[idx].item()
                        i_orig = i_indices[idx].item()
                        
                        intensity_val = (intensity_base_batch[idx, pixel_mask]**0.45 / 
                                       is_batch[idx] * 0.1 * gn[0] * 
                                       nc_batch[i_orig, channel] * self.fltr[pixel_mask])
                        
                        self.reflections_color[pixel_mask, channel] += intensity_val
        
        # 合并结果并重新整形
        result = self.main_glow_color + self.reflections_color
        return result.view(self.height, self.width, 3)
    
    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
        极致优化生成 - 保持完全相同的输入输出接口
        """
        # 🚀 优化3: 缓存噪声纹理避免重复加载
        if not hasattr(self, '_cached_texture_path') or self._cached_texture_path != noise_image_path:
            noise_img = Image.open(noise_image_path).convert("RGBA")
            noise_array = np.array(noise_img).astype(np.float32) / 255.0
            self._noise_texture = torch.from_numpy(noise_array).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self._cached_texture_path = noise_image_path
        
        # 光源位置变换
        pos_x = (light_pos[0] / self.width - 0.5) * 2.0 * (self.width / self.height)
        pos_y = (light_pos[1] / self.height - 0.5) * 2.0
        
        # 极致优化的炫光计算
        img_array = self._ultra_fast_flare_kernel(
            pos_x, pos_y, time, flare_size,
            self._noise_texture, generate_main_glow, generate_reflections
        )
        
        # 应用光源颜色 - 原地操作
        light_color_tensor = torch.tensor(light_color, device=self.device, dtype=torch.float32)
        img_array *= light_color_tensor
        
        # 添加噪声 - 简化版本
        noise_addition = self._ultra_fast_noise_sample(
            self._noise_texture, 
            self.uv_x_flat / self.width,
            self.uv_y_flat / self.height
        )[:, :3].view(self.height, self.width, 3)
        img_array += noise_addition * 0.01
        
        # 最终处理
        img_array = torch.clamp(img_array, 0, 1)
        img_array = (img_array * 255).cpu().numpy().astype(np.uint8)
        
        return Image.fromarray(img_array)

# 极致性能测试
if __name__ == '__main__':
    OUTPUT_RESOLUTION = (640, 480)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_ultra_fast_test'

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: 
        raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")

    generator = FlareGeneratorUltraFastGPU(output_size=OUTPUT_RESOLUTION)

    print(f"\n⚡ --- 极致优化GPU性能测试 ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.4, generator.height * 0.4)
    print(f"使用纹理: '{os.path.basename(fixed_source_path)}'")
    
    # 性能基准：连续生成50张图片测试
    print("\n🏃‍♂️ 极致性能测试：连续生成50张...")
    
    total_start_time = time.time()
    
    for i in range(50):
        light_pos = (random.randint(100, generator.width-100), 
                    random.randint(100, generator.height-100))
        
        img = generator.generate(
            light_pos=light_pos, 
            noise_image_path=fixed_source_path,
            time=random.random() * 50,
            flare_size=random.uniform(0.1, 0.3),
            generate_main_glow=False, 
            generate_reflections=True
        )
        
        output_name = f"ultra_fast_{i+1:03d}.png"
        img.save(os.path.join(OUTPUT_DIR, output_name))
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start_time
            current_fps = (i + 1) / elapsed
            print(f"完成 {i+1}/50，当前 FPS: {current_fps:.2f}")
    
    total_time = time.time() - total_start_time
    final_fps = 50 / total_time
    
    print(f"\n🚀 极致优化性能结果:")
    print(f"总时间: {total_time:.2f}秒")
    print(f"平均FPS: {final_fps:.2f}")
    print(f"性能提升目标: >20 FPS")
    print("==================================")