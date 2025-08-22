import numpy as np
from PIL import Image
import os
import random
import numba
from concurrent.futures import ThreadPoolExecutor
import math

# JIT加速的全局函数 (Numba要求)
@numba.jit(nopython=True, fastmath=True)
def _noise_sample_jit(noise_texture, u, v):
    """JIT加速的噪声采样 - 完全复刻GLSL texture()"""
    h, w, c = noise_texture.shape
    
    # 归一化坐标到纹理尺寸
    x = u * w
    y = v * h
    
    # 整数部分用于索引
    x0 = int(x) % w
    y0 = int(y) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    
    # 小数部分用于插值
    fx = x - int(x)
    fy = y - int(y)
    
    # 双线性插值 - 完全复刻GLSL采样
    result = np.zeros(4, dtype=np.float32)
    for i in range(min(c, 4)):
        v00 = noise_texture[y0, x0, i]
        v10 = noise_texture[y0, x1, i]
        v01 = noise_texture[y1, x0, i]
        v11 = noise_texture[y1, x1, i]
        
        # 双线性插值
        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx
        result[i] = v0 * (1 - fy) + v1 * fy
    
    return result

@numba.jit(nopython=True, fastmath=True)
def _flare_kernel_jit(uv, pos, seed, flare_size, noise_texture, generate_main_glow, generate_reflections):
    """JIT加速的炫光内核 - 完全复刻GLSL flare()函数"""
    
    # noise(seed-1.0) - 复刻GLSL噪声调用
    gn = _noise_sample_jit(noise_texture, seed - 1.0, 0.0)
    gn[0] = flare_size  # gn.x = size
    
    p = pos
    d = uv - p
    
    # 独立的效果层 - 完全复刻GLSL结构
    main_glow_color = np.zeros(3, dtype=np.float32)
    reflections_color = np.zeros(3, dtype=np.float32)
    
    # Part 1: 主光辉 (光源核心 + 光晕) - 完全复刻
    if generate_main_glow:
        d_length = math.sqrt(d[0]*d[0] + d[1]*d[1])
        # 计算光源核心: (0.01+gn.x*.2)/(length(d)+0.001)
        core_intensity = (0.01 + gn[0] * 0.2) / (d_length + 0.001)
        main_glow_color[0] += core_intensity
        main_glow_color[1] += core_intensity  
        main_glow_color[2] += core_intensity
        
        # 光晕: vec3(noise(atan(d.y,d.x)*256.9+pos.x*2.0).y*.25) * main_glow_color
        angle = math.atan2(d[1], d[0])
        halo_u = angle * 256.9 + pos[0] * 2.0
        halo_noise = _noise_sample_jit(noise_texture, halo_u, 0.0)
        halo_factor = halo_noise[1] * 0.25
        
        main_glow_color[0] *= (1.0 + halo_factor)
        main_glow_color[1] *= (1.0 + halo_factor)
        main_glow_color[2] *= (1.0 + halo_factor)
    
    # fltr计算 - 完全复刻GLSL
    uv_length = math.sqrt(uv[0]*uv[0] + uv[1]*uv[1])
    fltr = uv_length * uv_length * 0.5 + 0.5
    fltr = min(fltr, 1.0)
    
    # Part 2: 反射炫光 - 完全复刻复杂的GLSL循环
    if generate_reflections:
        for i in range(20):  # for (float i=.0; i<20.; i++)
            # vec4 n = noise(seed+i); vec4 n2 = noise(seed+i*2.1); vec4 nc = noise(seed+i*3.3);
            n = _noise_sample_jit(noise_texture, seed + i, 0.0)
            n2 = _noise_sample_jit(noise_texture, seed + i * 2.1, 0.0)
            nc = _noise_sample_jit(noise_texture, seed + i * 3.3, 0.0)
            
            # nc+=vec4(length(nc)); nc*=.65;
            nc_length = math.sqrt(nc[0]*nc[0] + nc[1]*nc[1] + nc[2]*nc[2] + nc[3]*nc[3])
            nc[0] += nc_length
            nc[1] += nc_length
            nc[2] += nc_length
            nc[3] += nc_length
            nc[0] *= 0.65
            nc[1] *= 0.65
            nc[2] *= 0.65
            nc[3] *= 0.65
            
            for j in range(3):  # RGB通道
                # float ip = n.x*3.0+float(j)*.1*n2.y*n2.y*n2.y; 
                ip = n[0] * 3.0 + j * 0.1 * n2[1] * n2[1] * n2[1]
                # float is = n.y*n.y*4.5*gn.x+.1;
                is_val = n[1] * n[1] * 4.5 * gn[0] + 0.1
                # float ia = (n.z*4.0-2.0)*n2.x*n.y;
                ia = (n[2] * 4.0 - 2.0) * n2[0] * n[1]
                
                # vec2 iuv = (uv*(mix(1.0,length(uv),n.w*n.w)))*mat2(cos(ia),sin(ia),-sin(ia),cos(ia));
                mix_factor = 1.0 + (uv_length - 1.0) * n[3] * n[3]
                iuv_x = uv[0] * mix_factor
                iuv_y = uv[1] * mix_factor
                
                # 2D旋转矩阵应用
                cos_ia = math.cos(ia)
                sin_ia = math.sin(ia)
                rotated_x = iuv_x * cos_ia + iuv_y * sin_ia
                rotated_y = -iuv_x * sin_ia + iuv_y * cos_ia
                
                # vec2 id = mix(iuv-p,iuv+p,ip);
                id_x = (rotated_x - p[0]) * (1.0 - ip) + (rotated_x + p[0]) * ip
                id_y = (rotated_y - p[1]) * (1.0 - ip) + (rotated_y + p[1]) * ip
                id_length = math.sqrt(id_x*id_x + id_y*id_y)
                
                # 反射强度计算 - 完全复刻GLSL公式
                # pow(max(.0,is-(length(id))),.45)/is*.1*gn.x*nc[j]*fltr
                intensity_base = max(0.0, is_val - id_length)
                if intensity_base > 0:
                    intensity = math.pow(intensity_base, 0.45) / is_val * 0.1 * gn[0] * nc[j] * fltr
                    reflections_color[j] += intensity
    
    # 最后合并 - return main_glow_color + reflections_color;
    result = np.zeros(3, dtype=np.float32)
    result[0] = main_glow_color[0] + reflections_color[0]
    result[1] = main_glow_color[1] + reflections_color[1]  
    result[2] = main_glow_color[2] + reflections_color[2]
    
    return result

class FlareGeneratorPerfectCPU:
    """
    完全复刻GLSL效果的高性能CPU炫光生成器
    - 100%复刻原GLSL算法
    - Numba JIT加速
    - 多线程并行化
    - 完整的噪声纹理采样
    """
    def __init__(self, output_size=(1920, 1080)):
        self.output_size = output_size
        self.width, self.height = output_size
        print("🚀 初始化高性能CPU炫光生成器 (完全复刻GLSL)")
        
        # 预编译JIT函数
        self._warm_up_jit()
    
    def _warm_up_jit(self):
        """预热JIT编译器"""
        print("🔥 JIT预编译中...")
        # 小尺寸测试以触发编译
        test_noise = np.random.random((32, 32, 4)).astype(np.float32)
        _ = _noise_sample_jit(test_noise, 0.5, 0.5)
        _ = _flare_kernel_jit(
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32), 
            1.0, 0.15, test_noise, True, True
        )
        print("✅ JIT预编译完成")
    
    def _render_chunk(self, args):
        """并行渲染块"""
        y_start, y_end, x_start, x_end, light_pos, noise_array, time, flare_size, light_color, generate_main_glow, generate_reflections = args
        
        chunk_height = y_end - y_start
        chunk_width = x_end - x_start
        chunk = np.zeros((chunk_height, chunk_width, 3), dtype=np.float32)
        
        # 归一化光源位置 - 复刻GLSL坐标变换
        pos = np.array([
            (light_pos[0] / self.width - 0.5) * 2.0 * (self.width / self.height),
            (light_pos[1] / self.height - 0.5) * 2.0
        ], dtype=np.float32)
        
        # 逐像素渲染
        for y in range(chunk_height):
            for x in range(chunk_width):
                global_y = y + y_start
                global_x = x + x_start
                
                # UV坐标 - 完全复刻GLSL main()中的坐标变换
                uv = np.array([
                    (global_x / self.width - 0.5) * 2.0 * (self.width / self.height),
                    (global_y / self.height - 0.5) * 2.0
                ], dtype=np.float32)
                
                # 调用炫光内核
                color = _flare_kernel_jit(uv, pos, time, flare_size, noise_array, generate_main_glow, generate_reflections)
                
                # 应用光源颜色 - vec3 color = flare(...) * u_light_color;
                chunk[y, x, 0] = color[0] * light_color[0]
                chunk[y, x, 1] = color[1] * light_color[1]
                chunk[y, x, 2] = color[2] * light_color[2]
                
                # 添加噪声 - color += noise(gl_FragCoord.xy).xyz * 0.01;
                noise_sample = _noise_sample_jit(noise_array, global_x / self.width, global_y / self.height)
                chunk[y, x, 0] += noise_sample[0] * 0.01
                chunk[y, x, 1] += noise_sample[1] * 0.01
                chunk[y, x, 2] += noise_sample[2] * 0.01
        
        return y_start, y_end, x_start, x_end, chunk
    
    def generate(self, light_pos, noise_image_path, time=0.0, flare_size=0.15, light_color=(1.0, 1.0, 1.0),
                 generate_main_glow=False, generate_reflections=True):
        """
        完全复刻GLSL的高性能CPU生成
        """
        print(f"🎨 生成炫光: 位置={light_pos}, 反射={generate_reflections}, 光源={generate_main_glow}")
        
        # 加载并预处理噪声纹理 - 完全复刻GLSL纹理格式
        noise_img = Image.open(noise_image_path).convert("RGBA")
        noise_array = np.array(noise_img).astype(np.float32) / 255.0
        
        # 多线程并行渲染设置
        num_threads = os.cpu_count()
        chunk_size = max(64, self.height // (num_threads * 2))
        
        chunks_args = []
        for y in range(0, self.height, chunk_size):
            y_end = min(y + chunk_size, self.height)
            chunks_args.append((
                y, y_end, 0, self.width, 
                light_pos, noise_array, time, flare_size, light_color,
                generate_main_glow, generate_reflections
            ))
        
        print(f"⚡ 并行渲染 {len(chunks_args)} 个块 (使用 {num_threads} 线程)")
        
        # 并行渲染
        img_array = np.zeros((self.height, self.width, 3), dtype=np.float32)
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self._render_chunk, chunks_args))
        
        # 合并结果
        for y_start, y_end, x_start, x_end, chunk in results:
            img_array[y_start:y_end, x_start:x_end] = chunk
        
        # 最终处理 - 限制范围并转换
        img_array = np.clip(img_array, 0, 1)
        img_array = (img_array * 255).astype(np.uint8)
        
        print("✅ 渲染完成")
        return Image.fromarray(img_array)

# 测试代码
if __name__ == '__main__':
    OUTPUT_RESOLUTION = (640, 480)
    TEXTURE_SOURCE_DIR = 'noise_textures'
    OUTPUT_DIR = 'R_flare_perfect_cpu_test'

    if not os.path.isdir(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    if not os.path.isdir(TEXTURE_SOURCE_DIR): 
        raise FileNotFoundError(f"错误：找不到图片源文件夹 '{TEXTURE_SOURCE_DIR}'。")
    
    available_textures = [f for f in os.listdir(TEXTURE_SOURCE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not available_textures: 
        raise FileNotFoundError(f"错误：文件夹 '{TEXTURE_SOURCE_DIR}' 中没有任何图片文件。")

    generator = FlareGeneratorPerfectCPU(output_size=OUTPUT_RESOLUTION)

    print(f"\n🎯 --- 高性能CPU炫光生成测试 ---")
    
    fixed_source_path = os.path.join(TEXTURE_SOURCE_DIR, random.choice(available_textures))
    fixed_light_pos = (generator.width * 0.4, generator.height * 0.4)
    print(f"使用噪声纹理: '{os.path.basename(fixed_source_path)}'")
    
    # 测试1: 纯反射炫光 (重点!)
    print("\n[测试 1/3] 高性能CPU反射炫光...")
    img1 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=False, generate_reflections=True)
    img1.save(os.path.join(OUTPUT_DIR, "01_perfect_cpu_reflections_only.png"))
    print(" -> 已保存: 01_perfect_cpu_reflections_only.png")

    # 测试2: 纯主光辉
    print("\n[测试 2/3] 高性能CPU主光辉...")
    img2 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=False)
    img2.save(os.path.join(OUTPUT_DIR, "02_perfect_cpu_main_glow_only.png"))
    print(" -> 已保存: 02_perfect_cpu_main_glow_only.png")

    # 测试3: 完整效果
    print("\n[测试 3/3] 高性能CPU完整效果...")
    img3 = generator.generate(light_pos=fixed_light_pos, noise_image_path=fixed_source_path,
                              generate_main_glow=True, generate_reflections=True)
    img3.save(os.path.join(OUTPUT_DIR, "03_perfect_cpu_all_effects_on.png"))
    print(" -> 已保存: 03_perfect_cpu_all_effects_on.png")

    print("\n🎉 ====================================")
    print(f"✅ 成功生成 3 张完美复刻GLSL效果的炫光图像！")
    print(f"📁 结果保存在: '{OUTPUT_DIR}'")
    print("🚀 渲染方式: 高性能CPU (JIT+多线程+完全复刻GLSL)")
    print("====================================")