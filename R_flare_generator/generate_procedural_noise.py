import os
import numpy as np
from PIL import Image
from perlin_noise import PerlinNoise
import random

# --- 配置 ---
TOTAL_IMAGES = 50
TARGET_DIR = "noise_textures"
IMAGE_SIZE = 512

# --- 参数范围，用于生成多样化的噪声 ---
# Perlin噪声的复杂度范围 (octaves)
PERLIN_OCTAVES_MIN = 2
PERLIN_OCTAVES_MAX = 32

# Voronoi噪声的细胞点数范围
VORONOI_POINTS_MIN = 10
VORONOI_POINTS_MAX = 250

# --- 核心生成函数 (保持不变) ---

def generate_perlin_image(filename, octaves, seed):
    """使用Perlin噪声算法生成一张灰度图"""
    noise = PerlinNoise(octaves=octaves, seed=seed)
    pic = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    
    # 这里我们只在函数入口打印一次，避免在循环中刷屏
    print(f"-> 正在生成 Perlin 噪声: {filename}")
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            noise_val = noise([i / IMAGE_SIZE, j / IMAGE_SIZE])
            pic[i][j] = (noise_val + 0.5) * 255

    img = Image.fromarray(pic).convert('L')
    img.save(os.path.join(TARGET_DIR, filename))

def generate_voronoi_like_image(filename, num_points, seed):
    """生成一个简单的、类似Voronoi图的图像"""
    np.random.seed(seed)
    points = np.random.randint(0, IMAGE_SIZE, (num_points, 2))
    pic = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    
    print(f"-> 正在生成 Voronoi 噪声: {filename}")
    for i in range(IMAGE_SIZE):
        for j in range(IMAGE_SIZE):
            distances = np.sqrt(np.sum((points - [i, j])**2, axis=1))
            min_dist = np.min(distances)
            pic[i][j] = min(255, min_dist * 2.5)

    img = Image.fromarray(pic).convert('L')
    img.save(os.path.join(TARGET_DIR, filename))


# --- 主程序：批量生成300个噪声 ---

if __name__ == "__main__":
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        print(f"已创建文件夹: '{TARGET_DIR}'")

    print(f"\n准备开始批量生成 {TOTAL_IMAGES} 个程序化噪声图片...")
    print("这个过程可能需要几分钟，请耐心等待。")
    
    # 我们将生成一半Perlin噪声，一半Voronoi噪声
    num_perlin = TOTAL_IMAGES // 2
    num_voronoi = TOTAL_IMAGES - num_perlin

    # --- 批量生成Perlin噪声 ---
    print(f"\n--- 正在生成 {num_perlin} 个 Perlin 噪声图像 ---")
    for i in range(1, num_perlin + 1):
        # 随机化参数
        seed = i
        octaves = random.randint(PERLIN_OCTAVES_MIN, PERLIN_OCTAVES_MAX)
        
        # 创建唯一的文件名
        filename = f"perlin_{i:03d}_oct{octaves}_seed{seed}.png"
        
        generate_perlin_image(filename, octaves=octaves, seed=seed)

    # --- 批量生成Voronoi噪声 ---
    print(f"\n--- 正在生成 {num_voronoi} 个 Voronoi 噪声图像 ---")
    for i in range(1, num_voronoi + 1):
        # 随机化参数
        seed = i + num_perlin # 确保seed不重复
        num_points = random.randint(VORONOI_POINTS_MIN, VORONOI_POINTS_MAX)
        
        # 创建唯一的文件名
        filename = f"voronoi_{i:03d}_pts{num_points}_seed{seed}.png"

        generate_voronoi_like_image(filename, num_points=num_points, seed=seed)
    
    print("\n--------------------")
    print(f"成功生成 {TOTAL_IMAGES} 个程序化噪声！")
    print(f"您现在可以将真实照片也添加到 '{TARGET_DIR}' 文件夹中，")
    print("然后运行 GLSL_flare.py 来生成炫光图像。")
    print("--------------------")