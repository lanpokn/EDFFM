import os
import requests
import time

TARGET_DIR = "noise_textures"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

# --- 精选自Poly Haven的高质量、多样化纹理 (CC0 许可) ---
# 我们选择1K分辨率的PNG格式漫反射贴图，作为高质量的信息源
POLY_HAVEN_TEXTURES = {
    "abstract_plaster.png": "https://dl.polyhaven.com/file/textures/plaster_rough_43d/png/1k/plaster_rough_43d_diff_1k.png",
    "fabric_carpet.png": "https://dl.polyhaven.com/file/textures/carpet_persian_2/png/1k/carpet_persian_2_diff_1k.png",
    "nature_mud_cracked.png": "https://dl.polyhaven.com/file/textures/cracked_mud_01/png/1k/cracked_mud_01_diff_1k.png",
    "nature_bark.png": "https://dl.polyhaven.com/file/textures/bark_brown_02/png/1k/bark_brown_02_diff_1k.png",
    "metal_painted.png": "https://dl.polyhaven.com/file/textures/painted_metal_shoddy/png/1k/painted_metal_shoddy_diff_1k.png",
    "abstract_voronoi.png": "https://dl.polyhaven.com/file/textures/voronoi_plaster/png/1k/voronoi_plaster_diff_1k.png",
    "nature_rocks.png": "https://dl.polyhaven.com/file/textures/rock_pitted_mossy/png/1k/rock_pitted_mossy_diff_1k.png",
    "fabric_denim.png": "https://dl.polyhaven.com/file/textures/denim_fabric/png/1k/denim_fabric_diff_1k.png",
    "nature_sand.png": "https://dl.polyhaven.com/file/textures/sand_01/png/1k/sand_01_diff_1k.png",
    "wood_planks.png": "https://dl.polyhaven.com/file/textures/wood_planks_weathered/png/1k/wood_planks_weathered_diff_1k.png",
    "abstract_circuit.png": "https://dl.polyhaven.com/file/textures/circuit_board/png/1k/circuit_board_diff_1k.png",
    "fabric_wool.png": "https://dl.polyhaven.com/file/textures/wool_big_stitch/png/1k/wool_big_stitch_diff_1k.png",
}

def download_textures(textures, directory):
    if not os.path.exists(directory):
        print(f"目标文件夹 '{directory}' 不存在，正在创建...")
        os.makedirs(directory)
    
    print(f"\n开始从 Poly Haven 下载 {len(textures)} 个纹理到 '{directory}' 文件夹...")
    
    success_count = 0
    skipped_count = 0
    total_size = 0
    
    for filename, url in textures.items():
        file_path = os.path.join(directory, filename)
        
        if os.path.exists(file_path):
            print(f"-> 跳过: '{filename}' 已存在。")
            skipped_count += 1
            continue
        
        try:
            print(f"-> 正在下载: '{filename}' ...", end='', flush=True)
            response = requests.get(url, headers=HEADERS, timeout=60, stream=True)
            response.raise_for_status()

            file_size = 0
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    file_size += len(chunk)
            
            total_size += file_size
            print(f" 完成 ({file_size / 1024:.1f} KB)")
            success_count += 1
            time.sleep(0.5)
            
        except requests.exceptions.RequestException as e:
            print(f"\n !! 错误: 下载 '{filename}' 失败。原因: {e}")
            
    total_mb = total_size / (1024 * 1024)
    print("\n--------------------")
    print("下载任务完成！")
    print(f"成功下载: {success_count} 个文件 (共 {total_mb:.2f} MB)")
    print(f"跳过已有: {skipped_count} 个文件")
    print("--------------------")

if __name__ == "__main__":
    download_textures(POLY_HAVEN_TEXTURES, TARGET_DIR)