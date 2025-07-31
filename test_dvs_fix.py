#!/usr/bin/env python3
"""
DVS模拟器问题诊断脚本
测试临时目录创建和配置修改是否正确
"""

import os
import tempfile
import shutil
import sys
import yaml

# 添加src路径
sys.path.append('src')
from dvs_flare_integration import DVSFlareEventGenerator

def test_dvs_simulator_setup():
    """测试DVS模拟器设置过程"""
    
    # 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("=== DVS模拟器设置测试 ===")
    
    # 初始化生成器
    generator = DVSFlareEventGenerator(config)
    print(f"✅ 模拟器路径: {generator.simulator_path}")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="flare_events_test_")
    print(f"✅ 创建临时目录: {temp_dir}")
    
    try:
        # 创建测试视频序列目录
        sequence_dir = os.path.join(temp_dir, "flare_sequence")
        os.makedirs(sequence_dir, exist_ok=True)
        print(f"✅ 创建序列目录: {sequence_dir}")
        
        # 创建测试帧和info.txt
        import numpy as np
        import cv2
        
        # 创建3个测试帧
        for i in range(3):
            # 创建640x480的测试图像
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_path = os.path.join(sequence_dir, f"{i:03d}.png")
            cv2.imwrite(frame_path, test_image)
        
        # 创建info.txt
        info_path = os.path.join(sequence_dir, "info.txt")
        with open(info_path, 'w') as f:
            for i in range(3):
                frame_path = os.path.join(sequence_dir, f"{i:03d}.png")
                timestamp = i * 33333  # 30fps
                f.write(f"{frame_path} {timestamp:012d}\n")
        
        print(f"✅ 创建测试数据: 3帧 + info.txt")
        
        # 测试配置修改
        print("\n=== 测试配置修改 ===")
        config_path = os.path.join(generator.simulator_path, "src/config.py")
        
        # 读取原始配置
        with open(config_path, 'r') as f:
            original_config = f.read()
        print("✅ 读取原始配置")
        
        # 修改配置
        generator._prepare_simulator_config(temp_dir)
        print("✅ 修改配置")
        
        # 检查修改后的配置
        with open(config_path, 'r') as f:
            modified_config = f.read()
        
        print(f"\n=== 配置对比 ===")
        if "__C.DIR.IN_PATH" in modified_config:
            # 提取IN_PATH行
            for line in modified_config.split('\n'):
                if "__C.DIR.IN_PATH" in line:
                    print(f"IN_PATH: {line.strip()}")
                if "__C.DIR.OUT_PATH" in line:
                    print(f"OUT_PATH: {line.strip()}")
        
        # 测试目录结构
        print(f"\n=== 目录结构测试 ===")
        print(f"临时目录内容:")
        for root, dirs, files in os.walk(temp_dir):
            level = root.replace(temp_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
        
        # 尝试运行DVS模拟器（限时测试）
        print(f"\n=== DVS模拟器运行测试 ===")
        try:
            # 切换到模拟器目录
            original_cwd = os.getcwd()
            os.chdir(generator.simulator_path)
            
            import subprocess
            result = subprocess.run([
                sys.executable, "main.py"
            ], capture_output=True, text=True, timeout=30)
            
            print(f"返回码: {result.returncode}")
            if result.stdout:
                print(f"标准输出: {result.stdout[:500]}")
            if result.stderr:
                print(f"标准错误: {result.stderr[:500]}")
                
            # 检查输出文件
            expected_output = os.path.join(temp_dir, "flare_sequence.txt")
            if os.path.exists(expected_output):
                print(f"✅ 找到输出文件: {expected_output}")
                with open(expected_output, 'r') as f:
                    lines = f.readlines()
                print(f"✅ 输出文件行数: {len(lines)}")
            else:
                print(f"❌ 未找到输出文件: {expected_output}")
                
            os.chdir(original_cwd)
            
        except subprocess.TimeoutExpired:
            print("❌ DVS模拟器运行超时")
            os.chdir(original_cwd)
        except Exception as e:
            print(f"❌ DVS模拟器运行错误: {e}")
            os.chdir(original_cwd)
        
        # 恢复配置
        with open(config_path, 'w') as f:
            f.write(original_config)
        print("✅ 恢复原始配置")
            
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"✅ 清理临时目录")

if __name__ == "__main__":
    test_dvs_simulator_setup()