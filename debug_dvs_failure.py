#!/usr/bin/env python3
"""
DVS模拟器失败根因分析脚本
专门诊断为什么DVS仿真会失败
"""

import os
import sys
import tempfile
import shutil
import subprocess
import yaml
import numpy as np
import cv2
import time
from pathlib import Path

# 添加src路径
sys.path.append('src')
from dvs_flare_integration import DVSFlareEventGenerator

def analyze_dvs_failures():
    """深度分析DVS模拟器失败的根本原因"""
    
    print("=== DVS模拟器失败根因分析 ===")
    
    # 1. 加载配置
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = DVSFlareEventGenerator(config)
    simulator_path = generator.simulator_path
    
    print(f"DVS模拟器路径: {simulator_path}")
    
    # 2. 检查当前配置状态
    config_path = os.path.join(simulator_path, "src/config.py")
    print(f"\n=== 当前配置文件检查 ===")
    
    with open(config_path, 'r') as f:
        current_config = f.read()
    
    # 提取路径
    in_path_line = None
    out_path_line = None
    for line in current_config.split('\n'):
        if "__C.DIR.IN_PATH" in line:
            in_path_line = line.strip()
        if "__C.DIR.OUT_PATH" in line:
            out_path_line = line.strip()
    
    print(f"当前IN_PATH: {in_path_line}")
    print(f"当前OUT_PATH: {out_path_line}")
    
    # 3. 创建多个临时目录测试
    print(f"\n=== 多次DVS调用测试 (连续5次) ===")
    
    success_count = 0
    failure_count = 0
    failure_details = []
    
    for test_num in range(5):
        print(f"\n--- 测试 {test_num + 1}/5 ---")
        
        # 创建新的临时目录
        temp_dir = tempfile.mkdtemp(prefix=f"dvs_test_{test_num}_")
        print(f"临时目录: {temp_dir}")
        
        try:
            # 创建测试序列
            sequence_dir = os.path.join(temp_dir, "flare_sequence")
            os.makedirs(sequence_dir, exist_ok=True)
            
            # 创建测试帧
            for i in range(3):
                test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                frame_path = os.path.join(sequence_dir, f"{i:03d}.png")
                cv2.imwrite(frame_path, test_image)
            
            # 创建info.txt
            info_path = os.path.join(sequence_dir, "info.txt")
            with open(info_path, 'w') as f:
                for i in range(3):
                    frame_path = os.path.join(sequence_dir, f"{i:03d}.png")
                    timestamp = i * 33333
                    f.write(f"{frame_path} {timestamp:012d}\n")
            
            print(f"✅ 创建测试数据完成")
            
            # 修改配置
            generator._prepare_simulator_config(temp_dir)
            
            # 检查配置修改后的状态
            with open(config_path, 'r') as f:
                modified_config = f.read()
            
            for line in modified_config.split('\n'):
                if "__C.DIR.IN_PATH" in line:
                    print(f"修改后IN_PATH: {line.strip()}")
                    configured_in_path = line.split('=')[1].strip().strip("'\"")
                if "__C.DIR.OUT_PATH" in line:
                    print(f"修改后OUT_PATH: {line.strip()}")
            
            # 验证目录存在性
            if os.path.exists(configured_in_path.rstrip('/')):
                print(f"✅ IN_PATH目录存在: {configured_in_path}")
            else:
                print(f"❌ IN_PATH目录不存在: {configured_in_path}")
            
            # 检查序列目录
            expected_sequence_path = os.path.join(configured_in_path.rstrip('/'), "flare_sequence")
            if os.path.exists(expected_sequence_path):
                print(f"✅ 序列目录存在: {expected_sequence_path}")
                files = os.listdir(expected_sequence_path)
                print(f"✅ 序列目录文件: {files}")
            else:
                print(f"❌ 序列目录不存在: {expected_sequence_path}")
            
            # 运行DVS模拟器
            print(f"🔄 运行DVS模拟器...")
            original_cwd = os.getcwd()
            os.chdir(simulator_path)
            
            start_time = time.time()
            result = subprocess.run([
                sys.executable, "main.py"
            ], capture_output=True, text=True, timeout=90)
            end_time = time.time()
            
            os.chdir(original_cwd)
            
            print(f"DVS运行时间: {end_time - start_time:.1f}秒")
            print(f"返回码: {result.returncode}")
            
            if result.returncode == 0:
                # 检查输出文件
                expected_output = os.path.join(temp_dir, "flare_sequence.txt")
                if os.path.exists(expected_output):
                    with open(expected_output, 'r') as f:
                        lines = f.readlines()
                    print(f"✅ 成功! 输出文件: {len(lines)} 行事件")
                    success_count += 1
                else:
                    print(f"❌ 失败: 输出文件不存在 {expected_output}")
                    failure_count += 1
                    failure_details.append(f"Test {test_num + 1}: 输出文件不存在")
            else:
                print(f"❌ 失败: 返回码 {result.returncode}")
                if result.stderr:
                    print(f"错误信息: {result.stderr[:200]}")
                failure_count += 1
                failure_details.append(f"Test {test_num + 1}: 返回码 {result.returncode}, 错误: {result.stderr[:100]}")
                
        except subprocess.TimeoutExpired:
            print(f"❌ 失败: 超时")
            failure_count += 1
            failure_details.append(f"Test {test_num + 1}: 超时")
            os.chdir(original_cwd)
            
        except Exception as e:
            print(f"❌ 失败: 异常 {e}")
            failure_count += 1
            failure_details.append(f"Test {test_num + 1}: 异常 {str(e)}")
            
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # 小延迟避免资源竞争
            time.sleep(0.5)
    
    # 4. 总结分析
    print(f"\n=== 测试结果统计 ===")
    print(f"成功次数: {success_count}/5 ({success_count/5*100:.1f}%)")
    print(f"失败次数: {failure_count}/5 ({failure_count/5*100:.1f}%)")
    
    if failure_details:
        print(f"\n=== 失败详情 ===")
        for detail in failure_details:
            print(f"  {detail}")
    
    # 5. 环境因素分析
    print(f"\n=== 环境因素分析 ===")
    
    # 检查/tmp目录权限
    tmp_dir = "/tmp"
    try:
        test_tmp = tempfile.mkdtemp(prefix="perm_test_")
        print(f"✅ /tmp 目录可写: {test_tmp}")
        shutil.rmtree(test_tmp)
    except Exception as e:
        print(f"❌ /tmp 目录权限问题: {e}")
    
    # 检查Python环境
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 检查DVS模拟器依赖
    try:
        os.chdir(simulator_path)
        import torch
        from easydict import EasyDict
        print(f"✅ DVS模拟器依赖正常")
        os.chdir(original_cwd)
    except Exception as e:
        print(f"❌ DVS模拟器依赖问题: {e}")
        os.chdir(original_cwd)
    
    # 6. 竞争条件分析
    print(f"\n=== 竞争条件分析 ===")
    
    if failure_count > 0:
        failure_rate = failure_count / 5
        if failure_rate <= 0.2:
            print("🔍 推测原因: 轻微竞争条件或资源争用")
        elif failure_rate <= 0.5:
            print("🔍 推测原因: 中等程度的系统资源问题")
        else:
            print("🔍 推测原因: 严重的配置或环境问题")
    else:
        print("✅ 无竞争条件，DVS模拟器完全稳定")
    
    return success_count, failure_count, failure_details

if __name__ == "__main__":
    try:
        analyze_dvs_failures()
    except Exception as e:
        print(f"分析过程出错: {e}")
        import traceback
        traceback.print_exc()