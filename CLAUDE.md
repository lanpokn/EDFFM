# EventMamba-FX Project Memory

## Project Overview
EventMamba-FX is a Feature-Augmented Mamba model for real-time event denoising and artifact removal from event camera streams. It combines physical insights from PFD (Polarity-Focused Denoising) with Mamba's sequence modeling capabilities.

**🔖 CORE INSPIRATION**: ext文件夹中的PFD论文《Polarity-Focused Denoising for Event Cameras》和C++代码是13维特征提取的核心灵感来源，提供了基于极性一致性和运动一致性的物理去噪原理。

## Environment Setup 🔧 CRITICAL
- **MUST USE**: `source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare`
- Environment already has all dependencies: PyTorch, Mamba SSM, etc.
- Python 3.10.18 with CUDA support

## 🚨 MEMORY SAFETY PROTOCOLS (CRITICAL)
- **NEVER change batch_size from 2**: Historical memory explosions caused terminal crashes
- **batch_size=2 is MANDATORY**: Verified safe configuration, tested up to 791MB max usage
- **sequence_length=64**: Fixed from previous value of 4 (too small for learning)
- **max_samples_debug=4**: Use for quick validation (debug模式限制样本数)
- **Memory monitoring**: Current safe range 400-800MB, warning at 1GB+

## Current System Status ✅ (Updated 2025-08-02)
- **Architecture**: Epoch-Iteration分离架构，先完整序列特征提取，再滑动窗口
- **Model Architecture**: 271,489 parameters, 11D PFD features, 3x3 neighborhoods
- **Data Pipeline**: 统一使用新架构，legacy代码已删除
- **Flicker Generation**: 优化帧数 (max_fps: 3000→1500, min_samples_per_cycle: 24→12)
- **Debug System**: Multi-resolution event visualization + 完整epoch分析
- **Memory Safety**: batch_size=2, sequence_length=64, max_samples_debug=4
- **Format Handling**: DVS [t,x,y,p] → Project [x,y,t,p] 自动转换

## 🚨 NEW: Epoch-Iteration Architecture (完全重构) ✅
**最新实现 (2025-08-02)**: 完全符合"先完整序列特征提取，再滑动窗口"的要求

### 核心架构对比
```
❌ LEGACY (已淘汰): 每次__getitem__重新生成数据
✅ NEW: Epoch-Iteration分离架构

EPOCH LEVEL (数据生成):
1. Generate Background Events (0.1-0.3s): Load DSEC random slice [N1, 4]
2. Generate Flare Events (0.1-0.3s): DVS simulation → [N2, 4] 
3. Merge & Sort: complete_events [N_total, 4] (时序完整)
4. ✅ CRITICAL: Feature Extraction ONCE on complete sequence
   - complete_events [N_total, 4] → long_feature_sequence [N_total, 11]
   - Generate labels [N_total] (0=background, 1=flare)
   - Calculate num_iterations = N_total - sequence_length + 1

ITERATION LEVEL (模型训练):
1. Sliding Window Sampling: features[start:start+64, :] from long_feature_sequence
2. Return: (feature_window [64, 11], label_window [64])
3. Batch Collation: [batch_size, sequence_length, 11]
4. Model Forward: Direct 11D feature processing
```

### 关键文件结构
- **核心实现**: `src/epoch_iteration_dataset.py` (新建)
- **数据集类**: `EpochIterationDataset` + `EpochIterationDataLoader`
- **配置控制**: `data_pipeline.use_epoch_iteration: true`
- **主程序**: `main.py` 自动选择架构
- **Legacy保留**: `src/mixed_flare_datasets.py` (向后兼容)

### DVS格式转换 (已验证正确)
```
DVS输出: [t, x, y, p] → 项目格式: [x, y, t, p]
极性转换: DVS (1/0) → DSEC (1/-1)
实现位置: _format_flare_events() 方法
```

## DVS-Voltmeter Physics Optimization (2025-07-31) 🎯
**基于ECCV 2022论文的深度物理分析，实现22,692x事件减少**:

### DVS物理模型 (Brownian Motion with Drift)
```
Eq.(10): ΔVd = (k1/(L+k2))·kdL·Δt + (k3/(L+k2))·√L·W(Δt) + k4·Δt + k5·L·Δt + k6·W(Δt)

物理参数意义:
- k1: 光-电转换敏感度 (核心参数，直接控制事件触发)
- k2: 暗电流阈值偏移 (分母项，增大可降低整体敏感度)
- k3: 光子噪声系数 (随机事件生成强度)
- k4: 温度漏电流 (背景事件基础率)
- k5: 寄生光电流 (亮度相关噪声)
- k6: 漏电流噪声 (随机噪声基础)
```

### 优化策略与结果 (修正版 2025-07-31)
```yaml
BEFORE (原始DVS346参数):
  dvs346_k: [1.0, 200, 0.001, 1e-8, 1e-9, 0.0001]
  事件密度: ~59,000 events/ms (过多)

EXTREME BUG (错误参数):
  dvs346_k: [2.5, 100, 0.01, 1e-7, 1e-8, 0.001]  
  事件密度: 2.6 events/ms (过低！有Bug)

FIXED (平衡优化参数):
  dvs346_k: [3.0, 50, 0.01, 1e-7, 5e-9, 0.001]  
  事件密度: 646-1618 events/ms (合理范围！)
  
优化效果: 36-91x事件减少，保持炫光+频闪场景的合理性
```

### 核心优化洞察 (修正版)
1. **k1敏感度优化**: 1.0→3.0 (3x提升，确保炫光事件生成)
2. **k2阈值优化**: 200→50 (4x降低，显著提高触发敏感度)  
3. **噪声项平衡**: k3,k5,k6适度调整，避免过度抑制
4. **Bug修复**: 极端参数导致0事件生成已解决

### 事件密度验证 (2025-07-31)
- **测试1**: 1617.8 events/ms (100ms, 161,777 events)
- **测试2**: 646.3 events/ms (100ms, 64,634 events)  
- **平均范围**: 600-1600 events/ms
- **评估**: ✅ 符合炫光+频闪场景的1K-10K events/ms目标范围

### IEBCS集成记录 (2025-07-31)
- ✅ **完整实现**: IEBCSFlareEventGenerator类，多时间窗口可视化
- ✅ **配置支持**: config.yaml完整IEBCS传感器参数  
- ✅ **调试系统**: 与DVS一致的0.5x/1x/2x/4x事件积累可视化
- ❌ **效果不佳**: 用户反馈"效果都不太好"，已切换回DVS优化方案

## 🚨 CRITICAL DATA PATH CORRECTIONS

### Correct Dataset Paths (Verified & Updated 2025-07-30)
**DSEC Events**: Pattern `{dsec_path}/*/events/left/events.h5` 
- ✅ Correctly searches all sequences under base path
- ✅ Currently finds 47 event files from 5 sequences used

**Flare7K Images**: Two separate Compound_Flare directories
- ✅ `Flare-R/Compound_Flare/`: 962 flare images  
- ✅ `Flare7K/Scattering_Flare/Compound_Flare/`: 5000 flare images
- ✅ **Total: 5962 flare images** (6x more than previously reported)
- ✅ Random selection from both directories during training

### Configuration (configs/config.yaml)
```yaml
data:
  # DSEC dataset (correct path structure)
  dsec_path: "/path/to/dsec/events/left/events.h5"
  resolution_w: 640  # DSEC standard
  resolution_h: 480  # DSEC standard
  
  # Flare7K dataset (correct subdirectories)
  flare7k_path: "/path/to/Flare7Kpp/"
  # Will look in: Flare7K/Scattering_Flare/ and Flare-R/Compound_Flare/
  
  # Training parameters
  sequence_length: 64
  time_window_us: 1000000  # 1 second windows
```

## Key Performance Optimizations ⚡

### 1. Flare Resolution Alignment (4-5x Speedup)
- **Problem**: Inconsistent flare image resolutions caused slow DVS simulation
- **Solution**: Force align all flares to DSEC resolution (640x480)
- **Files**: `src/flare_synthesis.py`, `src/dvs_flare_integration.py`
- **Impact**: 4-5x DVS simulation speedup, 80% memory reduction

### 2. Flare7K Diversity Transforms
- **Enhancement**: Realistic flare positioning (not always centered)
- **Transforms**: Rotation, scaling, translation, shear, flipping
- **Purpose**: Eliminate model bias toward centered flares

### 3. Memory-Efficient DSEC Loading
- **Innovation**: `src/dsec_efficient.py` for large-scale datasets
- **Method**: Metadata-only loading + binary search time windows
- **Result**: From 15GB+ down to <100MB memory usage

### 4. Natural Flare Transform Pipeline (2025-07-30) 🎯
- **核心改进**: 分离变换管道，消除不自然黑框问题
- **简化工作流**: 位置变换(大图) → 闪烁+运动(平移) → 自然裁剪
- **transform分离**: positioning_transform + final_crop_transform
- **天然画布**: 变换后的大图即为工作画布，无需额外创建
- **自然边界**: 炫光边缘自然过渡，无人工黑色边框

### 5. Simplified Movement on Natural Canvas (2025-07-30) ⚡
- **运动方式**: 直接在变换后大图上进行numpy平移操作
- **边界智能**: 限制工作区域为原图+120像素运动空间
- **运动范围**: 0-60像素随机距离，运动轨迹自然
- **最终裁剪**: PIL CenterCrop自然裁剪到目标分辨率
- **逻辑简化**: 去除不必要的大画布创建，提高效率

## Model Architecture
- **Feature Extractor**: 11D PFD features with 3x3 neighborhoods
- **Mamba Backbone**: 4 layers, d_model=128, d_state=16
- **Classification**: Binary output for flare removal
- **Total Parameters**: 271,489 (reduced from 271,745)

## PFD Features (11-Dimensional) - 优化后定义
**🚨 CRITICAL CHANGE**: 删除累积计数特征，仅保留原始PFD局部特征，避免泛化问题

| 维度 | 特征名称 | 物理含义 | 取值范围 | PFD关联 |
|------|----------|----------|----------|---------|
| 0-1 | x_center, y_center | 中心相对坐标 | [-1, 1] | ❌ 传统特征 |
| 2 | polarity | 事件极性 | {-1, 1} | ✅ PFD核心 |
| 3-4 | dt_norm, dt_pixel_norm | 对数时间间隔 | [0, 15] | ❌ 传统特征 |
| **5** | **Mf** | **极性频率 (时间窗口内)** | [0, 100] | ✅ **PFD核心** |
| **6** | **Ma** | **邻域极性变化总数** | [0, 100] | ✅ **PFD核心** |
| **7** | **Ne** | **活跃邻居像素数** | [0, 8] | ✅ **PFD核心** |
| **8** | **D** | **极性变化密度 Ma/Ne** | [0, 10] | ✅ **PFD核心** |
| **9** | **PFD-A评分** | **BA噪声检测 \|Mf-D\|** | [0, 100] | ✅ **PFD直接输出** |
| **10** | **PFD-B评分** | **频闪噪声检测 D** | [0, 10] | ✅ **PFD直接输出** |

**PFD特征占比**: 6/11 (54.5%) 为纯PFD特征，**删除了累积计数特征以避免训练→测试泛化问题**

## Running the Project 🚀 (Updated 2025-08-02)

### 🔍 分析模式 (推荐首次运行):
```bash
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare
python main.py --config configs/config.yaml --debug
# 配置: mode: analyze - 详细分析epoch数据流，不进行训练
```

### 🎯 训练模式:
```bash
# 修改configs/config.yaml: mode: train
python main.py --config configs/config.yaml
```

### 📊 评估模式:
```bash
# 修改configs/config.yaml: mode: evaluate
python main.py --config configs/config.yaml
```

**Debug Mode功能**:
- **🚨 炫光序列完整保存**: 保存所有DVS仿真器输入的炫光图像帧到 `output/debug_visualizations/flare_seq_XXX/original_frames/`
- **🔍 事件细分可视化**: 图像间隔分为5段，每段事件单独可视化到对应图像上
  - 格式: `frame_XXX_sub_YY_events.png` (YY为0-4的细分段)
  - 事件颜色: 负极性=蓝色，正极性=红色
  - 时间分辨率: 原帧间隔(2.5ms@400fps)的1/5 = 0.5ms精度
- **📊 详细元数据**: 每个序列的完整统计信息 (帧数、事件数、频率、极性分布等)
- **⚡ 快速测试**: 限制1个epoch，8个样本，避免长时间运行

**超参数调节**:
```python
# 在main.py中可调整细分数量 (默认5)
config['debug_event_subdivisions'] = 5  # 1-10均可，影响时间分辨率
```

**输出结构**:
```
output/debug_visualizations/
├── flare_seq_000/
│   ├── original_frames/          # 原始炫光图像序列
│   ├── event_visualizations/     # 事件可视化 (5x数量)
│   └── metadata.txt             # 序列统计信息
├── flare_seq_001/
└── ...
```

### Feature Testing:
```bash
python test_features.py
```

## ⚠️ Known Issues & Solutions

### ✅ CURRENT RESOLVED STATUS (2025-08-02 - 完全重构完成)
- **Epoch-Iteration架构**: ✅ 完全实现"先完整序列特征提取，再滑动窗口"
- **Legacy代码清理**: ✅ 删除mixed_flare_datasets，统一使用新架构
- **数据流修正**: ✅ 特征提取在epoch级别进行一次，iteration级别滑动窗口采样
- **帧数优化**: ✅ 减少flare仿真帧数 (max_fps: 3000→1500, samples: 24→12)
- **格式转换**: ✅ DVS [t,x,y,p] → Project [x,y,t,p] 自动处理
- **内存安全**: ✅ batch_size=2, sequence_length=64 严格限制
- **详细分析**: ✅ main.py集成epoch数据流分析功能

### 🎯 ENHANCED SYSTEM STATUS - FULLY FUNCTIONAL
- **✅ NATURAL BOUNDARIES**: No artificial black frames, natural flare edge transitions
- **✅ REALISTIC MOVEMENT**: Variable 0-60 pixel movement matching automotive scenarios
- **✅ COMPREHENSIVE DEBUG**: Multi-resolution event analysis + trajectory visualization
- **✅ OPTIMIZED PIPELINE**: Split transforms reduce complexity, improve efficiency  
- **✅ MEMORY STABLE**: Safe operation within 791MB limits

### ⚠️ Minor Notes
- **Debug Directory**: Located at `output/debug_visualizations/flare_seq_xxx/`
- **Visualization Types**: Original frames, multi-resolution events, movement trajectories  
- **Data Diversity**: Using 5962 flare images from both Compound_Flare directories

### Dependency Status - ALL WORKING
- **✅ V2CE**: 深度学习事件仿真器，完美时间精度，多分辨率可视化
- **✅ DVS-Voltmeter**: 物理仿真器（已被V2CE替代，但仍可用）
- **✅ DSEC Data**: Memory-efficient H5 loading from 47 event files across sequences
- **✅ Flare7K Data**: 5962 flare images loaded correctly from both Compound_Flare directories

### 仿真器对比分析 (2025-07-31)

#### V2CE (深度学习仿真器)
- **时间精度**: 100.0% 完美对齐（30ms→30.0ms）
- **事件密度**: ~3,000 events/ms (30ms→90K events)
- **处理速度**: ~16s (较快)
- **泛化性**: ⚠️ 可能存在数据集偏差问题
- **可视化**: 4种分辨率×48帧 = 360个文件/组

#### DVS-Voltmeter (物理仿真器) - 当前配置
- **时间精度**: ~120% (30ms→25ms，略有偏差)
- **事件密度**: ~59,000 events/ms (30ms→1.16M events)
- **处理速度**: ~9s (更快，帧数少)
- **泛化性**: ✅ 物理模型，理论上泛化性更好
- **可视化**: 4种分辨率×3帧 = 23个文件/组
- **优化参数**: K=[50.0,80,0.01,5e-6,1e-7,0.001], 100fps, 6samples/cycle

#### 使用建议
- **训练模型**: 推荐DVS-Voltmeter (更好泛化性，尽管事件数量较多)
- **快速原型**: 推荐V2CE (事件数量合理，处理简单)
- **研究对比**: 可同时使用两种仿真器验证模型鲁棒性

## 🚨 FUTURE DEVELOPMENT GUIDELINES (CRITICAL)

### ✅ APPROVED CHANGES - Safe to Modify
1. **Debug信息增强**: 
   - 添加更详细的日志输出
   - 增加训练过程可视化
   - 添加性能监控指标

2. **数据生成方法改进**:
   - 优化炫光事件生成算法
   - 改进DVS模拟参数
   - 增强炫光多样性变换
   - 调整随机化训练策略参数

3. **仿真性能优化** (2025-07-30):
   - **帧率提升**: 移除硬编码500fps限制，支持1600fps高精度仿真
   - **时长8x加速**: 所有事件时长缩短为原来1/8，大幅减少仿真时间
   - **质量改善**: 每周期采样从4-5个提升到14-17个，显著改善频闪质量

4. **真实光源频闪修复** (2025-07-30):
   - **消除黑屏**: 添加0-70%随机最低基线，避免完全黑屏
   - **线性变化**: 使用直观的三角波线性变化，消除低值停滞问题
   - **多分辨率调试**: 0.5x/1x/2x/4x事件可视化，全面评估仿真质量

### ❌ FORBIDDEN CHANGES - DO NOT MODIFY
**除非明确发现严重bug，以下部分严禁改动**:
- **batch_size=2**: 已验证的内存安全配置  
- **sequence_length=64**: 已优化的序列长度
- **11维特征提取核心逻辑**: 基于PFD物理原理 (删除累积特征)
- **Mamba模型架构**: 271,489参数已调优
- **数据流架构**: 数据集阶段特征提取 (物理意义)
- **内存优化代码**: dsec_efficient.py核心逻辑
- **DVS模拟器集成**: 已修复的配置路径管理

### 🔍 CHANGE CRITERIA
**只有在以下情况下才能修改禁止区域**:
1. 发现明确的严重bug (有可重现的错误证据)
2. 内存爆炸或训练崩溃
3. 数值不稳定性问题
4. 明确的性能退化

**修改前必须**:
- 完整备份当前工作代码
- 记录修改原因和预期效果
- 小规模测试验证

## 🎯 PROJECT COMPLETION STATUS
**当前版本为稳定基线，所有核心功能已验证工作正常**:
- ✅ 完整数据管线: DSEC + Flare7K + DVS模拟
- ✅ 11维PFD特征提取: 物理启发，删除累积特征避免泛化问题
- ✅ Mamba模型训练: 271,489参数优化架构
- ✅ 内存安全: batch_size=2防止崩溃
- ✅ 性能优化: 4-5x DVS加速，<100MB内存使用

## Troubleshooting
- Always activate environment first: `source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare`
- DSEC files: Ensure path includes `/events/left/events.h5`
- Flare7K files: Check subdirectories `Scattering_Flare/Compound_Flare/` and `Flare-R/Compound_Flare`
- For memory issues: Use `src/dsec_efficient.py` loader
- Resolution: Always verify 640x480 alignment for DSEC compatibility