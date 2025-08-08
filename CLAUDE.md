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
- **Model Architecture**: 271,489 parameters, 11D PFD features, 3x3 neighborhoods
- **Transform Pipeline**: Split positioning + final crop, natural flare boundaries
- **Movement Simulation**: 0-60 pixel random movement with realistic automotive speeds
- **Flicker Generation**: Linear triangle wave, baseline intensity constraints
- **Debug System**: Multi-resolution event visualization (0.5x/1x/2x/4x) + movement trajectories
- **Memory Efficient**: DSEC dataset integration with <100MB usage, 1440x1440→640x480 natural cropping
- **🚨 DEBUG TIMING**: main.py debug mode requires **300+ seconds** for complete DVS simulation and epoch generation

## Core Data Flow (FIXED & VERIFIED 2025-08-02) ✅
```
CORRECT Epoch-Iteration Training Pipeline:

🔄 EPOCH LEVEL (Data Generation - Once per Epoch):
1. Load DSEC background events: 100K-1M events in 0.1-0.3s window [N1, 4] 
   ⭐ CRITICAL: 时间戳自动归一化到0开始 (subtract t_min after loading)
2. Generate DVS flare events: Variable events in 0.1-0.3s [N2, 4] 
   ⭐ DVS events naturally start from 0 (no normalization needed)
3. Merge & sort by timestamp → long_sequence [N_total, 4] (完整物理序列)
4. ✅ PFD特征提取: long_sequence → long_feature_sequence [N_total, 11]
5. Generate labels: [N_total] (0=background, 1=flare)

⚡ ITERATION LEVEL (Model Training - Multiple per Epoch):
1. Sliding window sampling: long_feature_sequence → batch [sequence_length=64, 11]
2. Model forward: [batch_size, 64, 11] → [batch_size, 64, 1] probabilities
3. BCE Loss + backpropagation (每个batch执行)
4. Continue until long_feature_sequence consumed

🚨 CRITICAL BUG FIXES (2025-08-03 - COMPLETED):
- ❌ DSEC限制64事件 → ✅ 返回完整时间窗口内所有事件 (测试验证: 386万事件)
- ❌ 模型注释13维 → ✅ 修正为11维特征
- ❌ 人工sequence_length截断 → ✅ 自然长序列处理
- ❌ Config参数冲突 → ✅ 删除duration冗余参数，flare_synthesis统一控制
- ❌ DSEC时间戳未归一化 → ✅ 三级时间戳归一化修复完成：
  * dsec_efficient.py:159-160 - DSEC载入时减去t_min
  * epoch_iteration_dataset.py:311-314 - 炫光事件格式化时减去t_min  
  * epoch_iteration_dataset.py:229-232 - 时间窗口裁剪后重新归一化
- ❌ Debug可视化缺失 → ✅ 完整三种事件可视化系统已修复：
  * debug_epoch_000/background_events/ - DSEC背景事件多分辨率可视化
  * debug_epoch_000/flare_events/ - DVS炫光事件多分辨率可视化
  * debug_epoch_000/merged_events/ - 合并事件多分辨率可视化
- ✅ Loss反向传播：确认在iteration级别正确执行
- ✅ 时间戳验证成功: 背景事件(0-79279μs), 炫光事件(0-49978μs), 合并事件(0-79279μs)
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

### 优化策略与结果 (最新版 2025-08-08)
```yaml
BEFORE (原始DVS346参数):
  dvs346_k: [1.0, 200, 0.001, 1e-8, 1e-9, 0.0001]
  事件密度: ~59,000 events/ms (过多)

CURRENT (随机化优化参数):
  dvs346_k: [random(0.5-5.265), 20, 0.0001, 1e-7, 5e-9, 1e-5]  
  事件密度: 动态变化 (数据多样性增强)
  k1随机化: 每次炫光生成使用不同敏感度
  
优化效果: 数据多样性大幅提升，模拟真实场景变化
```

### 核心优化洞察 (最新版 2025-08-08)
1. **k1随机化**: random(0.5-5.265) 动态变化，模拟不同光照条件
2. **运动范围增强**: 0-180像素移动 (3x增强)，更丰富的运动模式
3. **数据多样性**: 每次训练产生不同强度的炫光事件
4. **背景事件**: 切换到简化的data/bg_events/*.h5结构，使用所有文件

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

### Correct Dataset Paths (Updated 2025-08-08)
**Background Events**: Simplified structure `data/bg_events/*.h5`
- ✅ 7 background event H5 files, using all files (no 5-file limit)
- ✅ Total: 3.27 billion background events, 291 time windows
- ✅ Memory-efficient loading with no file count restrictions

**Flare7K Images**: Two separate Compound_Flare directories
- ✅ `Flare-R/Compound_Flare/`: 962 flare images  
- ✅ `Flare7K/Scattering_Flare/Compound_Flare/`: 5000 flare images
- ✅ **Total: 5962 flare images** (6x more than previously reported)
- ✅ Random selection from both directories during training

### Configuration (configs/config.yaml) - Updated 2025-08-08
```yaml
data:
  # Background events (simplified structure)
  dsec_path: "data/bg_events"  # Simplified H5 files directory
  resolution_w: 640  # DSEC standard
  resolution_h: 480  # DSEC standard
  
  # Flare7K dataset (correct subdirectories)
  flare7k_path: "/path/to/Flare7Kpp/"
  # Will look in: Flare7K/Scattering_Flare/ and Flare-R/Compound_Flare/
  
  # Training parameters
  sequence_length: 64  # Sliding window size for iterations
  
  # 🚨 UNIFIED CONTROL: Duration parameters (no conflicts)
  randomized_training:
    background_duration_range: [0.1, 0.3]  # 100-300ms background windows
    
  flare_synthesis:
    duration_range: [0.05, 0.15]  # 50-150ms flare sequences (SINGLE CONTROL)
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

### 5. Enhanced Movement on Natural Canvas (2025-08-08) ⚡
- **运动方式**: 直接在变换后大图上进行numpy平移操作
- **边界智能**: 限制工作区域为原图+180像素运动空间
- **运动范围**: 0-180像素随机距离 (3x增强)，运动轨迹自然多样化
- **最终裁剪**: PIL CenterCrop自然裁剪到目标分辨率
- **逻辑简化**: 去除不必要的大画布创建，提高效率

## Model Architecture
- **Feature Extractor**: 11D PFD features with 3x3 neighborhoods
- **Mamba Backbone**: 4 layers, d_model=128, d_state=16
- **Classification**: Binary output for flare removal
- **Total Parameters**: 271,489 (reduced from 271,745)

## 🎯 **当前焦点：特征提取器性能优化** (基于PFDs.cpp)

### **特征提取性能问题现状** (2025-08-03)
- **原始问题**: O(N²)复杂度导致586万事件需要325秒处理
- **根本原因**: recent_events列表管理 + 重复搜索历史事件
- **目标性能**: 基于PFDs.cpp实现，应达到O(N)复杂度，~6秒处理时间

### **正确的11维PFD特征设计** (修正版)
#### ✅ **完整11维特征向量 (编号0-10)**：
| 维度 | 特征名称 | 物理含义 | 有效性 |
|------|----------|----------|--------|
| 0 | x_center | 归一化x坐标 | ✅ 局部空间特征 |
| 1 | y_center | 归一化y坐标 | ✅ 局部空间特征 |
| 2 | polarity | 事件极性 | ✅ PFD核心特征 |
| 3 | **dt_norm** | **相邻事件时间间隔** | ✅ **局部时序特征 (非全局!)** |
| 4 | **dt_pixel** | **像素级事件间隔** | ✅ **局部空间特征** |
| 5 | **Mf** | **像素极性频率** | ✅ **PFD核心 (时间窗口内)** |
| 6 | **Ma** | **邻域极性变化数** | ✅ **PFD核心 (时间窗口内)** |
| 7 | **Ne** | **活跃邻居数** | ✅ **PFD核心 (时间窗口内)** |
| 8 | **D** | **极性变化密度** | ✅ **PFD核心 (Ma/Ne)** |
| 9 | **PFD-A** | **BA噪声检测评分** | ✅ **PFD输出 \|Mf-D\|** |
| 10 | **PFD-B** | **频闪检测评分** | ✅ **PFD输出 D** |

**总计**: 11维特征 (0-10)，其中6个为纯PFD特征

#### ❌ **真正有问题的特征 (需避免)**：
- **全局累积计数**: 会随处理时间无限增长的特征
- **会话级状态**: 依赖整个处理会话历史的特征

### **关键洞察修正**：
- **dt_norm不是全局特征**: 它只是相邻事件的时间差，完全局部
- **所有时间特征都有界**: 事件相机物理限制确保合理范围
- **问题在实现而非设计**: PFDs.cpp证明O(N)复杂度可行

## Running the Project
### Training:
```bash
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare
python main.py --config configs/config.yaml
```

### Debug Mode (Complete Event Visualization):
```bash
# Run debug mode to save comprehensive event visualizations
python main.py --config configs/config.yaml --debug
```

**Debug Mode功能** (2025-08-04 完全修复):
- **🚨 TIMING REQUIREMENT**: main.py debug模式需要**180+秒**完成DVS仿真和epoch生成
- **✅ 炫光序列原始帧**: 完整的炫光图像序列
  - 原始炫光图像帧保存到 `output/debug_epoch_000/flare_sequence_frames/`
  - 131个RGB图像帧 (frame_000.png - frame_130.png)
  - **展示完整炫光运动和闪烁过程** - 直接用于PPT/视频展示
  - 来自DVS仿真器输入的真实炫光图像序列

- **✅ 完整事件可视化系统** (2025-08-04 全部修复): 
  - **最终输出结构**: `output/debug_epoch_000/` (统一、完整)
  - **background_events/**: DSEC背景事件 (126,172个事件)
  - **flare_events/**: DVS炫光事件 (1,031,034个事件) ✅ 已修复  
  - **merged_events/**: 合并训练序列 (智能颜色区分)
  - **epoch_metadata.txt**: 完整统计信息
  - **多分辨率**: 每种事件类型都有 0.5x/1x/2x/4x temporal窗口
  - **⚠️ 注意**: 需完整epoch生成才触发，DVS仿真时间较长

**输出结构** (2025-08-04 最终版本):
```
output/debug_epoch_000/              # 统一debug输出目录
├── background_events/               # DSEC背景事件可视化
│   ├── temporal_0.5x/              # 多分辨率时间窗口
│   ├── temporal_1x/
│   ├── temporal_2x/
│   └── temporal_4x/
├── flare_events/                    # ✅ DVS炫光事件可视化
│   ├── temporal_0.5x/              # 1,031,034个炫光事件
│   ├── temporal_1x/
│   ├── temporal_2x/
│   └── temporal_4x/
├── merged_events/                   # 合并训练序列可视化
│   ├── temporal_0.5x/              # 1,157,206个总事件
│   ├── temporal_1x/
│   ├── temporal_2x/
│   └── temporal_4x/
├── flare_sequence_frames/           # ✅ 新增：炫光序列原始帧
│   ├── frame_000.png               # 131个炫光图像帧
│   ├── frame_001.png               # 展示炫光的运动和闪烁
│   ├── ...
│   └── frame_130.png
└── epoch_metadata.txt              # 完整统计信息
```

**Debug配置优化** (快速测试):
```python
# 在main.py --debug模式下自动设置
config['debug_mode'] = True
config['training']['max_epochs'] = 1              # 限制epoch数
config['data']['max_samples_debug'] = 4           # 限制样本数
config['data']['randomized_training']['background_duration_range'] = [0.05, 0.1]  # 缩短背景
config['data']['flare_synthesis']['duration_range'] = [0.03, 0.08]  # 缩短炫光
```

### Feature Testing:
```bash
python test_features.py
```

## ⚠️ Known Issues & Solutions

### ✅ CURRENT RESOLVED STATUS (2025-08-03 - 时间戳归一化与Debug可视化修复完成)
- **🚨 DSEC数据加载Bug**: ✅ 删除sequence_length=64人工限制，现在返回完整时间窗口事件
- **🚨 模型架构不一致**: ✅ 修正模型注释从13维到11维特征，代码逻辑一致
- **🚨 Epoch-Iteration架构**: ✅ 修复epoch级别数据生成，支持真正的长序列处理
- **🚨 时间戳归一化Bug**: ✅ 三级修复完成，DSEC和炫光事件都从0开始，确保正确合并
- **🚨 Debug可视化缺失**: ✅ 完整修复三种事件可视化系统（背景+炫光+合并）
- **✅ Loss反向传播**: ✅ 验证确认在iteration级别正确执行，无需修改
- **DVS参数调优**: ✅ 已切换回DVS-Voltmeter，大幅优化参数减少事件数量
- **事件数量优化**: ✅ 从原始200K+ events/ms降至59K events/ms (3x减少)，仍比V2CE高20x
- **帧率优化**: ✅ 降低至100fps，6帧/30ms，显著减少计算负荷
- **参数配置**: ✅ K=[50.0,80,0.01,5e-6,1e-7,0.001] (10x+阈值提升)
- **Multi-Resolution Debug**: ✅ DVS多分辨率可视化(0.5x/1x/2x/4x)正常工作
- **Transform Pipeline**: ✅ Split positioning + natural cropping eliminates black borders
- **Memory Safety**: ✅ Verified stable with batch_size=2, max_samples_debug=4

### 🎯 ENHANCED SYSTEM STATUS - FULLY FUNCTIONAL
- **✅ NATURAL BOUNDARIES**: No artificial black frames, natural flare edge transitions
- **✅ REALISTIC MOVEMENT**: Variable 0-60 pixel movement matching automotive scenarios
- **✅ COMPREHENSIVE DEBUG**: Multi-resolution event analysis + trajectory visualization
- **✅ OPTIMIZED PIPELINE**: Split transforms reduce complexity, improve efficiency  
- **✅ MEMORY STABLE**: Safe operation within 791MB limits
- **✅ TIMESTAMP NORMALIZATION**: All event streams properly normalized to start from 0
- **✅ THREE-WAY DEBUG VISUALIZATION**: Background, flare, and merged events all visualized

### 🚨 DEBUG TASK LIST (Known Issues to Address)
**Verification Tasks**:
1. **📊 Timestamp Verification**: Confirm timestamp normalization across all edge cases
2. **🔍 Memory Usage Monitoring**: Track memory usage during long training runs

### ✅ RESOLVED ISSUES (2025-08-03)
- **🎯 Flare Events Visualization Issue**: ✅ FIXED - `debug_epoch_000/flare_events/` now displays complete flare patterns
  - Root cause: DVS format `[t,x,y,p]` vs project format `[x,y,t,p]` conversion missing in visualization
  - Solution: Added `_format_flare_events()` call in `_save_unified_debug_visualizations()`
  - Verification: flare_events files now ~20KB (vs previous small files), showing full flare shapes
  - Side benefit: Eliminated duplicate `flare_seq_*` outputs, unified to `debug_epoch_000`

### ⚡ **当前进行中：特征提取器重构** (2025-08-03)
- **现状**: 正在基于PFDs.cpp重写特征提取器，目标O(N)复杂度
- **🚨 当前特征提取器存在的bugs**:
  1. **维度错误**: 目前实现为6D或10D，应为11D (编号0-10)
  2. **性能问题**: O(N²)复杂度，recent_events列表管理低效
  3. **特征计算错误**: 部分PFD特征计算逻辑与PFDs.cpp不匹配
  4. **时间窗口管理**: 缺乏高效的固定大小循环缓冲区
- **已尝试方案**:
  1. 错误的10D PFD特征 → 处理速度25K events/s (仍需233秒处理586万事件)
  2. 简化6D特征 → 处理速度63K events/s (需93秒处理586万事件)
- **目标**: 实现正确的11D特征 + 达到PFDs.cpp性能标准 (~6秒处理586万事件)
- **下一步**: 修复特征维度为11D，完善PFD算法实现，可能需要Cython优化

### ⚠️ Minor Notes
- **Debug Directory**: Located at `output/debug_epoch_000/` (unified system)
- **Visualization Types**: Background, flare, and merged events with multi-resolution temporal windows
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