# EventMamba-FX Project Memory

## Project Overview
EventMamba-FX is a Feature-Augmented Mamba model for real-time event denoising and artifact removal from event camera streams. 

## Environment Setup 🔧 CRITICAL
- **MUST USE**: `source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare`
- Environment already has all dependencies: PyTorch, Mamba SSM, etc.
- Python 3.10.18 with CUDA support

## 🚀 CURRENT SYSTEM STATUS: UNIFIED DUAL-MODE TBPTT ARCHITECTURE (2025-08-08)

### ✅ 完成的重大架构重构
EventMamba-FX已从传统epoch-iteration架构成功重构为**工业级双模式TBPTT架构**，实现了"长序列工厂 + 序列消化器"设计模式。

### 🎯 双模式设计核心

#### 模式一：数据预生成 (Generate Mode)
- **流程**: 纯数据生成，生成完毕后自动退出
- **自动存档**: 每生成一个长序列，自动存储为.h5文件
- **防重名机制**: 时间戳+索引+随机后缀+UUID四级防重名保护
- **存档路径**: `data/generated_h5/{train|val}/sequence_timestamp_index[_suffix].h5`

#### 模式二：模型训练 (Load Mode)  
- **流程**: 从H5文件加载预生成数据进行训练
- **优点**: 启动快，CPU占用低，训练稳定可复现
- **断点续训**: 正确恢复best_val_loss和全部训练状态

### 🏗️ 核心架构组件

#### 1. UnifiedSequenceDataset (src/unified_dataset.py)
- **统一接口**: 配置驱动的模式切换
- **H5存档**: 完整的序列特征和标签存储
- **防重名**: 多级文件名冲突检测和解决
- **错误处理**: 存档失败时的graceful fallback

#### 2. Enhanced Trainer (src/trainer.py) 
- **TBPTT双循环**: 外循环处理长序列，内循环chunk切分
- **全局步数控制**: 基于global_step的精确训练进度
- **智能断点续训**: 自动检测和恢复最新checkpoint
- **工业级检查点**: 基于步数的文件命名和状态保存

#### 3. 配置系统 (configs/config.yaml)
```yaml
data_pipeline:
  mode: 'generate'  # 'generate' 或 'load' 
  h5_archive_path: 'data/generated_h5'

training:
  epochs: 10  # 训练轮数
  chunk_size: 8192  # TBPTT截断长度
  num_long_sequences_per_epoch: 1000  # 每epoch序列数（训练）
  validate_every_n_steps: 20000
  save_every_n_steps: 20000

evaluation:
  num_long_sequences_per_epoch: 50  # 验证序列数
```

## 📊 当前验证状态

### ✅ Generate模式 - 完全验证
- ✅ TBPTT双循环训练正常工作
- ✅ H5自动存档成功，防重名机制有效
- ✅ 断点续训正常工作
- ✅ 内存使用安全(<1GB)
- ✅ 输出清理完成 - 无冗余debug信息

### ✅ Load模式 - 当前运行
- ✅ 从Generate模式产生的H5文件加载训练正常
- ✅ 断点续训：正确恢复best_val_loss历史最佳值
- ✅ 修复tqdm嵌套显示冲突：禁用验证时进度条

## 🎯 数据生成管线

### DSEC背景事件
- **源文件**: `data/bg_events/*.h5` (7个文件)
- **总容量**: 32.7亿事件，291个时间窗口
- **时间窗口**: 100-300ms随机长度
- **格式**: 自动时间戳归一化从0开始

### DVS炫光事件
- **仿真器**: DVS-Voltmeter物理仿真
- **参数优化**: k1随机化(0.5-5.265)，提升数据多样性
- **时间窗口**: 50-150ms随机长度
- **运动模式**: 0-180像素随机运动

### Flare7K图像数据集
- **图像源**: 2个Compound_Flare目录，共5962张炫光图像
- **路径1**: `Flare-R/Compound_Flare/` (962张)
- **路径2**: `Flare7K/Scattering_Flare/Compound_Flare/` (5000张)

## 🔧 特征提取与模型

### 当前特征设置
- **特征维度**: 4D快速特征 [x_norm, y_norm, dt, polarity]
- **处理方式**: 纯NumPy向量化，毫秒级完成
- **PFD状态**: 暂时禁用，11D PFD特征设计已完备待恢复

### 模型架构
- **参数量**: 25,359,361个可训练参数
- **架构**: Mamba backbone，12层，d_model=512，d_state=64
- **输出**: 二元分类（事件去噪）

## 🚀 运行指南

### Generate模式 (当前默认)
```bash
source /home/lanpoknlanpokn/miniconda3/bin/activate event_flare
python main.py --config configs/config.yaml
```

### Debug模式 (清理输出版本)
```bash
python main.py --config configs/config.yaml --debug
```

### Load模式 (需先运行Generate生成数据)
```bash
# 1. 修改configs/config.yaml: data_pipeline.mode: 'load'
# 2. 运行训练
python main.py --config configs/config.yaml
```

## 📝 重要变更记录 (2025-08-08)

### 新增功能
- ✅ **统一双模式架构**: 完整的generate/load模式支持
- ✅ **H5防重名机制**: 多级文件名冲突解决
- ✅ **输出清理**: 移除冗余debug信息，保留核心训练信息
- ✅ **断点续训增强**: 基于global_step的精确恢复

### 代码文件状态
- **新增**: `src/unified_dataset.py` - 双模式统一数据集
- **重构**: `src/trainer.py` - TBPTT + 断点续训
- **更新**: `main.py` - 统一数据加载器集成
- **清理**: `src/epoch_iteration_dataset.py` - 注释冗余print
- **清理**: `src/flare_synthesis.py` - 注释详细步骤输出
- **清理**: `src/dvs_flare_integration.py` - 修复语法错误，注释verbose输出

## 🔍 推理系统 (2025-08-11)

### ✅ 工业级流式推理架构
EventMamba-FX现已配备完整的**OOM-防护流式推理系统**，可处理任意大小的DSEC H5文件进行事件去噪。

#### 核心推理组件

1. **H5StreamReader (src/h5_stream_reader.py)**
   - **分块读取**: 默认处理块大小可配置（避免内存OOM）
   - **时间限制**: 支持仅处理文件前N秒的事件
   - **二分搜索**: 高效定位时间截止点

2. **Predictor (src/predictor.py)** 
   - **流式处理**: 逐块处理大文件，不一次性加载
   - **优化chunk_size**: 推理时使用10倍训练chunk_size (81,920)
   - **内存管理**: 显式清理GPU缓存，强制垃圾回收
   - **性能监控**: 实时显示特征提取vs推理时间比

3. **InferenceEventVisualizer (src/inference_visualizer.py)**
   - **时序可视化**: 支持1ms间隔的逐帧事件可视化
   - **对比分析**: 原始vs去噪事件的并排比较
   - **DSEC风格**: 黑底彩色点阵，红色ON/青色OFF事件

### 🚀 推理使用方法

**基本推理**（处理前1秒事件）:
```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input data/inference/zurich_city_12_a.h5 \
    --output data/inference/clean_output.h5 \
    --time-limit 1.0
```

**内存优化推理**:
```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input data/inference/zurich_city_12_a.h5 \
    --output data/inference/clean_output.h5 \
    --time-limit 0.1 \
    --block-size 1000000  # 减小块大小防止OOM
```

**推理+可视化管线**:
```bash
python run_inference_with_visualization.py
```

### 📊 验证结果 (zurich_city_12_a.h5前100ms)

- **原始事件数**: 1,817,979
- **去噪后事件数**: 1,644,775  
- **炫光事件移除**: 173,204 (9.53%)
- **推理性能**: 
  - 特征提取时间: 0.03s
  - 模型推理时间: 23.15s
  - **瓶颈**: 模型推理是主要性能瓶颈
- **可视化输出**: 300帧(100ms × 3类型)

### 🔧 推理系统特性

- ✅ **OOM防护**: 分块处理任意大小H5文件
- ✅ **时间控制**: 可限制处理特定时长的事件
- ✅ **内存优化**: 10倍chunk_size提升推理效率
- ✅ **可视化**: 完整的时序对比可视化
- ✅ **DSEC兼容**: 完美支持DSEC数据集格式

## 🎯 下一步开发重点

1. **推理性能优化**: 模型推理是主要瓶颈，考虑批处理优化
2. **DVS炫光事件Bug修复**: 当前merge事件中DVS炫光事件数量为0，需要调试DVS仿真器集成
3. **PFD特征恢复**: 在训练稳定后恢复11D物理特征

## 🚨 内存安全配置 (历史验证)
- **batch_size**: 固定为1 (TBPTT需求)
- **chunk_size**: 8192 (TBPTT截断长度)
- **debug模式**: 自动将序列数减少到8个进行快速验证

## 🔍 Debug系统状态
- **可视化系统**: 完整保留，包括多分辨率事件分析
- **输出清理**: 移除冗余步骤信息，保留核心状态
- **Debug目录**: `output/debug_visualizations/`

## 📈 性能基准
- **序列长度**: 通常60万-100万事件
- **生成时间**: 40-120秒/序列 (包含DVS仿真)
- **特征提取**: <0.02秒 (4D快速特征)
- **内存峰值**: <1GB (生成+训练全流程)
- **checkpoint恢复**: 自动检测，支持中断续训

---

*最后更新: 2025-08-08 - 完成TBPTT架构重构，输出清理，H5防重名机制*