# Event Flare Removal with Mamba Architecture

## 🎯 项目概述 (Project Overview)

本项目实现了基于 Mamba 架构的事件相机闪光去除系统。系统接收两个事件流：原始事件数据和包含闪光的事件数据，通过深度学习模型学习在单个事件级别上移除闪光事件，输出干净的事件流。

This project implements an event-based glare removal system using Mamba architecture. The system takes two event streams as input: original events and events with glare, and learns to remove glare events at the individual event level through deep learning.

## 🏗️ 系统架构 (System Architecture)

### 数据流 (Data Flow)
```
原始事件 + 闪光事件 → 特征提取 → Mamba网络 → 分类输出 (0=闪光, 1=干净)
Raw Events + Glare Events → Feature Extraction → Mamba Network → Classification (0=glare, 1=clean)
```

### 核心组件 (Core Components)

#### 1. 特征提取器 (Feature Extractor) - `src/feature_extractor.py`
- **输入**: 原始事件序列 `[x, y, t, p]` (坐标、时间、极性)
- **输出**: 增强特征向量 (默认32维)
- **特征包括**:
  - 归一化坐标 (x_norm, y_norm)
  - 基础事件属性 (t, p)
  - 时间差特征 (dt, dt_pixel)
  - 可扩展维度 (为更复杂特征预留空间)

```python
特征向量 = [x_norm, y_norm, t, p, dt, dt_pixel, ...扩展特征]
```

#### 2. Mamba 模型 (Mamba Model) - `src/model.py`
- **架构**: 嵌入层 → N层Mamba → 分类头
- **参数**:
  - `input_feature_dim`: 输入特征维度 (32)
  - `d_model`: Mamba内部维度 (128)
  - `n_layers`: Mamba层数 (4)
  - `d_state`: SSM状态空间维度 (16)

```python
输入: [batch, seq_len, 32] → 嵌入: [batch, seq_len, 128] 
→ Mamba层: [batch, seq_len, 128] → 输出: [batch, seq_len, 1]
```

#### 3. 数据加载器 (Dataset) - `src/datasets.py`
- **数据格式**: `x y t p label` (每行一个事件)
- **序列长度**: 可配置 (默认64个事件)
- **标签**: 0=闪光事件, 1=干净事件
- **特征提取**: 在数据加载时实时进行

#### 4. 训练器 (Trainer) - `src/trainer.py`
- **损失函数**: 二元交叉熵 (BCELoss)
- **优化器**: AdamW
- **验证**: 每轮训练后进行验证
- **保存**: 自动保存最佳模型

#### 5. 评估器 (Evaluator) - `src/evaluate.py`
- **指标**: 准确率、精确率、召回率、F1分数
- **阈值**: 0.5 (概率→二元分类)

## 📁 项目结构 (Project Structure)

```
event_flick_flare/
├── main.py                    # 主程序入口
├── configs/
│   └── config.yaml           # 配置文件
├── src/
│   ├── model.py              # Mamba模型定义
│   ├── datasets.py           # 数据加载和处理
│   ├── feature_extractor.py  # 特征提取器
│   ├── trainer.py            # 训练逻辑
│   └── evaluate.py           # 评估逻辑
├── data/
│   └── simulated_events/     # 示例数据
├── checkpoints/              # 模型检查点
├── requirements.txt          # Python依赖
├── mock_mamba.py            # Mock Mamba实现(测试用)
├── test_pipeline.py         # 管道测试脚本
└── readme_new.md            # 本文档
```

## ⚙️ 配置说明 (Configuration)

### 关键配置参数 (`configs/config.yaml`)

```yaml
# 运行模式
run:
  mode: train  # 'train' 或 'evaluate'

# 数据设置
data:
  sequence_length: 64      # 输入序列长度
  resolution_h: 260        # 事件相机高度
  resolution_w: 346        # 事件相机宽度

# 模型架构
model:
  input_feature_dim: 32    # 特征维度
  d_model: 128            # Mamba内部维度
  n_layers: 4             # Mamba层数
  d_state: 16             # SSM状态维度

# 训练参数
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

## 🚀 使用方法 (Usage)

### 1. 环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
数据格式为文本文件，每行一个事件：
```
x y t p label
100 150 1000 1 1
101 150 1100 1 0
...
```

### 3. 训练模型
```bash
# 使用默认配置训练
python main.py --config configs/config.yaml

# 或修改config.yaml中的参数后运行
python main.py
```

### 4. 评估模型
```bash
# 修改config.yaml: mode: 'evaluate'
python main.py --config configs/config.yaml
```

### 5. 测试管道
```bash
# 测试管道逻辑(无需重依赖)
python test_pipeline.py
```

## 📊 核心算法详解 (Core Algorithm Details)

### 特征工程 (Feature Engineering)
1. **空间归一化**: 将像素坐标归一化到[0,1]
2. **时间特征**: 计算相邻事件时间差
3. **像素级时间记忆**: 维护每个像素的极性相关时间戳
4. **可扩展设计**: 预留维度用于更复杂的PFD类特征

### Mamba架构优势
1. **长序列建模**: 比Transformer更高效处理长事件序列
2. **线性复杂度**: O(n)而非O(n²)的计算复杂度
3. **状态空间模型**: 天然适合处理时序事件数据
4. **并行训练**: 支持GPU并行加速

### 损失函数设计
- **二元交叉熵**: 适合二分类任务(闪光vs干净)
- **序列级别**: 每个事件独立分类
- **权重均衡**: 可通过采样平衡正负样本

## 🔬 实验和评估 (Experiments & Evaluation)

### 评估指标
- **准确率 (Accuracy)**: 整体分类正确率
- **精确率 (Precision)**: 预测为干净事件中真正干净的比例
- **召回率 (Recall)**: 实际干净事件中被正确识别的比例
- **F1分数**: 精确率和召回率的调和平均

### 性能监控
- 训练损失和验证损失曲线
- 每轮epoch的指标变化
- 最佳模型自动保存

## 🛠️ 扩展和定制 (Extensions & Customization)

### 1. 增加特征维度
在 `src/feature_extractor.py` 中的 `process_sequence` 方法中添加更多特征：
```python
# 示例：添加邻域统计特征
neighbor_count = count_neighbors(x, y, raw_events, i)
feature_vector = np.array([
    x_norm, y_norm, t, p, dt, dt_pixel, 
    neighbor_count,  # 新特征
    *np.zeros(output_dim - 7)
])
```

### 2. 调整模型架构
修改 `configs/config.yaml` 中的模型参数：
- 增加 `n_layers` 提升模型容量
- 调整 `d_model` 改变表示维度
- 修改 `d_state` 优化SSM性能

### 3. 自定义数据格式
在 `src/datasets.py` 中修改数据加载逻辑以支持不同的输入格式。

## 🐛 故障排除 (Troubleshooting)

### 常见问题
1. **内存不足**: 减少 `batch_size` 或 `sequence_length`
2. **训练太慢**: 减少 `n_layers` 或使用GPU
3. **精度不够**: 增加 `epochs` 或调整学习率
4. **依赖安装失败**: 使用conda或指定版本安装

### 调试技巧
- 使用 `test_pipeline.py` 验证数据和模型逻辑
- 检查 `mock_mamba.py` 是否在缺少mamba-ssm时正确加载
- 监控训练过程中的损失变化

## 📚 参考文献 (References)

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
2. Event-based Vision: A Survey
3. DVS Event Camera Data Processing Techniques

## 🤝 贡献指南 (Contributing)

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 📄 许可证 (License)

MIT License - 详见LICENSE文件

---

*本项目为事件相机闪光去除的研究实现，适用于学术研究和工程应用。*