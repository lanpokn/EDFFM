# 3x3邻域特征维度一致性分析报告

## 问题识别

用户提出了两个关键问题：

### 1. 滤波参数使用问题

**发现**: 在当前实现中，PFD的滤波参数（tau, sigma, var_threshold等）确实**没有用于实际滤波决策**，而只是：

- `delta_tf`: 用于Mf计算的时间窗口（实际使用）
- `tau`, `sigma`: **未用于滤波判断**，仅作为注释中的参考
- `var_threshold`, `neighbor_threshold`: **完全未使用**

**原因**: 我们的方法是将PFD的计算逻辑作为**特征提取**，让神经网络学习复杂的去噪决策，而不是直接应用PFD的简单阈值判断。

**结论**: 这些参数应该移除，只保留实际使用的`delta_tf`。

### 2. 3x3邻域特征维度不一致问题

**核心问题**: 如果实现3x3邻域，不同位置的像素会有不同的邻居数量：

- **内部点**: 9个邻居 (3x3完整)
- **边缘点**: 6个邻居 (3x2或2x3)  
- **角点**: 4个邻居 (2x2)

这会导致特征向量维度不一致！

## 解决方案分析

### 方案1: 固定维度输出（推荐）

将邻域特征聚合为固定维度的统计量：

```python
def extract_3x3_features(x, y, maps):
    # 获取有效邻居
    neighbors = get_valid_neighbors(x, y, 3)
    
    # 聚合为固定维度特征
    ma_total = sum(maps['Mf'][ny, nx] for ny, nx in neighbors)
    ne_count = len([1 for ny, nx in neighbors if maps['Count'][ny, nx] > 0])
    
    # 固定输出维度的特征
    features = {
        'ma_3x3': ma_total,
        'ne_3x3': ne_count, 
        'd_3x3': ma_total / ne_count if ne_count > 0 else 0,
        'neighbor_count': len(neighbors),  # 有效邻居数
        'avg_activity': ma_total / len(neighbors) if neighbors else 0,
        'max_activity': max([maps['Mf'][ny, nx] for ny, nx in neighbors], default=0)
    }
    return features
```

**优势**:
- 输出维度固定，网络结构一致
- 包含边界信息（neighbor_count）
- 计算高效

### 方案2: 零填充法

```python
def extract_3x3_padded_features(x, y, maps):
    features = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                features.append(maps['Mf'][ny, nx])
            else:
                features.append(0)  # 边界外填零
    return features  # 始终返回9维特征
```

**问题**: 零填充可能引入人工偏差

### 方案3: 相对位置编码

```python
def extract_3x3_relative_features(x, y, maps):
    center_activity = maps['Mf'][y, x]
    relative_features = []
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # 跳过中心点
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                # 相对于中心点的特征差异
                relative_features.append(maps['Mf'][ny, nx] - center_activity)
            else:
                relative_features.append(-center_activity)  # 边界相对差异
    
    return relative_features  # 8维（排除中心点）
```

## 网络性能影响分析

### 理论分析

**维度不一致的影响**:
1. **参数学习困难**: 网络无法为不同维度输入学习一致的权重
2. **批处理问题**: 同一batch内特征维度不同，无法有效并行
3. **泛化能力下降**: 边界和内部的不同处理可能导致位置偏差

### 实验验证设计

```python
# 测试不同位置特征的一致性
def test_feature_consistency():
    positions = [
        (1, 1),      # 角点 - 4邻居
        (1, 50),     # 边缘 - 6邻居  
        (50, 50),    # 内部 - 8邻居
        (100, 150)   # 内部 - 8邻居
    ]
    
    for pos in positions:
        features = extract_features(pos)
        print(f"Position {pos}: {len(features)} dimensions")
        # 检查维度一致性
```

## 推荐实现方案

基于分析，推荐使用**方案1**（固定维度聚合）：

```python
def compute_3x3_pfd_features(self, x, y, maps):
    """计算3x3邻域的固定维度PFD特征"""
    neighbors = self.get_valid_neighbors(x, y, 3)
    
    if not neighbors:
        return [0] * 6  # 返回固定6维零特征
    
    # 收集邻域统计信息
    neighbor_mfs = [maps['Mf'][ny, nx] for ny, nx in neighbors]
    neighbor_counts = [maps['Count'][ny, nx] for ny, nx in neighbors]
    
    # 聚合为固定维度特征
    ma_total = sum(neighbor_mfs)
    ne_count = len([c for c in neighbor_counts if c > 0])
    
    return [
        ma_total,                                    # 邻域总活动
        ne_count,                                    # 活跃邻居数
        ma_total / ne_count if ne_count > 0 else 0, # 平均密度
        len(neighbors),                              # 有效邻居数（边界指示器）
        max(neighbor_mfs) if neighbor_mfs else 0,   # 最大活动
        min(neighbor_mfs) if neighbor_mfs else 0    # 最小活动
    ]

def get_valid_neighbors(self, x, y, size):
    """获取有效邻居坐标"""
    neighbors = []
    radius = size // 2
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue  # 排除中心点
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                neighbors.append((nx, ny))
    return neighbors
```

## 性能预期

**优势**:
- 维度完全一致，网络训练稳定
- 保留边界信息（有效邻居数）
- 计算开销适中
- 特征具有明确的物理意义

**可能的性能提升**:
- 边界效应的显式建模
- 更丰富的空间上下文信息
- 更好的噪声-信号区分能力

## 结论

1. **移除未使用的滤波参数**，保持代码简洁
2. **采用固定维度聚合方案**实现3x3邻域
3. **显式处理边界情况**，避免维度不一致
4. **保留边界指示特征**，让网络自适应学习边界处理策略

这样既解决了特征维度一致性问题，又充分利用了空间邻域信息。

---

*分析完成时间: 2025年1月*
*建议优先级: 高 - 涉及网络结构稳定性*