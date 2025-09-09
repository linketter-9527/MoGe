# NaN Detection in Training

## 概述

本实现提供了一个自定义的 `NanDetectionHook`，用于在训练过程中检测损失值和梯度中的 NaN 和 Inf 值。当检测到数值异常时，会输出警告信息到日志中。

## 功能特性

- ✅ 检测损失值中的 NaN 和 Inf
- ✅ 检测梯度中的 NaN 和 Inf  
- ✅ 详细的参数级警告信息
- ✅ 与 MMCV 训练框架无缝集成
- ✅ 支持分布式训练

## 使用方法

### 1. 自动启用（推荐）

在 `tools/train_dnf.py` 中已经自动集成了 NaN 检测功能。当运行训练脚本时，会自动启用：

```bash
python tools/train_dnf.py configs/your_config.py
```

### 2. 手动配置

如果需要在配置文件中手动启用，可以在配置文件中添加：

```python
optimizer_config = dict(
    type='NanDetectionHook',
    detect_loss_nan=True,
    detect_grad_nan=True,
    grad_clip=dict(max_norm=35, norm_type=2)  # 可选：梯度裁剪
)
```

### 3. 参数说明

- `detect_loss_nan` (bool): 是否检测损失值中的 NaN/Inf（默认：True）
- `detect_grad_nan` (bool): 是否检测梯度中的 NaN/Inf（默认：True）
- `grad_clip` (dict): 梯度裁剪配置（可选）

## 检测机制

### 损失值检测
在每次训练迭代后，检查损失值是否包含 NaN 或 Inf：
```python
if torch.isnan(loss).any() or torch.isinf(loss).any():
    logger.warning(f'NaN/Inf detected in loss at iteration {iter}')
```

### 梯度检测
在反向传播后，检查所有参数的梯度：
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            logger.warning(f'NaN/Inf detected in gradient for {name}')
```

## 输出示例

当检测到数值异常时，日志中会显示类似以下信息：

```
WARNING - NaN or Inf detected in loss at iteration 123. Loss value: nan
WARNING - NaN detected in gradient for parameter backbone.conv1.weight at iteration 123
WARNING - Inf detected in gradient for parameter neck.conv2.bias at iteration 123
WARNING - NaN/Inf gradients detected in parameters: ['backbone.conv1.weight', 'neck.conv2.bias']
```

## 故障排除

### 常见 NaN/Inf 原因
1. **学习率过高** - 尝试降低学习率
2. **梯度爆炸** - 启用梯度裁剪 (`grad_clip`)
3. **数据预处理问题** - 检查输入数据范围
4. **损失函数问题** - 检查损失计算
5. **数值不稳定操作** - 检查模型中的数学运算

### 调试建议

1. **启用梯度裁剪**:
```python
optimizer_config = dict(
    type='NanDetectionHook',
    detect_loss_nan=True,
    detect_grad_nan=True,
    grad_clip=dict(max_norm=10, norm_type=2)
)
```

2. **降低学习率**:
```python
optimizer = dict(type='AdamW', lr=1e-5, weight_decay=0.01)
```

3. **检查数据**: 验证输入数据是否包含异常值

## 文件结构

```
mmseg/
└── core/
    └── hooks/
        ├── __init__.py          # 注册 NanDetectionHook
        └── nan_detection_hook.py # NaN 检测钩子实现

tools/
└── train_dnf.py                 # 自动启用 NaN 检测
```

## 测试

运行测试脚本验证功能：
```bash
python test_nan_detection.py
```

## 注意事项

1. NaN 检测会增加轻微的训练开销（约 1-2%）
2. 在检测到 NaN 时，训练不会自动停止，只会输出警告
3. 建议结合梯度裁剪使用以提高训练稳定性
4. 对于生产环境，可以考虑在检测到 NaN 时自动保存检查点

## 自定义扩展

如果需要更复杂的处理逻辑（如自动停止训练、保存检查点等），可以继承 `NanDetectionHook` 并重写相关方法。