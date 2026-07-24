# v8 Jump Model 综合诊断总结

**日期**: 2026-07-22

## 核心发现

### Step 1: 数据

- ✅ 训练期内**存在重大回撤**（>15%），市场有真实的熊市期

### Step 2: 代码

- **15/15** 个训练组合成功检测到 bear 状态
- 平均 centroids 距离: 1.2695

### Step 3: 参数

- **n_restarts**: bear_days 范围 [712, 713]
- **train_window**: bear_days 范围 [254, 761]
- **random_state**: bear_days 范围 [712, 713]
- **jump_penalty**: bear_days 范围 [194, 712]

## Step 4 实验设计建议

部分训练组合能检测到 bear 状态，**Walk-forward 可用但需要优化参数**。

**建议**: 增加 n_restarts 到 10 重做 Walk-forward。
