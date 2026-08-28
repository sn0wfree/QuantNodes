# CTA均值回归模块核心逻辑分析

## 一、策略架构

```
15个因子 → 独立计算信号 → inv_vol加权合成 → vol_targeting → 执行
```

## 二、信号生成流程

### 2.1 因子信号计算

```python
# 每个因子独立计算信号
for name, cls, input_type in FACTORS:
    factor = cls()
    if continuous and has raw_value:
        sig = factor.raw_value(data)  # 连续值
        z = rolling_zscore(sig, 252)  # 时序z-score
        z = clip(z, -3, 3) / 3      # 归一化到[-1,1]
    else:
        sig = factor.compute(data)    # 离散{-1, 0, +1}
```

### 2.2 inv_vol加权合成

```python
# 关键: 波动率低的因子权重更高
for t in range(0, T, step):
    x = factor_signals[t-window:t]  # 过去window期的因子值
    std = np.nanstd(x, axis=0)      # 每个因子的波动率
    w = 1.0 / std                    # 波动率倒数
    w = w / w.sum()                  # 归一化
    weights[t:t+step] = w            # 应用权重

# 合成信号 = Σ(因子信号 × 权重)
module_signal = Σ(factor_i × weight_i)
```

### 2.3 方向信号

```python
# 信号 → 方向
direction = sign(module_signal)  # {-1, 0, +1}
```

## 三、仓位管理

### 3.1 波动率目标化

```python
# 每品种: 仓位 ∝ target_vol / realized_vol
weight_per_unit = target_vol / rolling_vol
weight_per_unit = clip(weight_per_unit, max_leverage)

# 最终仓位 = 方向 × 权重
position = direction × weight_per_unit
```

### 3.2 资金分配

```python
# 当日活跃品种等分资金
n_active = (weights != 0).sum()
capital_per_product = equity / n_active

# 手数 = 权重 × 资金 / (价格 × 合约乘数)
lots = weights × capital_per_product / notional_per_lot
```

## 四、核心机制解析

### 4.1 为什么inv_vol有效？

```python
# 等权: 所有因子同等重要
weight_equal = 1/15 = 6.7%

# inv_vol: 波动率低的因子权重更高
# 假设因子A波动率=0.1, 因子B波动率=0.5
weight_A = (1/0.1) / (1/0.1 + 1/0.5) = 83%
weight_B = (1/0.5) / (1/0.1 + 1/0.5) = 17%
```

**原理**：
- 波动率低 → 信号稳定 → 噪音少 → 应该信任 → 权重高
- 波动率高 → 信号不稳定 → 噪音多 → 应该怀疑 → 权重低

### 4.2 为什么始终在场？

```python
# 信号方向
direction = sign(module_signal)

# 15个因子中:
# - 4个真正均值回归 (Bias/CCI/WilliamsR/HalfLife)
# - 2个趋势跟踪 (Bollinger/PriceLocation)
# - 3个K线形态 (Doji/Hammer/Engulfing)
# - 6个其他 (Stochastic/Squeeze/CloseLocation/OpenLocation/Gap/SpreadMR)

# 大多数时候, 至少有一些因子给出信号
# 所以module_signal很少为0
# 所以direction很少为0
# 所以始终在场
```

### 4.3 为什么趋势市更赚钱？

```python
# 趋势市:
# - 趋势因子 (Bollinger/PriceLocation) 给出强信号
# - 均值回归因子 (CCI/WilliamsR) 给出弱信号
# - inv_vol加权: 趋势因子权重高, 均值回归因子权重低
# - 结果: 整体信号跟随趋势 → 赚钱

# 震荡市:
# - 趋势因子给出弱信号
# - 均值回归因子给出强信号
# - inv_vol加权: 均值回归因子权重高
# - 但均值回归因子本身不赚钱 (OOS Sharpe -0.325)
# - 结果: 整体信号跟随均值回归 → 亏钱
```

## 五、与"均值回归"名字的矛盾

| 特征 | 真正的均值回归 | 这个策略 |
|---|---|---|
| 持仓时间 | 短期 (几天) | 始终在场 |
| 交易方向 | 仅反转 | 双向 |
| 趋势市表现 | 差 | **好** |
| 震荡市表现 | 好 | **差** |
| 因子组成 | 纯反转 | 混合 |

**结论**：这是一个"始终在场+双向交易+波动率目标化"的策略，不是真正的均值回归。

## 六、收益来源分解

```
总收益 = 做多贡献 + 做空贡献 - 交易成本

10.6% = 11.0% + 6.7% - 约2% - 退出损失
```

### 6.1 做多贡献 (11.0%)

- 来源: 趋势跟踪因子 + 部分均值回归因子
- 机制: 价格上涨时做多赚钱
- 胜率: 49.3%

### 6.2 做空贡献 (6.7%)

- 来源: 趋势跟踪因子 + 部分均值回归因子
- 机制: 价格下跌时做空赚钱
- 胜率: 45.0%

### 6.3 空仓亏损 (-36.8%)

- 来源: 退出时机错误
- 机制: 离场后价格继续朝有利方向移动
- 教训: 始终在场比择时更重要

## 七、关键参数

| 参数 | 值 | 作用 |
|---|---|---|
| target_vol | 0.20 | 目标波动率20% |
| max_leverage | 2.0 | 最大杠杆2倍 |
| aggregation | inv_vol | 波动率倒数加权 |
| agg_window | 252 | 权重计算窗口 (1年) |
| agg_update | 21 | 权重更新频率 (月频) |
| execution_lag | 1 | T+1执行 |
| min_position_change | 0.30 | 仓位变化<30%不调仓 |

## 八、与简单MR的对比

| 维度 | CTA MR (10.6%) | 简单MR (-0.5%) |
|---|---|---|
| 因子数 | 15个 | 4个 |
| 加权 | inv_vol | 等权 |
| 仓位管理 | vol_targeting | 固定 |
| 交易方向 | 双向 | 仅做多 |
| 换手控制 | min_position_change | 无 |
| 始终在场 | 是 (1%空仓) | 否 |

## 九、核心insight

1. **inv_vol加权是关键**：从5.9%提升到10.6%
2. **始终在场比择时重要**：空仓亏损-36.8%
3. **双向交易增加收益**：做空贡献6.7%
4. **波动率目标化控制风险**：仓位按波动率调整
5. **趋势市更赚钱**：与"均值回归"名字矛盾

## 十、对铜策略的启示

| 启示 | 应用 |
|---|---|
| inv_vol加权 | 因子权重优化 |
| 始终在场 | 不要频繁切换 |
| 双向交易 | 可以做空 |
| vol_targeting | 仓位管理 |
| 趋势因子有效 | 趋势跟踪可能更好 |
