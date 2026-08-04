# v10.3 I: 连续风险-趋势混合方案

## 1. 问题

G 方案的二元切换（D/F模式）导致权重突变，风险模型只在 F 模式生效。
H 方案改善有限（Sharpe +0.7%），因为风险模型被月度调仓频率压制。

## 2. 新架构

**核心思想**: 用风险概率连续调节仓位大小，而不是切换模式。

```
旧: risk_prob → 二元切换 (D/F) → 权重突变
新: risk_prob + trend_strength → 连续混合 → 权重渐变
```

## 3. blended_min_defensive 函数

```python
def blended_min_defensive(trend_strength: float, risk_prob: float) -> float:
    """连续混合的防御底仓.

    Args:
        trend_strength: [0,1], 0=弱趋势, 1=强趋势
        risk_prob: [0,1], 0=低风险, 1=高风险

    Returns:
        min_defensive: [0.25, 0.90]
    """
    risk_factor = risk_prob
    trend_factor = 1 - trend_strength
    combined = 0.6 * risk_factor + 0.4 * trend_factor
    return 0.25 + combined * (0.90 - 0.25)
```

## 4. 行为示例

| 场景 | trend | risk | min_def | v1.0 |
|------|-------|------|---------|------|
| 强趋势+低风险 | 0.9 | 0.2 | 0.34 | ~34% |
| 强趋势+高风险 | 0.9 | 0.8 | 0.58 | ~58% |
| 弱趋势+低风险 | 0.1 | 0.2 | 0.52 | ~52% |
| 弱趋势+高风险 | 0.1 | 0.8 | 0.76 | ~76% |

## 5. 调仓频率

- 始终周频调仓 (每月 4-5 次)
- 无二元模式切换
- 每次调仓重新计算 blended_min_defensive

## 6. 实现改动

文件: `dynamic_weight_schemes.py`

新增:
- `blended_min_defensive(trend_strength, risk_prob)` 函数
- `scheme_i_blended(prices)` 函数
- 参数 `V10_3I_RISK_WEIGHT = 0.6` (风险权重)

## 7. 预期收益

1. 权重变化平滑 (无二元切换突变)
2. 风险模型持续生效 (不再只在 F 模式)
3. 趋势+风险双信号 (比单一信号更稳健)
4. 更低的换手 (连续变化 vs 二元切换)
