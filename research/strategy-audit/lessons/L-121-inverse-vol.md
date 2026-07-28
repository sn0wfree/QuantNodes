---
id: L-121
title: 逆波动率加权是稳健基线
severity: MEDIUM
auto_checkable: agent
category: methodology
related_lessons: [L-106]
related_daily: [L-20260709-8]
source: 05_LESSONS_LIBRARY.md
---

# L-121: 逆波动率加权是稳健基线

## 一句话总结
逆波动率加权 (OOS Calmar 0.488 → 0.589, +20.7%) 是与 v1/v3/v5 全策略一致的稳健基线。

## 问题描述
```python
σ = std(log_ret_60d) × √252
weights ∝ 1 / max(σ, vol_floor)
```
高波动品种自动降权。

## 检测 prompt (给 Agent 的检查清单)

1. **加权方法**:
   - 是否使用 `weights = 1/σ`?
   - vol_floor 是否设置 (避免 σ=0 时除零)?

2. **vol_floor 默认值**:
   - 推荐 0.01 (年化波动率下限)

3. **max_weight**:
   - 可放宽到 0.30 (高波动自动降权)

## 正确做法

```python
def inverse_vol_weights(returns: pd.DataFrame, vol_window: int = 60) -> pd.Series:
    """逆波动率加权 (NaN-safe)."""
    log_ret = np.log(returns / returns.shift(1))
    vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    inv_vol = 1.0 / vol.clip(lower=0.01)  # vol_floor
    weights = inv_vol / inv_vol.sum()
    return weights.clip(upper=0.30)  # max_weight
```

## 历史教训来源
- 首次发现: v3 时期逆波动 + v5.1 升级 (`9157b31`)