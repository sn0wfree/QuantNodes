---
id: L-131
title: HMM 在 A 股 2018-2026 几乎只识别"震荡市"
severity: MEDIUM
auto_checkable: manual
category: methodology
related_lessons: []
related_daily: [L-20260709-10]
source: 05_LESSONS_LIBRARY.md
---

# L-131: HMM 在 A 股只识别震荡市

## 一句话总结
HMM regime-based meta-allocator 难以发挥作用, 必须配合 distance prior。

## 问题描述
无先验 HMM = 几乎只识别震荡市。
distance prior 公式:
```python
POTENTIAL = {0:0.0, 1:0.5, 2:1.0}  # bear / transition / bull
rate[i,j] = exp(-α × d_eff(i,j))
d_eff = d + γ × (potential[i] - potential[j])
```

## 检测 prompt (给 Agent 的检查清单)

1. **HMM 是否有 distance prior**:
   - 若是无先验 HMM, 几乎无用
   - 必须用 distance prior HMM

2. **POTENTIAL 配置**:
   - bear=0.0, transition=0.5, bull=1.0

## 正确做法

```python
def hmm_with_distance_prior(returns, alpha=1.0, gamma=0.5):
    """HMM with distance prior (识别 bear/transition/bull)."""
    POTENTIAL = {0: 0.0, 1: 0.5, 2: 1.0}
    hmm = GaussianHMM(n_components=3)

    # 设置转移率基于距离 + 潜力差
    for i in range(3):
        for j in range(3):
            d = abs(i - j)
            d_eff = d + gamma * (POTENTIAL[i] - POTENTIAL[j])
            hmm.transmat_[i, j] = np.exp(-alpha * d_eff)
    return hmm
```

## 历史教训来源
- 首次发现: v4 HMM 深度研究 (`64dcba4`, `bcc3b9a`)