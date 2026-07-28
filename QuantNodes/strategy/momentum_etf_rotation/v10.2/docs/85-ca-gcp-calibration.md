# 85 · CA-GCP 超参校准 (343 Grid + 早停)

## 摘要

CA-GCP 默认超参（k=8, η=0.5, τ=60）在 38 ETF 池上过保守（marginal cov 99.9% vs 95% 目标）。本文实现 343 组合 grid search（k ∈ 7 维 × η ∈ 7 维 × τ ∈ 7 维），用 val 期覆盖率为准则选 Pareto 最优，并加入"连续 50 组合无新最优即停"的早停策略。

## 1. 性能瓶颈分析

`CAGCPipeline.predict()` 复杂度：
```
O(N_assets × T_test × |neighbors| × |calib_days| × log(|pool|))
= O(38 × 252 × 9 × 252 × log(9 × 252))
≈ O(2.4 亿)  # 单次 fit+predict
```

343 grid × 30s/次 = **3 小时**（不可接受）。

## 2. 加速方案

| 优化 | 加速比 | 实现 |
|------|--------|------|
| **预排序 pool + 预计算 cumsum** | 5-10× | `weighted_quantile_v2` |
| **批量构造 pool (向量化)** | 3-5× | `pipeline._pool_scores_v2` |
| **NumPy `np.searchsorted` 二分** | 2× | 替代 linear search |
| **早停 (50 阈值)** | 1.5-3× | grid 提前结束 |

**目标**：单次 fit+predict 从 ~30s 降到 < 3s，343 grid 从 3h 降到 < 30min。

## 3. Grid 范围（343 组合）

```python
k   ∈ [2, 4, 6, 8, 12, 16, 24]      # 邻居数（论文默认 8）
η   ∈ [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]  # 调制器敏感度
τ   ∈ [20, 30, 45, 60, 90, 120, 180]    # 时间衰减尺度（天）
```

## 4. 评估准则（Pareto Score）

```python
score = +10.0 * extreme_cov   # 极端日覆盖权重最高
        -5.0 * pa_std         # 越小越好
        -1.0 * (width_bps / 1000)  # 宽度惩罚
```

## 5. 早停策略

**规则**：连续 50 组合没有产生新 Pareto 最优 → 停止。

**理由**：
- grid 是笛卡尔积，遍历顺序影响搜索方向
- 50 组合足够覆盖"局部最优盆地"
- 总能跑前 50 组合保证有 baseline

## 6. 输出

| 文件 | 角色 |
|------|------|
| `data/results/calibration_grid.csv` | 343 行实验结果 |
| `data/results/best_params.json` | Pareto 最优超参 |
| `v10.2/__init__.py` 读取 `best_params.json` 作为默认 | 生产级集成 |

## 7. 论文参考

Parker & Zhang (2026) Sec. 5.7:
> "CA-GCP is not sensitive to precise hyperparameter choices, reinforcing its practicality."

→ 我们希望验证 ETF 池上是否同样"不敏感"，如果 grid 跑完发现最优在中间值附近，说明需要重新校准。

## 8. 局限

- 仅 1 个 val 期（2022-04 ~ 2023-04），未 walk-forward
- 343 组合可能仍不足以找到全局最优
- `sharpness_p` 未参与 grid（固定 1.0）
- 早停 50 阈值是经验值，未校准