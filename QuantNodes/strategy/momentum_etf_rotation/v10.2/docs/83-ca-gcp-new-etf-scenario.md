# 83 · CA-GCP 新 ETF 借用强度评估 (场景 B)

## 摘要

当轮动池中加入**新成立的 ETF**（如 AI、新能源主题），其历史只有 20-60 天，per-asset CP 的 95% 分位数因校准样本不足而剧烈抖动。CA-GCP 通过**借邻居的校准分数**缓解此问题，但"借多少"取决于邻居质量。本文实现论文 Sec. 5.4 的"借用强度评分"，并对比 4 种方法在 scarce-asset 设置下的覆盖率稳定性。

## 1. 问题定义

**设定**：38 ETF 池，其中 5 只被人工截断到 H=20/40/80/160 天校准期（模拟"新 ETF"），其余 33 只保留完整 252 天。

**对比方法**：
| 方法 | 校准策略 | 借用邻居？ |
|------|---------|-----------|
| PerAsset-CP | 单资产 raw residuals | ❌ |
| Vol-CP (VAC-FF) | 单资产 vol-normalized | ❌ |
| CA-GCP full | 跨邻居加权池化 | ✅ 全部邻居 |
| CA-GCP conditional | 仅借用 `neighbor_quality > 阈值` 的邻居 | ⚖️ 条件借用 |

**评估指标**：
- Scarce-asset Cov Std（论文主指标）：稀缺资产的覆盖率标准差，越小越稳
- Marginal Coverage（应接近 95%）
- Interval Width（宽度）

## 2. 借用强度评分

**定义**（基于论文 Sec. 5.4 + 本项目的 `sharpness_p=1.0`）：

```python
NeighborQuality(target=v):
    n_neighbors = len(N(v))
    weighted_corr_sum = Σ corr(v, v')^p for v' in N(v)
    effective_sample_size = (252 + |N(v)| * 252) * (weighted_corr_sum / n_neighbors)
    borrow_recommendation:
        "strong"  if weighted_corr_sum >= 5.0
        "moderate" if 2.0 <= weighted_corr_sum < 5.0
        "weak"    if weighted_corr_sum < 2.0
```

**直觉**：
- `weighted_corr_sum` 高 = 邻居高度相关 = 借用可信
- `effective_sample_size` 反映"借数据后的等效校准样本数"
- `weak` 邻居借用反而引入噪声，应回退 PerAsset-CP

## 3. 预期结果（论文 Table II 复刻）

| H (calib days) | Method | Scarce Cov (%) | Cov Std | Width (bps) |
|----------------|--------|----------------|---------|-------------|
| 20 | PerAsset-CP | 95.86 | 4.36 | 575 |
| 20 | Vol-CP | 96.80 | 3.58 | 686 |
| **20** | **CA-GCP** | **95.79** | **0.82** | **501** |
| 40 | PerAsset-CP | 93.80 | 4.16 | 437 |
| 40 | Vol-CP | 92.80 | 3.96 | 442 |
| **40** | **CA-GCP** | **95.68** | **0.86** | **496** |
| 80 | Vol-CP | 94.60 | 2.48 | 481 |
| **80** | **CA-GCP** | **95.71** | **0.81** | **496** |
| 160 | Vol-CP | 95.03 | 1.45 | 497 |
| **160** | **CA-GCP** | **95.78** | **0.79** | **499** |

**核心收益**：H=20 时，CA-GCP 的 Cov Std 比 Vol-CP **低 4.4 倍**（0.82 vs 3.58），同时宽度比 Vol-CP **窄 27%**（501 vs 686 bps）。

## 4. 文件清单

| 文件 | 角色 |
|------|------|
| `ca_gcp/validators/neighbor_quality.py` | 借用强度评分 API |
| `experiments/09_new_etf_scenario.py` | scarce-asset 对比实验 |
| `tests/test_neighbor_quality.py` | 单元测试 |
| `data/results/scarce_table.csv` | 实验输出（参考论文 Table II）|
| `data/results/borrow_recommendations.csv` | 每个资产的借用建议 |

## 5. 局限

- 当前 fixed H 值扫描，未做 walk-forward
- "new ETF" 是模拟的（截断老 ETF），非真新 ETF 数据
- conditional 策略的阈值 `strong=5.0 / moderate=2.0` 未在 calib 期 grid search

## 6. 论文参考

Parker & Zhang (2026), Sec. 5.4 Table II:
> "CA-GCP holds the standard deviation of scarce-asset coverage below 0.9% at every level of H, while the per-asset methods fluctuate between 1.5% and 4.4%."