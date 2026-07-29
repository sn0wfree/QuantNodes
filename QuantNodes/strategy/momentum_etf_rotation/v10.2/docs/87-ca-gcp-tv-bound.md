# 87 · CA-GCP 理论覆盖保证 (TV 距离上界)

## 摘要

前 4 个 phase 用实证覆盖率验证 CA-GCP（marginal cov 99.9%, extreme 99.8%）。本文实现论文公式 9 的**解析上界**：基于 Total-Variation (TV) 距离的 coverage gap bound，对比理论最坏情况 vs 实证结果。

## 1. 理论背景

论文公式 9 (基于 Barber et al. 2023, Annals of Statistics):

```
|coverage_gap(v)| ≤ Σ_{v' ∈ N(v)} w_{v',v} × TV(P_{s̃_{v'}}, P_{s̃_v})
                  └──── 加权平均 ────┘   └──── 总变差距离 ──┘
```

**TV 距离定义**：
```
TV(P, Q) = sup_A |P(A) - Q(A)| = (1/2) Σ_x |P(x) - Q(x)|
```

**两个关键设计**（论文 Sec. 4.6）：
1. **波动率归一化** → 消除最大异质性来源 → 归一化分数在不同资产间分布接近
2. **proximity 权重** `corr^p` → 集中到最相似 peer → 加权 TV 进一步缩小

## 2. TV 距离估计方法（经验 CDF 差）

**离散分布**：
```python
def total_variation_distance_ecdf(scores_p, scores_q):
    """
    Sort samples from P and Q together, compute empirical CDFs,
    TV = sup over x of |F_P(x) - F_Q(x)|.
    """
    all_pts = np.concatenate([scores_p, scores_q])
    sorted_pts = np.unique(all_pts)

    F_p = np.searchsorted(np.sort(scores_p), sorted_pts, side='right') / len(scores_p)
    F_q = np.searchsorted(np.sort(scores_q), sorted_pts, side='right') / len(scores_q)

    return float(np.max(np.abs(F_p - F_q)))
```

**优势**：
- 理论最严谨（直接实现 TV 定义）
- 无超参（无 bin size 选择）
- 计算快（O((n+m) log(n+m))）

**对比其他方法**（Q4 回答 c）：
- 直方图离散：bin size 敏感
- KDE：核宽敏感
- **经验 CDF 差：理论严谨，最稳健**

## 3. API 设计

```python
# ca_gcp/validators/theoretical_bound.py

@dataclass
class TheoreticalBound:
    """Per-asset theoretical coverage gap bound."""
    code: str
    tv_bound: float                  # Σ w_v × TV(P_v, P_neighbor_v)
    empirical_gap: float             # 1 - marginal_coverage (实测)
    bound_empirical_ratio: float     # tv_bound / empirical_gap

def theoretical_coverage_bound(
    pipeline: CAGCPipeline,
    scores_calib: pd.DataFrame,
    codes: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """每个资产的理论覆盖 gap 上界 (论文公式 9)."""
```

## 4. 实验设计

**对比**：
1. **理论覆盖 gap bound**: 公式 9 算的最坏情况
2. **实证覆盖 gap**: `1 - marginal_coverage`（实测）
3. **Bound / Empirical ratio**: 理论界是否合理（应 ≥ 1）

**输出**：
- `data/results/theoretical_bound.csv`: 38 资产 × (tv_bound, empirical_gap, ratio)
- `data/results/tv_distance_heatmap.png`: 资产对 TV 距离热图

## 5. 论文参考

Parker & Zhang (2026) Sec. 4.6:
> "Two design choices keep this penalty small in practice. First, volatility normalization (Sec. IV-C) removes the largest source of heterogeneity... Second, the proximity weights concentrate mass on the most similar peers..."

> "the realized coverage of CA-GCP is close to nominal throughout our experiments" — 我们的实验验证 bound vs empirical 的接近程度。

## 6. 局限

- 经验 CDF TV 距离在小样本下高估（n=252 calib days）
- bound 是最坏情况，实际通常远低于 bound
- 未验证 bound 是否对所有 regime 严格成立
- 假设 exchangeability 在池内近似成立（论文也承认）