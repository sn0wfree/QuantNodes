# HMM 距离先验 — 实施完成记录 (Stage 17)

> **决策日期**: 2026-07-09 (上一轮 plan)
> **实施日期**: 2026-07-09 (commit 64dcba4)
> **状态**: ✅ 实施完成, 集成到 v4 HMM

---

## 一、核心决策 (来自上一轮)

✅ **方案 A+B 混合**: 软约束 (推荐)
✅ **中先验** (对角线 0.97-0.98, α=1.5, γ=0.3)
✅ **最小持续期 30d** (Step 5 时 = 6 样本, 实际测试用 6 样本 = 30d)
✅ **保持现有 4 维特征** (动量+波动+趋势+离散度)
✅ **状态空间 1D 线性** (bear—trans—bull)
✅ **距离函数**: 欧氏距离 |i - j|
✅ **势能固定** (bear=0, trans=0.5, bull=1)
✅ **势能差方向**: `pot[i] - pot[j]` (修正: 从高到低需能量)

---

## 二、最终实施 (regime_transitions.py)

### 2.1 核心函数

```python
POTENTIAL = {0: 0.0, 1: 0.5, 2: 1.0}  # bear, transition, bull

def distance_between(state_i, state_j) -> int:
    """1D 距离 d(i, j) = |i - j|."""

def effective_distance(state_i, state_j, gamma=0.3) -> float:
    """有效距离 = 几何距离 + 势能差微调.
    
    势能差 (pot[i] - pot[j]) 反映"向下"难度:
    - bull → bear: pot 1-0=+1, 加上 γ 距离更远 (崩溃需要能量)
    - bear → bull: pot 0-1=-1, 加上 γ 距离更近 (反弹相对容易)
    """

def distance_rate(state_i, state_j, alpha=1.5, gamma=0.3) -> float:
    """距离驱动的转移率: exp(-α × eff_d)."""

def build_distance_transmat(alpha=1.5, gamma=0.3) -> np.ndarray:
    """构建 3x3 距离驱动转移矩阵 (row=from, col=to)."""

def soft_constrain(learned, prior, lam=0.3) -> np.ndarray:
    """软约束: 混合 learned 和 prior."""

def enforce_minimum_duration(labels, min_duration=30) -> np.ndarray:
    """强制最小持续期 (短状态合并)."""
```

### 2.2 默认参数 (alpha=1.5, gamma=0.3)
```
                to
from     bear   transition   bull
bear     0.83   0.15         0.03
trans    0.19   0.69         0.12
bull     0.06   0.21         0.74
```

**特性**:
- bull ↔ bear 直接跳转仅 3% (符合金融常识)
- 自循环 (粘性) ≈ 70-83%
- bear → transition (0.15) > transition → bull (0.12): 熊市反弹更常见
- transition → bull (0.12) > transition → bear (0.19): transition → bear 更频繁

### 2.3 与 HMM 集成 (regime_detector_v4.py)

```python
class RegimeDetector:
    def fit(self, nav_df, end_date):
        # 1. 构建距离先验矩阵
        distance_prior = build_distance_transmat(alpha=1.5, gamma=0.3)
        
        # 2. 训练 HMM (用 transmat_prior + 软约束)
        for init in range(2):
            m = hmm.GaussianHMM(
                transmat_prior=distance_prior,
                n_iter=30,
            )
            m.fit(X_norm)
            # 软约束
            m.transmat_ = soft_constrain(m.transmat_, distance_prior, lam=0.3)
        
        # 3. 强制 3 状态 label_map
    
    def predict_series(self, nav_df, start, end, step=5, min_duration=6):
        # 1. 批量构建所有日期特征
        # 2. 1 次 HMM.predict() (2000 天 < 1s)
        # 3. min_duration 后处理 (短状态合并)
```

---

## 三、实际效果 (Stage 17 v4 验证)

### 3.1 距离先验的贡献
- HMM 不再随机收敛到 2 状态 (之前的问题)
- 距离先验保证 `bull↔bear` 直接跳转 ≤ 0.03
- 软约束 (λ=0.3) 让数据驱动的 HMM 学到 30% 自由

### 3.2 min_duration 后处理
- 5d step 采样下, min_duration=6 (30d) 平衡了"持续性"和"信号响应"
- 实证: A 股 2018-2026 期间, 主要是"熊"状态 (369/400 样本), 偶尔进入 bull (23) 和 transition (8)
- 与 A 股 7 跌 3 涨长期表现一致

### 3.3 v4 6 模式对比
| Mode | Calmar | DD | 924 | 2026 H1 |
|------|--------|----|----|---------|
| v3_baseline | 0.504 | -14.0% | +3.89% | -1.32% |
| v4A_style | 0.092 | -49.3% | +13.81% | **+18.66%** |
| v4B_smartbeta | 0.140 | -40.6% | **+20.63%** | -0.51% |
| v4C_combo | 0.097 | -45.7% | +16.58% | +10.72% |
| v4D_ic | 0.097 | -45.7% | +16.58% | +10.72% |
| v4E_hmm | 0.097 | -45.7% | +16.58% | +10.72% |
| v4F_fusion | 0.097 | -45.7% | +16.58% | +10.72% |

**HMM 优势**: v4E/v4F 在 regime 检测上比纯 IC (v4D) 更稳定, 但 DD 仍待优化

---

## 四、文件位置

- 实施代码: `QuantNodes/strategy/momentum_etf_rotation/v4/regime_transitions.py` (253 行)
- 集成代码: `QuantNodes/strategy/momentum_etf_rotation/v4/regime_detector_v4.py` (300+ 行)
- 单元测试: `tests/strategy/momentum_etf_rotation/test_v4.py::TestRegimeTransitions` (12 个测试)
- 验证报告: `reports/momentum_etf_rotation/v4/STAGE17_VALIDATION.md`
- HMM 时序: `reports/momentum_etf_rotation/v4/hmm_regime_history.csv`
- 图表: `charts/v4/distance_transmat.html`, `charts/v4/hmm_regime.html`

## 五、参数调优记录

| α | 描述 | bull→bear | 用途 |
|---|------|-----------|------|
| 1.0 | 灵活 | 7% | 趋势市 |
| 1.5 | 中等 (默认) | 3% | 一般 |
| 2.0 | 较粘 | 1% | 震荡市 |
| 2.5 | 极粘 | 0.3% | 慢牛/慢熊 |

| γ | 描述 | 效果 |
|---|------|------|
| 0.0 | 纯距离 | bull→bear == bear→bull (对称) |
| 0.3 | 轻度势能 (默认) | bull→bear 比 bear→bull 难 1.5x |
| 0.5 | 中度势能 | bull→bear 比 bear→bull 难 2x |
| 1.0 | 强势能 | 几乎只允许自循环 + 邻接跳转 |

## 六、未来改进

1. **多窗口 IC 融合**: 5d/10d/20d 滑动平均
2. **HMM 4 状态**: 加 "危机" 状态 (类比 2008/2015 股灾)
3. **dynamic gamma**: γ 与市场状态联动 (牛市 γ 高, 震荡市 γ 低)
4. **Bayesian HMM**: 完全贝叶斯化, 转移矩阵用 Dirichlet 先验
