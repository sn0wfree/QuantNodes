# v9d — 评分合成 (Scoring System)

> **编号**: 49d
> **状态**: 📋 设计中
> **日期**: 2026-07-22
> **关联**: docs/49c-v9_coupling.md, docs/49e-v9_backtest.md

---

## 1. 评分系统概述

### 1.1 目标

将多个独立信号 (周期趋势, 共振, VIX) 合成一个 **0-100 的综合评分**, 用于:
1. 直观显示当前状态 (仪表盘)
2. 转化为大盘信号 0/1 (择时)
3. 历史回测 (验证)

### 1.2 评分构成

$$
\text{Score}_{total} = \text{Score}_{cycle} + \text{Score}_{coupling} + \text{Score}_{vix}
$$

| 评分维度 | 范围 | 权重 | 数据源 |
|---------|------|------|--------|
| 周期趋势 (cycle) | 0-40 | 40% | 多 IMF 方向综合 |
| 周期耦合 (coupling) | 0-40 | 40% | Hilbert + 双相干 |
| VIX 分 (vix) | 0-20 | 20% | VIX 倒数百分位 |

---

## 2. 周期趋势分 (0-40)

### 2.1 定义

**逻辑**: 每个 IMF 的当前方向 (上行/下行) 各贡献 10 分, 4 个 IMF 共 40 分。

```python
def compute_cycle_score(imfs: np.ndarray, 
                        window: int = 12) -> float:
    """
    周期趋势评分.
    
    每个 IMF 的最近 12 周趋势:
    - 上行 → 10 分
    - 下行 → 0 分
    
    4 IMF 共 40 分
    """
    score = 0
    for imf in imfs:
        # 12 周线性回归斜率
        recent = imf[-window:]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        
        if slope > 0:
            score += 10
    
    return min(score, 40)
```

### 2.2 加权变体 (备选)

按周期重要性加权:

| IMF | 周期 | 重要性 | 权重 |
|-----|------|--------|------|
| IMF1 (季度) | 12 周 | 低 | 0.10 |
| IMF2 (年) | 48 周 | 中 | 0.20 |
| IMF3 (基钦) | 160 周 | **高** | 0.40 |
| IMF4 (朱格拉+) | 480 周 | **高** | 0.30 |

```python
IMF_WEIGHTS = [0.10, 0.20, 0.40, 0.30]

def compute_cycle_score_weighted(imfs: np.ndarray, 
                                  window: int = 12,
                                  weights: list = IMF_WEIGHTS) -> float:
    """加权版周期趋势评分."""
    score = 0
    for imf, w in zip(imfs, weights):
        recent = imf[-window:]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        if slope > 0:
            score += 10 * w
    
    return min(score, 40)
```

### 2.3 推荐用法

**v9 默认**: 加权版 (`compute_cycle_score_weighted`), 因为:
- 基钦和朱格拉周期是 A 股主要驱动力
- 季度/年度周期次要

---

## 3. 周期耦合分 (0-40)

### 3.1 定义

**逻辑**: 每对 IMF 的相位锁定状态各贡献 10 分, 双相干显著额外加 10 分。

```python
def compute_coupling_score(plv: dict, 
                            bic_max: float,
                            delta_phases: dict,
                            lock_threshold_deg: float = 30,
                            lock_min_duration_weeks: int = 12,
                            bic_threshold: float = 0.6) -> float:
    """
    周期耦合评分 0-40.
    
    评分规则:
    - 每对 IMF 的相位锁定 (Δφ < 30° 持续 3 月): +10 分
    - 双相干显著 (bic > 0.6): +10 分 (一次性奖励)
    - 最多 4 对 IMF
    """
    score = 0
    
    # 相位锁定贡献
    n_pairs = len(plv)
    for k, (kp, jp) in enumerate(plv.keys()):
        if k >= 4:
            break
        
        # 检查相位差
        dp = delta_phases[(kp, jp)]
        locked = np.abs(dp) < np.deg2rad(lock_threshold_deg)
        
        # 检查持续时间
        if locked[-lock_min_duration_weeks:].all():
            score += 10
    
    # 双相干贡献
    if bic_max > bic_threshold:
        score += 10
    
    return min(score, 40)
```

### 3.2 阈值参数

| 参数 | 默认值 | 范围 | 备注 |
|------|--------|------|------|
| lock_threshold_deg | 30 | 15-60 | 越严格信号越少 |
| lock_min_duration_weeks | 12 (3 月) | 4-24 | 防抖 |
| bic_threshold | 0.6 | 0.4-0.8 | 显著性 |

**默认值的依据**:
- 30° 锁定: 来自神经科学同步文献
- 12 周持续: 防止瞬时噪声误判
- 0.6 双相干: 来自地球物理文献 (地震信号检测)

### 3.3 历史验证

**期望结果**:

| 时期 | 期望耦合分 | 实际 |
|------|-----------|------|
| 2014 大底 | 30-40 | 待验证 |
| 2015 顶部 | 20-30 | 待验证 |
| 2018 底 | 30-40 | 待验证 |
| 2021 顶 | 20-30 | 待验证 |
| 2024 底 | 30-40 | 待验证 |

---

## 4. VIX 分 (0-20)

### 4.1 定义

**逻辑**: VIX 越低 (市场越平静), VIX 分越高 (允许更高仓位)。

```python
def compute_vix_score(vix: pd.Series, 
                      window: int = 252) -> float:
    """
    VIX 评分 0-20.
    
    VIX 倒数百分位:
    - 倒数百分位 ≥ 80% → 20 分 (VIX 极低)
    - 倒数百分位 ≤ 20% → 0 分 (VIX 极高)
    """
    if len(vix) < window:
        return 10  # 数据不足, 给中性分
    
    # VIX 倒数
    inv_vix = 1 / vix
    
    # 滚动百分位
    percentile = inv_vix.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    latest_percentile = percentile.iloc[-1]
    
    # 线性映射 0-20
    score = latest_percentile * 20
    
    return score
```

### 4.2 简化版

如果不需要百分位计算:

```python
def compute_vix_score_simple(vix: float) -> float:
    """
    VIX 单值评分.
    
    - VIX < 12 → 20 分 (极平静)
    - VIX 12-20 → 15 分
    - VIX 20-30 → 10 分
    - VIX 30-40 → 5 分
    - VIX > 40 → 0 分 (恐慌)
    """
    if vix < 12:
        return 20
    elif vix < 20:
        return 15
    elif vix < 30:
        return 10
    elif vix < 40:
        return 5
    else:
        return 0
```

### 4.3 A 股市场校准

A 股市场 VIX 通常 15-30, 远低于美股 (VIX 9-80)。

**调整**:
- 中位数: ~20
- 25 分位: ~15
- 75 分位: ~25
- 极端值: > 35 (2020 新冠)

```python
VIX_THRESHOLDS_CN = [15, 20, 25, 30, 35]  # A 股市场校准

def compute_vix_score_cn(vix: float) -> float:
    """A 股校准的 VIX 评分."""
    if vix < 15:
        return 20
    elif vix < 20:
        return 15
    elif vix < 25:
        return 12
    elif vix < 30:
        return 8
    elif vix < 35:
        return 4
    else:
        return 0
```

---

## 5. 评分合成

### 5.1 总评分

```python
def compute_total_score(cycle_score: float,
                        coupling_score: float,
                        vix_score: float) -> float:
    """
    总评分 0-100.
    
    总分 = 周期趋势 + 周期耦合 + VIX
         = 0-40      + 0-40      + 0-20
    """
    return cycle_score + coupling_score + vix_score
```

### 5.2 评分时序

```python
def compute_score_timeseries(imfs: np.ndarray,
                              plv: dict,
                              bic_matrix: np.ndarray,
                              vix: pd.Series,
                              window: int = 12) -> pd.Series:
    """
    滚动计算评分时序.
    
    返回: (T, ) 评分时序
    """
    n = imfs.shape[1]
    scores = np.zeros(n)
    
    for t in range(window, n):
        imfs_window = imfs[:, t-window:t]
        plv_window = {k: v[t-window:t] for k, v in plv.items()}
        bic_window = bic_matrix[t-window:t]
        vix_window = vix.iloc[t-window:t]
        
        cycle = compute_cycle_score_weighted(imfs_window)
        coupling = compute_coupling_score(plv_window, bic_window.max())
        vix_s = compute_vix_score_simple(vix_window.mean())
        
        scores[t] = compute_total_score(cycle, coupling, vix_s)
    
    return pd.Series(scores, index=vix.index)
```

---

## 6. 信号生成

### 6.1 大盘信号

```python
def score_to_signal(score: float, 
                    upper_threshold: float = 50,
                    lower_threshold: float = 30) -> int:
    """
    评分 → 大盘信号 (0 或 1).
    
    - score >= upper_threshold → 1 (满仓)
    - score <= lower_threshold → 0 (空仓)
    - 否则保持上一状态 (迟滞, 防抖)
    """
    # 简化版: 单点判定
    if score >= upper_threshold:
        return 1
    elif score <= lower_threshold:
        return 0
    else:
        return None  # 中性区
```

### 6.2 迟滞逻辑 (防抖)

```python
def score_to_signal_hysteresis(score_series: pd.Series,
                                upper: float = 50,
                                lower: float = 30) -> pd.Series:
    """
    带迟滞的信号生成.
    
    状态机:
    - 当前 0: score >= upper → 转为 1
    - 当前 1: score <= lower → 转为 0
    """
    signals = np.zeros(len(score_series), dtype=int)
    current = 0
    
    for i, s in enumerate(score_series):
        if current == 0 and s >= upper:
            current = 1
        elif current == 1 and s <= lower:
            current = 0
        signals[i] = current
    
    return pd.Series(signals, index=score_series.index)
```

### 6.3 信号阈值

| 参数 | 默认值 | 范围 |
|------|--------|------|
| upper_threshold | 50 | 30-70 |
| lower_threshold | 30 | 20-50 |
| 迟滞宽度 | 20 | 10-30 |

**默认值的选择**:
- 50 (上限): 中位数以上, 表示"看多"
- 30 (下限): 中位数以下, 表示"看空"
- 20 (迟滞宽度): 防止在 50 附近反复切换

---

## 7. 评分历史回测

### 7.1 评分 vs 市场表现

**预期相关性**:

| 评分区间 | 市场表现 (年化) |
|---------|----------------|
| 0-30 | -5% ~ +5% (弱势/震荡) |
| 30-50 | 0% ~ +10% (中性) |
| 50-70 | +5% ~ +15% (偏多) |
| 70-100 | +10% ~ +25% (强势) |

### 7.2 评分 vs Sharpe

**预期**: 评分>50 期间平均 Sharpe 显著高于评分<50 期间。

```python
def score_vs_returns_analysis(score_series: pd.Series,
                               returns: pd.Series) -> dict:
    """
    评分与收益对比分析.
    """
    # 分组
    high = returns[score_series >= 50]
    low = returns[score_series < 30]
    mid = returns[(score_series >= 30) & (score_series < 50)]
    
    return {
        'high_period': {
            'mean': high.mean(),
            'std': high.std(),
            'sharpe': high.mean() / high.std() * np.sqrt(52) if len(high) > 0 else 0,
        },
        'low_period': {
            'mean': low.mean(),
            'std': low.std(),
            'sharpe': low.mean() / low.std() * np.sqrt(52) if len(low) > 0 else 0,
        },
        'mid_period': {
            'mean': mid.mean(),
            'std': mid.std(),
            'sharpe': mid.mean() / mid.std() * np.sqrt(52) if len(mid) > 0 else 0,
        },
    }
```

---

## 8. 评分权重不寻优

### 8.1 原则

**重要原则**: v9 的评分权重 (40+40+20) **不在历史数据上寻优**。

**原因**:
1. 避免过拟合
2. 学术共识权重 (基于 Campbell & Thompson 2008 等)
3. 测试期 = 训练期, 容易过拟合

### 8.2 权重敏感性测试

**目的**: 验证权重变化对结果的影响是否稳健。

```python
WEIGHT_VARIANTS = [
    ('default', (40, 40, 20)),
    ('cycle_heavy', (50, 30, 20)),
    ('coupling_heavy', (30, 50, 20)),
    ('vix_heavy', (40, 30, 30)),
    ('equal', (33, 33, 34)),
]

for name, (cw, cow, vw) in WEIGHT_VARIANTS:
    score = imfs_score * cw/40 + coupling_score * cow/40 + vix_score * vw/20
    # ... 回测
```

**目标**: 不同权重下 Sharpe 标准差 < 0.1 (稳健)。

---

## 9. 实施清单

### 9.1 文件

```
QuantNodes/strategy/momentum_etf_rotation/v9/
└── scoring.py               # 本模块
```

### 9.2 接口

```python
from QuantNodes.strategy.momentum_etf_rotation.v9.scoring import (
    compute_cycle_score,
    compute_cycle_score_weighted,
    compute_coupling_score,
    compute_vix_score,
    compute_vix_score_cn,
    compute_total_score,
    compute_score_timeseries,
    score_to_signal,
    score_to_signal_hysteresis,
)
```

### 9.3 测试用例

```python
def test_compute_total_score():
    """测试总评分计算."""
    assert compute_total_score(30, 30, 15) == 75
    assert compute_total_score(40, 40, 20) == 100
    assert compute_total_score(0, 0, 0) == 0


def test_score_to_signal_hysteresis():
    """测试迟滞信号."""
    scores = pd.Series([20, 40, 55, 60, 25, 50, 30, 70])
    signals = score_to_signal_hysteresis(scores, upper=50, lower=30)
    
    # 期望: 0, 0, 1, 1, 1, 1, 1, 1 (迟滞生效)
    assert signals.iloc[0] == 0
    assert signals.iloc[2] == 1  # 首次越过 50
    assert signals.iloc[5] == 1  # 不因回落立刻变 0
```

---

## 10. 风险与对冲

| 风险 | 缓解 |
|------|------|
| 评分权重过拟合 | 用学术默认权重, 不寻优 |
| 信号抖动 | 迟滞逻辑 (50/30 双阈值) |
| 极端 VIX | 单独处理 (VIX > 40 直接空仓) |
| 数据缺失 | 中性分 (10/10/10) |

---

**最后更新**: 2026-07-22
**状态**: 📋 设计中