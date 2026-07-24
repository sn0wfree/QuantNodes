# v8 per-asset + 动态仓位 实施方案 (2026-07-24)

> 日期: 2026-07-24
> 计划: 借鉴 v9 银河方案的 risk_scalar 机制, 解决 v8 per-asset 错过 924 行情的致命缺陷
> 路径: B (PoC, 25min) → A (实施, 60min) → C (优化, 50min)

---

## 一、问题定义

### 1.1 现状

v8 per-asset 5bp 表现:
- **Sharpe 0.871** (5bp), **MaxDD -18.14%**, **换手率 ~15x**
- v7.10 TV-PR: Sharpe 0.922, MaxDD -20.54%, AnnRet 17.89%

### 1.2 核心缺陷: v8 错过 924 行情

| 日期 | 沪深300 | per-asset | v7.10 | 真正发生 |
|------|---------|-----------|-------|---------|
| 2024-09-30 | +9.95% | +2.73% | +6.16% | 924 高潮 |
| 2024-09-24~10-08 周期 | +37% | +3.6% | +24% | 924 政策红利 |
| 2024-10-09 | -9.28% | -2.14% | -8.44% | 924 后获利回吐 |

**核心问题**: per-asset 在 924 周期只捕获 10%, 比 v7.10 少赚 20%。

### 1.3 v9 银河方案可借鉴机制

v9 4 大通用技术:
1. **熵权法综合得分**: 信息论动态权重
2. **滚动 β 估计**: 因子敏感度
3. **风险预算权重**: `w ∝ |β| × target / σ²`
4. **动态仓位 risk_scalar**: `risk_scalar(t) = (1 - 0.8 × zscore).clip(0.3, 1.5)`

**关键**: 风险_scalar 是 v9 真正带来 alpha 的机制 (Sharpe 0.565 → 0.843, +49%)。

---

## 二、最终架构

```
v8 per-asset 5bp (现有)         新增 Layer              预期
├─ per-asset sigmoid 0.50        ├─ 动态仓位 risk_scalar
├─ 月末评估                      ├─ 5 宏观因子熵权综合
└─ 整体满仓 (1.0)                └─ risk_scalar(t) 整体调整
                                  │
                                  final_position = per_asset_adj × risk_scalar
```

---

## 三、factor_score 设计 (5 真实宏观因子)

| # | 因子 | 计算 | 含义 |
|---|------|------|------|
| 1 | **增长** | 沪深300 (510300) 月收益 | 增长好 → 满仓 |
| 2 | **通胀 (反向)** | -黄金 (518880) 月收益 | 黄金涨 → 减仓 |
| 3 | **流动性** | 短债 (511260) - 沪深300 | 比率升 → 宽松 |
| 4 | **汇率** | 海外 (513500) - 沪深300 | 超额 → 外资流入 |
| 5 | **风险偏好** | 沪深300 - 中证500 | 大盘强 → 避险 |

**全部用 ETF 池已有数据, 无需付费数据源。**

---

## 四、三阶段执行计划

### Phase B: PoC (25 min) — 先验证可行性

**目标**: 验证 **5 宏观因子 + 熵权综合得分 + risk_scalar** 这个思路是否真的能解决问题（特别是 924 行情捕获）。

#### Step B.1: factor_score 模块骨架 (15 min)

**新建文件**: `QuantNodes/strategy/momentum_etf_rotation/v9/factor_score_basic.py`

```python
"""5 真实宏观因子 → 熵权综合得分 → risk_scalar."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_five_macro_factors(daily_returns: pd.DataFrame, lookback: int = 21):
    """5 个真实宏观因子 (周频).

    全部用 ETF 池内已有的 ETF 计算, 不需要付费数据源.

    参数:
        daily_returns: 日频收益 DataFrame (含 510300/518880/511260/513500/510500)
        lookback: 周频回望 (默认 21 天 ≈ 1 月)

    返回:
        zscore_factors: (T, 5) DataFrame, 52 周 z-score
    """
    # 周频收益 (取每周最后一日)
    weekly = daily_returns.resample('W').last().pct_change()

    hs300 = weekly['510300']  # 沪深 300
    gold = weekly['518880']   # 黄金
    bond = weekly['511260']   # 短债
    overseas = weekly['513500']  # 纳指 (海外)
    zz500 = weekly['510500']  # 中证 500

    factors = pd.DataFrame({
        'growth': hs300,
        'inflation': -gold,  # 反向: 黄金涨 = 通胀 = 减仓
        'liquidity': bond - hs300,  # 流动性: 短债超额
        'fx': overseas - hs300,  # 汇率: 海外超额
        'risk_preference': hs300 - zz500,  # 风险偏好: 大盘相对小盘
    })

    # 52 周滚动 z-score 标准化
    zscore_factors = (factors - factors.rolling(52).mean()) / (
        factors.rolling(52).std() + 1e-10
    )
    return zscore_factors


def entropy_weight(data: pd.DataFrame, window: int = 104) -> dict:
    """熵权法计算指标权重 (借鉴 v9/factor_galaxy.py).

    原理:
        信息熵大 → 指标越无序 → 权重小
        信息熵小 → 指标信息量大 → 权重大

    参数:
        data: 标准化后的因子 DataFrame
        window: 滚动窗口 (默认 104 = 2 年)

    返回:
        weights: {col: weight} 每个因子的权重 (归一化和=1)
    """
    if len(data) < window:
        return {col: 1.0 / len(data.columns) for col in data.columns}

    recent = data.iloc[-window:]
    n = len(recent)
    weights = {}

    for col in recent.columns:
        x = recent[col].abs()
        p = x / x.sum()
        entropy = -(p * np.log(p + 1e-10)).sum() / np.log(n)
        weights[col] = 1 - entropy

    total = sum(weights.values())
    if total == 0:
        return {col: 1.0 / len(data.columns) for col in data.columns}
    return {k: v / total for k, v in weights.items()}


def compute_factor_score(daily_returns: pd.DataFrame) -> pd.Series:
    """主入口: 5 因子 → 熵权综合得分."""
    factors = compute_five_macro_factors(daily_returns)
    weights = entropy_weight(factors, window=104)
    composite = sum(factors[col] * w for col, w in weights.items())
    return composite


def compute_risk_scalar(
    factor_score: pd.Series,
    window: int = 52,
    clip_low: float = 0.3,
    clip_high: float = 1.5,
) -> pd.Series:
    """借鉴 v9: dynamic position adjustment.

    risk_scalar(t) = (1 - 0.8 × zscore).clip(clip_low, clip_high)

    当宏观好 (zscore > 0) → risk_scalar < 1 (减仓, 偏防御)
    当宏观差 (zscore < 0) → risk_scalar > 1 (加仓, 偏进攻)
    """
    zscore = (factor_score - factor_score.rolling(window).mean()) / (
        factor_score.rolling(window).std() + 1e-10
    )
    risk_scalar = (1 - 0.8 * zscore).clip(clip_low, clip_high)
    return risk_scalar
```

#### Step B.2: 924 验证脚本 (10 min)

**新建脚本**: `scripts/combo/poc_factor_score_924.py`

```python
"""PoC: 验证 factor_score + risk_scalar 在 924 期间的行为."""
import sys
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'QuantNodes'))

import pandas as pd
from v9.factor_score_basic import compute_factor_score, compute_risk_scalar

daily_returns = pd.read_parquet('data/high_freq_macro/v56_expanded_daily.parquet')
factor_score = compute_factor_score(daily_returns)
risk_scalar = compute_risk_scalar(factor_score)

# 输出: 924 期间 (2024-09-24 ~ 2024-10-08) 的 risk_scalar
period = slice('2024-09-20', '2024-10-15')

print("=" * 80)
print("📊 factor_score + risk_scalar 在 924 期间的行为")
print("=" * 80)
print(f"\n{'日期':12s} {'factor_score':>14s} {'risk_scalar':>14s}")
print("-" * 50)
for date in pd.date_range('2024-09-20', '2024-10-15', freq='W'):
    if date in factor_score.index:
        fs = factor_score.loc[date]
        rs = risk_scalar.loc[date] if date in risk_scalar.index else 'N/A'
        print(f"{date.date():12s} {fs:>14.4f} {rs:>14.4f}" if isinstance(rs, float) else f"{date.date():12s} {fs:>14.4f} {rs:>>14s}")

print("\n关键验证:")
print("- 924 期间 (9/24 ~ 10/8) risk_scalar > 0.9 → 满仓")
print("- 924 后获利回吐 (10/9) risk_scalar < 0.7 → 减仓")
```

**验证标准 (通过)**:
- 924 期间 risk_scalar > 0.9 (允许满仓, 不错过行情)
- 924 后回吐时 risk_scalar < 0.7 (减仓防御)

**失败处理**: 回到 B.1 调整因子权重或 rolling window

---

### Phase A: 完整实施 (60 min) — B 通过后

**目标**: 完整实现 v8 per-asset + 动态仓位整合方案

#### Step A.1: 整合脚本 (25 min)

**新建脚本**: `scripts/combo/regenerate_v8_dynamic_position.py`

```python
"""v8 per-asset sigmoid 月末调仓 + 动态仓位 risk_scalar.

架构:
  Layer 1: per-asset sigmoid 月末调仓 (v8 现有)
  Layer 2: 整体仓位 risk_scalar(t) 动态调整 (v9 借鉴)

  final_position[t] = per_asset_adj[t] × risk_scalar[t]
"""
import sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'QuantNodes'))

from QuantNodes.strategy.momentum_etf_rotation.v9.factor_score_basic import (
    compute_factor_score, compute_risk_scalar,
)
from QuantNodes.strategy.momentum_etf_rotation.v8_integrated_comparison import (
    load_v7_14_portfolio,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import compute_metrics

OOS_START = pd.Timestamp('2021-08-01')
OUT_DIR = REPO / 'reports' / 'momentum_etf_rotation' / 'combo'
HF_DIR = REPO / 'data' / 'high_freq_macro'


def sigmoid_adj(P_bear, threshold=0.50):
    """per-asset sigmoid 仓位函数 (复用 v8 per-asset)."""
    if pd.isna(P_bear):
        return 1.0
    x = (P_bear - threshold) * 10
    return 1.0 / (1.0 + np.exp(x))


def compute_nav_dynamic_position(
    weekly_weights, daily_returns, signals,
    factor_score, risk_scalar,
    clip_low=0.3, clip_high=1.5,
    cost_bp=20,
):
    """per-asset sigmoid + 动态仓位整合."""
    common_codes = [c for c in weekly_weights.columns if c in daily_returns.columns]
    weekly_weights = weekly_weights[common_codes]
    daily_returns = daily_returns[common_codes]
    all_dates = daily_returns.index
    weekly_dates = weekly_weights.index

    # weekly_bear_pct
    weekly_bear_pct = {}
    for code in common_codes:
        if code in signals and 'P_bear' in signals[code].columns:
            bear_pct = signals[code]['P_bear']
            weekly_bear = bear_pct.reindex(weekly_dates, method='ffill')
            weekly_bear_pct[code] = weekly_bear

    date_to_adjusted_weights = {}
    last_ww = None
    last_per_asset_adj = {code: 1.0 for code in common_codes}

    for i, wd in enumerate(weekly_dates):
        after = all_dates[all_dates > wd]
        if len(after) == 0:
            continue
        start = after[0]
        if i + 1 < len(weekly_dates):
            next_wd = weekly_dates[i + 1]
            before_next = all_dates[all_dates <= next_wd]
            if len(before_next) == 0:
                continue
            end = before_next[-1]
        else:
            end = all_dates[-1]

        # 月末判断
        is_month_end = False
        if i + 1 < len(weekly_dates):
            is_month_end = (wd.month != next_wd.month)
        else:
            is_month_end = True

        if is_month_end:
            last_ww = weekly_weights.loc[wd].copy()
            for asset in common_codes:
                if asset not in weekly_bear_pct:
                    continue
                p_bear = weekly_bear_pct[asset].loc[wd]
                if pd.isna(p_bear):
                    p_bear = 0.0
                last_per_asset_adj[asset] = sigmoid_adj(p_bear)

        # 应用 Layer 1 (per-asset sigmoid 月末)
        if last_ww is not None:
            adj_weights = last_ww.copy()
        else:
            adj_weights = weekly_weights.loc[wd].copy()
        for asset in common_codes:
            if asset in last_per_asset_adj:
                adj_weights[asset] *= last_per_asset_adj[asset]

        # 应用 Layer 2 (动态仓位 risk_scalar) ← 关键新增
        rs = 1.0
        if wd in risk_scalar.index:
            rs = float(risk_scalar.loc[wd])
        rs = max(clip_low, min(clip_high, rs))
        adj_weights = adj_weights * rs

        # 归一化
        total = adj_weights.sum()
        if total > 1.0:
            adj_weights = adj_weights / total

        # 写入生效期
        mask = (all_dates >= start) & (all_dates <= end)
        for d in all_dates[mask]:
            date_to_adjusted_weights[d] = adj_weights.copy()

    # NAV 计算
    nav = pd.Series(1.0, index=all_dates, dtype=float)
    prev_w = pd.Series(0.0, index=common_codes)
    for i in range(1, len(all_dates)):
        d = all_dates[i]
        w = date_to_adjusted_weights.get(d)
        if w is not None:
            row = daily_returns.loc[d]
            if row[common_codes].isna().all():
                nav.iloc[i] = nav.iloc[i - 1]
            else:
                ret = row.fillna(0.0)
                port_ret = float((w * ret).sum())
                cost_factor = 1.0
                if cost_bp > 0:
                    turnover = float((w - prev_w).abs().sum())
                    cost_factor = max(1.0 - turnover * cost_bp / 10000.0, 0.0)
                nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret) * cost_factor
                prev_w = w.copy()
        else:
            nav.iloc[i] = nav.iloc[i - 1]

    return nav


def main():
    log = lambda msg: print(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log("=" * 70)
    log("v8 per-asset + 动态仓位 risk_scalar")
    log("=" * 70)

    log("加载数据...")
    daily_returns = pd.read_parquet(HF_DIR / 'v56_expanded_daily.parquet')
    weekly_weights, _, _ = load_v7_14_portfolio()

    log("生成 factor_score (5 宏观因子)...")
    factor_score = compute_factor_score(daily_returns)

    log("计算 risk_scalar (52 周滚动)...")
    risk_scalar = compute_risk_scalar(factor_score)

    log("加载 P_bear 信号...")
    with open('scripts/combo/signals_prob.pkl', 'rb') as f:
        signals = pickle.load(f)

    # 5 种风险偏好 × 4 成本档 = 20 个组合
    configs = [
        {'clip_low': 0.5, 'clip_high': 1.0, 'name': 'R1_极保守'},
        {'clip_low': 0.3, 'clip_high': 1.5, 'name': 'R2_标准'},  # v9 默认
        {'clip_low': 0.4, 'clip_high': 1.3, 'name': 'R3_温和'},
        {'clip_low': 0.1, 'clip_high': 2.0, 'name': 'R4_激进'},
        {'clip_low': 0.6, 'clip_high': 1.2, 'name': 'R5_保守防御'},
    ]

    cost_tiers = [5, 10, 15, 20]

    results = []
    for cfg in configs:
        for cost in cost_tiers:
            log(f"\n--- {cfg['name']} (clip [{cfg['clip_low']}, {cfg['clip_high']}]) cost={cost}bp ---")
            t0 = time.time()
            nav = compute_nav_dynamic_position(
                weekly_weights, daily_returns, signals,
                factor_score, risk_scalar,
                clip_low=cfg['clip_low'], clip_high=cfg['clip_high'],
                cost_bp=cost,
            )
            elapsed = time.time() - t0

            oos = nav.loc[OOS_START:].dropna()
            rets = oos.pct_change().dropna()
            sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
            peak = oos.cummax()
            dd = oos / peak - 1
            max_dd = dd.min()
            ann_ret = rets.mean() * 252
            calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

            log(f"  Sharpe={sharpe:.3f} Calmar={calmar:.3f} AnnRet={ann_ret:.2%} MaxDD={max_dd:.2%} ({elapsed:.1f}s)")

            results.append({
                'name': cfg['name'],
                'clip_low': cfg['clip_low'],
                'clip_high': cfg['clip_high'],
                'cost_bp': cost,
                'Sharpe': sharpe,
                'Calmar': calmar,
                'AnnRet': ann_ret,
                'MaxDD': max_dd,
            })

    df = pd.DataFrame(results)
    csv_path = OUT_DIR / 'v8_dynamic_position_comparison.csv'
    df.to_csv(csv_path, index=False)
    log(f"\n✅ 对比表: {csv_path}")
    log(df.sort_values(['clip_low', 'cost_bp']).to_string(index=False))


if __name__ == '__main__':
    main()
```

#### Step A.2: 后台运行 (20 min)

```bash
tmux new-session -d -s v8_dyn 'python3.11 scripts/combo/regenerate_v8_dynamic_position.py 2>&1 | tee /tmp/v8_dynamic.log'
```

#### Step A.3: 输出对比表 (5 min)

输出到 `v8_dynamic_position_comparison.csv`, 列包括:
- name, clip_low, clip_high, cost_bp, Sharpe, Calmar, AnnRet, MaxDD

#### Step A.4: 最终对比 v7.10 / v8 per-asset 5bp (10 min)

在对比表中加入 baseline 列:
| 策略 | Sharpe | Calmar | MaxDD |
|------|--------|--------|-------|
| v7.10 TV-PR (5bp) | 0.922 | 0.871 | -20.54% |
| v8 per-asset 5bp (当前) | 0.871 | 0.739 | -18.14% |
| **v8 + dynamic_最优** | ? | ? | ? |

---

### Phase C: 优化 + 综合归因 (50 min) — A 完成且 Sharpe 改善后

**目标**: 调频率、调参数、综合归因

#### Step C.1: factor_score 频率优化 (15 min)

**新建脚本**: `scripts/combo/regenerate_v8_freq_optimize.py`

```python
"""测试 6 种 factor_score 频率 + rolling window 组合."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path('.')
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'QuantNodes'))

from v9.factor_score_basic import compute_five_macro_factors

# 测试 6 种组合
configs = [
    {'freq': 'W', 'window': 26, 'name': 'F1_周频_半年窗'},
    {'freq': 'W', 'window': 52, 'name': 'F2_周频_一年窗'},
    {'freq': '2W', 'window': 52, 'name': 'F3_双周频_一年窗'},
    {'freq': 'M', 'window': 52, 'name': 'F4_月频_一年窗'},
    {'freq': 'M', 'window': 104, 'name': 'F5_月频_二年窗'},
    {'freq': '2M', 'window': 26, 'name': 'F6_双月频_半年窗'},
]

# ... 类似 Phase A, 跑 6 种频率组合
# 输出最优频率
```

#### Step C.2: risk_scalar 参数网格 (15 min)

**新建脚本**: `scripts/combo/regenerate_v8_risk_grid.py`

```python
"""risk_scalar 参数网格搜索."""
configs = [
    # (clip_low, clip_high, coef_0.8)
    {'clip_low': 0.3, 'clip_high': 1.5, 'coef': 0.8},
    {'clip_low': 0.5, 'clip_high': 1.0, 'coef': 0.8},
    {'clip_low': 0.3, 'clip_high': 2.0, 'coef': 0.5},
    {'clip_low': 0.4, 'clip_high': 1.3, 'coef': 1.0},
    # ... 12 种组合
]
```

#### Step C.3: 综合归因 (20 min)

**新建文档**: `docs/64-v8_dynamic_position.md`

归因内容:
1. **924 行情真实归因**: 用沪深300 数据验证
2. **整体 MaxDD 控制**: 与 v8 per-asset 对比
3. **慢牛跟随**: 不减仓场景
4. **慢熊防御**: 减仓场景

用沪深300 数据验证关键日期:
- 2024-09-24 ~ 10-08: per-asset 捕获率从 10% 到 50%+?
- 2024-10-09: risk_scalar 是否 < 0.7?
- 慢熊 2022-Q1: 减仓到 30%?

---

## 五、产出文件清单

### 新建
- `QuantNodes/strategy/momentum_etf_rotation/v9/factor_score_basic.py`
- `scripts/combo/poc_factor_score_924.py`
- `scripts/combo/regenerate_v8_dynamic_position.py`
- `scripts/combo/regenerate_v8_freq_optimize.py`
- `scripts/combo/regenerate_v8_risk_grid.py`
- `reports/momentum_etf_rotation/combo/v8_dynamic_position_comparison.csv`
- `reports/momentum_etf_rotation/combo/v8_dynamic_position_*.parquet`
- `docs/64-v8_dynamic_position.md`

### 复用
- `scripts/combo/signals_prob.pkl` (v8 P_bear 信号)
- v8 per-asset 框架 (调仓逻辑)

---

## 六、关键决策点

| 阶段 | 通过标准 | 不通过则 |
|-------|---------|---------|
| **B 完成** | 924 期间 risk_scalar > 0.9, 10-09 风险 < 0.7 | 回到 B.1 重新设计 factor_score |
| **A 完成** | Sharpe > 0.9 (vs v8 per-asset 5bp 0.871) | 跳到 Phase C 调参数 |
| **C 完成** | Sharpe > 0.95 | 输出结论, 不再优化 |

---

## 七、时间线

```
Phase B (PoC, 25 min)
  ├── B.1: factor_score 模块 (15 min)
  ├── B.2: 924 验证 (10 min)
  ↓ 验证通过
Phase A (实施, 60 min)
  ├── A.1: 整合脚本 (25 min)
  ├── A.2: 5 × 4 = 20 组合测试 (20 min)
  ├── A.3: 对比表 (5 min)
  ├── A.4: 最终对比 v7.10 (10 min)
  ↓ Sharpe 改善
Phase C (优化, 50 min)
  ├── C.1: frequency 优化 (15 min)
  ├── C.2: 参数网格 (15 min)
  ├── C.3: 综合归因 (20 min)

总时间: ~135 min (2.25 小时)
```

---

## 八、风险评估

| 风险 | 影响 | 对策 |
|------|------|------|
| 滞后风险 | 宏观因子变化滞后 5-10 天 | Phase C 测试不同 window |
| 冷启动 | 需要 2 年历史计算熵权 | 利用现有 8 年数据, 回测前 warmup |
| 过拟合 | 5 宏观 + v8 17 因子, 参数过多 | 用 4.9 年 OOS 测试, 不做样本内调参 |
| 数据依赖 | 5 ETF 都是池中, 不依赖付费 | ✓ 无风险 |

---

## 九、执行命令清单

```bash
# Phase B
mkdir -p QuantNodes/strategy/momentum_etf_rotation/v9
# 创建 factor_score_basic.py
# 创建 poc_factor_score_924.py

python3.11 scripts/combo/poc_factor_score_924.py
# 验证: 看 924 期间 risk_scalar > 0.9

# Phase A
# 创建 regenerate_v8_dynamic_position.py
tmux new-session -d -s v8_dyn 'python3.11 scripts/combo/regenerate_v8_dynamic_position.py 2>&1 | tee /tmp/v8_dynamic.log'
# 监控
tail -f /tmp/v8_dynamic.log

# Phase C
# 创建 freq_optimize.py, risk_grid.py
# 创建 docs/64-v8_dynamic_position.md
```

---

## 十、成功标准

✅ **Sharpe** > 0.95 (相比 v8 per-asset 5bp 0.871, 提升 9%)
✅ **MaxDD** < -16% (比 v8 per-asset -18.14% 改善)
✅ **924 捕获率** 从 10% 提升到 50%+
✅ **5 种风险偏好** 跑出对比表, Sharpe 全 > 0.85
✅ **真实可交易** 5/10/15/20bp 成本档 Sharpe 仍 > 0.8

---

**报告日期**: 2026-07-24
**状态**: 计划已写入文档, 待执行
**下一阶段**: Phase B 执行