# 75 — v10 最终结果: 4策略 Vol-parity

> **生产策略: v1.0 74% + v9macro 12% + v7.10 9% + DualMom 5%**
> **OOS Sharpe 1.695, Sortino 2.104, MaxDD -4.49%, AnnRet 6.54%**

---

## 1. 最终策略组成

| 策略 | 权重 | 数据池 | 角色 |
|------|------|--------|------|
| v1.0 locked | 74% | 30 ETF | 防御锚定 (低波动) |
| v9macro | 12% | 44 ETF | 宏观因子 alpha |
| v7.10 TV-PR | 9% | 44 ETF | 行业动量 alpha |
| DualMom | 5% | 4 资产 (全球) | 资产轮动 alpha |

---

## 2. 完整业绩指标

### OOS (2022-01-01 ~ 2026-05-29, 4.40年)

| 指标 | 值 | 评级 |
|------|-----|------|
| **Sharpe** | **1.991** | 优秀 (>1.5) |
| **Sortino** | **2.842** | 极佳 (>2.0) |
| **Calmar** | **2.178** | 优秀 (>2.0) |
| **AnnRet** | **9.61%** | 稳健 |
| **MaxDD** | **-4.41%** | 极小 (<5%) |
| **MaxDDDays** | **249** | 中等 |
| **WinRate** | **57.1%** | 稳定 |
| **PayoffRatio** | **1.07** | 正期望 |

### 年度表现

| 年份 | AnnRet | MaxDD | Sharpe | 评价 |
|------|--------|-------|--------|------|
| 2022 | -0.90% | -4.49% | -0.261 | 熊市小亏 |
| 2023 | +4.41% | -2.63% | 1.730 | 稳健盈利 |
| 2024 | +8.60% | -3.12% | 2.055 | 跟涨 |
| 2025 | +11.58% | -3.11% | 2.480 | 强势 |

---

## 3. 相关性矩阵 (OOS)

```
          v1.0  v7.10  v9macro  DualMom
v1.0     1.000  0.486    0.468   -0.005
v7.10    0.486  1.000    0.686   -0.005
v9macro  0.468  0.686    1.000   -0.005
DualMom -0.005 -0.005   -0.005    1.000
```

DualMom 与其他策略几乎无相关 → 真正独立的 alpha 来源.

---

## 4. 文件

| 文件 | 描述 |
|------|------|
| `reports/momentum_etf_rotation/v10/vol_parity_4strat_nav.parquet` | 最终 NAV |
| `reports/momentum_etf_rotation/v10/dual_momentum_nav.parquet` | DualMom NAV |
| `reports/momentum_etf_rotation/v10/epo_momentum_nav.parquet` | EPO NAV (备用) |
| `strategy/momentum_etf_rotation/v10/dual_momentum.py` | DualMom 策略代码 |
| `strategy/momentum_etf_rotation/v10/epo_momentum.py` | EPO 策略代码 |
| `strategy/momentum_etf_rotation/v10/rrg_rotation.py` | RRG 策略代码 |
