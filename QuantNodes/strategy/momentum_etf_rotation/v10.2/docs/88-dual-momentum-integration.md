# 88 - Dual Momentum + CA-GCP Integration

## Overview

验证 CA-GCP 风控层是否能改善 dual momentum 策略的风险收益特征。

**核心问题**: dual momentum 是单资产集中持仓（100% 1 只 ETF），CA-GCP 能否在 panic 时提供保护？

## Dual Momentum 简介

- **4 资产**: 510300（沪深300）、513100（纳指）、518880（黄金）、511260（国债）
- **信号**: 52 周绝对+相对动量，选 1 只资产全仓
- **调仓**: 月频（month-end），hold 1 个月
- **现有 NAV**: 2018-01-02 → 2026-06-30，3.29×

## 集成方案: Option C（月频 + 日频保护）

```
月度调仓日:
  dual_momentum_signal → 新目标权重
  CA-GCP 检查 → 如果 panic, 临时减仓

非调仓日:
  CA-GCP 日频监控
  如果 stress > red 或 width_z > red → 临时降仓至 panic_scale (0.3)
  如果 stress > yellow 或 width_z > yellow → 临时降仓至 yellow_scale (0.85)
  否则维持现有持仓

每次交易（无论调仓日 or panic 日）→ 扣 10bp 交易成本
```

## 3 个候选

### 候选 1: dual_mom 纯 vs dual_mom + CA-GCP
最简单对比：有无 CA-GCP 的差异。

### 候选 2: 4 策略对比表
v1.0 / v7.10 / v9macro / DualMom × ±CA-GCP = 8 列

### 候选 3: Sector CA-GCP on dual_mom
如果候选 1 正效果，尝试 Sector 级 CA-GCP

## Walk-Forward 4 Fold 校准

**严格 Train→Calib→Test 分离，无数据泄露**:

| Fold | Train | Calib | Test |
|------|-------|-------|------|
| 1 | 2018-01 → 2022-01 | 2022-01 → 2022-10 | 2022-10 → 2023-09 |
| 2 | 2018-01 → 2022-08 | 2022-08 → 2023-05 | 2023-05 → 2024-04 |
| 3 | 2018-01 → 2023-03 | 2023-03 → 2024-01 | 2024-01 → 2024-12 |
| 4 | 2018-01 → 2023-10 | 2023-10 → 2024-08 | 2024-08 → 2025-07 |

- 每折: fit(Train) → grid search(Calib) → predict(Test)
- Test 期数据**不参与**校准
- 4 fold × 3 候选 = 12 次 predict_fast 调用

## 预期

- dual_mom 波动较大（单资产集中），CA-GCP 可能有效
- 相比 scheme_e_hybrid（已有分散化），CA-GCP 效果可能更明显
- 月频调仓 + 日频保护 = 低换手 + 风险响应

## Data Flow

```
daily_prices (4 资产日线)
  → dual_momentum_signal (月频) → target_weights
  → CAGCP intervals (日频) → risk_filter → adjusted_weights
  → compute_nav (每日, 扣 10bp cost)
```
