# v7.0 5 Macro Dynamic 业绩统计表 (Stage 30.5 + Phase A + Phase B)

> **统一口径**: 5-fold walk-forward OOS (复用 v6.2_ir_expanding_5fold.py 切分) | 2018-01-01 ~ 2026-06-30
> **5 折 OOS 区间**:
> - Fold 1: 2020-01-01 ~ 2020-12-31 (1.0y)
> - Fold 2: 2021-01-01 ~ 2021-12-31 (1.0y)
> - Fold 3: 2022-01-01 ~ 2023-06-30 (1.5y, 熊市)
> - Fold 4: 2023-07-01 ~ 2024-12-31 (1.5y, 修复)
> - Fold 5: 2025-01-01 ~ 2026-06-30 (1.5y, 牛市)
> **运行脚本**:
> - `scripts/v7_0_macro_oos.py` (5 方案 × 5 fold OOS)
> - `scripts/v7_0_stress_test.py` (3 极端事件)
> - `scripts/v7_0_hmm_lag_test.py` (HMM 滞后)
> - `scripts/v7_0_integration_test.py` (加成本退化)
> - `scripts/v7_0_macro_oos_52etf.py` (41 ETF 对比)
> **可视化**: `reports/momentum_etf_rotation/v7/v7_0_state_timeline.html`

---

## 1. 实验设计

### 1.1 统一时间区间

| 阶段 | 区间 | 长度 | 说明 |
|------|------|------|------|
| 训练 + 预热 | 2018-01-01 ~ 2019-12-31 | 2 年 | Fold 1 训练期 (HMM + 因子初始化) |
| OOS 1 | 2020-01-01 ~ 2020-12-31 | 1.0 年 | 牛市恢复 |
| OOS 2 | 2021-01-01 ~ 2021-12-31 | 1.0 年 | 震荡市 |
| OOS 3 | 2022-01-01 ~ 2023-06-30 | 1.5 年 | 熊市 (A 股深跌 + 港股暴跌) |
| OOS 4 | 2023-07-01 ~ 2024-12-31 | 1.5 年 | 修复期 |
| OOS 5 | 2025-01-01 ~ 2026-06-30 | 1.5 年 | 牛市 |

**注**: v7.0 HMM 训练起点 2018-08-28 (iFinD PMI 数据起点), Fold 1 OOS 2020-01 之前用等权 fallback。

### 1.2 统一 ETF 池 (7 只, 手工选)

| 来源 | 数量 | 代码 | 角色 |
|------|------|------|------|
| A 股宽基 | 3 | 510300, 510500, 159915 | 价值/中盘/成长 |
| 商品 | 1 | 518880 | 黄金 (抗胀) |
| 行业 | 1 | 512760 | 半导体 (技术景气, 2019-06+) |
| 海外 | 1 | 513100 | 纳指 (海外科技) |
| SmartBeta | 1 | 510880 | 红利 (高分红价值) |
| **合计** | **7** | - | 7 池 (vs v1v5 52 池) |

### 1.3 数据源 (新增)

| 维度 | 来源 | 频率 | 起始 |
|------|------|------|------|
| ETF NAV | 本地 parquet | 日 | 2018-01-01 |
| 5 宏观因子 | iFinD (PMI/CPI/M2/CN10Y/US10Y) | 月/日 | 2018-01-01 |
| HMM 5 状态 | 月度 HMM on 5 macro zscore | 日级 timeline | 2018-08-28 |

**PIT 防护**: 调仓日 d 只能用 `release_date <= d` 的 macro 数据 (PMI lag=1, CPI lag=10, M2 lag=12, CN10Y/US10Y lag=0)。

### 1.4 成本与 cap

- **双边手续费**: 0.1% (10 bps) × 月度 12 次 = 1.2%/年 drag
- **流动性 cap**: 单 ETF 30% 权重上限 (迭代 cap+redistribute)
- **换手 cap**: 单 ETF 月度换手 30%

### 1.5 5 方案定义

| # | 方案 | 数学 | 业界对应 |
|---|------|------|----------|
| A | **Top-K (K=5)** | state × ETF forward 21d 均值 → top 5 等权 | 中泰/中银 |
| B | **Black-Litterman** | prior + Q + Σ + Ω (Ledoit-Wolf shrinkage) | 学术/中信 |
| C | **Macro Beta (K=5)** | 5 macro × 7 ETF 滚动 252d 回归 → predicted → top 5 | **中信动态加权** |
| D | **Momentum (63d)** | state 内 ETF 动量排名 → top 5 | 海通风格轮动 |
| E | **Inverse Vol** | 1/vol 权重 + 30% cap | 风险平价 |

---

## 2. v6.x + v7.0 核心指标 (5-fold OOS mean)

| 排名 | 阶段 | 版本 | 池 | 年化收益率 | 年化波动 | Sharpe | DD | **5-fold OOS Calmar** |
|------|------|------|-----|-----------|---------|--------|-----|---------------------|
| ⭐ | **Stage 30.5** | **C. Macro Beta** | **7** | **26.07%** | **19.75%** | **1.32** | **-10.28%** | **6.29** |
| 2 | Stage 30.5 | A. Top-K (K=5) | 7 | 25.34% | 18.10% | 1.40 | -12.18% | 6.07 |
| 3 | Stage 30.5 | D. Momentum | 7 | 26.88% | 20.52% | 1.31 | -13.35% | 6.84 ⚠ |
| 4 | Stage 30.5 | B. Black-Litterman | 7 | 24.60% | 17.32% | 1.42 | -16.83% | 4.58 |
| 5 | Stage 30.5 | E. Inverse Vol | 7 | 23.33% | 15.66% | 1.49 | -11.30% | 112.16 ⚠ |
| 6 | **Stage 29** | **v6.2 ir_expanding** | **44** | **18.86%** | **18.66%** | **1.01** | **-16.73%** | **1.512** |
| 7 | Stage 30.5 | baseline 等权 7 ETF | 7 | 17.38% | 4.14% | 4.19 | -10.42% | 1.33 |
| 8 | **Stage 29** | **v6.1 IC12** | **44** | **13.67%** | **20.59%** | **0.66** | **-20.65%** | **0.867** |

> ⚠ **D. Momentum**: 5-fold OOS mean Calmar 6.84 高, 但 fold 3 (熊市) ann -5.64%, 不鲁棒
> ⚠ **E. Inverse Vol**: 5-fold OOS mean Calmar 112.16 极高, 但 fold 5 (牛市) DD=-0.07% 异常, fold 3 ann -0.18%, 不鲁棒

**鲁棒赢家筛选**: ann_min > 0 AND calmar_min > 0 (5 fold 全正)
- ✅ C. Macro Beta (ann_min 4.36%, calmar_min 0.51)
- ✅ A. Top-K (ann_min 0.35%, calmar_min 0.03)
- ✅ B. Black-Litterman (ann_min 5.79%, calmar_min 0.85)
- ❌ D. Momentum (fold 3 ann -5.64%)
- ❌ E. Inverse Vol (fold 3 ann -0.18%)
- ❌ v6.1 (fold 3 ann -10.89%)
- ❌ v6.2 (fold 3 ann -0.27%)

---

## 3. 关键发现

### 3.1 v7.0 C. Macro Beta 是 OOS 最佳 (5-fold mean Calmar 6.29)

**5 方案平均对比 (5-fold OOS)**:

| 维度 | v6.2 | v7.0 C. Beta | 增量 | 解释 |
|------|------|--------------|------|------|
| OOS Calmar | 1.512 | **6.29** | **+4.78** | 5-fold mean |
| 年化% | 18.86% | **26.07%** | +7.21pp | 5-fold mean |
| Sharpe | 1.01 | 1.32 | +0.31 | |
| DD | -16.73% | **-10.28%** | -6.45pp | **DD 改善 38%** |

**根因**:
1. v7.0 直接建模 5 macro → ETF 收益, 比 v6.2 (量价因子) 信息量大
2. v6.2 折 3 (2022-2023.6) 完全失效 (ann -0.27%), v7.0 折 3 仍 +4.36%
3. v7.0 调仓频率与 HMM 状态匹配, 熊市切换到 defensive 状态

### 3.2 41 ETF 池不可用 (Phase B 验证)

| 方案 | 7 ETF Calmar | 41 ETF Calmar | Δ |
|------|--------------|---------------|----|
| A. Top-K | 5.31 | 0.35 | -4.96 |
| B. BL | 3.98 | 0.10 | -3.88 |
| C. Beta | 6.29 | 0.34 | -5.95 |
| D. Momentum | 6.70 | -0.21 | -6.91 |
| E. IV | 112.16 | 1.39 | -110.77 |

**根因** (详见 `v7_0_52etf_decision.txt`):
1. 行业相关度高 (512480/512170/512400 等高度相关)
2. 流动性差 (日均 5000万-1亿, ffill 引入噪声)
3. 波动率高 (行业 ETF 30-40%, 加权后波动放大)
4. 业界对应: 中信 ETF 池也是 7-15 个, 不是 50+

**结论**: 退回 7 ETF 池。

### 3.3 极端市压测 (3 事件)

| 事件 | baseline | A | B | C | D | E |
|------|----------|---|---|---|---|---|
| 2020-02 疫情暴跌+反弹 | -1.91% | -2.50% | -2.50% | -6.75% | **+1.04%** | -3.80% |
| 2022-04 港股暴跌 | -8.25% | -4.62% | **+1.15%** | -3.92% | -3.92% | -4.84% |
| 2024-09 政策反转 | +24.11% | +17.34% | **+20.67%** | +14.02% | +18.35% | +13.93% |
| **3 事件平均** | +4.65% | +3.41% | **+6.44%** | +1.12% | +5.16% | +1.76% |

→ **B. BL 极端市最稳健** (3 事件平均 +6.44%, 仅在港股暴跌取得正收益)

### 3.4 HMM 滞后影响 (反直觉)

| 方案 | lag_0 | lag_5 | 趋势 | 解释 |
|------|-------|-------|------|------|
| A. Top-K | 16.94% | **19.60%** | ⬆️ +15% | 滞后过滤假信号 |
| B. Black-Litterman | 16.85% | **24.45%** | ⬆️ +45% | 同上 |
| C. Macro Beta | 16.45% | 16.15% | → 稳定 | 回归模型对滞后不敏感 |
| D. Momentum | 18.42% | 17.13% | ⬇️ -7% | 动量对滞后敏感 |
| E. Inverse Vol | 15.84% | 15.84% | → 不变 | 与 HMM 无关 |

→ 实盘中 macro 数据延迟 1-3 日是**正常**的, 反而可作为 regularizer 过滤假信号。

### 3.5 加交易成本退化 (Phase A 集成)

| 方案 | plain ann | with_cost ann | Δ pp | plain Calmar | with_cost Calmar | Δ |
|------|-----------|---------------|------|--------------|-------------------|----|
| A. Top-K | 25.34% | 25.24% | -0.11 | 6.07 | 6.03 | -0.04 |
| B. Black-Litterman | 24.60% | 24.41% | -0.18 | 4.58 | 4.55 | -0.03 |
| **C. Macro Beta** | 26.07% | 25.81% | -0.26 | **6.29** | **6.16** | -0.13 |
| D. Momentum | 26.88% | 26.69% | -0.18 | 6.84 | 6.77 | -0.08 |
| E. Inverse Vol | 23.33% | 23.31% | -0.02 | 112.16 | 98.45 | -13.71 |

→ 加成本后 **C. Beta 仍为鲁棒赢家** (calmar 6.16, -2% 退化)

### 3.6 业界对应 (Stage 30 调研)

| 机构 | 口径 | 19% 目标 | v7.0 C. Beta (5-fold mean) |
|------|------|----------|----------------------------|
| 中信证券 | 宽基 ETF + 动态加权 | 19.0% | **26.07%** ✅ 超 7pp |
| 国泰海通 | 复苏/过热/衰退/中性, 5 年滚动 | 22.67% | 超 3pp |
| 海通证券 | 红利+创业板 | 14.94% | **超 11pp** |
| 银河证券 | 宽基 ETF + 熵权法 | 19.05% | **超 7pp** |

→ v7.0 C. Beta **5-fold OOS mean 26.07%** 已**超过业界 4 大研究流派的 19% 目标**。

---

## 4. 5-Fold OOS 单折明细 (C. Macro Beta)

| Fold | OOS 区间 | 年化% | Calmar | DD% | Sharpe | 备注 |
|------|----------|-------|--------|-----|--------|------|
| 1 | 2020 全年 | +51.17 | 8.39 | -6.10 | 2.35 | 牛市恢复 |
| 2 | 2021 全年 | +18.56 | 13.65 | -1.36 | 1.96 | 震荡市 |
| 3 | 2022-2023.6 | +4.36 | 0.51 | -8.60 | 0.40 | 熊市最低 |
| 4 | 2023.7-2024 | +9.50 | 0.92 | -10.28 | 0.48 | 修复期 |
| 5 | 2025-2026.6 | +46.77 | 7.97 | -5.87 | 1.43 | 牛市 |
| **Mean** | - | **26.07** | **6.29** | **-10.28** | **1.32** | 5/5 正 |

**vs v6.2 单折对比**:

| Fold | v6.2 ann | v6.2 Calmar | C. Beta ann | C. Beta Calmar | 增量 ann | 增量 Calmar |
|------|----------|--------------|-------------|----------------|----------|-------------|
| 1 | +36.76 | 3.022 | +51.17 | 8.39 | +14.41 | +5.37 |
| 2 | +11.85 | 1.409 | +18.56 | 13.65 | +6.71 | +12.24 |
| 3 | **-0.27** | **-0.016** | **+4.36** | **0.51** | **+4.63** | **+0.53** ⭐ |
| 4 | +8.26 | 0.636 | +9.50 | 0.92 | +1.24 | +0.28 |
| 5 | +37.71 | 2.510 | +46.77 | 7.97 | +9.06 | +5.46 |

→ v7.0 C. Beta 在**所有 5 fold 都优于 v6.2**, 特别在熊市 fold 3 显著改善 (-0.27% → +4.36%)

---

## 5. 演进路径总结

```
v6.1 IC12 (Stage 27, IC12 加权)        — 5-fold OOS mean Calmar 0.867
  │ + 量价因子 (Stage 22 11 因子)
  │ + 因子正交化 Gram-Schmidt (Stage 27)
  │ + ir_expanding 加权 (Stage 29)
  v6.2 ir_expanding                    — 5-fold OOS mean Calmar 1.512 ⭐ PROMISING
  │ + 5 macro 因子 (PMI/CPI/M2/CN10Y/US10Y, iFinD, Stage 30 POC)
  │ + HMM 5 状态 (recovery/overheat/neutral/stagflation/recession, Stage 30.2)
  │ + 5 动态配置方案 (Stage 30.5, 拒绝 vol_target 防御版 + 写死 TAA)
  │   - A. Top-K      (排名 + 等权)
  │   - B. BL         (prior + views, Ledoit-Wolf shrinkage)
  │   - C. Macro Beta (5 macro × 7 ETF 回归) ⭐ 鲁棒赢家
  │   - D. Momentum   (state 内动量)
  │   - E. IV         (1/vol 权重)
  v7.0 C. Macro Beta                  — 5-fold OOS mean Calmar 6.29 🏆
```

**关键单点贡献** (5-fold OOS mean Calmar):
| 阶段 | OOS Calmar | 增量 |
|------|-----------|------|
| v6.1 IC12 | 0.867 | 起点 |
| **+ v6.2 ir_expanding** | **1.512** | +0.65 |
| **+ v7.0 5 macro + HMM + 5 dynamic** | **6.29** | **+4.78** ⭐⭐⭐ |

Stage 30 宏观增强是 v6.2 之后**最大突破**, 单点贡献 +4.78 5-fold mean Calmar。

---

## 6. 推荐组合 (5 方案等权)

| 组合 | 5-fold ann | 5-fold Calmar | 5-fold DD |
|------|-----------|---------------|-----------|
| C. Beta 100% | 26.07% | 6.29 | -10.28% |
| C. Beta 50% + B. BL 50% | 25.34% | **5.44** | **-13.56%** (双 hedge) |
| C. Beta 70% + B. BL 30% | 25.61% | 5.74 | -12.30% |
| C. Beta 100% + cost | 25.81% | 6.16 | -10.28% (实盘预期) |

**推荐**: C. Macro Beta 70% + B. BL 30% (实盘对冲组合, DD 容忍 -12.3%)

---

## 7. 结论

1. **v7.0 C. Macro Beta 是 5-fold OOS mean Calmar 6.29 的鲁棒赢家** (5/5 fold 全正, 26.07% 年化, -10.28% DD, 1.32 Sharpe)
2. **超过业界 4 大研究流派的 19% 目标** (中信/银河 19% / 国泰海通 22.67% / 海通 14.94%)
3. **5 方案中 C. Beta 5/5 fold 鲁棒**, B. BL 极端市最稳健 (推荐 hedge 30%)
4. **D. Momentum 折 3 亏 -5.64%, E. IV 折 5 异常, 淘汰**
5. **41 ETF 池不可用** (calmar 退化 -99%, 行业相关度+流动性差, 退回 7 ETF)
6. **HMM 滞后 1-5 日对 B. BL/A. Top-K 反而有利** (反直觉发现, 实盘数据延迟 = regularizer)
7. **加成本后 C. Beta 仍为赢家** (calmar 6.16, -2% 退化)
8. **未来方向**:
   - C. Beta 加 recession state 切黄金 (避免 -6.7% 疫情)
   - 模拟盘 1-3 月 (iFinD 实时数据, Sharpe ≥ 1.5 验证)
   - 41 ETF 池 PCA 调优后重测 (中期)
   - 引入因子模型 + 风险预算 (替代 raw ETF 池)

---

## 8. 附录

### 8.1 生成文件 (Stage 30.5 + Phase A + B)

| 文件 | 用途 |
|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/v7/dynamic_allocation.py` | A. Top-K |
| `QuantNodes/strategy/momentum_etf_rotation/v7/black_litterman.py` | B. BL |
| `QuantNodes/strategy/momentum_etf_rotation/v7/macro_beta.py` | C. Beta |
| `QuantNodes/strategy/momentum_etf_rotation/v7/state_momentum.py` | D. Momentum |
| `QuantNodes/strategy/momentum_etf_rotation/v7/state_inverse_vol.py` | E. IV |
| `QuantNodes/strategy/momentum_etf_rotation/v7/transaction_cost.py` | Phase A1 |
| `QuantNodes/strategy/momentum_etf_rotation/v7/liquidity_cap.py` | Phase A2 |
| `scripts/v7_0_macro_oos.py` | 5-fold OOS |
| `scripts/v7_0_stress_test.py` | 3 极端事件 |
| `scripts/v7_0_hmm_lag_test.py` | HMM 滞后 |
| `scripts/v7_0_data_sla_test.py` | iFinD SLA |
| `scripts/v7_0_integration_test.py` | 加成本集成 |
| `scripts/v7_0_select_52_etf.py` | 41 ETF 筛选 |
| `scripts/v7_0_macro_oos_52etf.py` | 41 ETF OOS |
| `scripts/v7_0_52etf_decision.py` | 41 ETF 决策 |
| `tests/test_v7_0_dynamic.py` | 24 单元测试 |
| `tests/test_v7_0_phase_a.py` | 11 单元测试 |

### 8.2 报告产物 (12 个)

| 文件 | 内容 |
|------|------|
| `reports/.../v7/v7_0_macro_oos_5fold.csv` | 5 方案 × 5 fold (25 行) |
| `reports/.../v7/v7_0_macro_oos_summary.csv` | 5 策略 × 8 指标 |
| `reports/.../v7/v7_0_macro_oos_winner.txt` | OOS 决策 |
| `reports/.../v7/v7_0_stress_test.csv` | 3 事件 × 6 策略 |
| `reports/.../v7/v7_0_hmm_lag.csv` | 5 方案 × 5 滞后 |
| `reports/.../v7/v7_0_data_sla.csv` | 5 因子 × 6 天 |
| `reports/.../v7/v7_0_data_sla_summary.txt` | SLA 报告 |
| `reports/.../v7/v7_0_integration_test.csv` | 5 方案 × 2 配置 × 5 fold |
| `reports/.../v7/v7_0_52etf_metrics.csv` | 52 ETF 指标 |
| `reports/.../v7/v7_0_52etf_universe.csv` | 41 ETF 通过 |
| `reports/.../v7/v7_0_52etf_oos_5fold.csv` | 5 方案 × 2 池 × 5 fold |
| `reports/.../v7/v7_0_52etf_decision.txt` | 41 ETF 决策 |
| `reports/.../v7/v7_0_state_timeline.html` | 5 状态 plotly 报告 |
| `reports/.../v7/v7_0_PERFORMANCE_TABLE.md` | v1v5 风格统计表 (本文) |

### 8.3 测试覆盖

| 类别 | 测试数 | 状态 |
|------|--------|------|
| v7 0 dynamic (5 方案) | 24 | ✓ pass |
| v7 0 phase A (cost + cap) | 11 | ✓ pass |
| v7 0 macro factors (PIT) | 35 | ✓ pass |
| v7 0 regime (HMM) | 11 | ✓ pass |
| v7 0 backtest (vol_target, 弃用) | 7 | ⏭ skipped |
| **总计** | **81 pass + 7 skipped** | ✓ |

无新增回归, 6 个历史预存 v4 失败 (与 v7.0 无关)。

### 8.4 引用

- 中信证券: 三维宏观状态监测体系 (2026)
- 国泰海通: 五维方法 (复苏/过热/衰退/中性, 5 年滚动窗口)
- 海通证券: 宏观调整的近一季风格动量差
- 银河证券: 宽基 ETF + 熵权法 (19.05%)
- Black & Litterman (1992): 原版 BL 模型
- Idzorek (2005): τ=0.05 推荐
- Ledoit-Wolf (2003): Covariance shrinkage
- Markowitz (1952): 投资组合理论
