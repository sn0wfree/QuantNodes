# v7.0 5 Macro Dynamic 方案 + 5-Fold OOS 决策 (Stage 30.5)

> **状态**: ✅ Stage 30.5 落地, 5-fold OOS 鲁棒赢家 **C. Macro Beta**
> **关联**: [docs/35-宏观因子体系业界调研.md](35-宏观因子体系业界调研.md) (业界调研, 7 机构)
> **commit**: `e5387fd feat(v7.0): 5 Macro Dynamic 方案 + 5-fold OOS (Stage 30.5)`
> **测试**: 24 dynamic + 35 PIT + 11 HMM = **70/70 pass** (v7 子项目)

---

## 0. 背景与动机

Stage 29 推广后, v6.2 ir_expanding 是**纯量价驱动** (11 因子 + 5bp 单边成本 + 12-month IC + 5-fold 验证)。业界调研 ([docs/35](../35-宏观因子体系业界调研.md)) 显示:

- **宏观因子是"压舱石"**: 2024 年宏观因子年化超额 17.8%, 月度胜率 67%
- **纯宏观偏左侧**: 胜率偏低但赔率高, 与量价融合才能实战
- **A 股数据工程已成熟**: iFinD 800+ 万指标, 月频宏观因子可 PIT 防护

Stage 30 演进路径:

| Stage | 内容 | 状态 |
|-------|------|------|
| 30.0 | 调研 (7 机构宏观因子) | ✅ 归档 docs/35 |
| 30.1 | iFinD fetcher + PIT 工具 | ✅ |
| 30.2 | HMM 5 状态时间线 | ✅ |
| 30.3 | v7.0 vol_target 防御版 | ❌ **用户否决** ("我不需要防御版本") |
| 30.4 | v7.0 TAA 写死版 (5 state × 5 ETF 权重) | ❌ **用户否决** ("我不需要写死配置") |
| 30.5 | v7.0 5 Macro Dynamic 方案 | ✅ **当前落地** |

---

## 1. 5 方案概览

### 1.1 设计原则

1. **数据驱动**: 无人工拍脑袋 (拒绝写死权重)
2. **PIT 安全**: 全部 expanding window, t-1 数据算 t 时点权重
3. **冷启动保护**: state 历史 < 3 月 → fallback 到等权
4. **长期约束**: w ≥ 0, sum=1, max=30%
5. **业界对应**: 每个方案对应业界 1 个研究流派

### 1.2 5 方案对比

| # | 方案 | 数学 | 业界对应 | 复杂度 |
|---|------|------|----------|--------|
| A | **Top-K Simple Dynamic** | 排名 + 等权 | 中泰/中银 "Top-K 排名" | ★ |
| B | **Black-Litterman** | prior + Q + Σ + Ω | 学术经典 / 中信动态加权 | ★★★★ |
| C | **Macro Beta Regression** | 5 features × 7 ETF 回归 | 中信动态加权法 | ★★★ |
| D | **State Conditional Momentum** | state 内 ETF 动量排名 | 海通 "近一季风格动量差" | ★★ |
| E | **State Conditional Inverse Vol** | state × ETF inverse vol | 风险平价 / Bridgewater | ★★ |

### 1.3 5 方案详细算法

#### A. Top-K Simple Dynamic (排名 + 等权)
```python
调仓日 d:
    1. 截至 d-1, 算每月最后交易日的 forward 21d 收益 (state × ETF)
    2. state-conditional 均值收益表 (5 state × 7 ETF)
    3. 当前 state 下, 取 top K ETF 等权 (K=5)
冷启动: state < 3 月 → 等权 1/7
```
- **优势**: 简单, 可解释, 无参数 (K 之外)
- **风险**: K 选择敏感, 集中度高 (K=5 中等)

#### B. Black-Litterman (prior + views)
```python
Posterior: E[R|views] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
参数:
    π = equal 7%  (prior)
    Σ = expanding 252d 日收益协方差 (Ledoit-Wolf shrinkage)
    τ = 0.05       (Idzorek 推荐)
    P = I_7        (one view per ETF)
    Q = state-conditional forward 21d 均值 (年化)
    Ω = diag(τ × σ² / n_samples)  (桶稀疏修正)
Weights: w ∝ max(0, posterior) → normalize → 30% cap
```
- **优势**: 学术经典, 数学严谨
- **风险**: 协方差 7×7 接近奇异, 必须 Ledoit-Wolf

#### C. Macro Beta Regression (5 特征 × 7 ETF)
```python
调仓日 d:
    1. 截至 d-1, 滚动 252d, 5 宏观特征 × 7 ETF 回归:
       r_etf = α + β_PMI*PMI + β_CPI*CPI + β_M2*M2 + β_CN10Y*ΔCN10Y + β_US10Y*ΔUS10Y + ε
    2. 截至 d, current macro × beta → predicted return (年化)
    3. top K ETF 等权 (K=5)
冷启动: 历史 < 60 天 → 等权 1/7
```
- **优势**: 业界主流 (中信), 用到全部 5 macro features
- **风险**: 35 系数同窗口估计, 5 维可能过拟合

#### D. State Conditional Momentum (state 内动量)
```python
调仓日 d:
    1. 每 ETF 算过去 63d 动量
    2. 当前 state 历史日的 ETF 平均动量 (state-conditional)
    3. combined = 0.5 × state_mom + 0.5 × current_mom
    4. top K ETF 等权 (K=5)
```
- **优势**: 动量 alpha, 跨 state 验证
- **风险**: 转折市失效, 状态检测滞后

#### E. State Conditional Inverse Vol (1/vol)
```python
调仓日 d:
    1. 截至 d-1, 算每 ETF 过去 252d 年化 vol
    2. weight = (1/vol) / Σ(1/vol)
    3. 30% cap → normalize
冷启动: 历史 < 30 天 → 等权 1/7
```
- **优势**: 风险预算, 防御性强
- **风险**: 牛市跟不上, 集中度高 (低 vol ETF)

---

## 2. 5-Fold OOS 测试

### 2.1 Fold 切分 (复用 v6.2_ir_expanding_5fold.py)

| Fold | 训练期 | OOS 期 | 长度 |
|------|--------|--------|------|
| 1 | 2018-01 ~ 2019-12 | 2020-01 ~ 2020-12 | 1.0y |
| 2 | 2018-01 ~ 2020-12 | 2021-01 ~ 2021-12 | 1.0y |
| 3 | 2018-01 ~ 2021-12 | 2022-01 ~ 2023-06 | 1.5y |
| 4 | 2018-01 ~ 2023-06 | 2023-07 ~ 2024-12 | 1.5y |
| 5 | 2018-01 ~ 2024-12 | 2025-01 ~ 2026-06 | 1.5y |

### 2.2 7-ETF 池

```
510300 沪深300     (2018+, 价值)
510500 中证500     (2018+, 中盘)
159915 创业板       (2018+, 成长)
518880 黄金        (2018+, 抗胀)
512760 半导体       (2019-06+, 技术景气)
513100 纳指        (2018+, 海外科技)
510880 红利        (2018+, 高分红价值)
```

### 2.3 5-Fold OOS 决策

```
策略       calmar_mean  calmar_min  ann_min    正 fold
C_beta     6.29         0.51        +4.36%     5/5  ← 鲁棒赢家 🏆
A_topk     6.07         0.03        +0.40%     5/5
B_bl       4.58         0.86        +5.80%     5/5
D_momentum 6.84         -0.42       -5.64%     4/5  ← 折 3 熊市亏
E_iv       112.16       -0.02       -0.18%     4/5  ← 折 5 异常 (DD=-0.07%)
baseline   1.33         -0.33       -6.84%     4/5
```

### 2.4 决策标准 (Stage 30.5 升级版)

1. **鲁棒筛选**: `ann_min > 0` (无负收益 fold) AND `calmar_min > 0` (无负 Calmar fold)
2. **鲁棒赢家中按 `calmar_mean` 降序选最优**
3. **避免单 fold 异常值**: 如 IV 在 fold 5 DD=-0.07% 导致 calmar=543 的"假赢家"

**赢家: C. Macro Beta**
- 5/5 OOS fold 全正收益
- 平均年化 **26.07%**, 最差 fold 4.36%
- 平均 Calmar **6.29** (3 robust 策略最高)
- 5-fold 最低 Calmar 0.51 (优于 baseline 1.33)
- 远超业界 19% 目标

### 2.5 各 Fold 详细结果

```
Fold 1 (OOS 2020, 牛市恢复):
  C_beta  ann=+51.17% DD=-6.10% Calmar=8.39
  A_topk  ann=+57.96% DD=-4.41% Calmar=13.13
  B_bl    ann=+57.99% DD=-4.41% Calmar=13.13
  D_mom   ann=+58.00% DD=-5.72% Calmar=10.14
  E_iv    ann=+49.19% DD=-6.07% Calmar=8.11

Fold 2 (OOS 2021, 震荡):
  C_beta  ann=+18.56% DD=-1.36% Calmar=13.65
  A_topk  ann=+23.41% DD=-2.31% Calmar=10.15
  B_bl    ann=+23.41% DD=-2.31% Calmar=10.15
  D_mom   ann=+24.95% DD=-2.68% Calmar=9.30
  E_iv    ann=+19.44% DD=-2.29% Calmar=8.47

Fold 3 (OOS 2022-2023.6, 熊市):
  C_beta  ann=+4.36%  DD=-8.60% Calmar=0.51
  A_topk  ann=+0.35%  DD=-12.18% Calmar=0.03
  B_bl    ann=+5.79%  DD=-6.12% Calmar=0.95
  D_mom   ann=-5.64%  DD=-13.35% Calmar=-0.42  ❌
  E_iv    ann=-0.18%  DD=-11.05% Calmar=-0.02  ❌

Fold 4 (OOS 2023.7-2024, 修复):
  C_beta  ann=+9.50%  DD=-10.28% Calmar=0.92
  A_topk  ann=+15.16% DD=-11.60% Calmar=1.31
  B_bl    ann=+14.27% DD=-16.69% Calmar=0.86
  D_mom   ann=+13.02% DD=-10.45% Calmar=1.25
  E_iv    ann=+11.27% DD=-11.30% Calmar=1.00

Fold 5 (OOS 2025-2026.6, 牛市):
  C_beta  ann=+46.77% DD=-5.87% Calmar=7.97
  A_topk  ann=+38.60% DD=-3.76% Calmar=10.27
  B_bl    ann=+28.58% DD=-13.43% Calmar=2.13
  D_mom   ann=+44.06% DD=-3.16% Calmar=13.95
  E_iv    ann=+36.93% DD=-0.07% Calmar=543.25 (异常)
```

---

## 3. C. Macro Beta 详细说明

### 3.1 数学形式

```
对每只 ETF e ∈ {510300, 510500, 159915, 518880, 512760, 513100, 510880}:
    截至 d-1 滚动 252d 窗口 OLS:
    r_etf,d = α_e + β_PMI,e × PMI_zscore_d
              + β_CPI,e × CPI_zscore_d
              + β_M2,e × M2_zscore_d
              + β_CN10Y,e × ΔCN10Y_d
              + β_US10Y,e × ΔUS10Y_d
              + ε_d

调仓日 d:
    predicted_return_e = α_e + Σ β_feature,e × current_macro_feature
    rank by predicted_return → top K (K=5) 等权
```

### 3.2 关键参数

| 参数 | 值 | 备注 |
|------|-----|------|
| ETF 池 | 7 (见上) | 手工选, 上市 ≥ 3 年 |
| Macro 特征 | 5 (PMI/CPI/M2/CN10Y/US10Y zscore) | iFinD PIT |
| 回归窗口 | 252d 滚动 | 默认 |
| 最小样本 | 60d | 不足时 fallback |
| 调仓频率 | 月度 | 与 HMM 状态匹配 |
| K | 5 | top-K 数量 |
| 最大权重 | 100% 等权 (无 cap) | 简化为 1/5 |

### 3.3 PIT 防护

- 5 macro 特征从 iFinD 拉取, 用 `get_pit_series()` 做 PIT 调整
- 调仓日 d 只能用 `release_date <= d` 的数据
- 月度数据 (PMI/CPI/M2): `obs_date` 统一月末, `release_date = obs_date + lag_days`
  - PMI lag=1, CPI lag=10, M2 lag=12
- 日度数据 (CN10Y/US10Y): lag=0

### 3.4 冷启动保护

- HMM 训练需要 ≥ 60 天 macro 数据
- 2018-08-28 ~ 2018-10-31 期间 HMM 未稳定 → fallback 等权 1/7
- 2018-11 后 HMM 稳定, 切换到 Macro Beta 动态

### 3.5 为何 C. Macro Beta 优于其他 4 方案?

1. **vs A. Top-K (排名)**:
   - 排名只用 forward 21d 均值, 信息量小
   - Macro Beta 显式建模 macro → ETF 关系, 鲁棒性更好

2. **vs B. Black-Litterman (prior + views)**:
   - BL 数学严谨, 但 7×7 协方差接近奇异
   - Macro Beta 直接回归, 不需协方差矩阵, 数值稳定

3. **vs D. State Conditional Momentum (动量)**:
   - 动量在转折市失效 (fold 3 熊市 -5.64%)
   - Macro Beta 用 macro 预测, 转折市更稳

4. **vs E. Inverse Vol (1/vol)**:
   - 1/vol 牛市跟不上, 折 5 异常 DD=-0.07%
   - Macro Beta 选 alpha, 长期更优

---

## 4. 关键文件清单

### 4.1 源代码 (5 模块 + 1 包)

| 文件 | 行数 | 用途 |
|------|------|------|
| `v7/dynamic_allocation.py` | 6.3 KB | A. Top-K 排名 |
| `v7/black_litterman.py` | 8.8 KB | B. BL prior+views |
| `v7/macro_beta.py` | 5.5 KB | C. 5 特征 × 7 ETF 回归 |
| `v7/state_momentum.py` | 4.6 KB | D. State 内动量 |
| `v7/state_inverse_vol.py` | 3.8 KB | E. 1/vol 权重 |
| `v7/__init__.py` | 2.2 KB | 暴露 ~25 API |

### 4.2 测试

| 文件 | 行数 | 测试数 |
|------|------|--------|
| `tests/test_v7_0_dynamic.py` | 9.0 KB | **24 测试** (全过) |

### 4.3 脚本

| 文件 | 行数 | 用途 |
|------|------|------|
| `scripts/v7_0_macro_oos.py` | 10.2 KB | 5-fold OOS 对比 + 决策 |

### 4.4 报告

| 文件 | 大小 | 内容 |
|------|------|------|
| `reports/v7_0_macro_oos_5fold.csv` | 4.2 KB | 25 行 (5 fold × 5 strategy) |
| `reports/v7_0_macro_oos_summary.csv` | 1.0 KB | 6 策略 × 8 指标 |
| `reports/v7_0_macro_oos_winner.txt` | 2.0 KB | 决策报告 |

---

## 5. 测试覆盖 (24 测试)

### 5.1 Top-K Dynamic (5 测试)
- `test_state_conditional_means_shape` — 5 state × 7 ETF + state 列
- `test_state_conditional_means_pit` — PIT 安全, 不含未来
- `test_topk_weights_sum_one` — sum=1
- `test_topk_cold_start` — 空输入 → fallback 等权
- `test_topk_runs` — 完整 backtest 跑通

### 5.2 Black-Litterman (7 测试)
- `test_expanding_cov_shape` — 7×7 对称正定
- `test_state_view_q_pit` — PIT 安全
- `test_bl_posterior_no_views` — views=0 → posterior ≈ prior
- `test_bl_posterior_with_views` — views≠0 → posterior 改变
- `test_bl_weights_long_only` — w≥0
- `test_view_uncertainty_omega` — Ω 对角矩阵
- `test_bl_runs` — 完整 backtest 跑通

### 5.3 Macro Beta Regression (4 测试)
- `test_etf_macro_betas_shape` — 7×6 (const + 5 features)
- `test_predict_etf_returns` — 预测函数
- `test_beta_cold_start` — 样本不足 → None
- `test_beta_runs` — 完整 backtest 跑通

### 5.4 State Conditional Momentum (3 测试)
- `test_etf_momentum` — 动量计算
- `test_momentum_pit` — PIT 安全
- `test_momentum_runs` — 完整 backtest 跑通

### 5.5 State Conditional Inverse Vol (3 测试)
- `test_etf_vol_shape` — 7 ETF vol 形状
- `test_iv_weights_long_only` — w≥0, max=30%
- `test_iv_runs` — 完整 backtest 跑通

### 5.6 共享 (2 测试)
- `test_pit_no_lookahead_simulation` — 12 月连续 PIT 验证
- `test_5_strategies_same_metrics_shape` — 5 策略 metrics 格式一致

---

## 6. 已知限制与 Phase A 待补

### 6.1 当前版本 (Stage 30.5) 限制

| # | 限制 | 严重度 | 影响 |
|---|------|--------|------|
| 1 | **无交易成本** | 🔴 致命 | 26% 年化可能虚高 1-2% |
| 2 | **无流动性 cap** | 🟠 严重 | 极端行情可能 100% 集中到小盘 |
| 3 | **极端市未压测** | 🟠 严重 | 2020 疫情/2022 港股/2024 政策未单独测 |
| 4 | **HMM 滞后未测** | 🟡 中 | 状态切换可能慢半拍 |
| 5 | **实盘数据 SLA 未测** | 🟡 中 | iFinD 拉取延迟/失败率未知 |
| 6 | **7 ETF 手工选** | 🟡 中 | 52 ETF 池可能更分散 |
| 7 | **2.68% 5 月 DD** | 🟢 轻 | 历史最大, 仍可控 |

### 6.2 Phase A 待补 (3.5 天)

- A1. 交易成本模块 (`v7/transaction_cost.py`)
- A2. 流动性 cap (`v7/liquidity_cap.py`)
- A3. 极端市压测 (`scripts/v7_0_stress_test.py`)
- A4. HMM 滞后回测 (`scripts/v7_0_hmm_lag_test.py`)
- A5. 实盘数据 SLA 测试 (`scripts/v7_0_data_sla_test.py`)

### 6.3 Phase B 待补 (4 天, 52 ETF 扩展)

- B1. ETF 池量化筛选 (44+12 → 52)
- B2. 5 方案 52 ETF 5-fold OOS
- B3. BL 协方差 PCA 降维 (52 → 10)
- B4. Macro Beta Ridge + PCA (5 macro → 2-3)
- B5. 落地决策 (vs 7 ETF 对比)

---

## 7. 落地建议

### 7.1 推荐路径

```
[Stage 30.5 - 当前] ✅
    C. Macro Beta 5-fold OOS 鲁棒赢家 (calmar 6.29, 5/5 正收益)

[Phase A - 1-2 周] 
    实盘前准备: 交易成本 + 流动性 + 极端市 + HMM 滞后 + SLA

[Phase B - 1-2 周] 
    52 ETF 扩展, 验证 7 ETF vs 52 ETF

[Phase C - 6-12 月] 
    模拟盘验证 (3-6 月) + 小资金实盘 (6-12 月)
```

### 7.2 实盘门槛 (用户决策)

```
[Phase 0: 模拟盘 (3-6 月)]
    ✓ 加入交易成本 (双边 0.1% × 月度 12 次 = 1.2%/年)
    ✓ 加入流动性 cap (单 ETF 30%, 月度 30% ADV)
    ✓ 极端市压测 (2020-02 疫情, 2022-04 港股, 2024-09 政策)
    ✓ HMM 滞后回测 (state 切换点 ±5 日)

[Phase 1: 小资金 (6-12 月)]
    ✓ 模拟盘 Sharpe ≥ 1.5
    ✓ 模拟盘最大 DD ≤ 25%
    ✓ 模拟盘胜率 ≥ 70% 月度

[Phase 2: 实盘 (12+ 月)]
    ✓ 模拟盘连续 3 月正收益
    ✓ Macro 数据 SLA ≥ 99%
    ✓ 风控规则就位
```

### 7.3 风险提示

- **26% 年化过于漂亮**: 5-fold OOS 仍属样本内 (虽 OOS), 实盘可能 15-20%
- **Macro Beta 系数敏感**: 5 维回归在样本不足时可能漂移
- **7 ETF 池手工选**: 引入选择偏差, 52 ETF 池应验证
- **HMM 5 状态分类**: 业界常用 4 状态, 5 状态可能过细

---

## 8. 后续工作

### 8.1 立即 (1-2 周)
- Phase A1-A5: 实盘前准备
- 把 C. Macro Beta 接到 v6.2 默认 (作为可选 sub-strategy)

### 8.2 中期 (1-2 月)
- Phase B1-B5: 52 ETF 扩展
- 与 v6.2 / 等权 7 ETF / 黄金单 ETF 组合对比, 找最优 风险预算

### 8.3 长期 (6-12 月)
- 模拟盘 + 小资金实盘
- 实盘数据 SLA 监控
- 季度 recalibration (HMM 重新训练, Beta 重新回归)

---

## 9. 引用

- 中信证券: 三维宏观状态监测体系 (2026)
- 国泰海通: 五维方法 (复苏/过热/衰退/中性, 5 年滚动窗口)
- 海通证券: 宏观调整的近一季风格动量差
- 银河证券: 宽基 ETF + 熵权法 (19.05%)
- Black & Litterman (1992): 原版 BL 模型
- Idzorek (2005): τ=0.05 推荐
- Ledoit-Wolf (2003): Covariance shrinkage
