# Stage 17-22 完整研究索引

> **更新日期**: 2026-07-09
> **范围**: Stage 17 (v4 诊断) → Stage 18 (v4 整合) → Stage 19 (LW 增强) → Stage 22 (v5 行业量价)
> **目标**: 完整记录从 v4 失败诊断 → v4 升级 → LW 可选模式 → v5 新子策略的完整研究链

---

## 一、研究时间线

| 阶段 | Commit | 关键产出 | 状态 |
|------|--------|---------|------|
| Stage 17 | `6de674f` | 4 份深度研究 (1300+ 行) | ✅ 完成 |
| Stage 17.5 | `7ca6c92` | 多策略 TAA 研究 | ✅ 完成 |
| Stage 18 v5 实验 | `9f4dfd9` | v5 子策略实验 (4+5 改进) | ⚠️ 已并入 v4 |
| **Stage 18 v4 整合** | `7d5004c` | **v4 + v5 合并, v5 删除** | ✅ **完成** |
| **Stage 19 LW** | `e17098c` | **Nagel Ledoit-Wolf + λ 收缩** | ✅ **完成** |
| Stage 19 未来扩展 | `ec4fded` | LW 8+ 因子 ready 文档 | ✅ 完成 |
| **Stage 22 v5** | `3e2100f` | **行业量价因子行业轮动 (v5)** | ✅ **完成** |
| Stage 22.5 详细统计 | `40c5cc8` | 年化收益 + 波动详细统计 | ✅ 完成 |
| **本索引更新** | (本次) | **总索引** | 🔄 |

---

## 二、Stage 17 — 4 份深度研究

**目标**: 摸清 v4 失败原因, 找 Stage 18 优化方向

### 2.1 COMPLEMENTARITY_RESEARCH.md (互补性)
- **核心**: v4A 和 v4B 高度相关 (0.78-0.90), 分散价值低
- **结论**: v3 (动量) 和 v4 才是真正互补 (相关 0.66)
- **推荐**: TCA (趋势切换) + VT (波动率) 调节

### 2.2 STYLE_ROTATION_RESEARCH.md (风格轮动)
- **核心**: 当前 L60_T3 (60d 动量 + Top-3) 是次优的
- **数据**: L60_T3 = Calmar 0.218, L120_T1 = Calmar **0.919** (4x 改善)
- **推荐**: 120d 动量 + Top-1, 或 Long-biased 5/20/120 T1

### 2.3 SMART_BETA_ALPHA_DECAY.md (Smart β 衰减)
- **核心**: Smart β 是低 beta 工具 (beta 0.60), 不是 alpha 工具
- **数据**: 2021-2024 累计 alpha +50%, 2025 之后急速衰减
- **推荐**: 静态 2 ETF (512040 + 515100)

### 2.4 FACTOR_TIMING_EFFECTIVENESS.md (因子择时)
- **核心**: 6 因子只有 value 稳定正 IC (+0.044, hit 60%)
- **结论**: 静态单买 512040 = Calmar 0.638, 比复杂 IC 择时更好
- **推荐**: 静态 value + div ETF, IC 择时失效

### 2.5 MULTI_STRATEGY_CORRELATION.md (TAA 多策略)
- **核心**: v3 + cash + TAA 收益 Calmar 0.918
- **方法**: trend following (v3 60d>3% 切到 v4A+v4B, 否则 cash)

---

## 三、Stage 18 — v4 整合 (v4 + v5 实验合并)

**核心**: 把 v5 4+5 改进整合到 v4 默认, 删除 v5 模块

### 3.1 风格轮动 v4 升级 (4 改进)
| 改进 | 诊断基础 | Stage 18 默认 |
|------|---------|---------------|
| 多窗口 Long-biased 5/20/120/180 | 单窗口 L=120 Calmar 0.016 | 5/20/120/180 (0.10/0.20/0.30/0.40) |
| 强制 dividend 底仓 20% | 5 风格相关 0.86-0.90, dividend 唯一分散器 | 0.20 |
| Top-2 选择 | Top-1 准确率 34.5% → Top-2 53.4% | 2 |
| Sideways regime filter | Sideways -2.50% ann, 70% 时间 | 50% 仓位 |

### 3.2 因子择时 v4 升级 (5 改进)
| 改进 | 诊断基础 | Stage 18 默认 |
|------|---------|---------------|
| 因子特异 forward_window | 统一 20d 是错的, 各因子最优 FW 完全不同 | m=120/r=60/v=40/d=180/q=252 |
| 因子特异 lag 平滑 | 5 因子 lag1 0.48-0.69 高持续 | m/v/d/q=4w, r=1 |
| Regime-conditioned 因子选择 | bull/bear/sideways 因子 IC 不同 | bull: m+v, bear: v+d+q, sideways: v |
| 删除 low_vol 因子 | IC vs forward 相关 -0.454, 反指 | use_low_vol=False |
| IC 质量过滤 (|IC|<0.05 → 0) | 84-94% 噪声率 | 0.05 |

### 3.3 升级效果 (8y 2018-2026)

| 策略 | 原 v4 | Stage 18 v4 | 提升 |
|------|-------|-------------|------|
| v4 风格 Calmar | 0.016 | **0.439** | 27x |
| v4 因子 Calmar | 0.092 | **0.613** | 6.7x |
| v3 70% + v4 因子 30% Calmar | - | **0.715** | - |
| 三策略 33/33/34 Calmar | - | **0.677** | - |

**核心文件**:
- `STAGE18_V4_FINAL.md` (270 行)
- `SUB_STRATEGY_DIAGNOSTIC.md` (368 行, 诊断基础)
- `style_rotation_diagnostic.py` + `factor_timing_diagnostic.py` (650 行)
- `v4_merged_verify.py` (验证脚本)

---

## 四、Stage 19 — Nagel 风格 LW 增强 (可选模式)

**核心**: 基于 Nagel 团队《Optimal Factor Timing in a High-Dimensional Setting》, 实施 Ledoit-Wolf 协方差 + λ 权重收缩

### 4.1 论文核心方法
- **Ledoit-Wolf 协方差收缩**: `cov_lw = (1-δ)·S + δ·F`, δ 自适应
- **MVO 权重**: `w ∝ cov_lw⁻¹ · μ` (long-only + L1 norm)
- **λ 权重收缩**: `w = (1-shrink)·w_mvo + shrink·w_equal`, `shrink = λ/(1+λ)`
- **滚动验证**: 训练 60m + 验证 12m

### 4.2 实施结果

| 模式 | 8y Calmar | OOS Calmar | OOS Sharpe |
|------|----------|-----------|-----------|
| v4 IC^2 (默认) | **0.613** | 0.581 | 0.65 |
| LW λ=10 | 0.531 | 0.476 | 0.60 |
| LW λ=100 | 0.536 | 0.476 | 0.60 |
| LW 滚动 λ | 0.468 | 0.476 | 0.60 |

### 4.3 关键发现
- **LW 在我们 5 ETF 类别设置下不显著优于 IC^2** (论文用 10 Barra 风格因子, 高维, LW 必要)
- **LW 优势在高维 (8+ 因子)**: 我们 5 因子, IC^2 集中度更有效
- **论文 A 股复现 λ=30-100 偏保守** 也在我们这观察到 (LW 滚动 33 次选 0, 21 次选 100)

### 4.4 未来扩展 (Stage 19.5)
- **触发条件**: 因子数 ≥ 8 / 多信号 ≥ 3 / 因子相关性 > 0.5
- **代码已就绪**: `lw_factor_timing.py`, `lw_factor_timing_integration.py`
- **重新启用**: 改 `lw_enabled=True, lw_lambda_mode="rolling"`, 跑 `v4_lw_integrated_test.py` 验证

**核心文件**:
- `STAGE19_LW_INTEGRATION.md` (264 行)
- `lw_factor_timing.py` (192 行, LW 核心)
- `lw_factor_timing_integration.py` (220 行, 集成)
- `factor_timing_v4.py` (+130 行, `lw_enabled/lw_lambda_mode` 字段)

---

## 五、Stage 22 — v5 行业量价因子 (新子策略)

**核心**: 实施华西证券《行业有效量价因子与行业轮动策略》 (2022-08-22)

### 5.1 11 个量价因子 (6 大类)

| 大类 | 因子 | IC 均值 (论文) |
|------|------|----------------|
| 动量 | 二阶动量 / 动量期限差 | 0.044 / 0.046 |
| 交易波动 | 成交金额波动 / 成交量波动 | 0.054 / 0.040 |
| 换手率 | 换手率变化 | 0.022 |
| 多空对比 | 多空对比总量 / 多空对比变化 | 0.063 / 0.025 |
| 量价背离 | 量价排序协方差 / 量价相关系数 / 一阶量价背离 | 0.028 / 0.034 / 0.037 |
| 量幅同向 | 量幅同向 | 0.021 |

### 5.2 数据准备 (Sina API)
- 端点: `https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData`
- 返回: `[{day, open, high, low, close, volume}]`
- 速率: ~10 req/s
- 拉取: 44 ETF × 2200 天 = 30s
- 新增: `common/data_sina.py` (140 行)
- 落盘: `data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet` (44×5×2064)

### 5.3 v5 SubStrategy 升级
- 升级为完整 SubStrategy (继承 v4.sub_strategy_v4)
- 完整 select/weight/run_step 接口
- 可与 v3/v4 在 multi_strategy 框架下组合
- 位置: `v5/industry_rotation_v5.py` (250 行)
- 11 因子: `v5/industry_factors.py` (270 行, 从 v4 移动)

### 5.4 实施结果 (8y 2018-2026 + OOS 2022-2026)

| 策略 | 8y Calmar | OOS Calmar | OOS Sharpe | 年化收益 | 年化波动 |
|------|----------|-----------|-----------|---------|---------|
| v3 baseline | 0.484 | 1.012 | 1.29 | 6.76% | 7.76% |
| v4 因子 (5 因子) | 0.613 | 0.581 | 0.65 | 11.15% | 18.03% |
| v4 风格 | 0.439 | - | - | 10.06% | 16.15% |
| **v5 量价 (11 因子)** | **0.643** | **0.600** | 0.67 | **17.59%** | 21.24% |
| v3 70% + v5 30% | 0.614 | 0.790 | 0.91 | 10.90% | 12.20% |
| v3 80% + v5 20% | 0.619 | **0.850** | **1.01** | 9.65% | 10.54% |
| **v3 33% + v4f 33% + v5 34%** | **0.753** | 0.747 | 0.84 | **12.56%** | **14.15%** |

### 5.5 Year-by-year 详细 (v5 量价)

| 年 | 收益 | 年化收益 | 年化波动 | Sharpe | DD |
|----|------|---------|---------|--------|-----|
| 2019 | +28.53% | +28.62% | 18.41% | 1.55 | -13.12% |
| 2020 | +39.60% | +39.74% | 19.86% | 2.00 | -14.95% |
| 2024 | **+87.36%** | **+87.62%** | 19.85% | **4.41** | -8.86% |
| 2025 | +31.08% | +31.30% | 22.85% | 1.37 | -19.34% |
| 2026 | -10.10% | -19.74% | 19.86% | -0.99 | -19.74% |

### 5.6 月度收益分布 (v5)
- mean **+1.50%**, std 5.20%
- min -12.96%, max **+32.67%** (大牛市月)
- skew +1.69, kurt **+11.83**
- VaR 5% -6.12%, CVaR 5% -9.06%
- **月胜率 61.4%**

### 5.7 Top-N 扫描 (Top-5 最优)

| Top-N | Ann | Calmar |
|-------|-----|--------|
| 3 | 12.89% | 0.536 |
| **5 (论文)** | **17.59%** | **0.643** |
| 7 | 14.89% | 0.519 |
| 10 | 12.49% | 0.461 |

### 5.8 相关性 (v5 是 3 策略中最独立的分散器)

| | v3 | v4 风格 | v4 因子 | **v5 量价** |
|---|----|---------|---------|------------|
| v5 量价 | 0.54 | **0.44** | **0.44** | 1.00 |

**核心文件**:
- `v5/STAGE22_V5_REPORT.md` (440 行)
- `v5/industry_rotation_v5.py` (250 行)
- `v5/industry_factors.py` (270 行)
- `scripts/industry_rotation_backtest.py` (390 行, Stage 19 实施)
- `scripts/v5_backtest.py` (250 行, Stage 22 集成)
- `scripts/v5_stats_report.py` (250 行, 详细统计)
- `data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet` (1.99 MB)
- `reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf` (1.77 MB)
- `reports/momentum_etf_rotation/v4/papers/README.md`

---

## 六、Stage 22 行业轮动 详细统计

### 6.1 同比汇总

| 策略 | 年化收益 | 年化波动 | Sharpe | Max DD | Calmar |
|------|---------|---------|--------|--------|--------|
| v3 baseline | 6.76% | 7.76% | 0.92 | -13.97% | 0.484 |
| v4 风格 | 10.06% | 16.15% | 0.70 | -22.95% | 0.439 |
| v4 因子 | 11.15% | 18.03% | 0.70 | -18.18% | 0.613 |
| **v5 量价** | **17.59%** | **21.24%** | **0.89** | -27.36% | **0.643** |
| v3 80% + v5 20% | 9.65% | 10.54% | **0.96** | -15.59% | 0.619 |
| **v3 33% + v4f 33% + v5 34%** | **12.56%** | **14.15%** | 0.94 | -16.67% | **0.753** |

### 6.2 滚动年化收益

| 策略 | 1y | 2y | 3y | 5y |
|------|-----|-----|-----|-----|
| v5 量价 | **+24.67%** | **+55.02%** | **+69.77%** | **+127.88%** |
| v3 80% + v5 20% | +12.90% | +26.54% | +33.53% | +56.68% |
| v3 33% + v4f 33% + v5 34% | +16.53% | +35.39% | +47.32% | +85.71% |

**核心发现**: v5 5y 滚动年化 +127.88% (v3 baseline 仅 +34.50%), 月度极值 +32.67% (2024 牛市)

---

## 七、最终生产推荐

### 7.1 生产配置 (基于 OOS Walk-Forward 2022-2026)

| 风险偏好 | 配置 | 8y Calmar | OOS Calmar | OOS Sharpe |
|---------|------|-----------|------------|-----------|
| **最稳健** | v3 80% + v5 20% | 0.619 | **0.850** | **1.01** ⭐⭐⭐ |
| **平衡** | v3 70% + v5 30% | 0.614 | 0.790 | 0.91 ⭐ |
| **分散** | v3 33% + v4f 33% + v5 34% | **0.753** | 0.747 | 0.84 ⭐ |
| **进取** | v3 50% + v4f 25% + v5 25% | 0.733 | 0.788 | 0.91 ⭐ |

### 7.2 单策略部署

| 风险偏好 | 策略 | 8y Calmar | OOS Calmar |
|---------|------|-----------|------------|
| 防御 | v3 baseline | 0.484 | **1.012** |
| 平衡 | v5 量价 | 0.643 | 0.600 |
| 进取 | v3 33% + v4f 33% + v5 34% | 0.753 | 0.747 |

---

## 八、文件清单 (按时间倒序)

### 8.1 Stage 22.5 详细统计 (commit `40c5cc8`)
- `scripts/v5_stats_report.py` (250 行, 年化收益 + 波动 + DD + 月度分布)
- `reports/momentum_etf_rotation/v5/stats_summary.csv` (汇总表)
- `reports/momentum_etf_rotation/v5/STAGE22_V5_REPORT.md` (+220 行)

### 8.2 Stage 22 v5 (commit `3e2100f`)
- `QuantNodes/strategy/momentum_etf_rotation/v5/__init__.py` (60 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/industry_factors.py` (270 行)
- `QuantNodes/strategy/momentum_etf_rotation/v5/industry_rotation_v5.py` (250 行)
- `scripts/v5_backtest.py` (250 行)
- `reports/momentum_etf_rotation/v5/STAGE22_V5_REPORT.md` (190 行)
- 删除: `QuantNodes/strategy/momentum_etf_rotation/v4/industry_factors.py`

### 8.3 Stage 19 LW 增强 (commit `e17098c`)
- `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing.py` (192 行)
- `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing_integration.py` (220 行)
- `QuantNodes/strategy/momentum_etf_rotation/v4/factor_timing_v4.py` (+130 行)
- `scripts/lw_factor_timing_backtest.py` (350 行)
- `scripts/v4_lw_integrated_test.py` (220 行)
- `reports/momentum_etf_rotation/v4/STAGE19_LW_INTEGRATION.md` (264 行)

### 8.4 Stage 19 未来扩展 (commit `ec4fded`)
- `reports/momentum_etf_rotation/v4/STAGE19_LW_INTEGRATION.md` (+50 行, 8+ 因子 ready)

### 8.5 Stage 18 v4 整合 (commit `7d5004c`)
- `QuantNodes/strategy/momentum_etf_rotation/v4/style_rotation_v4.py` (+178 行)
- `QuantNodes/strategy/momentum_etf_rotation/v4/factor_timing_v4.py` (+57 行)
- `scripts/v4_merged_verify.py` (190 行)
- `reports/momentum_etf_rotation/v4/STAGE18_V4_FINAL.md` (270 行)
- 删除: `QuantNodes/strategy/momentum_etf_rotation/v5/` (3 文件)

### 8.6 Stage 17.5 多策略 (commit `7ca6c92`)
- `reports/momentum_etf_rotation/v4/MULTI_STRATEGY_CORRELATION.md` (313 行)
- `reports/momentum_etf_rotation/v4/STAGE17_V4_FINAL.md` (后续改名)

### 8.7 Stage 17 v4 诊断 (commit `6de674f`)
- `COMPLEMENTARITY_RESEARCH.md` (305 行)
- `STYLE_ROTATION_RESEARCH.md` (306 行)
- `SMART_BETA_ALPHA_DECAY.md` (241 行)
- `FACTOR_TIMING_EFFECTIVENESS.md` (283 行)
- `STAGE17_RESEARCH_INDEX.md` (219 行, 旧版索引)

---

## 九、数据基础

| 数据 | 路径 | 规模 | 用途 |
|------|------|------|------|
| ETF NAV (close) | `data/real/etf_nav_2018-01-01_2026-06-30.parquet` | 44 ETF × 2058 天 | v3 / v4 |
| ETF OHLCV | `data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet` | 44 × 5 × 2064 | v5 行业量价 |
| Smart β ETF | `data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet` | 12 × 2058 | v4 因子 |
| Stage 17 NAV | `reports/momentum_etf_rotation/v4/stage17_navs.parquet` | 7 策略 | 验证 |
| v4 merged NAV | `reports/momentum_etf_rotation/v4/v4_merged_navs.parquet` | v4_style + v4_factor | 验证 |
| v5 NAV | `reports/momentum_etf_rotation/v5/v5_navs.parquet` | v3+v4s+v4f+v5 | 验证 |
| 论文 PDF | `reports/momentum_etf_rotation/v4/papers/huaxi_industry_rotation.pdf` | 1.77 MB | 理论依据 |

---

## 十、下一步

1. **生产部署**: v3 80% + v5 20% (OOS Calmar 0.850, Sharpe 1.01) ⭐⭐⭐
2. **Stage 23**: IC 加权复合因子 (替代等权, 期望 Calmar 0.7+)
3. **Stage 24**: 加交易成本回测 (v5 月换手率 161% 较高)
4. **Stage 25**: 把 v5 接入 multi_strategy_v4 框架 (v4Mode.V5 模式)
5. **Stage 26**: LW 框架 ready (8+ 因子时启用, 触发条件见 STAGE19 §8.1)
