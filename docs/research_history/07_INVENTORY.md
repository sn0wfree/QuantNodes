# 07 — 关键资产索引（Inventory）

> **范围**：momentum_etf_rotation 策略研发产生的关键资产（代码 / 数据 / 报告 / HTML）
> **用途**：快速定位"哪个版本在哪里"、"哪些工具可复用"、"哪些文档可参考"

---

## 一、策略代码（按版本号）

### V0–V3 时期（动量 → slope_r² → 多策略架构）

| 路径 | 行数 | 用途 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/v1/portfolio_v1.py` | 323 | CICC Stage 8 原始 4 步组合管理 |
| `QuantNodes/strategy/momentum_etf_rotation/v1/momentum_v1.py` | 123 | 纯涨幅动量排名 + 52 周高点距离 |
| `QuantNodes/strategy/momentum_etf_rotation/v1/backtest_v1.py` | 172 | 月度调仓循环 |
| `QuantNodes/strategy/momentum_etf_rotation/v1/strategy_versions_v1.py` | 38 | v1 VERSIONS 字典 |
| `QuantNodes/strategy/momentum_etf_rotation/v2/momentum_v2.py` | 268 | 动量 + slope×R² + hybrid |
| `QuantNodes/strategy/momentum_etf_rotation/v2/portfolio_v2.py` | 759 | 4 步 + VT + Cost + TF + Caps + RotationConfig |
| `QuantNodes/strategy/momentum_etf_rotation/v2/backtest_v2.py` | 250 | 月度调仓 + HMM + VT + Cost + CICC 对照 |
| `QuantNodes/strategy/momentum_etf_rotation/v2/strategy_v2.py` | 33 | V2Strategy |
| **`QuantNodes/strategy/momentum_etf_rotation/v3/sub_strategy_v3.py`** | 228 | **SubStrategy 抽象基类（v4-v7 全部继承）** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v3/reversion_v3.py` | 251 | 均值反转子策略 |
| `QuantNodes/strategy/momentum_etf_rotation/v3/industry_rotation_v3.py` | 254 | 行业轮动子策略 |
| `QuantNodes/strategy/momentum_etf_rotation/v3/sub_weighting_v3.py` | 195 | 3 种子策略权重法 + 合并 |
| `QuantNodes/strategy/momentum_etf_rotation/v3/multi_strategy_v3.py` | 417 | 多策略主回测 |
| `QuantNodes/strategy/momentum_etf_rotation/v3/strategy_v3.py` | 48 | V3Strategy |

### V4–V6 时期（多因子诊断 + IC 加权 + 正交化）

| 路径 | 行数 | 用途 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/v4/style_rotation_v4.py` | - | 风格轮动 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/smart_beta_v4.py` | - | Smart β |
| `QuantNodes/strategy/momentum_etf_rotation/v4/factor_ic.py` | - | 6 因子 IC |
| `QuantNodes/strategy/momentum_etf_rotation/v4/factor_timing_v4.py` | - | 因子择时 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/regime_transitions.py` | - | HMM 距离先验 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/regime_detector_v4.py` | - | HMM 检测 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/multi_strategy_v4.py` | - | 6 模式主回测 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing.py` | - | LW 因子择时 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/lw_factor_timing_integration.py` | - | LW 集成 |
| `QuantNodes/strategy/momentum_etf_rotation/v4/universe_v4.py` | - | 5 风格组 + 7 Smart β |
| **`QuantNodes/strategy/momentum_etf_rotation/v5/industry_factors.py`** | - | **11 量价因子** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v5/industry_rotation_v5.py` | - | V5 完整 SubStrategy |
| `QuantNodes/strategy/momentum_etf_rotation/v5_1/industry_rotation_v5_1.py` | - | 逆波动率加权 |
| `QuantNodes/strategy/momentum_etf_rotation/v6/industry_rotation_v6.py` | - | V6 单策略（7 档风控消融）|
| `QuantNodes/strategy/momentum_etf_rotation/v6_1/factor_weighting.py` | - | IC-IR 加权（expanding 12/24/36 月 + 6 月平滑）|
| `QuantNodes/strategy/momentum_etf_rotation/v6_1/industry_rotation_v6_1.py` | - | V6.1 SubStrategy |
| **`QuantNodes/strategy/momentum_etf_rotation/v6_2/factor_orthogonal.py`** | - | **Gram-Schmidt + 5 种 sort_method（DEPRECATED）** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v6_2/industry_rotation_v6_2.py` | - | V6.2 SubStrategy（研究版本）|

### V7–V10 时期（TV-PR + 引擎收敛 + Vol-parity）

| 路径 | 行数 | 用途 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/v7/dynamic_allocation.py` | - | 5 Macro Dynamic A.Top-K |
| `QuantNodes/strategy/momentum_etf_rotation/v7/black_litterman.py` | - | B.BL |
| `QuantNodes/strategy/momentum_etf_rotation/v7/macro_beta.py` | - | C.Beta（5-fold 赢家）|
| `QuantNodes/strategy/momentum_etf_rotation/v7/state_momentum.py` | - | D.Momentum |
| `QuantNodes/strategy/momentum_etf_rotation/v7/state_inverse_vol.py` | - | E.IV |
| `QuantNodes/strategy/momentum_etf_rotation/v7/macro_data.py` | - | 9 宏观因子 |
| `QuantNodes/strategy/momentum_etf_rotation/v7/factor_risk_parity.py` | - | Bootstrap-Lasso + FRP |
| `QuantNodes/strategy/momentum_etf_rotation/v7/transaction_cost.py` | 90 | 交易成本（A1）|
| `QuantNodes/strategy/momentum_etf_rotation/v7/liquidity_cap.py` | 110 | 流动性 cap（A2）|
| `QuantNodes/strategy/momentum_etf_rotation/v7/dcc_regime_overlay.py` | - | DCC 6 维特征 + regime overlay |
| `QuantNodes/strategy/momentum_etf_rotation/v7/graph_distance_factors.py` | - | 图谱距离因子 |
| `QuantNodes/strategy/momentum_etf_rotation/v7/correlation_distance_factors.py` | - | 相关性距离因子 |
| `QuantNodes/strategy/momentum_etf_rotation/v7/enhanced_factors.py` | - | f12-f17 增强量价 |
| **`QuantNodes/strategy/momentum_etf_rotation/v7/tvpr_estimator.py`** | - | **TV-PR 估计器（ADMM）** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v7/data_loader_v7_6.py` | - | X[T,N,K] 面板构造 |
| **`QuantNodes/strategy/momentum_etf_rotation/v7/macro_substrategy_v7_6.py`** | - | **v7.6 完整 SubStrategy（生产）** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v8/jump_model.py` | - | Jump Model（regime-aware）|
| `QuantNodes/strategy/momentum_etf_rotation/v9/macro_layer.py` | - | 宏观层（5 宏观因子）|
| `QuantNodes/strategy/momentum_etf_rotation/v9/factor_galaxy.py` | - | 因子银河（熵权法）|
| `QuantNodes/strategy/momentum_etf_rotation/v9/factor_score_basic.py` | - | 因子打分基础 |
| `QuantNodes/strategy/momentum_etf_rotation/v9/citic_rotation.py` | - | 中信行业轮动 |
| **`QuantNodes/strategy/momentum_etf_rotation/v10/dual_momentum.py`** | - | **Antonacci GEM 模型（4 大类资产）** ⭐ |
| `QuantNodes/strategy/momentum_etf_rotation/v10/epo_momentum.py` | - | EPO 动量 |
| `QuantNodes/strategy/momentum_etf_rotation/v10/rrg_rotation.py` | - | RRG 四象限 |
| `QuantNodes/strategy/momentum_etf_rotation/v10/dynamic_weight_schemes.py` | - | 5 方案动态权重 |

**注**: `v10/` 现只保留 4 策略主体. 5 层架构已迁移到 `v11/`.

### v11（5 层架构 + ACT-1/2/3, 从 v10 迁移）

| 路径 | 用途 |
|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/v11/config_v11.py` | V11Config + 所有 LayerConfig |
| `QuantNodes/strategy/momentum_etf_rotation/v11/macro_layer.py` | Layer 1: 宏观择时 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/industry_layer.py` | Layer 2A: 行业轮动 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/style_layer.py` | Layer 2B: 风格轮动 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/factor_layer.py` | Layer 2C: 因子选股 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/risk_layer.py` | Layer 3: Jump Model |
| `QuantNodes/strategy/momentum_etf_rotation/v11/risk_layer_v11.py` | ACT-2/3: Kelly + 回撤控制 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/position_layer.py` | Layer 4: 动态仓位 |
| `QuantNodes/strategy/momentum_etf_rotation/v11/portfolio_layer.py` | Layer 5: 组合构建 |
| **`QuantNodes/strategy/momentum_etf_rotation/v11/v11_strategy.py`** | V11 主入口 (5 层串联 + ACT-1/2/3) |
| **`QuantNodes/strategy/momentum_etf_rotation/v11/backtest_v11.py`** | v11 回测引擎 |
| `scripts/v11/v11_backtest.py` | v11 回测脚本 |

### 通用工具（`common/`）

| 路径 | 行数 | 用途 | 被复用 |
|------|------|------|--------|
| **`QuantNodes/strategy/momentum_etf_rotation/common/strategy_engine.py`** | 190 | **BaseStrategy + StrategyEngine（最简策略引擎）** ⭐ | v1-v10 所有策略 |
| `QuantNodes/strategy/momentum_etf_rotation/common/covariance.py` | 120 | 4 方法协方差（sample / LW / EWMA / diagonal）| v3-v10 |
| `QuantNodes/strategy/momentum_etf_rotation/common/risk_parity.py` | 110 | solve_risk_parity + solve_max_diversification | v3-v10 |
| `QuantNodes/strategy/momentum_etf_rotation/common/regime_detector.py` | 150 | HMMRegimeDetector + get_regime_params | v2-v4 |
| `QuantNodes/strategy/momentum_etf_rotation/common/data_loader.py` | - | 通用数据加载 | v1-v10 |
| `QuantNodes/strategy/momentum_etf_rotation/common/rd_utils.py` | - | R&D 工具（提升自 `scripts/quant/`）| v4-v10 |
| `QuantNodes/strategy/momentum_etf_rotation/common/walk_forward.py` | **990** ⭐ | **Walk-Forward 框架（通用）** | v7.6-v10 OOS |
| `QuantNodes/strategy/momentum_etf_rotation/common/ic_utils.py` | - | 截面 vs 时序 IC 统一 | v7.6+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/extended_metrics.py` | - | 17 指标 | 全部 |
| `QuantNodes/strategy/momentum_etf_rotation/common/backtest_utils.py` | - | 调仓 + 成本 + NAV 计算 | 全部 |
| `QuantNodes/strategy/momentum_etf_rotation/common/backtest_engine.py` | - | 通用回测引擎 | v3+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/config_runner.py` | 270 | YAML 配置驱动 | 全部 |
| `QuantNodes/strategy/momentum_etf_rotation/common/brinson.py` | - | Brinson 归因 | v8+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/contribution.py` | - | 贡献分析 | v8+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/fi_plus.py` | - | FI 增强（stub）| - |
| `QuantNodes/strategy/momentum_etf_rotation/common/data_sina.py` | 140 | Sina API 数据 | v5+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/data_eastmoney.py` | 144 | Eastmoney API 数据 | v5+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/data_tencent.py` | - | Tencent API 数据 | v1+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/portfolio.py` | - | 根级 portfolio 兼容桩 | - |
| `QuantNodes/strategy/momentum_etf_rotation/common/momentum.py` | 268 | 根级 momentum 兼容桩 | v1+ |
| `QuantNodes/strategy/momentum_etf_rotation/common/backtest.py` | 250 | 根级 backtest 兼容桩 | v1+ |

### 顶级根路径

| 路径 | 行数 | 用途 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/strategy_versions.py` | 196 | v0.0~v1.0 + v7.10 注册表 |
| `QuantNodes/strategy/momentum_etf_rotation/__init__.py` | - | re-export 桩 |

### YAML 配置模板

```
QuantNodes/strategy/momentum_etf_rotation/strategies/
├── v1.0.yaml
├── v2.yaml
├── v3.yaml
├── v4.yaml
├── v6.yaml
└── v7.10.yaml
```

---

## 二、scripts/ 脚本库（按功能）

### 数据准备（与策略直接相关）

| 路径 | 用途 |
|------|------|
| `scripts/fetch_real_etf_panel.py` | Tencent ETF 前复权面板 |
| `scripts/fix_ohlcv_adjust.py` | OHLCV 50% 阈值复权 |
| `scripts/fetch_proxy_indices.py` | iFinD 19 proxy 指数 |
| `scripts/fetch_proxy_indices_wind.py` | Wind proxy |
| `scripts/build_proxy_panel.py` | Proxy 对齐 v56 |
| `scripts/extract_macro_data.py` | gold SQLite 提取宏观 |

### IC 评估

| 路径 | 用途 |
|------|------|
| `scripts/calc_factor_ic.py` | 统一计算 IC |
| `scripts/factor_timing_diagnostic.py` | IC 窗口衰减 + 自相关 + regime |

### V4–V6 评估

| 路径 | 用途 |
|------|------|
| `scripts/eval_v6_2_overfitting.py` | 过拟合综合评估 ⭐ |
| `scripts/test_v6_2_starting_points.py` | 10 起点测试 ⭐ |
| `scripts/generalization_test_v6_2.py` | 5-fold walk-forward ⭐ |
| `scripts/v6_2_phase1_ablation.py` / `phase3_qr_ablation.py` / `phase4_warmup_ablation.py` / `phase4_grid_ablation.py` / `v6_2_ir_expanding_5fold.py` | 5 个消融 |
| `scripts/v5_1_ablation.py` | 4 项 S 消融 |
| `scripts/lw_factor_timing_backtest.py` | LW 回测 |
| `scripts/style_rotation_diagnostic.py` | 风格轮动诊断 |
| `scripts/style_rotation_grid.py` | 风格轮动网格 |
| `scripts/factor_timing_diagnostic.py` | 因子择时诊断 |
| `scripts/industry_rotation_backtest.py` | 行业轮动回测 |

### V7 验证

| 路径 | 用途 |
|------|------|
| `scripts/v7_0_stress_test.py` | A3 极端市压测 |
| `scripts/v7_0_hmm_lag_test.py` | A4 HMM 滞后 |
| `scripts/v7_0_data_sla_test.py` | A5 SLA |
| `scripts/v7_0_select_52_etf.py` | B1 52 ETF 量化筛选 |
| `scripts/v7_0_macro_oos.py` | 5-fold OOS |
| `scripts/v7_0_macro_oos_52etf.py` | 7/41 ETF 对比 |
| `scripts/v7_0_52etf_decision.py` | B5 决策 |
| `scripts/v7_3_bootstrap_sensitivity.py` | bootstrap 5×3 档 |
| `scripts/v7_6_validation.py` | v7.6 完整验证 |
| `scripts/v7_6_sensitivity_*.py` | 8 阶段敏感性 |
| `scripts/v7_6_topn5_validation.py` | top_n=5 起点 CV% |
| `scripts/v7_6_tf_regime_test.py` | TF + Regime 加固 |
| `scripts/v7_6_regime_combo_test.py` | regime_combo 防御 |
| `scripts/eval_v7_6_validation.py` | v7.6 综合评估 |
| `scripts/v7_10_beta_timing_test.py` | β[t-1] vs β[t] |
| `scripts/v7_10_monday_open_test.py` | 周一开盘执行 |
| `scripts/run_v7_7_backtest.py` | v7.7 树模型 |
| `scripts/v7_7_adaptive_test.py` | v7.7 自适应 |
| `scripts/v7_7_phase2_rolling.py` | Phase 2 滚动 |

### V8 / V9 / V10 / Combo

| 路径 | 用途 |
|------|------|
| `scripts/v8_*` | Jump Model 实验 |
| `scripts/v9_*` | 宏观周期 / 银河 / 中信 |
| `scripts/v10/*` | V10 ETF 轮动 |
| `scripts/combo/combine_a_*.py` | Combo A+ |
| `scripts/combo/combine_b1_*.py` | Combo B1 (X 维度扩展)|
| `scripts/combo/combine_d_*.py` | Combo D (3 源加权)|
| `scripts/combo/combine_e_*.py` | Combo E (3 策略加权)|
| `scripts/combo/combine_e_pbear_dynamic.py` | P_bear 动态 |
| `scripts/combo/regenerate_v*_nav_*.py` | 无前视重算 |
| `scripts/combo/standard_comparison.py` | 标准对比 |
| `scripts/combo/full_sample_metrics.py` | 全样本指标 |
| `scripts/combo/export_v9_navs.py` | v9 NAV 导出 |

### 公平对比 + 前复权

| 路径 | 用途 |
|------|------|
| `scripts/fix_ohlcv_adjust.py` | 9 只 ETF 拆合股前复权 |
| `scripts/compare_cicc_vs_stage12a.py` | CICC vs Stage 12A 4 配置 |
| `scripts/chart_cicc_vs_stage12a.py` | CICC vs 图表 |
| `scripts/validate_stage16a.py` | V3 验证 |
| `scripts/chart_stage16a.py` | V3 图表 |
| `scripts/build_stage_charts.py` | 14 个 Stage 17-22 图表 |

### 因子工程（`scripts/quant/factors/`）

| 路径 | 用途 |
|------|------|
| `scripts/quant/factors/` | 因子库基础 |
| `scripts/quant/strategies/` | 策略 yaml 配置 |

### 量化研究（`scripts/research/`）

| 路径 | 用途 |
|------|------|
| `scripts/research/aggregate_alpha_results.py` | alpha 聚合 |
| `scripts/research/analyze_alpha_results.py` | alpha 分析 |
| `scripts/research/cross_validate_factor.py` | 因子交叉验证 |
| `scripts/research/deep_compare_alphalens.py` | alphalens 深度对比 |
| `scripts/research/merge_alpha_data.py` | alpha 数据合并 |
| `scripts/research/run_101_alphas_v2.py` | 101 alpha |
| `scripts/research/validate_alphalens.py` | alphalens 验证 |
| `scripts/research/validate_backtrader.py` | backtrader 验证 |
| `scripts/research/validate_qn_nodes.py` | qn_nodes 验证 |

### 根目录通用工具

| 路径 | 用途 |
|------|------|
| `walk_forward.py` | Walk-Forward 顶层入口 |
| `research.py` | 研究工具顶层入口 |

---

## 三、数据资产（`data/`）

### ETF 主面板

| 路径 | 用途 |
|------|------|
| `data/real/etf_nav_2018-01-01_2026-06-30.parquet` | ETF NAV 主面板 |
| `data/real/per_etf/*.parquet` | per-ETF 缓存 |
| `data/real/fetch_log.json` | 拉取日志 |
| `data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet` | OHLCV 前复权（关键）⭐ |

### Proxy + 宏观

| 路径 | 用途 |
|------|------|
| `data/high_freq_macro/v56_proxy_indices_daily.parquet` | Proxy panel |
| `data/high_freq_macro/_proxy_nan_table.csv` | 26 ETF pre/post IPO NaN |
| `data/high_freq_macro/_proxy_etf_map.csv` | ETF→Proxy 映射 |

### V5 / V7 / V10 数据产物

| 路径 | 用途 |
|------|------|
| `data/real/etf_ohlcv_2018-01-01_2026-06-30.parquet` | V5 OHLCV（Sina）|
| `data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet` | 12 ETF Smart β |
| `data/real/v56_expanded_daily.parquet` | 56 资产扩展池 |
| `data/real/v7_10_*` | v7.10 数据产物 |

### 数据源限制（已记录）

- iFinD 试用 token 起点 2021-07-01
- 黄金 / 能源化工 / 豆粕：既有跨资产指数
- 日经 / 港股通科技：无直接 proxy
- 节假日：A 股/港股/美股对齐后保留 NaN
- 数据版本演化：gold DB → Excel、月频实际利率 → FRED 日频 DFII10

---

## 四、报告资产（`reports/momentum_etf_rotation/`）

### 业绩 HTML（核心输出）

| 路径 | 备注 |
|------|------|
| `combo/STRATEGY_ITERATION_RECORD.html` | 9 策略精简版（生产入口）⭐ |
| `combo/STRATEGY_ITERATION_RECORD_v10.html` | v10 专用 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260714.html` | 过程性（7-14）|
| `combo/STRATEGY_ITERATION_RECORD_v2_20260717.html` | 7-17 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260718.html` | 7-18 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260720.html` | 7-20 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260722.html` | 7-22 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260723.html` | 7-23 |
| `combo/STRATEGY_ITERATION_RECORD_v2_20260724.html` | 最新（7-24）|
| `combo/UNIFIED_V1V5_EVOLUTION.html` (6.5 MB) | v1-v5 9 策略 9 图 |
| `combo/UNIFIED_V1V5_REPORT.md` (224 行) | 完整 markdown 报告 |
| `combo/V1V5_NAV_CURVES.html` (~8 MB) | 9 策略纯 NAV + HS300 |
| `combo/V1V5_NAV_CURVES_v2_20260710.html` | 7-10 |
| `combo/V1V5_NAV_CURVES_v2_20260714.html` | 7-14 |
| `index.html` | v1/v2/v3 一站式仪表盘 |

### 阶段报告

| 路径 | 行数 | 主题 |
|------|------|------|
| `STAGE_SUMMARY.md` (in docs/) | 236 | Stage 12A 之前总览 |
| `STAGE32_PLAN.md` | 121 | V7.10 硬化 5 P0 |
| `STAGE33_PLAN.md` | 145 | 新因子 + HMM + 跨资产 + 代码清理 |
| `STAGE17_22_INDEX.md` | 380+ | Stage 17→22 完整链 |
| `STAGE17_22_CHARTS.html` (7 MB) | - | Stage 17-22 14 图表 |

### v4 报告（Stage 17–19）

| 路径 | 行数 |
|------|------|
| `v4/STAGE17_PLAN.md` | 195 |
| `v4/HMM_DISTANCE_PLAN.md` | 153 |
| `v4/IC_PERFORMANCE_REPORT.md` | 100 |
| `v4/COMPLEMENTARITY_RESEARCH.md` | 305 |
| `v4/STYLE_ROTATION_RESEARCH.md` | 306 |
| `v4/SMART_BETA_ALPHA_DECAY.md` | 241 |
| `v4/FACTOR_TIMING_EFFECTIVENESS.md` | 283 |
| `v4/MULTI_STRATEGY_CORRELATION.md` | 313 |
| `v4/SUB_STRATEGY_DIAGNOSTIC.md` | 368 |
| `v4/STAGE17_RESEARCH_INDEX.md` | 219 |
| `v4/STAGE17_VALIDATION.md` | 173 |
| `v4/STAGE18_V4_FINAL.md` | 270 |
| `v4/STAGE19_LW_INTEGRATION.md` | 264 |
| `v4/INDUSTRY_ROTATION_REPORT.md` | - |
| `v4/v4_smart_beta_optimization_report.md` | - |
| `v4/v4_stage27_report.md` | - |
| `v4/v4_stage27_43etf_report.md` | - |
| `v4/v4_full_sensitivity_report.md` | - |
| `v4/papers/huaxi_industry_rotation.pdf` | 1.77 MB, 23 页 |

### v5 / V6 / V6.2 报告

| 路径 | 行数 |
|------|------|
| `v5/STAGE22_V5_REPORT.md` | 410 |
| `v5/stats_summary.csv` | - |
| `v6_2/STAGE29_PROMOTION.md` | - |
| `v6_2/__init__.py` | PROMISING → 研究版本 |

### v7 报告（Stage 32）

| 路径 | 行数 |
|------|------|
| `v7/bootstrap_sensitivity/sensitivity_summary.md` | - |
| `v7_3_factor_loadings.csv` / `v7_3_*` | - |
| `v7_6_validation.md` | - |
| `v7_6_factor_ic_report.md` | - |
| `v7_6_sensitivity_report.md` | - |
| `v7_6_oos_validation.md` | - |
| `v7_6_stop_loss_test.md` | - |
| `v7_6_tf_regime_test.md` | - |
| `v7_6_topn5_validation.md` | - |
| `v7_6_optimal_verify.md` | - |
| `v7_6_regime_combo_test.md` | - |
| `v7_7_phase1_results.md` | - |
| `v7_7_phase2_results.md` | - |
| `v7_9_oos_validation.md` | - |
| **`v7_10_oos_validation.md`** | 118 ⭐ |
| **`v7_10_cv_test.md`** | 58 ⭐ |
| `v7_10_beta_timing.md` | 42 |
| `v7_10_monday_open_comparison.md` | 35 |
| `v7_10_step_test.md` | 41 |
| `v7_10_*.png` × 10 | - |

### v8 / v9 / v10 报告

| 路径 | 备注 |
|------|------|
| `v8_3state_experiment/` | 111 行 summary |
| `v8_correct/` / `v8_diagnostic/` / `v8_integrated/` | - |
| `v8_jump_model_*` / `v8_probabilistic_*` | - |
| `v9/current_cycle_state.md` / `cycle_decomposition_report.md` | - |
| `v9/dashboard.html` / `phase_dashboard.png` / `score_timeseries.png` | - |
| `v9/macro_*` / `factor_galaxy_*` / `citic_*` | - |
| `v9/dynamic_risk_parity_*` / `multi_asset_*` / `factor_allocator_*` | - |
| **`v9/strategy_factor_analysis.md`** | 184 行（9 策略核心机制 + 10 因子）⭐ |
| `v10/*.png` / `v10_*` 报告 | - |
| `v10/dual_momentum_nav.parquet` / `epo_momentum_nav.parquet` / `rrg_rotation_nav.parquet` | - |
| `v10/dynamic_nav_*.parquet` × 6 / `dynamic_weights_*.parquet` × 5 | - |
| `v10/v10_report.md` / `v10_compare_report.md` / `v10_tvpr_sensitivity_report.md` | - |
| `v10/vol_parity_4strat_nav.parquet` | ⭐⭐⭐ |

### Combo 报告 + 数据产物

| 路径 | 备注 |
|------|------|
| `combo/final_strategy_comparison.csv` | - |
| `combo/standard_comparison_*.csv` | - |
| `combo/full_sample_metrics.csv` | - |
| `combo/equal_weight_baseline.parquet` | - |
| `combo/combo_navs_unified52.parquet` | - |
| **`combo/v6_navs.parquet`** | V6 7 档 NAV ⭐ |
| `combo/v6_1_ablation_navs.parquet` | V6.1 7 组消融 |
| **`combo/v6_2_ablation_navs.parquet`** | V6.2 6 组消融 ⭐ |
| **`combo/v6_2_ir_expanding_5fold.csv`** | 5-fold walk-forward 结果 ⭐ |
| **`combo/v6_2_starting_points.csv`** | 10 起点 CV% 结果 ⭐ |
| `combo/v7_10_v56_*.parquet` × 4 (5bp/10bp/15bp/20bp) | - |
| `combo/v7_14_nav*.parquet` | - |
| `combo/figs/*.png` | - |
| `combo/archive/stage27-v6.2-baseline` / `v5.1-baseline` | - |
| `combine_*.csv` + `*.parquet` × 30+ | B1/E/A+/pbear_dynamic 全套 |

### 数据产物导航

| 路径 | 用途 |
|------|------|
| `reports/momentum_etf_rotation/v4/stage17_navs.parquet` | Stage 17 NAV |
| `reports/momentum_etf_rotation/v4/v4_merged_navs.parquet` | v4 合并 |
| `reports/momentum_etf_rotation/v4/v4_lw_integrated_navs.parquet` | LW 集成 |
| `reports/momentum_etf_rotation/v4/lw_factor_timing_navs.parquet` | LW 因子 NAV |
| `reports/momentum_etf_rotation/v4/hmm_regime_history.csv` | HMM 时序状态 |
| `reports/momentum_etf_rotation/v5/v5_navs.parquet` | V5 NAV |

---

## 五、设计文档（`docs/`）

### TV-PR / 策略核心

| 路径 | 行数 | 主题 |
|------|------|------|
| `38-v7_3_macro_only.md` | 819 | v7_macro_baseline 锁定声明 |
| `39-v7_6_tvpr.md` | 254 | TV-PR 数学公式 + 因子定义 + 实验设计 ⭐ |
| `40-v7_6_sensitivity.md` | 330 | 10 阶段敏感性测试设计 ⭐ |
| `41-v7_6_factor_ic_and_enhancement.md` | 411 | 因子 IC 评估框架 ⭐ |
| `42-v7_6_l1_penalty_fix.md` | - | L1 罚项修复 |
| `43-v7_7_lgbm.md` | 368 | PyCaret 25 模型对比 ⭐ |
| `44-StrategyResearch设计文档.md` | 612 | 通用策略自动研究框架 ⭐ |
| **`45-StrategyResearch工具复用设计文档.md`** | 452 | **7 工具复用 + 41 算子** ⭐⭐⭐ |
| `46-v8_ml_design.md` | 410 | v8 ML 因子择时 5 方向 |
| `47-49x_v9` | - | V9 周期诊断全套 |
| `16-TV-PR迭代记录v7_6到v7_9.md` | 249 | TV-PR 全期迭代记录 ⭐ |
| `35-宏观因子体系业界调研.md` | - | 宏观因子调研 |
| `34-CICC动量ETF轮动-缺失模块重写计划.md` | - | CICC 复刻 |
| `36-37-*` | - | - |
| `50-v9_current_cycle_state.md` | - | 当前周期状态 |
| `51-v9_brinson_attribution.md` | - | Brinson 归因 |
| `52-v9_citic_strategies.md` | - | 中信策略 |
| **`53-v9_strategy_factor_analysis.md`** | - | **9 策略核心机制 + 10 因子** ⭐ |
| `54-v1_v9_strategy_summary.md` | 108 | v1-v9 全版本演进 + 10 因子 |
| `55-v10_architecture_design.md` | - | v10 架构设计 |
| `56-v4_improvement_plan.md` | - | v4 改进计划 |
| **`57-v10_final_design.md`** | 288 | **用户确认版 v10 5 层架构** ⭐⭐⭐ |
| `58-v8_vs_v9_fair_comparison.md` | 176 | v8 vs v9 公平对比 |
| `59-v56_pct_change_fix.md` | - | v56 收益率修复 |
| `60-strategy_analysis.md` | - | 策略分析 |
| `62-v8_rebalance_frequency_research.md` | - | v8 调仓频率研究 |
| `63-final_summary.md` | - | - |
| `64-v8_dynamic_position.md` / `64-v8_dynamic_position_plan.md` | - | v8 动态仓位 |
| `65-v9_macro_level_final.md` | - | v9 宏观级别 |
| `66-full_sample_comparison.md` | - | 全样本比较 |
| `67-v8_dynamic_position_master.md` | - | v8 动态仓位主控 |
| `68-standard_comparison.md` | - | 标准比较 |
| `69-v7_10_v9_macro_combination.md` | - | v7.10 + v9 macro 组合 |
| `70-three_strategy_combination.md` | - | 三策略组合 |
| `71-pbear_dynamic_weighting.md` | - | P_bear 动态权重 |
| `72-vol_parity_method_record.md` | - | Vol-parity 方法记录 |
| `73-v10_research_start.md` | - | v10 研究起点 |
| `74-v10_research_plan.md` | - | v10 研究计划 |
| `75-v10_results.md` | - | v10 结果 |
| `76-dynamic_weight_schemes.md` | - | 动态权重方案 |

### 通用文档

| 路径 | 主题 |
|------|------|
| `docs/README.md` | 文档入口 |
| `docs/INDEX.md` | 索引 |
| `AGENTS.md` | 项目总 AGENTS.md |
| `README.md` | 项目总 README.md |
| `research_history/` | **本次复盘目录** |

---

## 六、测试 / 验证脚本（按重要性）

### 单元测试

| 路径 | 行数 | 用例数 |
|------|------|-------|
| `tests/strategy/momentum_etf_rotation/test_slope_r2.py` | - | 20 |
| `tests/strategy/momentum_etf_rotation/test_v1_0_regression.py` | - | 19 |
| `tests/strategy/momentum_etf_rotation/test_fused_signal.py` | - | - |
| `tests/strategy/momentum_etf_rotation/test_trend_filter.py` | - | - |
| `tests/strategy/momentum_etf_rotation/test_vol_targeting.py` | - | - |
| `tests/strategy/momentum_etf_rotation/test_regime_detector.py` | - | - |
| `tests/strategy/momentum_etf_rotation/test_concentration.py` | - | 9 |
| `tests/strategy/momentum_etf_rotation/test_cost_model.py` | - | 10 |
| `tests/strategy/momentum_etf_rotation/test_cov_rp.py` | - | 18 |
| `tests/strategy/momentum_etf_rotation/test_v4.py` | - | **43** ⭐ |
| `tests/strategy/momentum_etf_rotation/test_industry_rotation_v5_1.py` | - | 24 |
| `tests/strategy/momentum_etf_rotation/test_industry_rotation_v6.py` | - | 18 |
| `tests/strategy/momentum_etf_rotation/test_v6_1_v6_2.py` | - | **38** ⭐ |
| `tests/strategy/momentum_etf_rotation/test_reversion_v3.py` | - | 12 |
| `tests/strategy/momentum_etf_rotation/test_industry_rotation_v3.py` | - | 12 |
| `tests/strategy/momentum_etf_rotation/test_sub_weighting_v3.py` | - | 14 |
| `tests/strategy/momentum_etf_rotation/test_multi_strategy_v3.py` | - | 12 |
| `tests/strategy/momentum_etf_rotation/test_v8/` | - | v8 + stop_loss 11 tests |
| `tests/strategy/momentum_etf_rotation/test_v9/` | - | v9 + v7.10 工厂函数 8 tests |

**总测试**：181 + 43 + 38 + 12 = 271+ passed

### 集成测试脚本

| 路径 | 行数 | 用途 |
|------|------|------|
| `scripts/validate_stage16a.py` | 165 | V3 多策略验证 |
| `scripts/chart_stage16a.py` | 197 | V3 图表 |
| `scripts/eval_v6_2_overfitting.py` | - | v6.2 过拟合评估 ⭐ |
| `scripts/test_v6_2_starting_points.py` | - | v6.2 起点测试 ⭐ |
| `scripts/generalization_test_v6_2.py` | - | v6.2 5-fold ⭐ |
| `scripts/v7_10_beta_timing_test.py` | - | β[t-1] vs β[t] |
| `scripts/v7_10_monday_open_test.py` | - | 周一开盘执行 |

---

## 七、Quick Reference

### 生产首选

- **V10 4 策略 Vol-parity**：OOS Sharpe 1.991
  - `QuantNodes/strategy/momentum_etf_rotation/v10/portfolio_layer.py`
- **V1.0 locked**：极致防御
  - `QuantNodes/strategy/momentum_etf_rotation/v2/portfolio_v2.py`
- **V7.10 TV-PR**：激进 alpha
  - `QuantNodes/strategy/momentum_etf_rotation/v7/macro_substrategy_v7_6.py`

### 工具入口

- **回测引擎**：`common/strategy_engine.py`
- **YAML 配置**：`common/config_runner.py`
- **Walk-Forward**：`common/walk_forward.py`
- **IC 工具**：`common/ic_utils.py`
- **协方差**：`common/covariance.py`
- **风险平价**：`common/risk_parity.py`
- **HTML 生成**：`combo/nav_curves_html.py`

### 业绩呈现入口

- **生产 HTML**：`reports/momentum_etf_rotation/combo/STRATEGY_ITERATION_RECORD.html`
- **过程性 HTML**：`STRATEGY_ITERATION_RECORD_v2_2026MMDD.html`（按日）

### 文档查询

- **本目录**：`docs/research_history/`（9 份文档）
- **阶段报告**：`reports/momentum_etf_rotation/STAGE*_PLAN.md`、`v*/*.md`
- **设计文档**：`docs/16-*` ~ `docs/76-*`

---

## 八、目录结构总览

```
QuantNodes/
├── AGENTS.md                                    # 项目总则
├── README.md
├── walk_forward.py                              # Walk-Forward 顶层入口
├── research.py                                  # 研究工具顶层入口
├── QuantNodes/strategy/momentum_etf_rotation/   # ⭐ 主策略代码
│   ├── __init__.py
│   ├── momentum.py
│   ├── portfolio.py
│   ├── backtest.py
│   ├── strategy_versions.py                     # v0.0~v1.0 + v7.10 注册表
│   ├── common/                                  # ⭐ 通用工具
│   ├── v1/ v2/ v3/ v4/ v5/ v5_1/               # 各版本策略
│   ├── v6/ v6_1/ v6_2/
│   ├── v7/                                      # TV-PR + 标准化
│   ├── v8/ v9/ v10/                             # 收敛版本
│   ├── strategies/                              # YAML 模板
│   └── *.py
├── scripts/                                     # 数据 + 评估脚本
│   ├── fetch_real_etf_panel.py
│   ├── fix_ohlcv_adjust.py
│   ├── fetch_proxy_indices.py
│   ├── extract_macro_data.py
│   ├── calc_factor_ic.py
│   ├── factor_timing_diagnostic.py
│   ├── combo/                                   # 多策略组合
│   ├── quant/                                   # 因子库
│   └── research/                                # 量化研究
├── tests/strategy/momentum_etf_rotation/        # 测试
├── reports/momentum_etf_rotation/               # ⭐ 报告 + HTML
│   ├── v1/ v2/ v3/ v4/ v5/ v6_2/ v7/ v8/ v9/ v10/
│   ├── combo/                                   # ⭐ 业绩 HTML
│   ├── common/
│   ├── docs/                                    # 阶段文档
│   ├── STAGE32_PLAN.md
│   ├── STAGE33_PLAN.md
│   ├── STAGE17_22_INDEX.md
│   ├── README.md
│   └── index.html
├── data/                                        # 缓存（部分被 .gitignore）
│   ├── real/
│   └── high_freq_macro/
├── research/                                    # 策略研究子模块
│   └── strategy-research/
└── docs/                                        # ⭐ 本次复盘 + 设计文档
    ├── README.md
    ├── INDEX.md
    ├── research_history/                        # ⭐ 本次复盘 9 份
    ├── 16-TV-PR迭代记录v7_6到v7_9.md          # ⭐
    ├── 34-CICC动量ETF轮动-缺失模块重写计划.md
    ├── 35-宏观因子体系业界调研.md
    ├── 38-v7_3_macro_only.md                   # ⭐
    ├── 39-v7_6_tvpr.md                          # ⭐
    ├── 40-v7_6_sensitivity.md                   # ⭐
    ├── 41-v7_6_factor_ic_and_enhancement.md     # ⭐
    ├── 43-v7_7_lgbm.md                          # ⭐
    ├── 44-StrategyResearch设计文档.md           # ⭐
    ├── 45-StrategyResearch工具复用设计文档.md   # ⭐⭐⭐
    ├── 46-v8_ml_design.md
    ├── 49-v9_cycle_timing.md                    # ⭐
    ├── 53-v9_strategy_factor_analysis.md       # ⭐
    ├── 54-v1_v9_strategy_summary.md
    ├── 55-v10_architecture_design.md
    └── 57-v10_final_design.md                   # ⭐⭐⭐
```

---
