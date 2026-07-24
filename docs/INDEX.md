# QuantNodes 文档索引

> 自动维护的文档总目录, 按主题分组

---

## 🏆 最终推荐 (2026-07-24)

| # | 风险偏好 | 推荐 | Sharpe (OOS) | Sortino | AnnRet | MaxDD |
|---|---------|------|-------------|---------|--------|-------|
| **1** ⭐ | **综合最优** | ⭐ **E 混合动态权重** | **2.932** | **3.802** | **17.01%** | -6.11% |
| 2 | 极保守 | C 回撤控制 | 2.616 | 3.182 | 10.02% | -3.33% |
| 3 | 静态基准 | 4策略 Vol-parity | 2.461 | 3.047 | 9.51% | -4.42% |

> **E 混合**: 市场状态切换 + 回撤控制 + 波动率缩放
> **C 回撤控制**: MaxDD -3.33%, Calmar 3.013 (最保守)

---

## 📁 v8-v9 动态仓位方案 (已完成)

| 文档 | 内容 |
|------|------|
| **docs/72-vol_parity_method_record.md** ⭐ | Vol-parity 组合方法完整记录 |
| **docs/71-pbear_dynamic_weighting.md** | P_bear/LEVEL/FLOW 动态加权失败 |
| **docs/70-three_strategy_combination.md** | 3 策略组合 Vol-parity 成功 |
| **docs/69-v7_10_v9_macro_combination.md** | 4 整合实验 + v7.10 最优 |
| **docs/68-standard_comparison.md** | 25 策略标准化对比 (9×9) |
| **docs/67-v8_dynamic_position_master.md** | 综合主文档 (5 Phase 历程) |
| [docs/66-full_sample_comparison.md](66-full_sample_comparison.md) | 21 策略综合对比 (legacy) |
| [docs/65-v9_macro_level_final.md](65-v9_macro_level_final.md) | Phase D 突破 (Sharpe 1.014) |
| [docs/64-v8_dynamic_position.md](64-v8_dynamic_position.md) | Phase A/C 失败报告 |
| [docs/64-v8_dynamic_position_plan.md](64-v8_dynamic_position_plan.md) | 三阶段计划 |
| [docs/63-final_summary.md](63-final_summary.md) | 综合汇总 |

---

## 📁 v10 研发起点

| 文档 | 内容 |
|------|------|
| **docs/76-dynamic_weight_schemes.md** ⭐ | 动态权重分配: E混合方案 Sharpe 2.932 |
| **docs/75-v10_results.md** | 4策略 Vol-parity: Sharpe 1.991 |
| **docs/74-v10_research_plan.md** | v10 研发计划: 3 个独立策略 (DualMom/EPO/RRG) |
| **docs/73-v10_research_start.md** | v10 研发起点 + 已知约束 + 方向 + 代码模板 |

---

## 🛠️ 核心脚本

| 脚本 | 用途 | 输出 |
|------|------|------|
| **`scripts/combo/combine_e_3strategies.py`** | ⭐ 3 策略 Vol-parity 组合 | 1 CSV + 4 NAV |
| **`scripts/combo/combine_e_pbear_dynamic.py`** | 动态加权 108 组合 | 1 CSV + 1 NAV |
| **`scripts/combo/standard_comparison.py`** | 25 策略 × 9 区间标准对比 | 5 CSV |
| **`scripts/combo/standard_visualization.py`** | 3 张可视化 | 3 PNG |
| **`scripts/combo/regenerate_v7_10_4costs.py`** | v7.10 4 档成本 | 4 NAV + 1 CSV |
| `scripts/combo/combine_a_plus_v7_10_v9_macro.py` | A+ 方案 (失败) | 1 CSV |
| `scripts/combo/combine_b1_v7_14_v9_in_x.py` | B1 方案 (失败) | 1 CSV |
| `scripts/combo/combine_d_3source_avg.py` | D 方案 (失败) | 1 CSV |
| `scripts/combo/combine_a_plus_plus_pbear_only.py` | A++ 方案 (失败) | 1 CSV |

---

## 📊 数据输出

`reports/momentum_etf_rotation/combo/`:

### 核心产出
| 文件 | 描述 |
|------|------|
| `combine_e_3strategies_grid.csv` | 3 策略组合 11 权重组 |
| `combine_e_3strat_v710_重_0.{60,70,80,90}_C5.parquet` | 4 个权重 NAV |
| `combine_e_pbear_dynamic_grid.csv` | 动态加权 108 组合 |
| `standard_comparison_wide.csv` | 25 策略 × 91 列 |
| `v7_10_v56_{5,10,15,20}bp.parquet` | v7.10 4 成本档 |
| `v7_10_v56_4costs_comparison.csv` | v7.10 4 档对比 |

### 可视化
| 文件 | 描述 |
|------|------|
| `figs/standard_sharpe_heatmap.png` | 25×9 Sharpe 热图 |
| `figs/nav_curves_4strats.png` | 4 策略 NAV 曲线 |
| `figs/scenario_sharpe_bars.png` | 7 场景柱状 |

---

## 📅 项目时间线

| 日期 | 里程碑 |
|------|--------|
| 2026-07-24 09:20 | Phase B PoC 启动 |
| 2026-07-24 09:30 | Phase A 整合 (5 ETF, 失败) |
| 2026-07-24 09:35 | Phase C 参数网格 (仍失败) |
| 2026-07-24 09:38 | Phase D 加 v9 macro (Sharpe 1.014) |
| 2026-07-24 09:44 | Phase E 4 成本 + 21 策略综合 |
| 2026-07-24 10:00 | Phase F 3 策略 Vol-parity (Sharpe 1.535) ⭐ |
| 2026-07-24 10:30 | Phase G 动态加权失败 (Sharpe < 1.41) |
| 2026-07-24 11:00 | 文档 63-72 完成, 准备 v10 |

---

**最后更新**: 2026-07-24 11:00
