# 华泰时序 CTA 稳健性验证计划 (Phase 10)

## 一、背景

Phase 6-9 已完成华泰时序 CTA 策略的核心实现与优化：
- Phase 6: 因子/状态/退出/策略/消融五层架构，86 测试通过
- Phase 7: 宏观闸门 (DXY+US10Y+VIX+CRB)、DrawdownCircuitBreaker、低波品种杠杆上限
- Phase 8: confirm_window 敏感性、扩展至 31 品种、聚焦 9 品种组合 (Calmar=1.572)
- Phase 9: 组合总杠杆上限 (portfolio_max_leverage)

**最近发现**：样本内/外拆分测试显示聚焦 9 品种组合在 2025-2026 样本外 Calmar=-0.057，提示**品种选择存在过拟合风险**。

为系统化验证策略鲁棒性，开展 Phase 10：四联稳健性测试。

## 二、目标

| 测试 | 目的 | 预期产出 |
|------|------|----------|
| **T1 WF 三目标** | 验证目标函数选择对参数稳定性的影响 | 3 份 WF CSV + 汇总 |
| **T2 宏观双向门控** | 验证多/空门控对趋势决策的对称增益 | 2 份新模块 + 单元测试 |
| **T3 宏观 7 模式** | 验证新增维度 (CS+COR) 是否带来正向增益 | 1 份对比表 |
| **T4 成本 5 场景** | 验证策略在成本扰动下的稳定性 | 1 份敏感性表 |
| **T5 综合报告** | 汇总四测结论，给出生产配置建议 | `docs/robustness_validation.md` |

## 三、详细任务

### T1: Walk-Forward 三目标对比 (9 聚焦品种)

**范围**：sn / AU / CU / ZN / AG / sf / AL / eb / NI

**修改**：`QuantNodes/strategy/cta/optimization/walk_forward.py`
- `grid_search()` 增加 `objective: str` 参数 (calmar / sharpe / max_dd)
- max_dd 目标取负值 (越小越好)

**WF 配置**：
- 窗口：2 年训练 + 1 年验证
- 时间：2020-01 ~ 2026-09
- 参数网格：46,656 组合，使用 `n_samples=300` 采样

**输出**：
- `reports/cta_ts_mom/walkforward_objective_{calmar,sharpe,maxdd}.csv`
- `reports/cta_ts_mom/wf3obj_summary.csv`

### T2: 宏观过滤器扩展

**T2a 短空门控扩展** (`macro_gate.py`)
- 新增 `use_credit_spread` / `use_copper_oil_ratio` 开关
- 新增 `cn_us_spread` / `gold_oil_corr` 两个 parquet 分支
- 顺风判断：
  - cn_us_spread 120日动量 > 0 → 禁空
  - gold_oil_corr 120日变化 < -0.3 → 允许空

**T2b 独立多头门控模块** (`macro_long_gate.py`)
- 新建 `regime/macro_long_gate.py`
- 复用 6 个宏观维度的反向逻辑
- `HuataiTimingConfig` 新增 `long_gate_enabled` / `long_gate_mode`
- `_entry_signals()` trend_up 分支接 long_gate

**单元测试**：
- `tests/strategy/cta/test_macro_gate.py` (新增)
- `tests/strategy/cta/test_macro_long_gate.py` (新增)

### T3: 宏观过滤器回测 (7 模式)

| # | short_gate | long_gate | 备注 |
|---|-----------|-----------|------|
| M0 | off | off | 纯基线 |
| M1 | original (4D) | off | 当前默认 |
| M2 | +CS | off | 加信用利差 |
| M3 | +COR | off | 加铜油比 |
| M4 | +CS+COR | off | 6 维短空 |
| M5 | original | external | 双向门控 |
| M6 | +CS+COR | external | **终极配置** |

**输出**：`reports/cta_ts_mom/macro_filter_comparison.csv`

### T4: 成本敏感性 5 场景

| 场景 | commission | slippage | 倍数 |
|------|-----------|----------|------|
| S0 | 0.0005 | 0.0005 | 0.5x |
| S1 | 0.00075 | 0.00075 | 0.75x |
| S2 | 0.001 | 0.001 | 1.0x |
| S3 | 0.0015 | 0.0015 | 1.5x |
| S4 | 0.002 | 0.002 | 2.0x |

**输出**：`reports/cta_ts_mom/cost_sensitivity_5scenarios.csv`

**判断标准**：CAGR 从 S0 → S4 衰减 < 30% 视为成本鲁棒。

### T5: 综合报告

**输出**：`docs/robustness_validation.md`

**章节**：
1. 摘要 (结论矩阵 + 鲁棒性评分)
2. Walk-Forward 三目标对比
3. 宏观过滤器对比 (双向门控)
4. 成本敏感性
5. 综合结论与生产建议

## 四、Acceptance Criteria

| 项目 | 通过标准 |
|------|----------|
| T1 全部完成 | 3 份 CSV + 汇总表，无 OOM/超时 |
| T2 a/b 测试通过 | `pytest test_macro_gate.py test_macro_long_gate.py` 全绿 |
| T3 全部 7 模式 | `macro_filter_comparison.csv` 7 行 |
| T4 全部 5 场景 | `cost_sensitivity_5scenarios.csv` 5 行 |
| T5 报告完整 | `docs/robustness_validation.md` 5 章节齐全 |
| ruff 干净 | `ruff check QuantNodes/strategy/cta/` 无警告 |

## 五、时间预算

| 任务 | 预计耗时 |
|------|----------|
| T1 WF 三目标 | 25 min |
| T2 a/b 宏观扩展 | 16 min |
| T3 宏观回测 | 10 min |
| T4 成本 5 场景 | 10 min |
| T5 综合报告 | 5 min |
| **合计** | **~66 min** |

## 六、风险点

| 风险 | 检测 | 回退 |
|------|------|------|
| WF 网格爆炸 | n_samples 限制 300 | 用前 1 品种试跑确认 < 3min |
| CS/COR 数据 < 5 年 | `len(s) > 120` 检查 | 不足时降级跳过 |
| macro_long 触发过频 | M5/M6 持仓覆盖率 < 50% | 调高阈值或禁用 |
| 成本 2x 业绩崩塌 | S4 Calmar < 1.0 | 提示换手过高 |
| WF 不传宏观参数 | OOS Sharpe < IS 30% | 同步修复入口 |