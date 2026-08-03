# 82. 策略迭代路线图 (Phase 0 → 1 → 2)

**作者**: opencode
**日期**: 2026-08-03
**分支**: `refactor/v10-to-v11-migration`
**状态**: Phase 0 进行中

---

## 1. 背景与动机

最近 7 个 commit (`28dd5cb` → `20529a0`) 集中在 CA-GCP 早期预警系统：

- 启发式 v1 + 全评估框架 (walk-forward / bootstrap / ROC / 校准)
- ML v2–v5（5 类标签 + autoencoder + regime routing）
- 64 个 unit test（heuristic + ML helpers）

但 **所有数字都没有可信的对照基准**：HTML 策略排行榜的虚高 Sharpe 还未审计复核过。

`docs/77-v0_v10_codebase_audit.md` 已发现 9 个审计问题，**最关键的两条**：

1. `combo/build_all_strategy_navs.py` 把 v11 周频数据用 `bdate_range + ffill` 扩展成日频，又用 `first_valid.cumprod` 稀释 vol 分母 → 2,986 obs vs 实际 2,058 trading days
2. `ca_gcp_ew_full_eval.py:bootstrap_metrics` 4 次写同一 CSV → 最终 confidence-signal 0/0 覆盖前置结果

CA-GCP v4 y_E XGB F1=0.546 是 **recall 撑起来的玩具**（AUC ≈ 0.49 接近随机），且 5/10 事件 recall 里有 3 个是 **bullish reversal**（模型不该报警才对）。

**项目经验教训**（来自历史文档）：

- 独立 sleeve + 动态权重 > 多层叠加（v11 反而不如 v10）
- 简单 trend/hysteresis > HMM/ADX/复杂融合
- 动态仓位贡献 v9 71% alpha（Brinson attribution）
- DualMom 与 v10 其他 sleeve 几乎不相关（-0.005），贡献最大

---

## 2. 路线图总览

```
Phase 0 (1-2 周) ── 规范评估管线重建 [强制前置]
       ↓
Phase 1 (2-3 周) ── CA-GCP v6 utility-aligned [核心研究]
       ↓
Phase 2 (2-3 周) ── v10.3 约束动态分配器 [生产候选]
```

**不接受跳过 Phase 0**。Phase 1 与 Phase 2 顺序：**先 CA-GCP v6 → 后 v10.3**（v10.3c 依赖 v6 风险预算）。

---

## 3. Phase 0：规范评估管线重建

### 3.1 目标

所有策略 NAV 在真实交易日的交集上计算。HTML 数字与审计数字一致。

### 3.2 Bug 根因

`combo/build_all_strategy_navs.py:188-197`：

- `first_valid = df.apply(lambda c: c.first_valid_index()).max()` → 取交集起点（OK）
- 但 `load_v11_daily()` 用 `bdate_range(start=s.index[0], end=s.index[-1])` + `ffill` → v11 起始日 2014（周频数据），与 2018-01-02 起的策略混合后 → 2014-2018 期间 v11 NAV 平 → 0 收益日污染 vol 分母

### 3.3 执行动作

| # | 任务 | 验证 |
|---|------|------|
| 0.1 | 审计 `build_all_strategy_navs.py` 找到所有 "NDA日 → 0 收益" 污染源 | 单元测试：构造 synthetic 污染源，验证修复后 vol 不变 |
| 0.2 | 重写为 trading-day-only 严格交集（`.reindex(bdate_range, method=None)` + `dropna()`） | 修复前后 vol 差 ≤ 1% |
| 0.3 | 统一 `OOS_START` 文本（HTML 写 "2022-2026"，代码用 "2021-08-01"） | HTML 显示 = 代码实际 |
| 0.4 | 修复 `ca_gcp_ew_full_eval.py:bootstrap_metrics` → 输出 `bootstrap_ci_{model}.csv` | 4 文件均存在 |
| 0.5 | 重建 `all_strategy_navs.parquet` + `all_strategy_metrics.parquet` + HTML | 修复后 v1.0 / v10-DynD / v10 Vol-parity 排名与 `docs/77` 审计一致（±0.05） |
| 0.6 | 加 `tests/combo/test_canonical_eval.py` 守门 | 5 unit tests |

### 3.4 接受标准

- [ ] NAV parquet shape ≈ 2,058 trading days（不是 2,986）
- [ ] v1.0 audited Sharpe ≈ 1.46（不是 1.78）
- [ ] v10-DynD audited Sharpe ≈ 1.46（不是 1.65）
- [ ] v10 Vol-parity audited Sharpe ≈ 1.43（不是 1.62）
- [ ] 4 个 bootstrap CSV 均独立存在

### 3.5 提交序列

```
[1] fix(eval): canonical trading-day-only NAV rebuild + per-model bootstrap
[2] test(combo): canonical eval alignment regression
```

---

## 4. Phase 1：CA-GCP v6 utility-aligned

### 4.1 目标

把 v4 y_E 从 "recall 撑起来的 F1 玩具" 改成 "**真实 portfolio loss utility 输入**"。

### 4.2 核心问题

- v4 best model AUC ≈ 0.49（接近随机），F1=0.546 主要靠高 recall
- 5/10 事件 recall，但 3 个 "miss" 其实是 **bullish reversal**（模型不该报警才对）
- 当前 equal-weight overlay 不改善 Sharpe/Calmar
- `ca_gcp_ew_full_eval.py:bootstrap_metrics` 4 次写同一 CSV
- v5 transition-regime 路由输出 R=2.000（无效）
- 缺 utility / PR-AUC / 校准误差评估

### 4.3 两步法设计

#### Stage A — Downside probability（替换 y_E）

- 标签：未来 `max_dd(t → t+h)` 跨阈 + 恢复时长 > 阈值
- 模型：保留 v4 XGB + LightGBM ensemble
- 输出：`p_downside` (calibrated) + `recovery_days`

#### Stage B — Action calibration

- 风险预算：`risk_budget = clip(α · p_downside + β · recovery_factor, 0, 0.20)`
- 目标：最大化 `U = E[R] - λ · cvar_loss`, λ 校准
- 阈值：基于 portfolio utility 非 F1

#### 评估指标切换

- 不再只看 F1 → 改 PR-AUC + ECE + net portfolio utility
- 事件 recall **只**对 `is_downside=True` 事件计算
- 经济价值：v10-DynD + CA-GCP v6 risk budget vs v10-DynD 单独

### 4.4 执行动作

| # | 任务 | 验证 |
|---|------|------|
| 1.1 | `ca_gcp_ew_eval.py` 加 `is_downside` 字段到 `KNOWN_EVENTS` | 5 unit tests |
| 1.2 | 修复 v5 routing bug（输出 R > 1 现象） | 1 unit test + 数值验证 |
| 1.3 | 验证 `y_E`/`y_A`/`y_B`/`y_C`/`y_D` 标签无 future-leakage | 单元测试：模拟前看检测 |
| 1.4 | 写 `ca_gcp_training_v6.py`：2-stage（downside prob + action calibration） | 入口 smoke test |
| 1.5 | 加 `expect_calibrated_probability` / `expect_recovery_days` validation | 5 unit tests |
| 1.6 | `ca_gcp_ew_full_eval.py` 加 `pr_auc` / `ece` / `net_utility` 指标 | 3 unit tests |
| 1.7 | 跑 `ca_gcp_training_v6.py` 出 4 个模型 + 4 个 bootstrap CSV | 4 CSV 均存在 |
| 1.8 | 写 `scripts/ca_gcp_v6/overlay_v10dynd.py`（v10-DynD + risk budget 0-20%） | NAV + metrics |
| 1.9 | nested walk-forward：purge=5d, embargo=5d, train=4y, test=1y | 报告 |
| 1.10 | 写 `docs/83-ca_gcp_v6_utility_aligned.md` | 完整实验报告 |
| 1.11 | 更新 `tests/ca_gcp_ew/test_*.py` 覆盖新字段 | 5 新 unit tests |

### 4.5 接受标准

- [ ] AUC > 0.55（摆脱随机）
- [ ] PR-AUC > F1 单点（稳定的 ranking）
- [ ] ECE < 0.10（calibrated）
- [ ] `is_downside` 事件 recall ≥ 0.7（仅 7 个 true downside）
- [ ] v10-DynD + v6 risk budget OOS Sharpe ≥ v10-DynD + 0.03
- [ ] v10-DynD + v6 Calmar ≥ v10-DynD
- [ ] 4 个 bootstrap CSV 均独立存在
- [ ] overlay 期间 turnover ≤ v10-DynD × 1.5

### 4.6 提交序列

```
[3] fix(ca_gcp_ew): event taxonomy (is_downside) + bootstrap per-model
[4] fix(ca_gcp_ew): v5 routing bug (invalid recall)
[5] feat(ca_gcp_ew): v6 utility-aligned 2-stage model
[6] test(ca_gcp_ew): v6 unit tests
[7] docs(ca_gcp_ew): v6 utility-aligned experiment report
```

---

## 5. Phase 2：v10.3 约束动态分配器

### 5.1 目标

在 v10-DynD 基础上加约束，做 v10.3 — **不动 v10.3 名字前先有 3 套对比实验**。

### 5.2 3 个候选方案

| 方案 | 约束 | 思路 |
|------|------|------|
| **v10.3a** Conservative | min(v1.0)=25%, max turnover=30%/rebalance, hysteresis 5% | 防御底仓 + 换手控制 |
| **v10.3b** Trend-Adaptive | 在 3a 基础上 + 5日 trend 同 v10.2 TrendFilter | 顺势调节 offensive sleeve |
| **v10.3c** CA-GCP-Bounded | 在 3a 基础上 + CA-GCP v6 y_E 概率影响总风险预算 0-20% | 风险预算输入而非信号 |

### 5.3 共享规则

- 4 sleeve：v1.0 / v7.10 / v9macro / DualMom（沿用 v10）
- 权重导出：trailing 60d Sharpe + 60d momentum（沿用 v10.D 方案）
- 调仓频率：月频（25d 触发）
- 成本：5bp / side
- 验证：nested walk-forward (train=4y, test=1y, step=1y)

### 5.4 执行动作

| # | 任务 | 验证 |
|---|------|------|
| 2.1 | 在 `dynamic_weight_schemes.py` 加 `scheme_F_conservative` / `scheme_G_trend` / `scheme_H_cagcp` | 文档 + unit tests |
| 2.2 | 写 `scripts/v10_3/v10_3_backtest.py` 跑 3 方案 OOS（2021-08-01 ~ 2026-05-06） | 3 个 NAV parquet + 1 个 metrics parquet |
| 2.3 | 写 `scripts/v10_3/v10_3_walk_forward.py` nested walk-forward | 3 方案 × 5 折 = 15 段 |
| 2.4 | 写 `scripts/v10_3/v10_3_costs_turnover.py` 报告 turnover / max DD / 恢复时间 | 含 stress：2022 / 2024 反转 / 2025 |
| 2.5 | vs v10-DynD 现 OOS（审计后数字）做 bootstrap mean Sharpe diff | p-value 报告 |
| 2.6 | 加 `tests/strategy/momentum_etf_rotation/test_v10_3_*.py` smoke tests | 3 schemes × 5 tests |
| 2.7 | 写 `docs/84-v10_3_constrained_allocator.md` | 完整实验报告 |

### 5.5 接受标准

- [ ] audited OOS Sharpe ≥ v10-DynD + 0.05
- [ ] audited OOS Calmar ≥ v10-DynD + 0.05
- [ ] Max DD ≥ v10-DynD（不恶化）
- [ ] 与 v1.0 / v7.10 / DualMom 相关性比 v10-DynD 更低
- [ ] Turnover 满足上限
- [ ] 2022 / 2024 反转 / 2025 段均不崩
- [ ] bootstrap p-value < 0.1

### 5.6 提交序列

```
[8]  feat(v10.3): constrained dynamic allocator (schemes F/G/H)
[9]  test(v10.3): smoke tests for 3 schemes
[10] docs(v10.3): constrained allocator experiment report
```

---

## 6. 风险与决策点

| 风险 | 缓解 |
|------|------|
| 0.6 修复后 v1 vs v10-DynD 排名可能反转 | 接受现状，重写 doc |
| v10.3 3 方案都跑不出 v10-DynD | 接受，转去更低投入方向 |
| 滚动 trailing Sharpe 引入 look-ahead | 严格 1d lag + purged WF |
| 单元测试与 walk-forward 一致性 | Phase 0.7 + Phase 2.6 测试守门 |
| CA-GCP v6 仍 AUC < 0.55 | 接受 v6 为辅助，加 v10.3a/3b 后退路 |

---

## 7. 不做的事（明确边界）

- ❌ 不引入新因子（OHLCV 特征推迟到 Phase 2+）
- ❌ 不做 CFFEX sleeve（Phase 2 之后）
- ❌ 不做 agent factor 转换（不在本次路线）
- ❌ 不动 v11（v11 单独审计 review）
- ❌ 不改 HTML 颜色/排版（只改数字）

---

## 8. 完整提交序列

```
── Phase 0 ──
[1] fix(eval): canonical trading-day-only NAV rebuild + per-model bootstrap
[2] test(combo): canonical eval alignment regression

── Phase 1 ──
[3] fix(ca_gcp_ew): event taxonomy (is_downside) + bootstrap per-model
[4] fix(ca_gcp_ew): v5 routing bug (invalid recall)
[5] feat(ca_gcp_ew): v6 utility-aligned 2-stage model
[6] test(ca_gcp_ew): v6 unit tests
[7] docs(ca_gcp_ew): v6 utility-aligned experiment report

── Phase 2 ──
[8]  feat(v10.3): constrained dynamic allocator (schemes F/G/H)
[9]  test(v10.3): smoke tests for 3 schemes
[10] docs(v10.3): constrained allocator experiment report
```

每个 commit 独立 pytest + ruff 可验。

---

## 9. 进度跟踪

| Phase | # | 任务 | 状态 |
|-------|---|------|------|
| 0.0 | 写本文档 | DONE |
| 0.1 | 审计污染源 | TODO |
| 0.2 | 交易日期交集 | TODO |
| 0.3 | OOS_START 统一 | TODO |
| 0.4 | bootstrap 分离 | TODO |
| 0.5 | 重建 NAV + HTML | TODO |
| 0.6 | 守门测试 | TODO |
| 0.7 | Phase 0 commit | TODO |
| 1.1 | is_downside 字段 | TODO |
| 1.2 | v5 routing 修复 | TODO |
| 1.3 | label leakage 验证 | TODO |
| 1.4 | v6 2-stage 模型 | TODO |
| 1.5 | pr_auc/ECE/net_utility | TODO |
| 1.6 | 4 bootstrap CSV | TODO |
| 1.7 | v10-DynD overlay | TODO |
| 1.8 | nested walk-forward | TODO |
| 1.9 | docs/83 | TODO |
| 1.10 | tests | TODO |
| 1.11 | Phase 1 commit | TODO |
| 2.1 | schemes F/G/H | TODO |
| 2.2 | 3 方案 OOS | TODO |
| 2.3 | nested WF | TODO |
| 2.4 | cost/turnover 报告 | TODO |
| 2.5 | bootstrap p-value | TODO |
| 2.6 | smoke tests | TODO |
| 2.7 | docs/84 | TODO |
| 2.8 | Phase 2 commit | TODO |
