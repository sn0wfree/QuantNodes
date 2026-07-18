# Stage 32 — v7.10 硬化 P0 修补 + v6.2 DEPRECATED

> **日期**: 2026-07-17
> **决策**: 用户确认 Stage 32 主线为 v7.10 硬化 (P0 修补)
> **分支**: `stage/32-v710-hardening`
> **状态**: 🚧 进行中

---

## 1. 背景

v7.10 TV-PR (标准化+CV) 在 Stage 31 达到 OOS Calmar 2.144 / Sharpe 1.60, 但存在 4 个 P0 缺口阻塞生产化:

1. **止损桩**: `v7_6_with_stop_loss()` (L441) 仅为 stub, 未实现止损逻辑
2. **起点依赖未测**: v7.10 没有像 v6.2 一样跑 3 起点 CV% 验证
3. **无工厂入口**: `strategy_versions.py` 只有 v0.x/v1.0, v7.10 无法通过 `get_version("7.10")` 调用
4. **数据生成非自动化**: `generate_v7_10_data()` 需手动调用

同时, v6.2 ir_expanding 因 CV% 56.9% FAIL 正式降级 DEPRECATED。

### 1.1 用户决策

| 决策项 | 结论 |
|--------|------|
| Stage 32 主线 | A. v7.10 硬化 (P0 修补) |
| v6.2 命运 | 降级为 DEPRECATED |
| v7.10 vs v1.0 定位 | v1.0 主力实盘 + v7.10 研究参考 (两路并行) |
| v7.7 ML 路线 | 暂缓, 优先硬化 v7.10 |

---

## 2. 工作分解 (5 个 P0 任务)

### 2.1 Task 1: v7_6_with_stop_loss 实现 ⭐

- **目标**: 把 `macro_substrategy_v7_6.py:441` 的 stop_loss stub 替换为真实逻辑
- **复用**: v7.5 stop_loss 模式 (`v7/__init__.py:221`, 硬止损 -10% → 100% bonds)
- **参数**: `stop_loss_threshold=-0.10`, cooldown 5 周
- **集成**: 在 `construct_portfolio()` 输出 top-10 之后, 加 stop_loss check, 触发则全仓现金
- **滑点保护**: 止损日 cooldown 5 周, 避免震荡市来回止损
- **配置**: `V7_6Config.stop_loss_threshold` 默认 `None` (关闭), v7.10 默认 `-0.10`
- **验收**: ≥ 8 tests, OOS Calmar ≥ 0.597, DD 改善

### 2.2 Task 2: v7.10 起点依赖 CV% 测试

- **目标**: 验证 v7.10 是否有起点依赖 (v6.2 CV% 56.9% FAIL 的教训)
- **方法**: 3 起点 (2018-01 / 2020-01 / 2022-01), 6 年训练 + 3 年 OOS
- **指标**: CV% = std(OOS_Calmar) / mean(OOS_Calmar)
- **判定**: CV% < 25% → PASS; 25-50% → PROMISING; > 50% → DEPRECATED
- **输出**: `reports/momentum_etf_rotation/v7_10_cv_test.md`
- **验收**: 报告 + CV% 结论

### 2.3 Task 3: strategy_versions.py 接入 v7.10

- **目标**: `get_version("7.10")` 可用
- **实现**: 在 `strategy_versions.py` 添加 `v7_10_std_newλ()` 工厂函数
- **验收**: ≥ 5 tests, `get_version("7.10")` 返回 nav Series

### 2.4 Task 4: v7.10 数据生成自动化

- **目标**: `generate_v7_10_data()` 从手动调用变 CLI 一键
- **实现**: 新增 `quantnodes data gen-v7-10` 子命令 + 自动检测 + 生成
- **验收**: 4 个文件自动存在, 复现性验证

### 2.5 Task 5: v6.2 DEPRECATED

- **目标**: 标记 v6.2 为 DEPRECATED, 移除主力组合
- **实现**: `DeprecationWarning` + 文档标记 + combo 移除
- **验收**: deprecation warning 测试 + HTML 重生成

---

## 3. 验收门槛 (Stage 32 Gate)

| # | 验收项 | 通过标准 |
|---|--------|---------|
| 1 | stop_loss 实现 | ≥ 8 tests, Calmar ≥ 0.597 |
| 2 | 起点依赖 CV% | CV% < 50% |
| 3 | get_version("7.10") | 可用, ≥ 5 tests |
| 4 | 数据 pipeline | 一键化, 复现性 OK |
| 5 | v6.2 DEPRECATED | warning + combo 移除 |
| 6 | 全量测试 | ≥ 5800 tests pass |

不通过处理:
- CV% > 50% → v7.10 降 DEPRECATED, 触发 Stage 33 紧急 ML 路线

---

## 4. 代码改动清单

### 4.1 改动 (N 个生产文件)
- `QuantNodes/strategy/momentum_etf_rotation/v7/macro_substrategy_v7_6.py` — stop_loss 实现
- `QuantNodes/strategy/momentum_etf_rotation/strategy_versions.py` — v7.10 工厂函数
- `QuantNodes/strategy/momentum_etf_rotation/v7/data_loader_v7_6.py` — 自动化 pipeline
- `QuantNodes/strategy/momentum_etf_rotation/v6_2/` — deprecation warning
- `combo/nav_curves_html.py` — v6.2 DEPRECATED 标记
- `reports/momentum_etf_rotation/combo/STRATEGY_ITERATION_RECORD.html` — 重生成

### 4.2 新增 (N 个文件)
- `reports/momentum_etf_rotation/STAGE32_PLAN.md` — 本文档
- `reports/momentum_etf_rotation/v7_10_cv_test.md` — 起点依赖测试报告
- `scripts/v7_10_cv_test.py` — 起点依赖测试脚本
- `tests/strategy/momentum_etf_rotation/test_v7_6_stop_loss.py` — 止损测试
- `tests/strategy/momentum_etf_rotation/test_strategy_versions_v7_10.py` — 工厂函数测试

### 4.3 改动 (N 个测试)
- 新增 ≥ 15 tests (stop_loss 8 + v7.10 5 + deprecation 2)

---

## 5. 未来方向 (Stage 33+)

- B. v7.7 ML 替代 TV-PR (PyCaret 25 模型, 若 CV% FAIL 触发紧急启动)
- C. v8 宏观子策略 (docs/35 路径三)
- D. combo 重建 (v6.2 移除后主力重选)
- E. 文档整合 (Stage 12-31 补全)

---

**最后更新**: 2026-07-17
**状态**: 🚧 5 个 P0 任务进行中
