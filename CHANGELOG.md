# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`GroupAnalyzerNode` 支持 bool / 离散 / 轻度 ties 因子**
  (`QuantNodes/research/factor_test/nodes/group_analyzer_node.py:55`)
  - 原 `_calc_group_return` 在 n_unique < n_groups 时（如 `pl.when(cond)
    .then(-1).otherwise(+1)` 产出 30×-1 + 20×+1，或 3 unique 的离散因子）
    调用 `pd.qcut(..., labels=range(1, group+1), duplicates='drop')` 抛
    `ValueError: Bin labels must be one fewer than the number of bin edges`。
  - 改为策略模式：`_classify_factor` 按 dtype + n_unique 判别 → 2 个纯
    函数 handler：
    - `_group_ranked` — 连续或轻度 ties 因子（含 alpha-004 场景：
      7 unique × 50 行有大量 ties 但 n_unique >= group），统一用
      `pd.qcut(series.rank(method='first'), ...)` 破 tie。对无 ties
      纯连续因子与原 `pd.qcut(series, ...)` 行为 bitwise 等价
      （rank 单调变换保序），零回归。
    - `_group_discrete` — bool/离散因子 (n_unique <= 2 或 bool/integer
      dtype 且 n_unique <= 10)，按 value 比例分配组段 + 内部 seeded
      shuffle（`seed=yyyymmdd % 2**31`）保证可复现。
  - `_classify_factor` 返回种类从 3 (`continuous`/`low_tie`/`discrete`)
    合并为 2 (`ranked`/`discrete`)，dispatch 简化为 2 分支。
  - 原 `_group_continuous` (修复 ties 抛错) + `_group_low_tie` (修复
    ties 抛错) 合并为单个 `_group_ranked`。
  - `_calc_group_return` 改写为 dispatch 调度，循环内只调一次
    `factor_data.loc[t_i].dropna()`。
  - 新增 `tests/test_group_analyzer_bool.py` 覆盖 2 分支 + dispatch，
    含 alpha-004 真实 ties 场景。

### Changed
- **`FactorNeutralizeNode` 改 Chain of Responsibility (Phase 2.1)**
  (`QuantNodes/research/factor_test/nodes/factor_neutralize_node.py:65`)
  - 原 `_neutralize` 72 行 3 个 if/elif 分支 (industry only / risk only
    / both) 几乎相同 (90% 重复)，仅 X 设计矩阵组装不同。
  - 抽出 `nodes/neutralizers.py`：
    - `Neutralizer` (ABC) / `IndustryNeutralizer` / `RiskNeutralizer`
    - `build_neutralizer_chain(if_industry, if_risk, industry, risk_data)`
      → `List[Neutralizer]`，自动过滤 `is_active() == False` 的环节
    - `apply_neutralizer_chain(factor_i, chain)` 统一"按日期循环 +
      merge + OLS + 写残差"流程
  - `_neutralize` 退化为 4 行：构造 chain + 委托
  - 新增中性化类型 (如 `StyleNeutralizer`) 只需新增一个 `Neutralizer`
    子类，`_execute` 无需修改
  - 顺带修复 2 个 latent bugs (原代码从未被测试覆盖):
    1. `pd.get_dummies` 产出 bool dtype，`sm.add_constant` 报
       "numpy boolean subtract" 错误。chain 实现显式 `.astype(float)`
    2. 原 branch 3 (risk only) `pd.concat` 组装 X 时未转置，产生
       `(n_risks, n_stocks)` 而非预期的 `(n_stocks, n_risks)`，
       导致后续 `merge(left_index=True, right_index=True)` 错位。
       chain 实现统一 X 形状 (index=股票代码, columns=factors)
  - 新增 `tests/research/factor_test/nodes/test_neutralizer_chain.py`
    覆盖 5 类 (ABC / Industry / Risk / Chain build / Chain apply /
    Backward compat / E2E)，37 个测试

- **`FactorPreprocessNode` 改 Strategy pattern (Phase 2.2)**
  (`QuantNodes/research/factor_test/nodes/factor_preprocess_node.py:85`)
  - 原 `_preprocess_vectorized` 102 行硬编码 3 类 if 链 (missing fill /
    de-extreme / normalise)，每类多个 method 分支。
  - 抽出 `nodes/preprocess_strategies.py`：
    - `MissingFillStrategy` (ABC) / `PassThroughMissing` / `IndustryAverageMissing`
    - `DeExtremeStrategy` (ABC) / `PassThroughExtreme` /
      `MedianAbsoluteDeviationExtreme` / `PercentileShrinkExtreme`
    - `NormStrategy` (ABC) / `PassThroughNorm` / `ZScoreNorm` / `RankToNormalNorm`
    - 工厂函数 `build_missing_strategy / build_extreme_strategy /
      build_norm_strategy / build_preprocess_strategies`
  - `_preprocess_vectorized` 退化为 3 行 dispatch:
    `result = missing_s.apply(result, industry=industry)`
    `result = extreme_s.apply(result, ...)`
    `result = norm_s.apply(result)`
  - 新增策略类型 (如 winsorize) 只需新增一个 Strategy 子类，
    `_preprocess_vectorized` 不变
  - 与原 if 链 bitwise 一致 (向后兼容) — 4 个 `TestBackwardCompat` 测试
    验证 zscore / median+pct_shrink / pct_shrink 路径与原公式输出
    decimal=10 一致
  - 新增 `tests/research/factor_test/nodes/test_preprocess_strategies.py`
    覆盖 7 类 (ABC / 3 strategy / Factory / E2E / BackwardCompat)，
    38 个测试

### Added
- **端到端集成测试 (Option D 巩固)**
  (`tests/research/factor_test/e2e/test_pipeline_bool_factor.py`)
  - 15 个 e2e 测试验证 Phase 1+2 重构后的 3 个节点
    (preprocess → neutralize → group_analyzer) 在 bool / 离散 /
    连续因子上的端到端行为
  - 6 个 test class: bool / low_tie / continuous / group_counts /
    output_keys / floor_mode
  - 验证场景: alpha-004 风格 30×-1 + 20×+1、7 unique ties、
    不同 group 数 (2/3/5/10)、4 种 preprocess 组合、
    industry neutralize、floor_mode='last'
- **设计模式审计文档 (Option D)**
  (`docs/26-设计模式重构与审计.md`)
  - 总结 Phase 1+2 已应用的 7 个 GoF 模式 (Null Object / Decorator /
    Builder / Visitor / Chain of Responsibility / Strategy)
  - 调研结论: Abstract Factory 在 QuantNodes 适用度有限 (无"一族
    互相依赖的产品族"场景), 改用 Facade + Simple Factory
  - Phase 3 路线: CLI Command pattern (推荐下一步), DataSource
    Factory + Adapter, Operator Facade
  - 模式选择决策树: 9 种场景 → 推荐模式, 供未来参考

---

## [2.7.0] - 2026-06-21

LLM Operator Layer Upgrade — implements 4 PRs from `docs/25-LLM算子层升级设计.md`
for llmwikify Loop v4 integration. Adds Composite DAG abstraction level
between primitive ops (L0) and business semantics (L3).

### Added

- **PR-QN-1**: `CodeSandbox` accepts instance-level `allowed_imports` /
  `blocked_imports` parameters. Class-level whitelist/blacklist
  extensible without monkey-patching. Default behavior unchanged.
  See `docs/24-核心功能框架设计.md` §15.
- **PR-QN-2**: `PipelineRunner` accepts `extra_phases` plugin mechanism
  (`__init__(specs=...)` / `from_dict(extra_phases=...)`). Downstream
  systems (e.g. llmwikify Loop v4) can inject custom stages after the
  standard 12 phases. `run()` now iterates `self._specs` instead of
  hardcoding `PIPELINE_SPEC`. See `docs/24-核心功能框架设计.md` §16.
- **PR-QN-3a**: Composite DAG core — `@composite_operator` decorator,
  `ParamSpec` / `CompositeSpec` dataclasses, `_COMPOSITE_REGISTRY`
  isolated registry (no pollution to main `_OPERATOR_REGISTRY`),
  `load_composites_from_yaml()` with AST parsing + function name
  whitelist (rejects bare `exec` risk), `get_composite_doc_for_llm()`
  producing LLM-friendly markdown. See `docs/22-算子系统设计与规范.md` §17.
- **PR-QN-3b**: 20 built-in composite ops covering quant research
  common algorithms:
  - Neutralization (3): `industry_neutralize`, `market_neutralize`,
    `subindustry_neutralize`
  - Cross-sectional normalization (3): `zscore_xs`, `rank_xs`, `scale_xs`
  - Rolling regression (3): `rolling_beta`, `rolling_ols_simplified`,
    `rolling_residual`
  - Volatility (4): `parkinson_vol`, `garman_klass_vol`,
    `yang_zhang_vol`, `realized_vol`
  - Pairs trading (2): `pair_zscore`, `pair_ratio`
  - Winsorize/outlier (3): `winsorize`, `mad_outlier`, `zscore_clip`
  - Composite time-series (2): `decay_linear_xs`, `momentum_accel`

  Polars 1.40+ API adaptation notes in `docs/22` §17.6.1 (corrections
  from original design spec: `Expr.group_by` → `.over()`, `rolling_corr`
  → OLS closed-form, `window=` → `window_size=`).

### Fixed

- **L1**: `resample_trade_date` accepts position aliases
  (`'beg'` / `'start'` / `'first'` → `'begin'`, `'last'` → `'end'`),
  no longer raises on natural shorthand.
- **L2**: `offset_date` explicit overflow check, fixing pandas iloc
  negative-index wrap-around silent bug. `n=-1` previously returned the
  last date silently; now raises `IndexError` with details, or clips
  to boundary when `if_modify=True`.
- **L3**: `FactorTestReport` raises `ValueError` for unknown `format`
  at runtime. Previously silently skipped (writing `'html'` produced
  no file and no error).

### Documentation

- New `docs/25-LLM算子层升级设计.md` (1338 lines, full PR-QN-1/2/3 design).
- Updated `docs/24-核心功能框架设计.md` §15 (PR-QN-1), §16 (PR-QN-2).
- Updated `docs/22-算子系统设计与规范.md` §17 (Composite DAG chapter).

### Testing

Added 6 test files, **+100 tests** (4608 → 4716 passed):

- `tests/test_sandbox_allowed_imports.py` (+17, PR-QN-1)
- `tests/test_pipeline_plugin.py` (+10, PR-QN-2)
- `tests/test_composite_dag.py` (+24, PR-QN-3a)
- `tests/test_composite_dag_ops.py` (+43, PR-QN-3b, includes 20-parametrize)
- `tests/research/factor_test/utils/test_date_utils_edge_cases.py`
  (+10, L1 alias + L2 overflow)
- `tests/research/factor_test/nodes/test_score_report.py` (+3, L3 format)

### Changed (Internal)

- `QuantNodes/operators/__init__.py`: 8 new composite re-exports.
- `QuantNodes/research/factor_test/pipeline_runner.py`: `__init__`
  accepts `specs`, `from_dict` accepts `extra_phases`, `run()` uses
  `self._specs`.

---

## Historical Versions

Earlier version history available via `git log`. This project adopted
structured CHANGELOG from May 2026; versions before 2.7.0 lack structured
entries.

[Unreleased]: https://github.com/sn0wfree/QuantNodes/compare/v2.7.0...HEAD
[2.7.0]: https://github.com/sn0wfree/QuantNodes/compare/v2.6.0...v2.7.0
