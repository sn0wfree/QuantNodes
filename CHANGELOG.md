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
  - 改为策略模式：`_classify_factor` 按 dtype + n_unique 判别 → 3 个纯
    函数 handler：
    - `_group_continuous` — 连续因子，保持 `pd.qcut(series, ...)` 原
      行为零变化
    - `_group_low_tie` — 轻度 ties (3 <= n_unique < group)，用
      `pd.qcut(series.rank(method='first'), ...)` 强制分 N 组（bug
      修复，原路径崩溃）
    - `_group_discrete` — bool/离散因子 (n_unique <= 2 或 bool/integer
      dtype 且 n_unique <= 10)，按 value 比例分配组段 + 内部 seeded
      shuffle（`seed=yyyymmdd % 2**31`）保证可复现
  - `_calc_group_return` 改写为 match-case 调度，循环内只调一次
    `factor_data.loc[t_i].dropna()`。
  - 新增 `tests/test_group_analyzer_bool.py` 覆盖 3 分支 + dispatch。

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
