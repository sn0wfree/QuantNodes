# Changelog - QuantAlpha

All notable changes to QuantAlpha subpackage will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-06-23 (M1)

### Added

- **OperatorVocab 主类**：统一算子查询/调用/元数据化接口
  - `OperatorVocab.default()`：模块级单例
  - `OperatorVocab(config)`：自定义配置
  - `build_namespace(data, date_column, code_column, cross_sectional=True)`：构造 eval 沙箱
  - `list_operators(category=None)`：列出算子
  - `get_operator(name)`：按名称获取
  - `get_metadata(name)`：按名称获取元数据
  - `evaluate(formula, data, ...)`：端到端评估（先 build_namespace 再 eval）

- **5 个新算子**（修复 Alpha 101 关键缺口）：
  - `signedpower(x, a)` = `sign(x) * abs(x) ** a`
  - `ts_decay_linear(x, d)` = `decay_linear(x, d)` 别名
  - `IndNeutralize(x, ind_class)` = `industry_neutralize(x, ind_class)` 别名
  - `ts_skew(x, w)` = `rolling_skew(x, w)` 别名
  - `ts_kurt(x, w)` = `rolling_kurt(x, w)` 别名

- **算子元数据 schema 扩展**（从 5 字段到 12 字段）：
  - 旧字段：name, category, func, doc, signature, parameters
  - 新字段：difficulty, category_tags, default_window, requires_group_by, output_dtype, examples, composes_with

- **per-date over() 语义修复**：
  - `rank(x)` → `x.rank().over(date_column)`（per-date）
  - `zscore(x)` → `(x - x.mean().over(date)) / (x.std().over(date) + 1e-8)`（per-date）
  - `winsorize(x, l, u)` → per-date quantile clip
  - `IndNeutralize(x, ind)` → per-date demean
  - 提供 `cross_sectional=False` 关闭开关（兼容旧全局语义）

- **DeprecationWarning**：旧 4 个文件加 import-time warning：
  - `QuantNodes/research/factor_miner.py`
  - `QuantNodes/research/factor_evaluator.py`
  - `QuantNodes/research/mcts_search.py`
  - `QuantNodes/research/auto_researcher.py`

- **migration.md**：旧 API → 新 API 完整映射表

### Fixed

- **BUG 1（API 不存在）**：`pl.Series.rolling_corr` / `rolling_cov` 不存在
- **BUG 2（维度错误）**：`rank` / `zscore` 全局计算而非 per-date
- **BUG 3（静默失败）**：异常被 `except: return None` 吞掉 → 改为完整错误抛出

### Changed

- 旧 4 文件仍可用，行为完全兼容（旧 12-lambda 保留为内部实现，可通过 `cross_sectional=False` 切换回全局语义）

### Deprecated

- `QuantNodes.research.factor_miner.FactorMiner` → use `QuantNodes.research.quant_alpha.OperatorVocab`
- `QuantNodes.research.factor_evaluator.FactorEvaluator` → use `QuantNodes.research.quant_alpha.OperatorVocab`
- `QuantNodes.research.mcts_search.MCTSSearch` → use `QuantNodes.research.quant_alpha.mcts.MCTSSearch` (M2)
- `QuantNodes.research.auto_researcher.AutoResearcher` → use `QuantNodes.research.quant_alpha.AutoResearcher` (M5+)

### Performance

- 算子可用数：从 12 → 285（+ 23×）
- 元数据字段：从 5 → 12（+ 7 字段 LLM 友好）
- per-date 正确性：修复维度 bug 后 IC 计算准确

### Migration Path

- **Phase A** (M1, current): 旧 4 文件 + DeprecationWarning，新子包并行
- **Phase B** (M5+): 旧类变 thin wrapper，行为等价
- **Phase C** (v3.0.0): 旧实现归档到 `_legacy_3c/`，破坏性变更

---

[Unreleased]: https://github.com/sn0wfree/QuantNodes/compare/v2.7.0...HEAD
[0.1.0]: https://github.com/sn0wfree/QuantNodes/releases/tag/quant_alpha-v0.1.0
