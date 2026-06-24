# QuantAlpha

> 自动化因子挖掘引擎 — 从"人工设计"到"机器发现"的范式升级

[![Status](https://img.shields.io/badge/status-M1%20in%20progress-yellow)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)]()

---

## 1. 是什么

**QuantAlpha** 是 QuantNodes 的自动化因子挖掘子包，参考业界 4 大因子库演进链：

| 因子库 | 年份 | 范式 |
|--------|------|------|
| Alpha 101 | 2015 | 公式化因子 |
| Alpha 158/360 | 2020 | ML 友好特征 |
| AutoAlpha | 2020 | 层次化进化 |
| AlphaGen / Alpha-GPT | 2023+ | RL / LLM 驱动 |

完整规划见 [`docs/quant_alpha/PROJECT_PLAN.md`](../../docs/quant_alpha/PROJECT_PLAN.md)。

---

## 2. 当前状态：M1 (OperatorVocab)

### 2.1 解决的问题

QuantNodes 实际有 **285 个算子**（157 L0 + 109 talib + 20 L1 composite），但自动挖掘链路只能访问 **12 个硬编码 lambda**：

- `factor_evaluator.py:202-215` 的 namespace 是 12 个手写 lambda
- 3 个 latent bug：
  - `ts_corr` / `ts_cov` 用 `Series.rolling_corr`（Series 上不存在）
  - `rank` / `zscore` 全局计算（不是 per-date 截面）
  - 异常被 `except: return None` 静默吞掉

### 2.2 M1 交付

- ✅ **OperatorVocab**：统一算子查询/调用/元数据化接口
- ✅ **5 个新算子**：`signedpower` / `ts_decay_linear` / `IndNeutralize` / `ts_skew` / `ts_kurt`
- ✅ **per-date over() 修复**：默认走 per-date，可通过 `cross_sectional=False` 关闭
- ✅ **算子元数据 schema 扩展**：7 个新字段（difficulty / category_tags / default_window / requires_group_by / output_dtype / examples / composes_with）
- ✅ **DeprecationWarning**：旧 4 文件加 warning，进入 deprecation 周期
- ✅ **migration.md**：旧 API → 新 API 完整映射

### 2.3 快速开始

```python
from QuantNodes.research.quant_alpha import OperatorVocab
import polars as pl

# 构造数据
df = pl.DataFrame({
    "date": [...],
    "code": [...],
    "close": [...],
    "open": [...],
    "high": [...],
    "low": [...],
    "vol": [...],
})

# 构造 eval namespace（285 个算子 + per-date over()）
vocab = OperatorVocab.default()
namespace = vocab.build_namespace(
    data=df,
    date_column="date",
    code_column="code",
    cross_sectional=True,  # 默认 per-date
)

# 评估公式
result = vocab.evaluate(
    "rank(ts_argmax(signedpower(close, 2), 5))",
    data=df,
    date_column="date",
)

# 列出所有算子
all_ops = vocab.list_operators()
ts_ops = vocab.list_operators(category="time")
print(f"总算子数: {len(all_ops)}, 时序算子: {len(ts_ops)}")

# 查询元数据
meta = vocab.get_metadata("ts_argmax")
print(f"difficulty={meta.difficulty}, requires_group_by={meta.requires_group_by}")
```

### 2.4 旧 API 迁移

| 旧 API | 新 API | 行为差异 |
|--------|--------|----------|
| `from QuantNodes.research.factor_evaluator import FactorEvaluator` | `from QuantNodes.research.quant_alpha import OperatorVocab` | 行为等价 + 修复 3 bug |
| `FactorEvaluator(config).evaluate(candidate, data)` | `OperatorVocab.default().evaluate(candidate, data)` | 算子从 12 增到 285 |
| `eval(formula, namespace, data)` (12 ops) | `vocab.build_namespace(data)` (285 ops) | per-date over() 修复 |
| `rank: col.rank()` (全局) | `rank: col.rank().over(date_column)` (per-date) | 修复维度 bug |

完整迁移指南见 [`docs/quant_alpha/migration.md`](../../docs/quant_alpha/migration.md)。

---

## 3. 后续路线

| 里程碑 | 周 | 路线 | 交付物 |
|--------|----|----|--------|
| M1 | 1 | 路线 0 | OperatorVocab + 5 算子 + per-date over() |
| M2 | 2 | 路线 7 | MCTS + 5 通道反馈 |
| M3 | 3 | 路线 1+2 借鉴 | alpha101/158 设计文档 + few-shot |
| M4 | 4 | 路线 4 | PolarsAlphaCalculator 适配器 |
| M5 | 5-6 | 路线 6 启动 | 算子元数据回填 + 3 智能体 + 4 层 RAG |
| M6 | 7-10 | 路线 6 完成 | Alpha-GPT 完整工作流 |
| M7 | 11 | 整合 | 跨路线 A/B + v2.7.0 release |

---

## 4. 测试

```bash
# 单元测试
pytest tests/quant_alpha/ -v

# 覆盖率
pytest tests/quant_alpha/ --cov=QuantNodes.research.quant_alpha --cov-report=term-missing

# 集成测试（含旧 4 文件兼容）
pytest tests/quant_alpha/ tests/research/ -v
```

---

## 5. 文档

- [`docs/quant_alpha/PROJECT_PLAN.md`](../../docs/quant_alpha/PROJECT_PLAN.md) - 完整调研与规划
- [`docs/quant_alpha/migration.md`](../../docs/quant_alpha/migration.md) - 旧 API → 新 API 迁移指南
- [`CHANGELOG.md`](./CHANGELOG.md) - 子包变更日志

---

## 6. 许可

MIT License（与 QuantNodes 主项目一致）
