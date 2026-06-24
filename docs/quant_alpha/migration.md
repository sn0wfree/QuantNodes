# QuantAlpha 迁移指南

> 从旧 `QuantNodes.research.*` 自动挖掘 API 迁移到新的 `QuantNodes.research.quant_alpha.*` 子包

**版本**：v2.7.0+
**生效日期**：2026-06-23
**目标版本**：v3.0.0（计划 2026 Q4）

---

## 1. 迁移路线图

QuantAlpha 子包按方案 C 渐进合并，分 3 阶段：

| 阶段 | 时机 | 状态 | 兼容性 |
|------|------|------|--------|
| **Phase A** (本 PR) | 2026-06-23 → | ✅ 已完成 | 100% 向后兼容（零行为变化）|
| **Phase B** | v2.9+ (M5+) | 🔜 进行中 | 旧类变 thin wrapper，行为等价 |
| **Phase C** | v3.0.0 | 📅 计划中 | 旧实现归档到 `_legacy_3c/`，破坏性变更 |

---

## 2. 旧 API → 新 API 完整映射

### 2.1 算子查询与命名空间构建

| 旧 API | 新 API | 行为差异 |
|--------|--------|----------|
| `from QuantNodes.research.factor_evaluator import FactorEvaluator, EvalConfig` | `from QuantNodes.research.quant_alpha import OperatorVocab, OperatorVocabConfig` | 行为等价 + 修复 3 个 latent bug |
| `FactorEvaluator(eval_config)` | `OperatorVocab.default()` 或 `OperatorVocab(config)` | 行为等价 |
| 12-lambda hardcoded namespace（隐式）| `vocab.build_namespace(data, date_column='date', cross_sectional=True)` | 算子从 **12 → 162**（+13×）|

### 2.2 端到端公式评估

**旧 API**：
```python
from QuantNodes.research.factor_evaluator import FactorEvaluator, EvalConfig
from QuantNodes.research.factor_miner import FactorCandidate

evaluator = FactorEvaluator(EvalConfig(...))
result = evaluator.evaluate(
    candidate=FactorCandidate(name="x", formula="rank(close)"),
    data=df,
    date_column="date",
)
```

**新 API**：
```python
from QuantNodes.research.quant_alpha import OperatorVocab

vocab = OperatorVocab.default()
series = vocab.evaluate(
    formula="rank(close)",
    data=df,
    date_column="date",
    cross_sectional=True,  # 默认 per-date over(date)
)
```

### 2.3 per-date 截面语义（关键 BUG 修复）

**旧行为**（`factor_evaluator.py:202-215`）：
```python
"rank": lambda col: col.rank(),  # 全局 rank（错误！）
"zscore": lambda col: (col - col.mean()) / (col.std() + 1e-8),  # 全局 zscore（错误！）
```

**新行为**（`OperatorVocab.build_namespace` 默认）：
```python
def _rank_per_date(x):
    s = x.to_series() if isinstance(x, pl.Expr) else x
    tmp = pl.DataFrame({"_x": s, "_d": data[date_col]})
    return tmp.select(pl.col("_x").rank(method="average").over("_d"))["_x"]

# 默认 cross_sectional=True → per-date over(date_column)
```

**关闭 per-date（兼容旧行为）**：
```python
# 旧行为（全局）
series = vocab.evaluate("rank(close)", df, cross_sectional=False)
```

### 2.4 算子元数据

| 旧 API | 新 API | 差异 |
|--------|--------|------|
| 无元数据 | `OperatorMetadata`（12 字段）| +7 LLM 友好字段 |
| 无 `list_operators()` | `vocab.list_operators(category='time')` | 新增 |
| 无 `get_metadata()` | `vocab.get_metadata('ts_argmax')` | 新增（LLM 路线 6 需要）|

### 2.5 5 个新算子（Alpha 101 必需）

| 算子 | 等价实现 | 用途 |
|------|---------|------|
| `signedpower(x, a)` | `sign(x) * abs(x) ** a` | Alpha #1 等 |
| `ts_decay_linear(x, d)` | `decay_linear(x, d)` | Alpha #39 等 |
| `IndNeutralize(x, ind_class)` | `x - x.mean().over(ind_class)` | Alpha #101 等 |
| `ts_skew(x, w)` | `rolling_skew(x, w)` | 通用 |
| `ts_kurt(x, w)` | `rolling_kurt(x, w)` | 通用 |

---

## 3. 行为对比示例

### 3.1 rank(close) — per-date 修复

**旧行为**（全局 rank，错误）：
```python
df = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "C", "A", "B", "C"],
    "close": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
})
evaluator.evaluate(candidate, df)  # rank: [1, 2, 3, 4, 5, 6]（全局）
```

**新行为**（per-date rank，正确）：
```python
vocab.evaluate("rank(close)", df, cross_sectional=True)
# rank: [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]（每个日期内 1, 2, 3）
```

**回退旧行为**（如需）：
```python
vocab.evaluate("rank(close)", df, cross_sectional=False)
# rank: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]（全局，与旧行为一致）
```

### 3.2 Alpha 101 #1 简化版 — 新算子组合

**完整公式**（实际 Alpha 101 #1）：
```
rank(ts_argmax(signedpower(((close < open) ? stddev(returns, 20) : close), 2.), 5)) - 0.5
```

**新 API 执行**：
```python
# 旧 API：因 12-lambda 限制，无法执行（缺 signedpower / 三元 ? 语法）
# 新 API：162 算子可直接执行
result = vocab.evaluate(
    "rank(ts_argmax(signedpower(close, 2), 5))",  # 简化版
    data=df,
    date_column="date",
)
```

---

## 4. Phase 时间表

### Phase A（当前，2026-06-23+）
- ✅ 4 个旧文件加 `DeprecationWarning`（import 时触发一次）
- ✅ 新子包 `quant_alpha` 独立运行
- ✅ 旧 12-lambda 行为完全保留（通过 `cross_sectional=False`）
- ✅ 4718 旧测试零失败

### Phase B（v2.9+，M5+ 之后）
- 旧 `FactorEvaluator` / `FactorMiner` / `MCTSSearch` / `AutoResearcher` 类变 thin wrapper
- 内部调新 `quant_alpha.OperatorVocab` 等
- 行为完全等价（diff < 1e-10）
- 旧类签名不变
- 继续触发 DeprecationWarning

### Phase C（v3.0.0）
- 旧实现迁到 `QuantNodes/research/_legacy_3c/`
- `QuantNodes/research/__init__.py` 移除旧 re-export
- 5 个设计文档同步更新
- 旧 4718 测试中的 50% 删除，保留 2 个 smoke test 在 `_legacy_3c/tests/`
- CHANGELOG `v3.0.0` 标记 breaking change

---

## 5. 迁移检查清单

迁移到新 API 时，请检查：

- [ ] 所有 `from QuantNodes.research.factor_evaluator import ...` → `from QuantNodes.research.quant_alpha import ...`
- [ ] 所有 `from QuantNodes.research.factor_miner import ...` → `from QuantNodes.research.quant_alpha import ...`
- [ ] 所有 `from QuantNodes.research.mcts_search import ...` → 等待 M2 PR
- [ ] 所有 `from QuantNodes.research.auto_researcher import ...` → 等待 M5+ PR
- [ ] 公式中如有 `rank` / `zscore` / `winsorize`：默认 per-date 行为已修复，**这是 BREAKING CHANGE**
- [ ] 公式中如有 `ts_corr` / `ts_cov`：现在通过 L0 注册表调用，行为等价
- [ ] 公式中如有 `signedpower` / `ts_decay_linear` / `IndNeutralize` / `ts_skew` / `ts_kurt`：新算子，需要确保依赖已就绪

---

## 6. FAQ

### Q1: 旧代码会立即失效吗？
**A**: 不会。Phase A 期间旧代码继续运行，只是打印 DeprecationWarning。

### Q2: 旧测试会失败吗？
**A**: Phase A 已验证 4718 旧测试零失败。Phase B 还会保持零失败。Phase C 才删除部分测试。

### Q3: 我可以同时使用新旧 API 吗？
**A**: 可以。两者不冲突，新 API 是独立子包。

### Q4: 关闭 DeprecationWarning 的方法？
**A**: 旧文件头部 `warnings.warn(...)` 用 `stacklevel=2` 触发一次。可在 `pyproject.toml` 配置：
```toml
[tool.pytest.ini_options]
filterwarnings = [
    "ignore::DeprecationWarning:QuantNodes.research.*",
]
```

### Q5: 新 API 与 factor_test.PipelineRunner 兼容吗？
**A**: Phase A 时 `factor_test.PipelineRunner` 仍用旧 12-lambda。Phase B 时新旧双轨，Phase C 时完全切到新 API。

---

## 7. 引用

- [`docs/quant_alpha/PROJECT_PLAN.md`](./PROJECT_PLAN.md) — 完整调研与规划
- [`docs/Architecture-v2.6.md`](../Architecture-v2.6.md) — 主项目架构
- [`docs/Evolution-Framework.md`](../Evolution-Framework.md) — QuantaAlpha-inspired 演化框架
