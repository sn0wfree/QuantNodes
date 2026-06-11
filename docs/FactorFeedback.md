# FactorFeedback — 结构化反馈规格

> QuantaAlpha `CoSTEERSingleFeedback` 的 QuantNodes 适配版本
>
> Version: 1.0  |  Date: 2026-06-11

---

## 1. 概述

### 1.1 问题

现有节点返回 ad-hoc dict:

```python
# ICAnalyzerNode 返回
{
    'ic': pd.Series(...),
    'rank_ic': pd.Series(...),
    'ic_result': {...},
    ...
}
```

**问题**:
- LLM 无法解析 (结构不固定)
- 演化控制器无法判断「这个因子是否成功」
- 反馈信号散落各处, 难以聚合
- 没有持久化, 跨实验无法对比

### 1.2 目标

把节点返回的统一为 `FactorFeedback`:
- 5 通道结构化信号 (execution / shape / code / value / llm)
- LLM 可解析
- 可持久化 (Parquet + JSON)
- 节点可双轨兼容 (dict 或 FactorFeedback)

---

## 2. 数据结构

### 2.1 FeedbackChannel 枚举

```python
from enum import Enum

class FeedbackChannel(str, Enum):
    """5 通道反馈信号"""
    EXECUTION = "execution"    # 沙箱 stdout/stderr, exit code
    SHAPE = "shape"            # 输出形状 vs 预期
    CODE = "code"              # AST 静态检查
    VALUE = "value"            # 数值分布统计
    LLM = "llm"                # LLM 一致性评判
```

### 2.2 ChannelFeedback 单通道

```python
@dataclass
class ChannelFeedback:
    """单通道反馈信号"""
    channel: FeedbackChannel
    passed: bool                # 通道是否通过
    detail: str                 # 详细说明 (供 LLM 阅读)
    score: float = 1.0          # 0-1 分数 (供聚合)
    metadata: dict = field(default_factory=dict)  # 额外元数据
```

### 2.3 FactorFeedback 完整反馈

```python
@dataclass
class FactorFeedback:
    """完整因子反馈 - QuantaAlpha CoSTEERSingleFeedback 等价物"""
    factor_id: str                                # 因子唯一 ID (UUID)
    factor_name: str                              # 因子名称 (供报告)
    channels: dict[FeedbackChannel, ChannelFeedback] = field(default_factory=dict)
    decision: bool = False                        # 最终通过/失败 (全部通道通过)
    summary: str = ""                             # 一句话总结
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0                      # 总耗时
    metadata: dict = field(default_factory=dict)  # 额外元数据 (如 metrics)
```

---

## 3. 5 通道详细规格

### 3.1 EXECUTION 通道

**目的**: 因子代码是否成功执行。

**采集内容**:
- exit code (0=成功)
- stdout (截断到 500 字符)
- stderr (截断到 500 字符)

**采集函数**:

```python
def collect_execution(stdout: str, stderr: str, exit_code: int) -> ChannelFeedback:
    passed = exit_code == 0
    detail = f"exit={exit_code}\nstdout: {stdout[:500]}\nstderr: {stderr[:500]}"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(FeedbackChannel.EXECUTION, passed, detail, score)
```

**触发时机**: 因子代码在沙箱执行完成后。

**失败模式**:
- SyntaxError → exit != 0
- RuntimeError → exit != 0
- 超时 → exit != 0 (强制 kill)
- ImportError → exit != 0

### 3.2 SHAPE 通道

**目的**: 输出形状是否符合预期。

**采集内容**:
- actual_shape (tuple)
- expected_shape (tuple)
- 一致性

**采集函数**:

```python
def collect_shape(actual_shape: tuple, expected_shape: tuple) -> ChannelFeedback:
    passed = actual_shape == expected_shape
    detail = f"actual={actual_shape}, expected={expected_shape}"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(FeedbackChannel.SHAPE, passed, detail, score)
```

**预期形状约定**:
- 单因子回测: `(n_adj_dates, n_stocks)`
- 多因子: `(n_adj_dates, n_factors, n_stocks)`

**触发时机**: 因子执行完成后, 计算结果 DataFrame 之前。

### 3.3 CODE 通道

**目的**: AST 静态检查 (防过拟合)。

**检查项** (QuantaAlpha `quality_gate.complexity`):
- `symbol_length_threshold` = 200 (表达式长度)
- `base_features_threshold` = 5 (基础特征数: close/open/high/low/volume/mv_float 等)
- `free_args_ratio_threshold` = 0.5 (自由参数占比)

**采集函数**:

```python
def collect_code(expression: str) -> ChannelFeedback:
    try:
        tree = ast.parse(expression)
    except SyntaxError as e:
        return ChannelFeedback(FeedbackChannel.CODE, False, f"语法错误: {e}")

    symbol_length = len(expression)
    base_features = count_base_features(tree)  # close, open, high, low, ...
    free_args_ratio = calc_free_args_ratio(tree)

    violations = []
    if symbol_length > 200:
        violations.append(f"length={symbol_length}>200")
    if base_features > 5:
        violations.append(f"features={base_features}>5")
    if free_args_ratio > 0.5:
        violations.append(f"free_args={free_args_ratio:.2f}>0.5")

    passed = len(violations) == 0
    detail = "; ".join(violations) if violations else \
             f"OK (length={symbol_length}, features={base_features})"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(FeedbackChannel.CODE, passed, detail, score)
```

**触发时机**: 因子提交时, 在执行前 (避免浪费算力)。

### 3.4 VALUE 通道

**目的**: 输出数值的统计合理性。

**检查项**:
- NaN 比例 < 30%
- Inf 数量 == 0
- 标准差 > 1e-6 (避免常量)

**采集函数**:

```python
def collect_value(values: pd.Series) -> ChannelFeedback:
    nan_pct = values.isna().mean()
    inf_count = np.isinf(values).sum()
    mean_val = values.mean()
    std_val = values.std()

    violations = []
    if nan_pct > 0.3:
        violations.append(f"NaN={nan_pct:.2%}>30%")
    if inf_count > 0:
        violations.append(f"Inf={inf_count}>0")
    if std_val <= 1e-6:
        violations.append(f"std={std_val:.6f}<=1e-6")

    passed = len(violations) == 0
    detail = "; ".join(violations) if violations else \
             f"OK (NaN={nan_pct:.2%}, mean={mean_val:.4f}, std={std_val:.4f})"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(FeedbackChannel.VALUE, passed, detail, score)
```

**触发时机**: 因子执行完成后, 数值分析前。

### 3.5 LLM 通道

**目的**: LLM 判断 hypothesis ↔ description ↔ expression 一致性。

**检查流程** (QuantaAlpha `quality_gate.consistency`):
1. 构造 prompt: hypothesis + description + expression
2. LLM 返回 `{consistent: bool, reason: str, score: 0-1}`
3. 若 `consistent=False` 且 `max_correction_attempts > 0`, 触发自我修正循环

**采集函数**:

```python
class LLMJudge:
    def __init__(self, model="deepseek-v3", max_correction_attempts=3):
        self.model = model
        self.max_correction_attempts = max_correction_attempts

    def judge(self, hypothesis: str, description: str, expression: str) -> ChannelFeedback:
        prompt = f"""判断以下三者是否逻辑一致:
Hypothesis (研究假设): {hypothesis}
Description (因子描述): {description}
Expression (代码表达式): {expression}

返回 JSON: {{"consistent": true/false, "reason": "理由", "score": 0-1}}"""

        for attempt in range(self.max_correction_attempts + 1):
            try:
                response = self._call_llm(prompt)
                result = json.loads(response)
                return ChannelFeedback(
                    FeedbackChannel.LLM,
                    result['consistent'],
                    result['reason'],
                    result['score'],
                )
            except (json.JSONDecodeError, KeyError) as e:
                if attempt == self.max_correction_attempts:
                    return ChannelFeedback(
                        FeedbackChannel.LLM, False,
                        f"LLM 解析失败: {e}", 0.0,
                    )
                # 重试
                continue
```

**触发时机**: 因子提交时, 可选 (需 LLM token)。

---

## 4. FeedbackCollector 聚合器

### 4.1 接口

```python
class FeedbackCollector:
    """聚合多个通道的反馈信号"""

    def __init__(self, factor_id: str, factor_name: str):
        self.factor_id = factor_id
        self.factor_name = factor_name
        self._channels: dict[FeedbackChannel, ChannelFeedback] = {}
        self._t0 = time.perf_counter()

    def add(self, channel: FeedbackChannel, passed: bool,
            detail: str, score: float = 1.0, **metadata):
        """添加一个通道的反馈"""
        self._channels[channel] = ChannelFeedback(
            channel=channel, passed=passed, detail=detail,
            score=score, metadata=metadata,
        )

    def finalize(self, decision: bool = None, summary: str = "") -> FactorFeedback:
        """聚合所有通道, 返回 FactorFeedback"""
        if decision is None:
            decision = all(fb.passed for fb in self._channels.values())

        if not summary:
            failed = [ch.value for ch, fb in self._channels.items() if not fb.passed]
            summary = f"失败通道: {', '.join(failed)}" if failed else "全部通过"

        return FactorFeedback(
            factor_id=self.factor_id,
            factor_name=self.factor_name,
            channels=dict(self._channels),
            decision=decision,
            summary=summary,
            duration_ms=(time.perf_counter() - self._t0) * 1000,
        )
```

### 4.2 使用示例

```python
# 节点内部
collector = FeedbackCollector(factor_id, factor_name)

# Pre-execution
collector.add(FeedbackChannel.CODE, *collect_code(expression).values())

# Execution
exit_code, stdout, stderr = run_in_sandbox(expression)
collector.add(FeedbackChannel.EXECUTION, *collect_execution(stdout, stderr, exit_code).values())

# Post-execution
collector.add(FeedbackChannel.SHAPE, *collect_shape(actual, expected).values())
collector.add(FeedbackChannel.VALUE, *collect_value(values).values())

# Optional LLM
if config.consistency.enabled:
    collector.add(FeedbackChannel.LLM, *llm_judge.judge(h, d, e).values())

# Aggregate
feedback = collector.finalize()
return feedback
```

---

## 5. 序列化 / Serialization

### 5.1 Parquet (推荐用于分析)

**Schema** (一行一条反馈):

| 列 | 类型 | 说明 |
|----|------|------|
| `factor_id` | str | UUID |
| `factor_name` | str | 因子名 |
| `decision` | bool | 通过/失败 |
| `summary` | str | 一句话总结 |
| `duration_ms` | float | 总耗时 |
| `timestamp` | str | ISO 格式 |
| `exec_passed` | bool | EXECUTION 通道 |
| `exec_score` | float | |
| `exec_detail` | str | |
| `shape_passed` | bool | SHAPE 通道 |
| `shape_score` | float | |
| `shape_detail` | str | |
| `code_passed` | bool | CODE 通道 |
| `code_score` | float | |
| `code_detail` | str | |
| `value_passed` | bool | VALUE 通道 |
| `value_score` | float | |
| `value_detail` | str | |
| `llm_passed` | bool | LLM 通道 (可选) |
| `llm_score` | float | |
| `llm_detail` | str | |

**API**:

```python
# 保存
feedback.save_parquet(Path('feedback.parquet'))

# 加载
feedback = FactorFeedback.load_parquet(Path('feedback.parquet'))
```

### 5.2 JSON (推荐用于调试)

```json
{
  "factor_id": "uuid-xxx",
  "factor_name": "momentum_20d",
  "channels": {
    "execution": {"channel": "execution", "passed": true, "detail": "exit=0", "score": 1.0},
    "shape": {"channel": "shape", "passed": true, "detail": "actual=(20,30), expected=(20,30)", "score": 1.0},
    "code": {"channel": "code", "passed": true, "detail": "OK (length=120, features=3)", "score": 1.0},
    "value": {"channel": "value", "passed": true, "detail": "OK (NaN=0.05, mean=0.02, std=0.5)", "score": 1.0}
  },
  "decision": true,
  "summary": "全部通过",
  "timestamp": "2026-06-11T10:30:00",
  "duration_ms": 1234.5,
  "metadata": {"ic_mean": 0.05, "sharpe": 1.2}
}
```

---

## 6. 与现有节点的集成

### 6.1 自动包装

```python
# core/feedback.py
def ensure_feedback(result, factor_id, factor_name) -> FactorFeedback:
    """如果 result 不是 FactorFeedback, 自动从 dict 包装"""
    if isinstance(result, FactorFeedback):
        return result
    if isinstance(result, dict):
        # 从 dict 提取已知的指标
        metadata = {}
        for key in ['ic', 'sharpe', 'arr', 'mdd', 'calmar']:
            if key in result:
                metadata[key] = result[key]
        return FactorFeedback(
            factor_id=factor_id,
            factor_name=factor_name,
            decision=True,  # 默认通过 (无显式失败信号)
            summary=f"dict 返回, {len(result)} 个字段",
            metadata=metadata,
        )
    raise TypeError(f"节点返回类型不支持: {type(result)}")
```

### 6.2 节点升级路径

**Phase 1**: 现有节点不动, PipelineRunner 自动包装
**Phase 2**: 关键节点 (ICAnalyzer, GroupAnalyzer, LongShort) 返回 `FactorFeedback`
**Phase 3**: 所有节点统一返回 `FactorFeedback`

---

## 7. 测试覆盖 (20 tests)

### 7.1 单元测试

| 测试 | 验证 |
|------|------|
| `test_channel_enum_values` | 5 个通道枚举值 |
| `test_channel_feedback_creation` | 单通道创建 |
| `test_factor_feedback_creation` | 完整反馈创建 |
| `test_feedback_to_dict` | 字典序列化 |
| `test_feedback_from_dict` | 字典反序列化 |

### 7.2 通道采集器测试

| 测试 | 验证 |
|------|------|
| `test_collect_execution_success` | exit=0 通过 |
| `test_collect_execution_failure` | exit!=0 失败 |
| `test_collect_shape_match` | 形状一致 |
| `test_collect_shape_mismatch` | 形状不一致 |
| `test_collect_code_simple` | 简单表达式通过 |
| `test_collect_code_long` | 长表达式失败 |
| `test_collect_code_many_features` | 多特征失败 |
| `test_collect_code_syntax_error` | 语法错误 |
| `test_collect_value_normal` | 正常分布 |
| `test_collect_value_nan_heavy` | NaN 过多 |
| `test_collect_value_inf` | Inf 检测 |
| `test_collect_value_constant` | 常量检测 |

### 7.3 Collector 测试

| 测试 | 验证 |
|------|------|
| `test_collector_add_channels` | 添加多通道 |
| `test_collector_finalize_default_decision` | 默认全部通过 |
| `test_collector_finalize_explicit_decision` | 显式决策 |
| `test_collector_summary_generation` | 自动生成总结 |

### 7.4 LLM Judge 测试

| 测试 | 验证 |
|------|------|
| `test_llm_judge_passes` | 一致性通过 (mock) |
| `test_llm_judge_fails` | 一致性失败 (mock) |
| `test_llm_judge_correction_attempts` | 自我修正循环 |
| `test_llm_judge_parse_failure` | 解析失败处理 |

---

## 8. 性能考虑

### 8.1 Parquet 写入开销

- 单条反馈: ~5 KB (JSON) vs ~0.5 KB (Parquet 行)
- 1000 因子实验: ~500 KB, 写入 < 10ms
- 10,000 因子实验: ~5 MB, 写入 < 100ms

### 8.2 LLM 调用开销

- 单次 LLM 调用: ~1-3 秒
- 1000 因子实验: 1000 × 2s = 33 分钟
- **建议**: 默认 `consistency.enabled=False`, 关键决策时才打开

---

## 9. 未来扩展

| 扩展 | 优先级 | 说明 |
|------|--------|------|
| `OVERFITTING` 通道 (回测/样本外差异) | P2 | 检测过拟合 |
| `STABILITY` 通道 (子样本波动) | P2 | 因子稳定性 |
| `EXPLAINABILITY` 通道 (SHAP 值) | P3 | 因子可解释性 |
| 自定义通道 (`ChannelFeedback` 继承) | P3 | 业务特定信号 |

---

## 10. 参考

- QuantaAlpha `quantaalpha/coder/costeer/evaluators.py:10` — `CoSTEERSingleFeedback`
- QuantaAlpha `quantaalpha/coder/costeer/evaluators.py:76` — `CoSTEERMultiEvaluator`
- QuantaAlpha `quantaalpha/coder/costeer/evolving_agent.py:5` — `FilterFailedRAGEvoAgent`

---

*Last updated: 2026-06-11*
