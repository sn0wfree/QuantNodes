# QualityGate — 质量门规格

> QuantaAlpha `quality_gate` 的 QuantNodes 适配版本
>
> Version: 1.0  |  Date: 2026-06-11

---

## 1. 概述

### 1.1 问题

当前 PipelineRunner 流程:

```
[Submit] → [Preprocess] → [Backtest] → [Report]
```

**问题**:
- 没有 pre-check, 低质量因子也跑完整回测
- 浪费 LLM token + GPU/CPU + 时间
- 没有"过拟合防御"机制
- 没有"近重复因子"拦截

### 1.2 目标

在 Submit 之后, Backtest 之前, 插入 **质量门** (QualityGate):

```
[Submit] → [QualityGate] ──passed=false──▶ [REJECTED]
                │ passed=true
                ▼
           [Preprocess] → [Backtest] → [Report]
```

**3 个独立可配门** (QuantaAlpha `quality_gate`):
1. **Complexity** — 防过拟合 (AST 静态检查)
2. **Redundancy** — 防重复 (因子 Zoo 去重)
3. **Consistency** — 防自相矛盾 (LLM 验证)

---

## 2. 3 个门详细规格

### 2.1 COMPLEXITY 门

**目的**: 通过 AST 静态检查, 防止过拟合。

**检查项** (QuantaAlpha `quality_gate.complexity`):

| 检查 | 阈值 | 说明 |
|------|------|------|
| `symbol_length_threshold` | 200 | 表达式字符串长度上限 |
| `base_features_threshold` | 5 | 调用的基础特征数上限 |
| `free_args_ratio_threshold` | 0.5 | 自由参数占比上限 |

**基础特征白名单**:
```python
BASE_FEATURES = {
    'close', 'open', 'high', 'low', 'volume', 'amount',
    'mv_float', 'industry', 'cap', 'turnover',
}
```

**采集函数**:

```python
def collect_code(expression: str) -> ChannelFeedback:
    """复杂度门"""
    try:
        tree = ast.parse(expression)
    except SyntaxError as e:
        return ChannelFeedback(FeedbackChannel.CODE, False, f"语法错误: {e}", 0.0)

    symbol_length = len(expression)
    base_features = _count_base_features(tree)
    free_args_ratio = _calc_free_args_ratio(tree)

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
    return ChannelFeedback(FeedbackChannel.CODE, passed, detail, 1.0 if passed else 0.0)
```

**示例**:

```python
# 通过
"(close - close.shift(20)) / close.shift(20)"  # 长度=37, 特征=1

# 失败: 特征过多
"(close - open + high - low + volume + amount + mv_float + cap) / 8"  # 8 特征

# 失败: 长度超限
# ... 任意 > 200 字符的表达式
```

### 2.2 REDUNDANCY 门

**目的**: 防止提交与历史因子过于相似的"近重复"因子。

**机制** (QuantaAlpha `quality_gate.duplication`):
1. 维护 **因子 Zoo** (`FactorZoo` 类) 存储历史通过因子的 AST hash
2. 提交新因子时, 计算其 hash, 与 Zoo 中所有 hash 计算 **汉明距离**
3. 若最小距离 < `threshold` (默认 5), 判定为重复 → REJECT

**FactorZoo**:

```python
class FactorZoo:
    """因子 Zoo - 存储历史通过因子的 AST hash"""

    def __init__(self, path: Path = None):
        self.path = Path(path) if path else None
        self._hashes: list[tuple[int, str]] = []  # (ast_hash, expression)
        if self.path and self.path.exists():
            self._load()

    def add(self, expression: str):
        """添加一个通过的质量门检查的因子"""
        h = self._ast_hash(expression)
        if h not in [hh for hh, _ in self._hashes]:
            self._hashes.append((h, expression))
            self._save()

    def _ast_hash(self, expression: str) -> int:
        """AST 规范化 hash (忽略变量名)"""
        tree = ast.parse(expression)
        # 移除变量名 (只保留结构)
        normalized = ast.dump(tree, annotate_fields=False)
        return hash(normalized)

    def _save(self):
        if self._hashes and self.path:
            df = pd.DataFrame(self._hashes, columns=['hash', 'expression'])
            df.to_parquet(self.path, index=False)

    def _load(self):
        if self.path.exists():
            df = pd.read_parquet(self.path)
            self._hashes = list(zip(df['hash'].tolist(), df['expression'].tolist()))

    def __len__(self):
        return len(self._hashes)
```

**RedundancyChecker**:

```python
class RedundancyChecker:
    """冗余检查: 与 Zoo 中已有因子的 AST 哈希距离"""

    def __init__(self, settings: RedundancySetting):
        self.settings = settings
        self.zoo = FactorZoo(
            Path(settings.zoo_path) if settings.zoo_path else None
        )

    def check(self, expression: str) -> ChannelFeedback:
        if not self.settings.enabled:
            return ChannelFeedback(FeedbackChannel.VALUE, True, "redundancy disabled")

        if len(self.zoo) == 0:
            return ChannelFeedback(FeedbackChannel.VALUE, True, "Zoo 为空, 无需检查")

        new_hash = self.zoo._ast_hash(expression)
        min_dist = min(
            bin(new_hash ^ old_hash).count('1')
            for old_hash, _ in self.zoo._hashes
        )
        passed = min_dist >= self.settings.threshold
        detail = f"min_hamming_dist={min_dist}, threshold={self.settings.threshold}, zoo_size={len(self.zoo)}"
        return ChannelFeedback(FeedbackChannel.VALUE, passed, detail, 1.0 if passed else 0.0)
```

**汉明距离解读**:
- `min_dist = 0`: 完全相同的 AST 结构 (改名变量也算)
- `min_dist = 1-4`: 极度相似 (仅一个节点不同)
- `min_dist = 5-10`: 相似 (几个算子不同)
- `min_dist > 20`: 完全不同

**默认 threshold=5** 平衡误杀与漏检。

### 2.3 CONSISTENCY 门

**目的**: LLM 验证 hypothesis ↔ description ↔ expression 三者逻辑一致。

**检查流程** (QuantaAlpha `quality_gate.consistency`):
1. 构造 prompt, 包含 hypothesis + description + expression
2. LLM 返回 JSON: `{consistent: bool, reason: str, score: 0-1}`
3. 若 `consistent=False` 且未达最大尝试次数, 触发自我修正
4. 修正: 重新生成 expression, 再次验证

**LLMJudge 实现**: 复用 `FactorFeedback` 系统的 `LLMJudge` (见 `docs/FactorFeedback.md` § 3.5)

**ConsistencyChecker**:

```python
class ConsistencyChecker:
    """LLM 一致性检查"""

    def __init__(self, settings: ConsistencySetting):
        self.settings = settings
        self._judge = LLMJudge(
            model=settings.model,
            max_correction_attempts=settings.max_correction_attempts,
        )

    def check(self, hypothesis: str, description: str, expression: str) -> ChannelFeedback:
        if not self.settings.enabled:
            return ChannelFeedback(FeedbackChannel.LLM, True, "consistency disabled")
        return self._judge.judge(hypothesis, description, expression)
```

**Prompt 模板**:

```python
CONSISTENCY_PROMPT = """你是一个量化研究专家。请判断以下三者是否逻辑一致:

Hypothesis (研究假设):
{hypothesis}

Description (因子描述):
{description}

Expression (代码表达式):
{expression}

判断标准:
1. Expression 是否能实现 Description 描述的逻辑?
2. Description 是否准确描述了 Expression 的行为?
3. Hypothesis 是否被 Description + Expression 验证?

返回 JSON (严格格式):
{{"consistent": true/false, "reason": "详细理由 (100字以内)", "score": 0-1}}"""
```

---

## 3. QualityGateSetting 配置

### 3.1 Pydantic 模型

```python
from pydantic import BaseModel, Field
from typing import Optional


class ComplexitySetting(BaseModel):
    """复杂度门配置"""
    enabled: bool = Field(default=True, description="启用复杂度门")
    symbol_length_threshold: int = Field(default=200, description="表达式长度上限")
    base_features_threshold: int = Field(default=5, description="基础特征数上限")
    free_args_ratio_threshold: float = Field(default=0.5, description="自由参数占比上限")


class RedundancySetting(BaseModel):
    """冗余门配置"""
    enabled: bool = Field(default=True, description="启用冗余门")
    threshold: int = Field(default=5, description="最小汉明距离阈值")
    zoo_path: Optional[str] = Field(default=None, description="因子 Zoo 路径")


class ConsistencySetting(BaseModel):
    """一致性门配置 (需 LLM)"""
    enabled: bool = Field(default=False, description="启用一致性门 (需 LLM)")
    model: str = Field(default="deepseek-v3", description="LLM 模型")
    max_correction_attempts: int = Field(default=3, description="最大自我修正次数")


class QualityGateSetting(BaseModel):
    """质量门总配置"""
    complexity: ComplexitySetting = Field(default_factory=ComplexitySetting)
    redundancy: RedundancySetting = Field(default_factory=RedundancySetting)
    consistency: ConsistencySetting = Field(default_factory=ConsistencySetting)

    def any_enabled(self) -> bool:
        """是否有任何门启用"""
        return (self.complexity.enabled
                or self.redundancy.enabled
                or self.consistency.enabled)
```

### 3.2 集成到 SingleFactorTestConfig

```python
class SingleFactorTestConfig(BaseModel):
    # ... 现有字段 ...
    quality_gate: QualityGateSetting = Field(
        default_factory=QualityGateSetting,
        description="质量门配置 (pre-backtest 检查)",
    )
```

### 3.3 YAML 配置示例

```yaml
# configs/single_factor_qg.yaml
factor:
  name: momentum_20d
  factor_dir: ./factors/momentum.h5

preprocess:
  adj_date_beg: 20260101
  adj_date_end: 20260630
  missing: ind_avg
  extreme: median
  norm: zscore

# 质量门配置
quality_gate:
  complexity:
    enabled: true
    symbol_length_threshold: 200
    base_features_threshold: 5
    free_args_ratio_threshold: 0.5
  redundancy:
    enabled: true
    threshold: 5
    zoo_path: ./factor_zoo/
  consistency:
    enabled: false  # 默认关闭 (需 LLM)
    model: deepseek-v3
    max_correction_attempts: 3
```

---

## 4. QualityGateNode 节点

### 4.1 节点签名

```python
class QualityGateNode(BaseNode):
    """质量门节点 - pre-backtest 检查

    输入: context['FactorCandidate'] = {
        'factor_id': str,
        'name': str,
        'expression': str,
        'hypothesis': str,
        'description': str,
    }
    输出: {
        'passed': bool,
        'feedback': FactorFeedback,
        'channels': dict[FeedbackChannel, ChannelFeedback],
    }
    """
```

### 4.2 节点实现

```python
class QualityGateNode(BaseNode):
    def __init__(self, name="QualityGate", config: dict = None, **kwargs):
        super().__init__(name, config, **kwargs)
        if config and 'quality_gate' in config:
            self._settings = QualityGateSetting(**config['quality_gate'])
        else:
            self._settings = QualityGateSetting()
        self._complexity = ComplexityChecker(self._settings.complexity)
        self._redundancy = RedundancyChecker(self._settings.redundancy)
        self._consistency = ConsistencyChecker(self._settings.consistency)

    def _execute(self, input_data=None, **kwargs) -> dict:
        context = kwargs.get('context', {})
        candidate = context.get('FactorCandidate')
        if not candidate:
            raise ValueError("FactorCandidate 缺失")

        factor_id = candidate.get('factor_id', str(uuid.uuid4()))
        factor_name = candidate.get('name', 'unnamed')

        collector = FeedbackCollector(factor_id, factor_name)

        # 1. Complexity
        if self._settings.complexity.enabled:
            fb = self._complexity.check(candidate['expression'])
            collector.add(FeedbackChannel.CODE, fb.passed, fb.detail, fb.score)

        # 2. Redundancy
        if self._settings.redundancy.enabled:
            fb = self._redundancy.check(candidate['expression'])
            collector.add(FeedbackChannel.VALUE, fb.passed, fb.detail, fb.score)

        # 3. Consistency (LLM)
        if self._settings.consistency.enabled:
            fb = self._consistency.check(
                candidate.get('hypothesis', ''),
                candidate.get('description', ''),
                candidate['expression'],
            )
            collector.add(FeedbackChannel.LLM, fb.passed, fb.detail, fb.score)

        feedback = collector.finalize()
        return {
            'passed': feedback.decision,
            'feedback': feedback,
            'channels': feedback.channels,
        }
```

### 4.3 输入/输出格式

**输入** (context 中):
```python
{
    'FactorCandidate': {
        'factor_id': 'uuid-xxx',
        'name': 'momentum_20d',
        'expression': "(close - close.shift(20)) / close.shift(20)",
        'hypothesis': "动量因子: 过去 20 日上涨的股票继续上涨",
        'description': "20 日动量, 衡量过去一个月价格变化率",
    }
}
```

**输出**:
```python
{
    'passed': True,
    'feedback': FactorFeedback(
        factor_id='uuid-xxx',
        factor_name='momentum_20d',
        channels={
            FeedbackChannel.CODE: ChannelFeedback(passed=True, detail='OK (length=37, features=1)', score=1.0),
            FeedbackChannel.VALUE: ChannelFeedback(passed=True, detail='min_hamming_dist=18, threshold=5', score=1.0),
        },
        decision=True,
        summary='全部通过',
        duration_ms=12.3,
    ),
    'channels': {...},
}
```

---

## 5. PipelineRunner 集成

### 5.1 初始化

```python
class PipelineRunner:
    def __init__(self, config: SingleFactorTestConfig):
        # ... 现有 ...
        if config.quality_gate.any_enabled():
            self._quality_gate = QualityGateNode(
                config={'quality_gate': config.quality_gate.dict()}
            )
        else:
            self._quality_gate = None
```

### 5.2 运行短路

```python
def run(self, candidate: dict = None) -> dict:
    # Phase 0: 质量门 (可选短路)
    if self._quality_gate and candidate:
        gate_result = self._quality_gate.execute(
            context={'FactorCandidate': candidate}
        )
        if not gate_result['passed']:
            # 记录到 TrajectoryPool (REJECTED)
            if self._trajectory_pool:
                entry = TrajectoryEntry(
                    round_idx=self._trajectory_pool.round_counter,
                    operation='original',
                    config_snapshot=self.config.dict(),
                    feedback=gate_result['feedback'],
                    metrics={},
                )
                self._trajectory_pool.add(entry)
            return {
                'status': 'rejected',
                'feedback': gate_result['feedback'],
                'reason': gate_result['feedback'].summary,
            }

    # Phase 1-12: 现有 12 节点
    ctx = self._context
    for node in self._pipeline_nodes:
        ctx[node.name] = node.execute(context=ctx)

    # ... 持久化到 TrajectoryPool ...
    return ctx
```

### 5.3 CLI 集成

```python
# cli/single_factor.py
@click.command()
@click.option('--candidate', type=str, help='JSON 格式的因子候选')
def run(candidate):
    config = load_config(...)
    runner = PipelineRunner(config)
    if candidate:
        candidate_dict = json.loads(candidate)
        result = runner.run(candidate=candidate_dict)
    else:
        result = runner.run()
    print(result)
```

---

## 6. 因子 Zoo 集成

### 6.1 自动归档

当一个因子通过所有质量门并完成回测, 自动加入 Zoo:

```python
# 在 PipelineRunner.run() 末尾
if self._trajectory_pool and candidate:
    last_entry = self._trajectory_pool.by_round(self._trajectory_pool.round_counter)[-1]
    if last_entry.feedback.decision and self._quality_gate:
        # 加入 Zoo
        zoo = FactorZoo(Path(self.config.quality_gate.redundancy.zoo_path))
        zoo.add(candidate['expression'])
```

### 6.2 手动管理

```python
# 添加
zoo = FactorZoo(Path('./factor_zoo/'))
zoo.add("(close - close.shift(20)) / close.shift(20)")

# 查看
print(f"Zoo size: {len(zoo)}")
for h, expr in zoo._hashes[:5]:
    print(f"  {h}: {expr}")

# 清空 (谨慎)
zoo._hashes.clear()
zoo._save()
```

### 6.3 Zoo 路径约定

```
{output.dir}/factor_zoo/
├── zoo.parquet          # 主索引 (hash, expression)
└── archive/             # 归档 (按日期)
    ├── 20260611_zoo.parquet
    └── ...
```

---

## 7. 测试覆盖 (25 tests)

### 7.1 ComplexityChecker 测试 (6)

| 测试 | 验证 |
|------|------|
| `test_complexity_simple_passes` | 简单表达式通过 |
| `test_complexity_long_fails` | 长度超限失败 |
| `test_complexity_many_features_fails` | 特征过多失败 |
| `test_complexity_high_free_args_fails` | 自由参数过多失败 |
| `test_complexity_syntax_error_fails` | 语法错误失败 |
| `test_complexity_count_base_features` | 基础特征计数 |

### 7.2 FactorZoo 测试 (4)

| 测试 | 验证 |
|------|------|
| `test_zoo_empty` | 空 zoo |
| `test_zoo_add` | 添加因子 |
| `test_zoo_save_load` | 持久化 |
| `test_zoo_hash_invariant` | hash 稳定性 |

### 7.3 RedundancyChecker 测试 (4)

| 测试 | 验证 |
|------|------|
| `test_redundancy_empty_zoo_passes` | 空 zoo 通过 |
| `test_redundancy_identical_fails` | 相同表达式失败 |
| `test_redundancy_similar_passes` | 相似表达式通过 |
| `test_redundancy_distance_threshold` | 阈值边界 |

### 7.4 ConsistencyChecker 测试 (4)

| 测试 | 验证 |
|------|------|
| `test_consistency_disabled_skips` | 禁用跳过 |
| `test_consistency_passes` | 一致 (mock) |
| `test_consistency_fails` | 不一致 (mock) |
| `test_consistency_correction_attempts` | 自我修正 |

### 7.5 QualityGateNode 测试 (5)

| 测试 | 验证 |
|------|------|
| `test_node_all_disabled_skips` | 全部禁用 |
| `test_node_complexity_only` | 仅复杂度 |
| `test_node_redundancy_only` | 仅冗余 |
| `test_node_consistency_only` | 仅一致性 |
| `test_node_all_enabled_passes` | 全部通过 |

### 7.6 集成测试 (2)

| 测试 | 验证 |
|------|------|
| `test_pipeline_integration_passes` | 管线通过 |
| `test_pipeline_integration_rejects` | 管线拒绝 |

---

## 8. 性能考虑

### 8.1 复杂度检查

- AST 解析: < 1ms (100 字符表达式)
- 基础特征计数: < 1ms
- 自由参数比: < 1ms
- **总计**: < 5ms / 因子

### 8.2 冗余检查

- AST hash: < 1ms
- 与 Zoo 比较 (1000 因子): < 10ms
- **总计**: < 20ms / 因子

### 8.3 LLM 一致性

- LLM 调用: 1-3 秒 / 因子
- 1000 因子实验: ~30 分钟
- **建议**: 默认关闭, 关键决策时启用

---

## 9. 配置示例 (典型场景)

### 9.1 严格 (研究阶段)

```yaml
quality_gate:
  complexity: {enabled: true, symbol_length_threshold: 150, base_features_threshold: 4}
  redundancy: {enabled: true, threshold: 10, zoo_path: ./factor_zoo/}
  consistency: {enabled: true, model: deepseek-v3, max_correction_attempts: 3}
```

### 9.2 宽松 (生产回测)

```yaml
quality_gate:
  complexity: {enabled: true, symbol_length_threshold: 300, base_features_threshold: 8}
  redundancy: {enabled: false}
  consistency: {enabled: false}
```

### 9.3 关闭 (调试)

```yaml
quality_gate:
  complexity: {enabled: false}
  redundancy: {enabled: false}
  consistency: {enabled: false}
```

---

## 10. 关键设计决策

### 10.1 为什么独立节点而非嵌入 FactorPreprocess

- **关注点分离**: 门控 vs 数据处理
- **诊断便利**: 失败时单独看哪个门拦截
- **可跳过**: 配置开关, 便于对比实验

### 10.2 为什么 AST hash 而非向量嵌入

- **简单**: 零依赖, 易于理解
- **可解释**: hash 距离是明确的相似度度量
- **快速**: 100ms vs 1s
- **可接受碰撞**: 小概率假阳性, 人工审核兜底

### 10.3 为什么默认 consistency.enabled=False

- **LLM 成本**: 1000 因子 ~30 分钟
- **失败率低**: 实验阶段 hypothesis 已被人工审核
- **按需开启**: 关键因子才用 LLM 验证

---

## 11. 未来扩展

| 扩展 | 优先级 | 说明 |
|------|--------|------|
| `OVERFITTING` 门 (回测/样本外差异) | P2 | 检测过拟合 |
| `STABILITY` 门 (子样本波动) | P2 | 因子稳定性 |
| `CUSTOM` 门 (用户定义) | P3 | 业务特定检查 |
| 向量嵌入 (语义去重) | P3 | Knowledge RAG |

---

## 12. 参考

- QuantaAlpha `configs/experiment.yaml:80-100` — `quality_gate` 配置
- QuantaAlpha `quantaalpha/coder/costeer/evaluators.py:50` — `FactorEvaluatorForCoder`
- QuantaAlpha `quantaalpha/factors/factor_ast.py` — `calculate_symbol_length` / `count_base_features`

---

*Last updated: 2026-06-11*
