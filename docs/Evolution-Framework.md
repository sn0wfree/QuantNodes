# 演化框架设计 / Evolution Framework Design

> 从「单次回测管线」到「多轮演化实验编排器」的架构升级
>
> 借鉴自 QuantaAlpha (arXiv:2602.07085)，适配 QuantNodes 现有 BaseNode + Pipeline 抽象
>
> Version: 1.0  |  Date: 2026-06-11  |  Author: QuantNodes Team

---

## 1. 动机 / Motivation

### 1.1 当前 QuantNodes 的定位

QuantNodes 当前 `research/factor_test/` 是一个 **单因子回测管线**:

```
[LoadData] → [SamplePool] → [Tradability] → [AdjustDate] → [Preprocess]
   → [Neutralize] → [IC] → [Group] → [LongShort] → [Score]
   → [RiskCorr] → [Report]
```

**特点**:
- 12 个独立节点, 通过 Pipeline 组合
- 单次执行, 一次性出结果
- 节点返回 ad-hoc dict, LLM 无法解析
- 没有历史回溯, 没有演化能力

### 1.2 目标: 实验编排器

参考 **QuantaAlpha** (1080 ⭐, arXiv:2602.07085) 的设计, 把 QuantNodes 升级为 **多轮演化实验编排器**:

```
round 0: original     (N 个初始假设 → N 个因子)
round 1: mutation     (扰动最佳父辈 → 派生因子)
round 2: crossover    (组合两个父辈 → 派生因子)
...
```

**核心新增能力**:
1. **结构化反馈**: 5 通道信号 (execution / shape / code / value / llm)
2. **轨迹池**: 持久化每轮实验的完整谱系
3. **质量门**: Pre-backtest 拦截低质量因子 (复杂度 / 冗余 / 一致性)

### 1.3 设计原则

| 原则 | 体现 |
|------|------|
| **不破坏现有节点** | 新模块独立存在, 节点返回 dict 自动包装成 `FactorFeedback` |
| **单一职责** | 反馈是数据, 轨迹是存储, 质量门是节点 — 互不耦合 |
| **持久化优先** | 所有中间结果存 Parquet + JSON, 跨会话可恢复 |
| **LLM 友好** | 反馈结构化, 轨迹可序列化, 质量门可解释 |
| **配置驱动** | 演化轮次、选择策略、质量门阈值全部 YAML/Pydantic 配置 |

---

## 2. 架构总览 / Architecture

### 2.1 三组件关系

```
┌──────────────────────────────────────────────────────────────────┐
│                    Experiment Orchestrator                         │
│                                                                    │
│  ┌────────────┐    ┌──────────────┐    ┌────────────────┐        │
│  │ Quality    │───▶│   Pipeline   │───▶│  Trajectory    │        │
│  │ Gate       │    │   Runner     │    │  Pool          │        │
│  │ (pre-check)│    │  (12 nodes)  │    │  (persist)     │        │
│  └────────────┘    └──────────────┘    └────────────────┘        │
│        │                  │                      │                  │
│        ▼                  ▼                      ▼                  │
│  ┌────────────────────────────────────────────────────┐          │
│  │              FactorFeedback (5 channels)             │          │
│  │  execution │ shape │ code │ value │ llm             │          │
│  └────────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
FactorCandidate (hypothesis + description + expression)
       │
       ▼
[QualityGate] ──passed=false──▶ REJECTED (记录到 TrajectoryPool, decision=False)
       │ passed=true
       ▼
[PipelineRunner] 12 节点
       │
       ▼
backtest result (IC, Sharpe, ARR, ...)
       │
       ▼
[FactorFeedback] (5 通道聚合)
       │
       ▼
[TrajectoryEntry] (config + context + feedback + parent_ids)
       │
       ▼
[TrajectoryPool] 持久化 (Parquet + JSON)
       │
       ▼
[ParentSelector] best/random/weighted → 下一轮父辈
```

### 2.3 文件结构

```
QuantNodes/
├── core/
│   ├── feedback.py              # FactorFeedback 数据类
│   └── stop_event.py            # StopEvent + @measure_time
│
└── research/factor_test/
    ├── config.py                # 扩展: QualityGateSetting
    ├── pipeline_runner.py       # 扩展: 集成质量门
    │
    ├── feedback/                # NEW
    │   ├── __init__.py
    │   ├── channels.py          # 5 通道采集
    │   ├── collector.py         # FeedbackCollector
    │   └── llm_judge.py         # LLMJudge
    │
    ├── trajectory/              # NEW
    │   ├── __init__.py
    │   ├── entry.py             # TrajectoryEntry
    │   ├── pool.py              # TrajectoryPool
    │   ├── selector.py          # 5 种选择策略
    │   ├── lineage.py           # 谱系追踪
    │   └── storage.py          # Parquet/JSON 持久化
    │
    ├── quality_gate/            # NEW
    │   ├── __init__.py
    │   ├── node.py              # QualityGateNode
    │   ├── complexity.py        # AST 静态检查
    │   ├── redundancy.py        # AST hash 去重
    │   ├── consistency.py       # LLM 一致性
    │   └── zoo.py               # 因子 Zoo
    │
    └── tests/
        ├── test_feedback.py     # 20 tests
        ├── test_trajectory.py   # 30 tests
        └── test_quality_gate.py # 25 tests
```

---

## 3. 三组件详细规格

### 3.1 FactorFeedback — 结构化反馈

**目的**: 把节点返回的 ad-hoc dict 转为 LLM 可解析的结构化信号。

**5 通道**:
| 通道 | 采集内容 | 触发时机 |
|------|---------|----------|
| `execution` | 沙箱 stdout/stderr, exit code | 因子代码执行后 |
| `shape` | 输出形状 vs 预期 | 执行完成后 |
| `code` | AST 长度, 基础特征数, 自由参数比例 | 执行前 |
| `value` | NaN 比例, Inf 数量, 均值/标准差 | 执行完成后 |
| `llm` | hypothesis ↔ description ↔ expression 一致性 | 可选, 执行前 |

**数据结构** (见 `docs/FactorFeedback.md`):

```python
@dataclass
class ChannelFeedback:
    channel: FeedbackChannel
    passed: bool
    detail: str
    score: float = 1.0
    metadata: dict = field(default_factory=dict)

@dataclass
class FactorFeedback:
    factor_id: str
    factor_name: str
    channels: dict[FeedbackChannel, ChannelFeedback]
    decision: bool                              # 全部通道通过
    summary: str                                # 一句话总结
    timestamp: datetime
    duration_ms: float
    metadata: dict
```

**Parquet 序列化** (每条反馈一行):

```
factor_id | factor_name | decision | summary | duration_ms | timestamp
exec_passed | exec_score | exec_detail
shape_passed | shape_score | shape_detail
code_passed | code_score | code_detail
value_passed | value_score | value_detail
llm_passed | llm_score | llm_detail
```

### 3.2 TrajectoryPool — 演化轨迹池

**目的**: 持久化每轮实验的完整记录, 支持演化控制器追溯 + 父辈选择。

**数据结构** (见 `docs/TrajectoryPool.md`):

```python
@dataclass
class TrajectoryEntry:
    entry_id: str                               # UUID
    round_idx: int                              # 0=原始, 1=mutation, 2=crossover
    operation: str                              # 'original'|'mutation'|'crossover'
    config_snapshot: dict                       # 配置快照
    context_subset: dict                        # 关键 context (Parquet-able)
    feedback: FactorFeedback
    parent_ids: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict) # IC, Sharpe, ARR
    timestamp: datetime
```

**存储格式**:
- **元数据** → `trajectories.parquet` (一张大表, 追加写入)
- **完整 context/feedback** → `{entry_id}.json` (单文件, 易于调试)

**5 种选择策略** (QuantaAlpha `parent_selection_strategy`):
| 策略 | 行为 |
|------|------|
| `best` | 按指标 (默认 Sharpe) 排序取 Top-N |
| `random` | 从有效池随机抽取 |
| `weighted` | 按指标 softmax 加权采样 |
| `weighted_inverse` | 反向加权, 鼓励多样性 |
| `top_percent_plus_random` | Top 30% + 随机补足 |

**谱系追踪**:
- `children_of(parent_id)` → 该父辈的所有子代
- `lineage(entry_id)` → 从原始到当前的完整路径

### 3.3 QualityGate — 质量门

**目的**: 在回测前拦截低质量因子, 避免浪费计算资源。

**3 个门** (QuantaAlpha `quality_gate` 配置):

| 门 | 检查内容 | 触发条件 | 实现 |
|----|----------|----------|------|
| `complexity` | AST 长度 ≤ 200, 基础特征 ≤ 5, 自由参数 ≤ 50% | 表达式提交时 | AST 静态分析 |
| `redundancy` | 与因子 Zoo 的 AST 哈希距离 ≥ 5 | 提交时 | hash 比对 |
| `consistency` | hypothesis ↔ description ↔ expression 一致 | 可选, 需 LLM | LLM 评判 |

**QualityGateSetting** (见 `docs/QualityGate.md`):

```python
class ComplexitySetting(BaseModel):
    enabled: bool = True
    symbol_length_threshold: int = 200
    base_features_threshold: int = 5
    free_args_ratio_threshold: float = 0.5

class RedundancySetting(BaseModel):
    enabled: bool = True
    threshold: int = 5
    zoo_path: Optional[str] = None

class ConsistencySetting(BaseModel):
    enabled: bool = False
    model: str = "deepseek-v3"
    max_correction_attempts: int = 3

class QualityGateSetting(BaseModel):
    complexity: ComplexitySetting = Field(default_factory=ComplexitySetting)
    redundancy: RedundancySetting = Field(default_factory=RedundancySetting)
    consistency: ConsistencySetting = Field(default_factory=ConsistencySetting)
```

**QualityGateNode** 集成到 PipelineRunner:

```python
# pipeline_runner.py
if self.quality_gate_node and not self._run_quality_gate(ctx):
    return self._build_rejected_result()
```

---

## 4. 配置扩展 / Configuration

### 4.1 SingleFactorTestConfig 新增字段

```python
class SingleFactorTestConfig(BaseModel):
    # ... 现有字段 (factor / preprocess / analysis / output / data_path / load_keys) ...

    # NEW: 演化配置 (Phase 2)
    evolution: Optional[EvolutionSetting] = None

    # NEW: 质量门 (Phase 3)
    quality_gate: QualityGateSetting = Field(default_factory=QualityGateSetting)
```

### 4.2 EvolutionSetting (Phase 2 - 暂未实施)

```python
class EvolutionSetting(BaseModel):
    enabled: bool = False
    max_rounds: int = 3
    mutation_enabled: bool = True
    crossover_enabled: bool = True
    crossover_size: int = 2          # 2 parents → 1 child
    crossover_n: int = 2             # 每轮 crossovers 数
    parent_selection_strategy: Literal['best','random','weighted',
                                         'weighted_inverse',
                                         'top_percent_plus_random'] = 'best'
    top_percent_threshold: float = 0.3
    prefer_diverse_crossover: bool = True
```

### 4.3 YAML 示例

```yaml
# configs/single_factor_evol.yaml
factor:
  name: momentum_20d
  factor_dir: ./factors/momentum.h5
preprocess:
  adj_date_beg: 20260101
  adj_date_end: 20260630
  adj_mode: [M, end]
  missing: ind_avg
  extreme: median
  norm: zscore
  industry_neutral: true

analysis:
  ic: {min_group_size: 5}
  group: {groups: 5, factor_direction: 1, floor_mode: group, hedge: equal}
  longshort: {factor_direction: 1}

# NEW: 质量门
quality_gate:
  complexity:
    enabled: true
    symbol_length_threshold: 200
    base_features_threshold: 5
  redundancy:
    enabled: true
    threshold: 5
    zoo_path: ./factor_zoo/
  consistency:
    enabled: true
    model: deepseek-v3
    max_correction_attempts: 3

# NEW: 演化 (Phase 2)
evolution:
  enabled: true
  max_rounds: 3
  parent_selection_strategy: top_percent_plus_random
  top_percent_threshold: 0.3
```

---

## 5. 与现有系统的集成

### 5.1 节点返回值的兼容性

现有节点返回 ad-hoc dict, 新系统需要 `FactorFeedback`。解决方案: **自动包装**。

```python
# core/feedback.py
def ensure_feedback(result, factor_id, factor_name) -> FactorFeedback:
    """如果 result 不是 FactorFeedback, 自动从 dict 提取"""
    if isinstance(result, FactorFeedback):
        return result
    if isinstance(result, dict):
        return FactorFeedback.from_dict(result, factor_id, factor_name)
    raise TypeError(f"节点返回类型不支持: {type(result)}")
```

### 5.2 PipelineRunner 改造

最小侵入式集成:

```python
class PipelineRunner:
    def __init__(self, config: SingleFactorTestConfig):
        # ... 现有 ...
        self._quality_gate = QualityGateNode(
            config={'quality_gate': config.quality_gate.dict()}
        ) if config.quality_gate.enabled() else None
        self._trajectory_pool = TrajectoryPool(
            Path(config.output.dir) / 'trajectory'
        )

    def run(self, candidate: dict = None) -> dict:
        # Phase 0: 质量门
        if self._quality_gate and candidate:
            gate_result = self._quality_gate.execute(
                context={'FactorCandidate': candidate}
            )
            if not gate_result['passed']:
                return {'status': 'rejected', 'feedback': gate_result['feedback']}

        # Phase 1-12: 现有 12 节点
        ctx = self._context
        for node in self._pipeline_nodes:
            ctx[node.name] = node.execute(context=ctx)

        # Phase 13: 持久化到 TrajectoryPool
        feedback = self._build_feedback_from_ctx(ctx)
        entry = TrajectoryEntry(
            round_idx=self._trajectory_pool._round_counter,
            operation='original',
            config_snapshot=self.config.dict(),
            context_subset=self._extract_context_subset(ctx),
            feedback=feedback,
            metrics=self._extract_metrics(ctx),
        )
        self._trajectory_pool.add(entry)

        return ctx
```

### 5.3 演化循环主控 (Phase 2 - 暂未实施)

```python
# research/evolution/loop.py (未来)
class EvolutionLoop:
    def run(self, initial_directions: list[str]) -> list[TrajectoryEntry]:
        pool = TrajectoryPool(...)
        selector = ParentSelector(strategy='best')

        # round 0: 原始
        for direction in initial_directions:
            candidate = self._hypothesize(direction)
            self._run_one_round(pool, candidate, operation='original')

        for round_idx in range(1, self.config.evolution.max_rounds + 1):
            operation = 'mutation' if round_idx % 2 == 1 else 'crossover'
            parents = selector.select(pool, n=2 if operation == 'crossover' else 1)

            if operation == 'mutation':
                child = self._mutate(parents[0])
                self._run_one_round(pool, child, operation, parents=[parents[0]])
            else:
                child = self._crossover(parents[0], parents[1])
                self._run_one_round(pool, child, operation, parents=parents)

        return pool.best(top_n=10)
```

---

## 6. 实施路线图 / Roadmap

### 6.1 Phase 1: FactorFeedback (Week 1)

| Day | 工作 | 交付 |
|-----|------|------|
| 1 | `core/feedback.py` skeleton + 第一个测试 | dataclass 框架 |
| 2-3 | 5 通道采集器 (`channels.py`) | execution/shape/code/value |
| 4 | `FeedbackCollector` 聚合器 | 自动化决策 |
| 5 | LLMJudge (consistency 通道) | hypothesis 验证 |
| 6-7 | 20 个测试 + 与现有节点集成验证 | 20 tests pass |

**里程碑**: 节点可返回 `FactorFeedback`, 现有 12 节点不破坏。

### 6.2 Phase 2: TrajectoryPool (Week 2)

| Day | 工作 | 交付 |
|-----|------|------|
| 1 | `TrajectoryEntry` + Parquet 持久化 | 基础数据类 |
| 2 | `TrajectoryPool` 主类 (add/get/by_round/best) | CRUD |
| 3 | 5 种 `ParentSelector` 策略 | 选择器 |
| 4 | 谱系追踪 (`lineage`/`children_of`) | 演化链 |
| 5-6 | 30 个测试 + 集成到 PipelineRunner | 30 tests pass |
| 7 | 文档 + 演示脚本 | demo.py |

**里程碑**: PipelineRunner 自动记录每轮结果, 可追溯。

### 6.3 Phase 3: QualityGate (Week 3)

| Day | 工作 | 交付 |
|-----|------|------|
| 1 | `ComplexityChecker` + 单元测试 | AST 静态检查 |
| 2 | `FactorZoo` + `RedundancyChecker` | hash 去重 |
| 3 | `ConsistencyChecker` (包装 LLMJudge) | LLM 一致性 |
| 4 | `QualityGateNode` 主类 | 节点化 |
| 5 | 集成到 `PipelineRunner` | 短路逻辑 |
| 6-7 | 25 个测试 + 演示 | 25 tests pass |

**里程碑**: 提交低质量因子时 PipelineRunner 直接 REJECT, 不进入回测。

### 6.4 Phase 4: 演化主循环 (Future - Q3 2026)

- 演化主控 (`EvolutionLoop`)
- LLM-based mutation/crossover operators
- Knowledge RAG 检索
- 多进程评估池

---

## 7. 测试覆盖

| 模块 | 测试数 | 类型 | 优先级 |
|------|--------|------|--------|
| FactorFeedback | 20 | 单元 + 序列化 | P0 |
| TrajectoryPool | 30 | 单元 + 持久化 + 选择策略 | P0 |
| QualityGate | 25 | 单元 + AST + LLM mock + 集成 | P0 |
| **总计** | **75** | 现有 92 → 总 **167** | |

---

## 8. 关键设计决策

### 8.1 为什么独立 `FactorFeedback` 而非继承 BaseNode

- 反馈是 **数据**, 不是节点 — 节点应**返回**反馈, 而非**是**反馈
- 反馈可被 LLM 解析、持久化、网络传输 — 节点不能
- 现有 12 节点不破坏, 自动包装

### 8.2 为什么用 Parquet 而非 SQLite

- 与 QuantNodes 现有约定一致 (因子测试模块已用 Parquet)
- 依赖少, 部署简单
- 列式存储对分析查询友好 (IC 序列、sharpe 分布等)
- 未来可平滑迁移到 DuckDB/Parquet-as-DB

### 8.3 为什么 AST hash 而非向量嵌入

- 简单、可解释、零依赖
- 哈希碰撞可接受 (小概率假阳性)
- 向量嵌入留作 Phase 4 RAG 检索

### 8.4 为什么质量门独立节点而非嵌入 FactorPreprocess

- 节点关注点分离: 门控 vs 数据处理
- 失败时可单独诊断哪个门拦截
- 便于跳过质量门 (配置开关) 进行回测对比

---

## 9. 与 QuantaAlpha 的差异

| 维度 | QuantaAlpha | QuantNodes v2 (本设计) |
|------|-------------|------------------------|
| **核心抽象** | `Experiment` + `EvolvingItem` | `BaseNode` + `Pipeline` (沿用) |
| **数据格式** | H5 + Qlib | H5 + Parquet + iFinD API |
| **配置** | Flat YAML | Pydantic (沿用) |
| **演化循环** | 5 步 (propose→construct→calc→backtest→feedback) | 沿用 12 节点 + 质量门 |
| **LLM 集成** | 双模型 (chat + reasoning) | 沿用现有 LLMProvider |
| **存储** | Pickled KnowledgeBase | Parquet + JSON |
| **质量门** | 3 个独立可配 | 同 |

**关键设计差异**:
- 我们沿用 `BaseNode` 抽象, 不引入新层次
- 我们支持真实数据 (iFinD) + 合成数据双模式
- 我们用 Pydantic 配置, 更严格的类型检查
- 我们保持单进程优先, 多进程作为未来 P2 优化

---

## 10. 风险 & 缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| LLM 调用慢 → 阻塞测试 | 中 | 用 stub + fixture, `@pytest.mark.slow` 跳过 |
| AST hash 碰撞 | 低 | hash + 表达式双重存储, 接受小概率假阳性 |
| TrajectoryPool 写入并发 | 中 | `threading.Lock` 保护 Parquet append |
| 质量门过严 → 误杀 | 高 | 所有阈值可配置, 默认宽松 |
| 现有节点未返回 FactorFeedback | 中 | 自动包装, 双轨兼容 |
| Parquet 文件无限增长 | 低 | 定期归档 (按实验 ID 分目录) |
| Evolution 状态恢复复杂 | 高 | 状态化设计: TrajectoryPool 自带 `_load()` |

---

## 11. 参考资料 / References

1. **QuantaAlpha** (arXiv:2602.07085) — https://github.com/QuantaAlpha/QuantaAlpha
   - `quantaalpha/pipeline/loop.py` — 5 步主循环
   - `quantaalpha/coder/costeer/evaluators.py:10` — `CoSTEERSingleFeedback` 5 通道
   - `quantaalpha/coder/costeer/evolving_strategy.py:23` — `MultiProcessEvolvingStrategy`
   - `quantaalpha/coder/costeer/knowledge_management.py` — KnowledgeBase
   - `configs/experiment.yaml:42-77` — `evolution` 配置
   - `configs/experiment.yaml:80-100` — `quality_gate` 配置

2. **RD-Agent** — QuantaAlpha 的 fork 前身

3. **AlphaAgent** — 同领域工作

---

## 12. 附录: 完整文件清单

### 新增文件 (Week 1-3)

```
QuantNodes/
├── core/
│   ├── feedback.py              (150 LOC)
│   └── stop_event.py            (80 LOC)
└── research/factor_test/
    ├── feedback/                (300 LOC)
    │   ├── __init__.py
    │   ├── channels.py
    │   ├── collector.py
    │   └── llm_judge.py
    ├── trajectory/              (500 LOC)
    │   ├── __init__.py
    │   ├── entry.py
    │   ├── pool.py
    │   ├── selector.py
    │   ├── lineage.py
    │   └── storage.py
    ├── quality_gate/            (400 LOC)
    │   ├── __init__.py
    │   ├── node.py
    │   ├── complexity.py
    │   ├── redundancy.py
    │   ├── consistency.py
    │   └── zoo.py
    └── tests/
        ├── test_feedback.py     (250 LOC, 20 tests)
        ├── test_trajectory.py   (350 LOC, 30 tests)
        └── test_quality_gate.py (300 LOC, 25 tests)
```

### 修改文件

```
QuantNodes/research/factor_test/
├── config.py                    (+30 LOC, 新增 QualityGateSetting)
├── pipeline_runner.py           (+50 LOC, 集成质量门 + 轨迹池)
└── README.md                    (+update, 引用新文档)
```

**总计**: ~13 新文件, ~1900 LOC, 75 新测试。

---

*Last updated: 2026-06-11*
