# TrajectoryPool — 演化轨迹池规格

> QuantaAlpha `Trace.hist` 的 QuantNodes 适配版本
>
> Version: 1.0  |  Date: 2026-06-11

---

## 1. 概述

### 1.1 问题

当前 `PipelineRunner._context` 是 dict:
- Transient (执行结束即丢失)
- 没有谱系 (无法追溯"这个因子是怎么来的")
- 没有选择策略 (无法"从历史中挑最好的")
- 没有持久化 (跨实验无法对比)

### 1.2 目标

**TrajectoryPool** 持久化每轮实验的完整记录:
- 谱系追踪 (parent → child)
- 5 种选择策略 (best/random/weighted/weighted_inverse/top_percent_plus_random)
- Parquet + JSON 双重存储
- 跨会话可恢复
- LLM 可解析 (供未来 evolution controller 消费)

---

## 2. 数据结构

### 2.1 TrajectoryEntry

```python
@dataclass
class TrajectoryEntry:
    """单条演化轨迹"""
    entry_id: str                               # UUID
    round_idx: int                              # 0=原始, 1=mutation, 2=crossover
    operation: str                              # 'original'|'mutation'|'crossover'
    config_snapshot: dict                       # 配置快照 (JSON-safe)
    context_subset: dict                        # 关键 context (Parquet-able)
    feedback: FactorFeedback                    # 完整反馈
    parent_ids: list[str] = field(default_factory=list)  # 父辈 ID
    metrics: dict = field(default_factory=dict) # 评估指标 (IC, Sharpe, ARR, MDD)
    timestamp: datetime = field(default_factory=datetime.now)
```

### 2.2 字段语义

| 字段 | 类型 | 说明 |
|------|------|------|
| `entry_id` | UUID | 唯一标识, 跨会话稳定 |
| `round_idx` | int | 演化轮次, 0=原始, 1=mutation, 2=crossover |
| `operation` | str | 操作类型, 用于过滤 |
| `config_snapshot` | dict | SingleFactorTestConfig 的 dict 形式 |
| `context_subset` | dict | 关键 context (factor 数据、IC 序列等) |
| `feedback` | FactorFeedback | 5 通道反馈 (见 FactorFeedback.md) |
| `parent_ids` | list[str] | 父辈 entry_id, 演化控制用 |
| `metrics` | dict | IC, Rank IC, Sharpe, ARR, MDD, Calmar |
| `timestamp` | datetime | 实验时间 |

### 2.3 谱系关系

```
round 0:                    round 1:                  round 2:
   ┌────┐                   ┌────┐                    ┌────┐
   │ A0 │ ──────────────▶   │ A1 │ ─────┐            │ A2 │
   └────┘                   └────┘       │            └────┘
                                          │
   ┌────┐                   ┌────┐       │       ┌───▶ │
   │ B0 │ ──────────────▶   │ B1 │ ──────┴──────▶│ C2 │
   └────┘                   └────┘                └────┘
                                                        │
   ┌────┐                                              │
   │ C0 │ ──────────────────────────────────────────────┘
   └────┘

A1.parent_ids = [A0]
A2.parent_ids = [A1]
B1.parent_ids = [B0]
C2.parent_ids = [B1, A1]  (crossover)
C2.children = []
```

---

## 3. 存储格式

### 3.1 双层存储

**Layer 1: `trajectories.parquet`** (元数据表, 一行一条)
- 路径: `{base_dir}/trajectories.parquet`
- Schema: 见下表
- 写入模式: append (累积)
- 用途: 快速分析、过滤、统计

**Layer 2: `{entry_id}.json`** (完整记录)
- 路径: `{base_dir}/{entry_id}.json`
- 内容: 完整 config + context + feedback + metrics
- 写入模式: 单文件, 易于调试
- 用途: 深度分析、单条记录恢复

### 3.2 Parquet Schema

| 列 | 类型 | 说明 |
|----|------|------|
| `entry_id` | str | UUID |
| `round_idx` | int | 演化轮次 |
| `operation` | str | original/mutation/crossover |
| `parent_ids` | str | 逗号分隔 |
| `decision` | bool | 反馈通过/失败 |
| `duration_ms` | float | 执行耗时 |
| `ic_mean` | float | IC 均值 (可空) |
| `rank_ic_mean` | float | Rank IC 均值 (可空) |
| `sharpe` | float | 多空 Sharpe (可空) |
| `arr` | float | 年化收益 (可空) |
| `mdd` | float | 最大回撤 (可空) |
| `calmar` | float | Calmar 比率 (可空) |
| `timestamp` | str | ISO 格式 |
| `factor_name` | str | 因子名 |
| `summary` | str | 反馈总结 |

### 3.3 JSON 完整结构

```json
{
  "entry_id": "uuid-xxx",
  "round_idx": 0,
  "operation": "original",
  "config_snapshot": {
    "factor": {"name": "momentum_20d", "factor_dir": "..."},
    "preprocess": {"adj_date_beg": 20260101, ...},
    "analysis": {...}
  },
  "feedback": {
    "factor_id": "...",
    "factor_name": "momentum_20d",
    "channels": {...},
    "decision": true,
    "summary": "全部通过",
    "duration_ms": 1234.5,
    "metadata": {}
  },
  "context_subset": {
    "factor": "base64-encoded-parquet-or-pickle",
    "ic_series": "..."
  },
  "metrics": {
    "ic_mean": 0.05,
    "rank_ic_mean": 0.04,
    "sharpe": 1.2,
    "arr": 0.18,
    "mdd": 0.08,
    "calmar": 2.25
  },
  "parent_ids": [],
  "timestamp": "2026-06-11T10:30:00"
}
```

---

## 4. TrajectoryPool 主类

### 4.1 初始化

```python
class TrajectoryPool:
    """演化轨迹池"""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, TrajectoryEntry] = {}
        self._round_counter = 0
        self._parquet_path = self.base_dir / 'trajectories.parquet'
        self._lock = threading.Lock()
        self._load()
```

### 4.2 核心 API

```python
# CRUD
def add(self, entry: TrajectoryEntry) -> None
def get(self, entry_id: str) -> TrajectoryEntry
def by_round(self, round_idx: int) -> list[TrajectoryEntry]
def by_operation(self, operation: str) -> list[TrajectoryEntry]
def all(self) -> list[TrajectoryEntry]

# 选择
def best(self, top_n: int = 5, metric: str = 'sharpe') -> list[TrajectoryEntry]
def random(self, n: int) -> list[TrajectoryEntry]
def filter(self, decision: bool = None) -> list[TrajectoryEntry]

# 谱系
def children_of(self, parent_id: str) -> list[TrajectoryEntry]
def lineage(self, entry_id: str) -> list[TrajectoryEntry]

# 状态
@property
def round_counter(self) -> int
@property
def size(self) -> int
def reset(self) -> None
```

### 4.3 持久化

```python
def _persist(self, entry: TrajectoryEntry) -> None:
    """持久化单条轨迹"""
    with self._lock:
        # Layer 1: Parquet append
        row = entry.to_parquet_row()
        df = pd.DataFrame([row])
        if self._parquet_path.exists():
            existing = pd.read_parquet(self._parquet_path)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_parquet(self._parquet_path, index=False)
        else:
            df.to_parquet(self._parquet_path, index=False)

        # Layer 2: JSON single file
        json_path = self.base_dir / f'{entry.entry_id}.json'
        with open(json_path, 'w') as f:
            json.dump(self._entry_to_json_dict(entry), f, indent=2, default=str)
```

### 4.4 启动时加载

```python
def _load(self) -> None:
    """启动时加载已有轨迹 (仅元数据)"""
    if not self._parquet_path.exists():
        return
    df = pd.read_parquet(self._parquet_path)
    for _, row in df.iterrows():
        # 加载元数据
        json_path = self.base_dir / f'{row["entry_id"]}.json'
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            entry = self._json_to_entry(data)
            self._entries[entry.entry_id] = entry
    # 更新轮次计数器
    if self._entries:
        self._round_counter = max(e.round_idx for e in self._entries.values())
```

---

## 5. 5 种选择策略

### 5.1 策略枚举

```python
class SelectionStrategy(str, Enum):
    BEST = "best"
    RANDOM = "random"
    WEIGHTED = "weighted"
    WEIGHTED_INVERSE = "weighted_inverse"
    TOP_PERCENT_PLUS_RANDOM = "top_percent_plus_random"
```

### 5.2 行为对比

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `best` | 按指标 (默认 Sharpe) 排序取 Top-N | 收敛到最优解 |
| `random` | 从有效池随机抽取 | 多样性探索 |
| `weighted` | 按指标 softmax 加权采样 | 平衡收敛与探索 |
| `weighted_inverse` | 反向加权 (差的多采) | 跳出局部最优 |
| `top_percent_plus_random` | Top 30% + 随机补足 | QuantaAlpha 默认 |

### 5.3 ParentSelector 实现

```python
class ParentSelector:
    """父辈选择策略"""

    def __init__(self, strategy: str = 'best', metric: str = 'sharpe',
                 top_percent_threshold: float = 0.3, seed: int = None):
        assert strategy in [s.value for s in SelectionStrategy]
        self.strategy = strategy
        self.metric = metric
        self.top_percent_threshold = top_percent_threshold
        self._rng = np.random.RandomState(seed)

    def select(self, pool: TrajectoryPool, n: int = 1) -> list[TrajectoryEntry]:
        valid = [e for e in pool.all() if e.feedback.decision]
        if not valid:
            return []

        if self.strategy == 'best':
            return pool.best(top_n=n, metric=self.metric)
        elif self.strategy == 'random':
            indices = self._rng.choice(len(valid), size=min(n, len(valid)), replace=False)
            return [valid[i] for i in indices]
        elif self.strategy == 'weighted':
            return self._weighted_sample(valid, n, inverse=False)
        elif self.strategy == 'weighted_inverse':
            return self._weighted_sample(valid, n, inverse=True)
        elif self.strategy == 'top_percent_plus_random':
            top_n = max(1, int(len(valid) * self.top_percent_threshold))
            top = pool.best(top_n=top_n, metric=self.metric)
            if n <= top_n:
                return top[:n]
            remaining = [e for e in valid if e not in top]
            extra = self._rng.choice(len(remaining), size=min(n - top_n, len(remaining)), replace=False)
            return top + [remaining[i] for i in extra]

    def _weighted_sample(self, valid, n, inverse=False):
        scores = np.array([e.metrics.get(self.metric, 0) for e in valid])
        if inverse:
            scores = -scores  # 反向
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        indices = self._rng.choice(len(valid), size=min(n, len(valid)),
                                   replace=False, p=weights)
        return [valid[i] for i in indices]
```

---

## 6. 谱系追踪

### 6.1 父辈 → 子代

```python
def children_of(self, parent_id: str) -> list[TrajectoryEntry]:
    """返回指定父辈的所有子代"""
    return [e for e in self._entries.values() if parent_id in e.parent_ids]
```

### 6.2 完整谱系 (从原始到当前)

```python
def lineage(self, entry_id: str) -> list[TrajectoryEntry]:
    """返回完整谱系 (从原始到当前), 顺序: 最老 → 最新"""
    if entry_id not in self._entries:
        return []
    lineage = []
    current = self._entries[entry_id]
    lineage.insert(0, current)

    # 向上追溯 (BFS 防环)
    visited = {current.entry_id}
    while current.parent_ids:
        # 优先选第一个父辈 (crossover 时多父辈)
        next_parent_id = current.parent_ids[0]
        if next_parent_id in visited or next_parent_id not in self._entries:
            break
        current = self._entries[next_parent_id]
        lineage.insert(0, current)
        visited.add(current.entry_id)

    return lineage
```

### 6.3 谱系可视化 (未来)

```
DAG (有向无环图):
         [A0]──┐
              ├──▶ [A1]──┐
         [B0]──┤        ├──▶ [C2]
              ├──▶ [B1]──┘
         [C0]─────────────┘

JSON 输出:
{
  "entry_id": "C2",
  "lineage": [
    {"entry_id": "A0", "round": 0, "operation": "original"},
    {"entry_id": "A1", "round": 1, "operation": "mutation", "parent": "A0"},
    {"entry_id": "B1", "round": 1, "operation": "mutation", "parent": "B0"},
    {"entry_id": "C2", "round": 2, "operation": "crossover", "parents": ["A1", "B1"]}
  ]
}
```

---

## 7. 集成到 PipelineRunner

### 7.1 初始化

```python
class PipelineRunner:
    def __init__(self, config: SingleFactorTestConfig):
        # ... 现有 ...
        self._trajectory_pool = TrajectoryPool(
            Path(config.output.dir) / 'trajectory' / config.experiment_id
        )
```

### 7.2 运行后持久化

```python
def run(self, candidate: dict = None) -> dict:
    # ... 现有 12 节点执行 ...
    ctx = self._context

    # 提取 metrics
    metrics = self._extract_metrics(ctx)  # IC, Sharpe, ARR, etc.

    # 构造 feedback
    feedback = self._build_feedback_from_ctx(ctx)

    # 构造 entry
    entry = TrajectoryEntry(
        entry_id=str(uuid.uuid4()),
        round_idx=self._trajectory_pool.round_counter,
        operation='original',
        config_snapshot=self.config.dict(),
        context_subset=self._extract_context_subset(ctx),
        feedback=feedback,
        metrics=metrics,
    )
    self._trajectory_pool.add(entry)

    return ctx
```

### 7.3 实验隔离

每个 experiment_id 一个独立目录:

```
output/
├── trajectory/
│   ├── exp_001/
│   │   ├── trajectories.parquet
│   │   ├── uuid-001.json
│   │   ├── uuid-002.json
│   │   └── ...
│   ├── exp_002/
│   └── ...
```

---

## 8. 测试覆盖 (30 tests)

### 8.1 单元测试 (10)

| 测试 | 验证 |
|------|------|
| `test_entry_dataclass_basic` | 基础创建 |
| `test_entry_to_parquet_row` | Parquet 转换 |
| `test_pool_create_empty` | 空 pool 初始化 |
| `test_pool_add_single` | 添加单条 |
| `test_pool_add_multiple` | 批量添加 |
| `test_pool_by_round` | 按轮次过滤 |
| `test_pool_by_operation` | 按操作过滤 |
| `test_pool_reset` | 重置 |
| `test_pool_size_property` | size 属性 |
| `test_pool_round_counter` | 轮次计数器 |

### 8.2 选择 API 测试 (8)

| 测试 | 验证 |
|------|------|
| `test_pool_best_ordering` | best 排序 |
| `test_pool_best_empty_metric` | 缺指标处理 |
| `test_pool_filter_decision_true` | 过滤通过 |
| `test_pool_filter_decision_false` | 过滤失败 |
| `test_pool_random_n` | 随机抽取 |
| `test_pool_all_iteration` | 遍历所有 |
| `test_pool_get_existing` | 获取存在 |
| `test_pool_get_missing_raises` | 缺失异常 |

### 8.3 选择策略测试 (5)

| 测试 | 验证 |
|------|------|
| `test_selector_best` | best 策略 |
| `test_selector_random_distribution` | 随机分布 |
| `test_selector_weighted_distribution` | 加权分布 |
| `test_selector_weighted_inverse` | 反向加权 |
| `test_selector_top_percent_plus_random` | Top + 随机 |

### 8.4 谱系测试 (4)

| 测试 | 验证 |
|------|------|
| `test_children_of_single` | 单子代 |
| `test_children_of_multiple` | 多子代 (crossover) |
| `test_lineage_chain` | 单链谱系 |
| `test_lineage_branch` | 树状谱系 |
| `test_lineage_orphan` | 孤儿节点 |

### 8.5 持久化测试 (3)

| 测试 | 验证 |
|------|------|
| `test_persist_reload_roundtrip` | 持久化-恢复 |
| `test_parquet_schema` | Schema 正确性 |
| `test_concurrent_writes` | 并发安全 (Lock) |

---

## 9. 性能考虑

### 9.1 写入性能

- 单条写入 (Parquet append): ~5ms
- 1000 条累积: ~5s (一次性 reload)
- 建议: 批量提交, 或用 `pyarrow` 原生 append

### 9.2 读取性能

- Parquet 读取 10,000 行: < 100ms
- JSON 单文件读取: < 10ms
- 全量加载到内存: 适合 < 50,000 条

### 9.3 存储开销

- 单条 JSON: ~5 KB
- 1000 条实验: ~5 MB JSON + ~500 KB Parquet
- 10,000 条实验: ~50 MB JSON + ~5 MB Parquet
- **建议**: 定期归档 (按 experiment_id 分目录)

---

## 10. 未来扩展

| 扩展 | 优先级 | 说明 |
|------|--------|------|
| DuckDB 后端 (SQL 查询) | P2 | 复杂查询 |
| 向量嵌入 (语义谱系) | P3 | Knowledge RAG |
| 谱系可视化 (DAG 图) | P2 | 调试用 |
| 分布式存储 (HDFS/S3) | P3 | 大规模实验 |

---

## 11. 参考

- QuantaAlpha `quantaalpha/pipeline/loop.py:209` — `self.trace.hist.append((h, e, f))`
- QuantaAlpha `quantaalpha/coder/costeer/evolving_strategy.py:23` — `MultiProcessEvolvingStrategy`
- QuantaAlpha `configs/experiment.yaml:42-77` — `evolution` 配置
- QuantaAlpha `configs/experiment.yaml:73` — `parent_selection_strategy` 5 选项

---

*Last updated: 2026-06-11*
