# 自动化因子挖掘流水线设计

## 目标

构建统一的自动化因子挖掘系统，支持三种后端（Alpha-GPT / MCTS / Hybrid），使用 ClickHouse 真实数据 + MiniMax LLM，小规模验证（10-50 因子）。

## 架构

```
quantnodes alpha-mine --backend {alphagpt,mcts,hybrid}
    │
    ├── AlphaGptBackend (5 智能体迭代)
    │   ├── idea-generator → LLM → ideas
    │   ├── formula-translator → LLM → formulas
    │   ├── evaluator → PolarsAlphaCalculator → IC/IR
    │   ├── reflector → LLM → verdicts
    │   └── critic → LLM → final_pool
    │
    ├── MCTSSearchBackend (UCB1 树搜索)
    │   ├── seed_formulas (random/alpha101/user)
    │   ├── ExtensionOpPool (26+ 操作)
    │   ├── 5-channel feedback
    │   └── UCB1 selection/expand/evaluate/backprop
    │
    ├── HybridBackend (Alpha-GPT 种子 + MCTS 优化)
    │   ├── AlphaGptBackend.run() → seed_formulas
    │   └── MCTSSearchBackend.search(seed_formulas) → optimized_pool
    │
    └── Persistence (Wiki + JSON)
        ├── wiki: research/wiki.py
        └── json: output/alpha_mine_{timestamp}.json
```

## Phase 1: Alpha-GPT 子智能体规格（5 个 .md 文件）

### 文件列表

| 文件 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `.agent/agents/alpha-gpt-idea-generator.md` | 生成因子创意 | objective + previous_reflection | `{"ideas": [...]}` |
| `.agent/agents/alpha-gpt-formula-translator.md` | 创意 → 公式 | ideas + available_operators | `{"formulas": [...]}` |
| `.agent/agents/alpha-gpt-evaluator.md` | 评估公式 | formulas + data | `{"evaluations": [...]}` |
| `.agent/agents/alpha-gpt-reflector.md` | 反思改进 | evaluations + round_idx | `{"formula_feedback": [...]}` |
| `.agent/agents/alpha-gpt-critic.md` | 最终筛选 | all_evaluations + all_reflections | `{"final_pool": [...]}` |

### 设计要点

- 每个 .md 包含：角色定义、输入格式、输出格式（严格 JSON）、few-shot 示例、约束条件
- few-shot 示例来自 Alpha 101/158 设计哲学
- 输出格式与 `llm/parser.py` 的 schema validator 对齐

## Phase 2: 统一流水线

### 新增文件

```
QuantNodes/research/quant_alpha/pipeline/
├── __init__.py
├── pipeline.py          # 主编排器
├── alphagpt_backend.py  # Alpha-GPT 后端
├── mcts_backend.py      # MCTS 后端
├── hybrid_backend.py    # 混合后端
└── persistence.py       # Wiki + JSON 持久化
```

### Pipeline 主编排器

```python
class FactorMiningPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.backend = self._create_backend()
        self.persistence = FactorPersistence(config.output_dir)

    def run(self, data: pl.DataFrame) -> MiningResult:
        # 1. 运行后端
        factors = self.backend.mine(data, self.config)

        # 2. 互信息去重
        factors = self._deduplicate(factors, data)

        # 3. 持久化
        self.persistence.save(factors)

        return MiningResult(factors=factors, stats=...)
```

### AlphaGptBackend

```python
class AlphaGptBackend:
    def __init__(self, llm_client, config):
        self.workflow = AlphaGptWorkflow(
            config=AlphaGptConfig(
                objective=config.objective,
                iterations=config.iterations,
                pool_size=config.pool_size,
                ...
            ),
            data=data,
            llm_client=llm_client,
        )

    def mine(self, data, config) -> List[FactorSpec]:
        result = self.workflow.run()
        return [FactorSpec(formula=f.formula, ...) for f in result.final_pool]
```

### MCTSSearchBackend

```python
class MCTSSearchBackend:
    def __init__(self, vocab, config):
        self.search = MCTSSearch(
            vocab=vocab,
            config=MCTSSearchConfig(
                iterations=config.iterations,
                max_depth=config.max_depth,
                ...
            ),
        )

    def mine(self, data, config) -> List[FactorSpec]:
        # 获取种子公式
        seeds = self._get_seeds(config.seed_source, data)

        # 运行 MCTS 搜索
        result = self.search.search(
            data=data,
            seed_formulas=seeds,
            date_column=config.date_column,
        )

        return [FactorSpec(formula=n.formula, ...) for n in result.best_k_nodes]
```

### HybridBackend

```python
class HybridBackend:
    def __init__(self, llm_client, vocab, config):
        self.alphagpt = AlphaGptBackend(llm_client, config)
        self.mcts = MCTSSearchBackend(vocab, config)

    def mine(self, data, config) -> List[FactorSpec]:
        # 1. Alpha-GPT 生成种子
        seeds = self.alphagpt.mine(data, config)

        # 2. MCTS 局部搜索优化
        optimized = self.mcts.mine(data, config, seed_formulas=seeds)

        return optimized
```

## Phase 3: CLI 命令

### 文件

```
QuantNodes/cli/commands/alpha_mine.py
```

### 命令设计

```bash
quantnodes alpha-mine \
  --backend {alphagpt,mcts,hybrid} \
  --data-source {clickhouse,mock} \
  --llm-provider {minimax,deepseek,mock} \
  --seed-source {random,alpha101,user} \
  --output {wiki,json,both} \
  --iterations 5 \
  --pool-size 10 \
  --objective "maximize IC for 1-day forward return" \
  --user-formulas "rank(close/open-1),ts_mean(volume,20)" \
  --output-dir ./output
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--backend` | `hybrid` | 挖掘后端 |
| `--data-source` | `clickhouse` | 数据源 |
| `--llm-provider` | `minimax` | LLM 提供商 |
| `--seed-source` | `random` | MCTS 种子来源 |
| `--output` | `both` | 输出格式 |
| `--iterations` | `5` | 迭代次数 |
| `--pool-size` | `10` | 每轮创意数 |
| `--objective` | (required) | 挖掘目标 |
| `--user-formulas` | `None` | 用户输入的种子公式 |
| `--output-dir` | `./output` | 输出目录 |

## Phase 4: 互信息去重

### 实现

在 `persistence.py` 中实现：

```python
def deduplicate_factors(
    factors: List[FactorSpec],
    data: pl.DataFrame,
    max_mutual_ic: float = 0.7,
) -> List[FactorSpec]:
    """基于互信息去重"""
    if len(factors) <= 1:
        return factors

    # 计算所有因子的值
    calculator = PolarsAlphaCalculator(data)
    values = {}
    for f in factors:
        try:
            vals = calculator.evaluate_formula(f.formula)
            values[f.formula] = vals
        except Exception:
            continue

    # 贪心去重：按 IR 排序，逐个加入，检查互信息
    factors_sorted = sorted(factors, key=lambda f: f.ir or 0, reverse=True)
    selected = []
    for f in factors_sorted:
        if f.formula not in values:
            continue

        # 检查与已选因子的互信息
        is_diverse = True
        for s in selected:
            if s.formula not in values:
                continue
            mutual_ic = calculator.calc_mutual_IC(
                values[f.formula], values[s.formula]
            )
            if abs(mutual_ic) > max_mutual_ic:
                is_diverse = False
                break

        if is_diverse:
            selected.append(f)

    return selected
```

## Phase 5: 真实 LLM 集成

### 测试计划

1. **Alpha-GPT 端到端测试**
   - 使用 MiniMax API（通过 LLMGateway → nanobot）
   - 验证 JSON 解析 robustness
   - 验证公式生成质量

2. **MCTS + LLM 反馈测试**
   - 启用 `enable_llm=True` 反馈通道
   - 验证 LLM 对假设-表达一致性的判断

3. **混合模式测试**
   - Alpha-GPT 生成种子 → MCTS 优化
   - 验证种子质量对 MCTS 搜索的影响

## Phase 6: 进度报告

### 实现

在 `pipeline.py` 中添加进度事件：

```python
class MiningProgress:
    def __init__(self):
        self.listeners = []

    def on(self, event: str, callback):
        self.listeners.append((event, callback))

    def emit(self, event: str, data: dict):
        for ev, cb in self.listeners:
            if ev == event:
                cb(data)

# 事件类型
# - "round_start": {"round": 1, "total_rounds": 5}
# - "idea_generated": {"ideas": [...]}
# - "formula_translated": {"formulas": [...]}
# - "evaluation_complete": {"evaluations": [...]}
# - "reflection_done": {"verdicts": [...]}
# - "mining_complete": {"factors": [...], "stats": {...}}
```

## 测试计划

### 单元测试

```
tests/quant_alpha/test_pipeline.py
├── test_alphagpt_backend.py
├── test_mcts_backend.py
├── test_hybrid_backend.py
├── test_persistence.py
└── test_deduplication.py
```

### 端到端测试

```bash
# Mock 模式验证流程
quantnodes alpha-mine --backend alphagpt --llm-provider mock --data-source mock

# 真实模式验证
quantnodes alpha-mine --backend alphagpt --llm-provider minimax --data-source clickhouse

# MCTS 模式
quantnodes alpha-mine --backend mcts --seed-source random --data-source clickhouse

# 混合模式
quantnodes alpha-mine --backend hybrid --data-source clickhouse --llm-provider minimax
```

## 实现顺序

1. **Phase 1**: 创建 5 个子智能体 .md 文件
2. **Phase 2**: 构建统一流水线（pipeline/）
3. **Phase 3**: 构建 CLI 命令
4. **Phase 4**: 实现互信息去重
5. **Phase 5**: 真实 LLM 集成测试
6. **Phase 6**: 进度报告

## 预估工作量

| Phase | 文件数 | 代码行数 | 时间 |
|-------|--------|----------|------|
| Phase 1 | 5 | ~500 | 30min |
| Phase 2 | 6 | ~800 | 1h |
| Phase 3 | 1 | ~200 | 20min |
| Phase 4 | 1 | ~100 | 15min |
| Phase 5 | 1 | ~100 | 15min |
| Phase 6 | 1 | ~100 | 15min |
| 测试 | 5 | ~500 | 30min |
| **总计** | **20** | **~2300** | **~3h** |
