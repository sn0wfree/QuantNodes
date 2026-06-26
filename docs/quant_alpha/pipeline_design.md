# 端到端因子挖掘流水线设计

## 目标

构建统一的端到端因子挖掘流水线，连接 Alpha-GPT → MCTS → 去重 → Wiki。

## 架构

```
quantnodes alpha-pipeline
    │
    ├── Stage 1: Alpha-GPT (LLM 生成种子)
    │   └── output: List[FinalFormulaRecord]
    │
    ├── Stage 2: MCTS (种子优化)
    │   ├── input: seed_formulas from Stage 1
    │   └── output: List[MCTSNode]
    │
    ├── Stage 3: 合并去重
    │   ├── input: Stage 1 + Stage 2 结果
    │   └── output: List[FactorMetrics]
    │
    └── Stage 4: Wiki 持久化
        ├── input: 去重后的因子
        └── output: WikiFactor pages
```

## 配置

```python
@dataclass
class PipelineConfig:
    objective: str
    wiki_path: str = "wiki/"
    alphagpt_iterations: int = 3
    alphagpt_pool_size: int = 10
    mcts_iterations: int = 50
    max_mutual_ic: float = 0.7
    top_k: int = 10
    date_column: str = "date"
    code_column: str = "code"
    forward_returns: Tuple[int, ...] = (1, 5, 20)
```

## 结果

```python
@dataclass
class PipelineResult:
    alphagpt_result: Optional[AlphaGptResult]
    mcts_result: Optional[MCTSSearchResult]
    final_pool: List[FactorMetrics]
    wiki_pages: List[str]
    elapsed_seconds: float
    summary: Dict[str, Any]
```

## CLI 命令

```bash
quantnodes alpha-pipeline \
  --objective "capture A-share reversal effect" \
  --data data.parquet \
  --wiki-path wiki/ \
  --alphagpt-iterations 3 \
  --mcts-iterations 50 \
  --max-mutual-ic 0.7
```

## 测试计划

```
tests/quant_alpha/test_pipeline.py
├── TestPipelineConfig
├── TestAlphaPipeline
└── TestCLI
```
