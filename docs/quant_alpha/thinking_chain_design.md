# 思维链利用方案设计文档

**版本**: v1.0
**日期**: 2026-06-28
**作者**: LLM Pipeline
**状态**: 设计 → 实施中
**分支**: `feature/thinking-chain`

---

## 1. 背景

### 1.1 当前现状

MiniMax M3 在响应中默认输出 `<think>...</think>` 思维链块，包含：
- 任务理解（"The user wants me to..."）
- 经济假设（"20 日反转因子在 A 股效果显著"）
- 算子选择推理（"用 rank 而不是 zscore 因为..."）
- 风险评估（"T+1 制度下窗口需 ≥ 5 天"）
- 自我批评（"可能因停牌缺失"）

**当前 gateway 行为**: `re.sub(r"", "", content)` → **完全丢弃**

### 1.2 问题

- 30-50% LLM token 浪费在 thinking 块
- thinking 中含丰富**结构化信号**（hypothesis, mechanism, operator_rationale）
- 未被利用 → MCTS 通道评分只能用贫乏的 `description`（如 "20日反转因子"）
- MCTS 算子采样纯随机 → 收敛慢

### 1.3 目标

- **不浪费** LLM 思考
- 将 thinking 转化为**结构化信号**
- 信号喂给 MCTS → 加快收敛、产出更多有效因子

---

## 2. 方案概览

### 2.1 四层架构（采用 1+2+4，跳过 3）

```
Tier 1: 捕获 & 持久化
  ↓
Tier 2: 结构化推理 Prompt + parse_thinking_block()
  ↓
Tier 3: (跳过) 二轮 self-critique — 性价比低
  ↓
Tier 4: OpPrior 算子先验 → MCTS 加权采样
```

### 2.2 数据流

```
MiniMax M3
   │
   │ "<think>{...}</think>{JSON}"
   ▼
gateway._complete_direct()
   │
   ├─ thinking, content = split_thinking(raw)
   ├─ persist thinking → {output_dir}/llm_raw/{agent_id}_thinking_{ts}.txt
   └─ return (content, thinking)  ← tuple
   │
   ▼
alpha_gpt._call_llm()
   │
   ├─ parse_thinking_block(thinking) → ThinkingRecord
   │     {hypothesis, mechanism, operator_rationale, ..., mentioned_ops}
   │
   ├─ attach to records (IdeaRecord, FormulaRecord, ...)
   │
   └─ update OpPrior(mentioned_ops, ir)  ← Tier 4
   │
   ▼
MCTS _expand()
   │
   └─ op_pool.sample(prior=OpPrior)  ← 加权采样
```

---

## 3. Tier 1: 捕获 & 持久化

### 3.1 gateway.py 改造

```python
def _complete_direct(
    self,
    prompt: str,
    temperature: Optional[float] = None,
    *,
    agent_id: Optional[str] = None,
    persist_dir: Optional[Path] = None,
) -> Tuple[str, str]:
    """Returns (content, thinking)."""
    # ... existing OpenAI call ...
    
    thinking = ""
    think_match = _re.search(r"", raw_content, _re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
    
    content = _re.sub(r"<think>[\s\S]*?</think>", "", raw_content).strip()
    
    if persist_dir and thinking:
        from pathlib import Path
        persist_path = Path(persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        ts = int(_time.time() * 1000)
        (persist_path / f"{agent_id}_thinking_{ts}.txt").write_text(thinking, encoding="utf-8")
    
    return content, thinking
```

### 3.2 state.py records 加字段

```python
@dataclass
class IdeaRecord:
    # ... existing ...
    thinking: Optional[str] = None
    hypothesis: Optional[str] = None
    mechanism: Optional[str] = None
    mentioned_ops: List[str] = field(default_factory=list)

@dataclass
class FormulaRecord:
    # ... existing ...
    thinking: Optional[str] = None
    hypothesis: Optional[str] = None
    mentioned_ops: List[str] = field(default_factory=list)

@dataclass
class ReflectionRecord:
    # ... existing ...
    thinking: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
```

### 3.3 alpha_gpt._call_llm 改造

```python
def _call_llm(self, agent_id: str, prompt: str) -> Tuple[str, str]:
    """Returns (content, thinking)."""
    temperature = self._get_temperature_for_agent(agent_id)
    persist_dir = None
    if self.output_dir:
        persist_dir = Path(self.output_dir) / "llm_raw"
    
    if isinstance(self.llm_client, LLMGateway):
        content, thinking = self.llm_client._complete_direct(
            prompt, temperature=temperature,
            agent_id=agent_id, persist_dir=persist_dir,
        )
    else:
        # mock client (e.g., MockLLMClient)
        content = self.llm_client.complete(agent_id=agent_id, prompt=prompt, temperature=temperature)
        thinking = ""
    
    return content, thinking
```

### 3.4 向后兼容

- mock LLM（测试用）返回 `("", "")` → 现有 322 测试不破
- records 加 Optional 字段 → 旧代码不读则无影响
- `parse_idea_generator_output(raw)` 仍接收字符串 → 无破坏

---

## 4. Tier 2: 结构化推理 Prompt

### 4.1 4 个 prompt 模板改造

**思路**: 在 JSON 输出前要求 LLM 把推理写进 `<think>` 块，且结构化。

#### idea-generator prompt:

```python
def _build_idea_prompt(self, round_idx, prev_reflection):
    schema = '{...}'
    return f"""You are the Alpha-GPT IdeaGenerator.

# STEP 1: REASONING (in <think>)
Before generating JSON, output your reasoning in EXACTLY this structure inside <think>:

HYPOTHESIS: <one-sentence economic hypothesis>
MECHANISM: <why this should work in A-shares specifically, mention T+1/retail/stop-loss if relevant>
OPERATOR_RATIONALE: <why these specific operators from the allowed list>
PARAMETER_RATIONALE: <why this lookback window>
RISK: <what could fail, e.g., regime change, liquidity shock>
SUGGESTED_OPS: <comma-separated operator names you plan to use, e.g., rank,ts_std,div>

# STEP 2: JSON OUTPUT
Then output STRICT JSON (no markdown) matching: {schema}
"""
```

#### formula-translator prompt: 同模式

#### reflector / critic prompt: 简化版

### 4.2 parser.py parse_thinking_block()

```python
@dataclass
class ThinkingRecord:
    raw: str
    hypothesis: str = ""
    mechanism: str = ""
    operator_rationale: str = ""
    parameter_rationale: str = ""
    risk: str = ""
    suggested_ops: List[str] = field(default_factory=list)
    mentioned_ops: List[str] = field(default_factory=list)

def parse_thinking_block(
    thinking_text: str,
    op_vocab: Optional[Set[str]] = None,
) -> ThinkingRecord:
    """从 thinking 文本提取结构化字段 + 算子提及。"""
    if not thinking_text:
        return ThinkingRecord(raw="")
    
    result = ThinkingRecord(raw=thinking_text)
    
    field_pattern = (
        r"(HYPOTHESIS|MECHANISM|OPERATOR_RATIONALE|"
        r"PARAMETER_RATIONALE|RISK|SUGGESTED_OPS):\s*(.+?)"
        r"(?=\n[A-Z_]+:|$)"
    )
    matches = re.findall(field_pattern, thinking_text, re.DOTALL)
    for key, value in matches:
        value = value.strip()
        if key == "HYPOTHESIS":
            result.hypothesis = value
        elif key == "MECHANISM":
            result.mechanism = value
        elif key == "OPERATOR_RATIONALE":
            result.operator_rationale = value
        elif key == "PARAMETER_RATIONALE":
            result.parameter_rationale = value
        elif key == "RISK":
            result.risk = value
        elif key == "SUGGESTED_OPS":
            result.suggested_ops = [
                s.strip() for s in value.split(",") if s.strip()
            ]
    
    if op_vocab:
        ops_in_text = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", thinking_text))
        result.mentioned_ops = [op for op in ops_in_text if op in op_vocab]
    
    return result
```

### 4.3 接入 _step_idea_generator

```python
def _step_idea_generator(self, round_idx):
    prev_reflection = (
        self.state.all_reflections[-1].to_dict()
        if self.state.all_reflections
        else None
    )
    prompt = self._build_idea_prompt(round_idx, prev_reflection)
    content, thinking = self._call_llm("alpha-gpt-idea-generator", prompt)
    
    # 解析 JSON
    parsed = parse_idea_generator_output(content)
    if not parsed.ok:
        logger.warning("idea-generator parse failed: %s", parsed.error)
        return []
    
    # 解析 thinking（Tier 2）
    op_vocab = set(self._get_available_operators())
    thinking_record = parse_thinking_block(thinking, op_vocab=op_vocab)
    
    ideas_data = (parsed.data or {}).get("ideas", [])[:self.config.pool_size]
    ideas = []
    for i, d in enumerate(ideas_data):
        idea = IdeaRecord.from_dict(d, round_idx)
        idea.thinking = thinking or None
        idea.hypothesis = thinking_record.hypothesis or None
        idea.mechanism = thinking_record.mechanism or None
        idea.mentioned_ops = thinking_record.mentioned_ops
        ideas.append(idea)
    return ideas
```

### 4.4 MCTS LLM 通道用 hypothesis

```python
# mcts/search.py _evaluate()
def _evaluate(self, node, data, date_column, forward_return_column):
    # ... existing ...
    
    # 优先用 hypothesis 替代 description
    hypothesis = node.metadata.get("hypothesis") or node.formula
    
    fb = collect_all_channels(
        formula=node.formula,
        result=result,
        expected_length=expected_length,
        config=self.config.feedback_config,
        exception=exception,
        ic_decay=ic_metrics.get("ic_decay"),
        data=data,
        date_column=date_column,
        code_column=self.config.code_column,
        llm_client=self.config.llm_client,
        structured_logic=self.config.structured_logic,
        hypothesis=hypothesis,  # NEW
    )
```

---

## 5. Tier 4: OpPrior 算子先验

### 5.1 mcts/op_prior.py

```python
@dataclass
class OpPrior:
    """算子先验分布。
    
    Attributes:
        weights: op_name -> 权重 ∈ [floor, 1.0]
        alpha: 历史保留率（指数衰减）
        floor: 最小权重（避免零概率）
        total_updates: 累计更新次数
    """
    weights: Dict[str, float] = field(default_factory=dict)
    alpha: float = 0.7
    floor: float = 0.1
    total_updates: int = 0
    
    def update(self, ops: List[str], ir: float) -> None:
        """根据一次成功公式更新先验。"""
        if not ops or abs(ir) < 0.01:
            return
        strength = min(abs(ir) / 0.5, 1.0)  # |IR|=0.5 视为满强度
        for op in ops:
            old = self.weights.get(op, 0.5)
            new = old * self.alpha + strength * (1 - self.alpha)
            self.weights[op] = max(self.floor, new)
        self.total_updates += 1
    
    def sample_weights(self, all_ops: List[str]) -> np.ndarray:
        """返回与 all_ops 对齐的权重数组。"""
        return np.array([self.weights.get(op, 0.5) for op in all_ops])
    
    def mix(self, all_ops: List[str], mix_ratio: float = 0.5) -> np.ndarray:
        """先验与均匀的混合分布。"""
        prior = self.sample_weights(all_ops)
        uniform = np.ones(len(all_ops))
        mixed = mix_ratio * prior + (1 - mix_ratio) * uniform
        return mixed / mixed.sum()
    
    def save(self, path: Path) -> None:
        import json
        Path(path).write_text(json.dumps({
            "weights": self.weights,
            "alpha": self.alpha,
            "floor": self.floor,
            "total_updates": self.total_updates,
        }, indent=2), encoding="utf-8")
    
    @classmethod
    def load(cls, path: Path) -> "OpPrior":
        import json
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            weights=data["weights"],
            alpha=data.get("alpha", 0.7),
            floor=data.get("floor", 0.1),
            total_updates=data.get("total_updates", 0),
        )
```

### 5.2 MCTSSearchConfig 加字段

```python
@dataclass
class MCTSSearchConfig:
    # ... existing ...
    op_prior: Optional[OpPrior] = None
    prior_mix: float = 0.5  # 0.0=纯均匀, 1.0=纯先验
```

### 5.3 _expand() 加权采样

```python
def _expand(self, node, data, available_cols):
    if node.formula == "__ROOT__":
        return None
    
    # 选算子（加权采样 or 均匀）
    try:
        if self.config.op_prior is not None:
            all_ops = self.op_pool.all_operators()  # List[Op]
            op_names = [op.name for op in all_ops]
            weights = self.config.op_prior.mix(op_names, self.config.prior_mix)
            idx = self.rng.choice(len(all_ops), p=weights)
            op = all_ops[idx]
        else:
            op = self.op_pool.sample()
    except (ValueError, IndexError):
        return None
    
    # ... rest unchanged ...
```

### 5.4 pipeline.py 维护 OpPrior

```python
def __init__(self, config):
    # ... existing ...
    self._op_prior = OpPrior()
    prior_path = Path(self.config.output_dir) / "op_prior.json"
    if prior_path.exists():
        try:
            self._op_prior = OpPrior.load(prior_path)
            logger.info("Loaded OpPrior: %d ops, %d updates", 
                        len(self._op_prior.weights), self._op_prior.total_updates)
        except Exception as e:
            logger.warning("Failed to load OpPrior: %s", e)
    self._op_prior_path = prior_path

def _run_mcts(self, data, seed_formulas):
    try:
        # 注入 OpPrior
        mcts_config.op_prior = self._op_prior
        
        result = search.search(...)
        
        # 用本轮有效节点更新 OpPrior
        for node in result.valid_nodes:
            ops = re.findall(r"\b([a-zA-Z_]\w*)\s*\(", node.formula)
            ir = node.metadata.get("ir", 0.0)
            self._op_prior.update(ops, ir)
        
        # 持久化
        self._op_prior.save(self._op_prior_path)
        return result
    except Exception as e:
        logger.error("[Pipeline] MCTS failed: %s", e)
        return None
```

---

## 6. 实施检查点

| CP | 触发 | 验证 |
|----|------|------|
| CP1 | Tier 1 完成 | 现有 322 测试通过；mock 场景无 thinking 不破 |
| CP2 | Tier 2 完成 | 单逻辑 smoke test；IdeaRecord 含 hypothesis |
| CP3 | Tier 3 完成 | OpPrior 更新/持久化跑通 |
| CP4 | Tier 4 完成 | V5 vs V4 对比报告出炉 |

---

## 7. 风险 & 缓解

| 风险 | 触发 | 缓解 |
|------|------|------|
| 结构化 thinking 抢占 max_tokens | LLM 输出超长 | max_tokens=16384，预估 thinking 2000 chars + JSON 3000 chars 足够 |
| LLM 不遵守格式 | regex 提取失败 | 所有字段 Optional 缺失 → 空串 fallback |
| OpPrior 过拟合 | 单一逻辑跑多次 | α=0.7 + floor=0.1 强制混合 |
| 加字段破坏 API | 旧调用方 | Optional 字段，旧代码不读则无影响 |
| 并行 V4 与主线冲突 | 共享文件 | V4 用独立 `pipeline_output_v4/` |

---

## 8. 文件清单

### 新增
- `QuantNodes/research/quant_alpha/mcts/op_prior.py`
- `tests/quant_alpha/test_thinking_block.py`
- `tests/quant_alpha/test_op_prior.py`
- `tests/quant_alpha/run_4_logic_v4.py`
- `tests/quant_alpha/run_4_logic_v5.py`
- `docs/quant_alpha/v4_baseline_report.md`
- `docs/quant_alpha/v5_vs_v4_comparison.md`

### 修改
- `QuantNodes/ai/llm/gateway.py` — `_complete_direct` 返回 tuple + persist
- `QuantNodes/research/quant_alpha/llm/parser.py` — `parse_thinking_block()`
- `QuantNodes/research/quant_alpha/workflow/state.py` — 4 个 record 加字段
- `QuantNodes/research/quant_alpha/workflow/alpha_gpt.py` — `_call_llm` 改 tuple + prompt 加结构化指令
- `QuantNodes/research/quant_alpha/mcts/feedback.py` — `collect_llm_channel` 用 hypothesis
- `QuantNodes/research/quant_alpha/mcts/search.py` — `_expand` 加权采样 + MCTS LLM 通道
- `QuantNodes/research/quant_alpha/pipeline.py` — 维护 OpPrior 状态

---

## 9. 验证指标

- **有效因子数**: V5 vs V4 (期望 +20-30%)
- **最佳 IR**: V5 vs V4
- **平均 IR**: V5 vs V4
- **MCTS 收敛迭代数**: 同样目标数下的 iteration 数
- **OpPrior 收敛性**: 跑 4 逻辑后 top-5 算子权重的稳定性
