# 解释字段截断修复设计文档

**版本**: v1.0
**日期**: 2026-06-28
**作者**: LLM Pipeline
**状态**: 设计 → 实施中
**分支**: `fix/explanation-truncation`
**前置**: `feature/thinking-chain` (V5 暴露问题)

---

## 1. 问题

V5 实验中 `mean_reversion` / `volatility` 失败 0 因子，但实际 LLM 输出了完整 JSON —— 只是被 parser 忽略。

### 真实失败案例（V5 mean_reversion formula-translator 响应）

LLM 响应结构（gateway 已剥离 `<think>` 开头标签）：

```
{
  "round": 1,         ← 第一次输出，被 max_tokens 截断
  ...
}

Actually, looking at the system prompt...

```json
{"round": 1, "formulas": [{...}, {...}, {...}]}  ← 第二次完整输出
```
```

- 响应长度：2261 chars
- explanation 字段长度：444 / 406 / 522 chars（远超 80 字符约束）

### 根因

1. **Parser greedy regex 失败**：`re.search(r"\{[\s\S]*\}", raw)` 匹配从**第一个 `{` 到最后一个 `}`** → 跨过两个 JSON → 解析失败
2. **explanation 字段膨胀**：prompt 要求 <80 chars，但 LLM 把结构化字段塞进去 → 每次 ~500 chars
3. **Schema validator 严格**：`_validate_formula_translator` 要求 `idea_id` 字段存在，缺失直接 fail
4. **字段命名混淆**：idea-generator 用 `description` 字段，formula-translator 用 `explanation` 字段，LLM 容易把"为什么"塞进 `explanation`

## 2. 方案：4 层防御

### 防御层 P0：Robust JSON parser

**目标**：当 greedy regex 失败时，扫描所有 JSON 候选，**优先用最后一个**（最可能是 LLM 重写的完整版本）

**改动**：`QuantNodes/research/quant_alpha/llm/parser.py`

新增函数 `_find_last_valid_json()`，集成到 `parse_json_3layer()`：

```python
def _find_last_valid_json(raw, schema_validator=None):
    """扫描所有 JSON 对象，用最后一个满足 schema 的"""
    if not raw:
        return None
    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(raw):
        if ch != '{':
            continue
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and len(obj) >= 2:
            if schema_validator is None or schema_validator(obj) is None:
                candidates.append(obj)
    return candidates[-1] if candidates else None
```

集成到 `parse_json_3layer()` 作为 Layer 4 fallback（在 truncated recovery 之后）。

**预期效果**：mean_reversion / volatility 失败 → 成功。

### 防御层 P1：explanation post-process

**目标**：LLM 即使写出 500 字 explanation，代码强制截到 200 字

**改动**：`QuantNodes/research/quant_alpha/workflow/alpha_gpt.py:_step_formula_translator`

在 parse 成功后，对 formulas_data 列表处理：

```python
# Post-process: 强制截断 explanation 到 200 chars
for fd in formulas_data:
    if isinstance(fd, dict) and "explanation" in fd:
        expl = fd["explanation"]
        if isinstance(expl, str) and len(expl) > 200:
            fd["explanation"] = expl[:197] + "..."
```

**预期效果**：未来如果 LLM 再违反长度约束，输出体积被限制 → 减少截断概率。

### 防御层 P2：Schema validator 强化

**目标**：让 `_validate_formula_translator` 容忍字段缺失/过长

**改动**：`QuantNodes/research/quant_alpha/llm/parser.py:_validate_formula_translator`

```python
def _validate_formula_translator(obj):
    if "formulas" not in obj:
        return "missing 'formulas'"
    formulas = obj["formulas"]
    if not isinstance(formulas, list) or len(formulas) == 0:
        return "'formulas' empty or not list"
    for i, f in enumerate(formulas):
        if not isinstance(f, dict):
            return f"formulas[{i}] not dict"
        if "formula" not in f:
            return f"formulas[{i}] missing formula"
        # 容忍缺失 idea_id
        f.setdefault("idea_id", "")
        # 双重防御：截断 explanation
        if "explanation" in f and isinstance(f["explanation"], str):
            if len(f["explanation"]) > 200:
                f["explanation"] = f["explanation"][:197] + "..."
    return None
```

**预期效果**：LLM 输出不规范时也能 accept 有效公式。

### 防御层 P3：idea-generator 字段重命名

**目标**：避免 LLM 把 `description` 当作 `explanation` 处理

**改动**：`QuantNodes/research/quant_alpha/workflow/alpha_gpt.py:_build_idea_prompt`

- Schema: `"description"` → `"rationale"`
- IdeaRecord.from_dict(): `d.get("description", ...)` → `d.get("rationale") or d.get("description", "")`

**向后兼容**：旧 mock LLM 输出 `description` 字段仍可工作（`rationale` 为空时 fallback 到 `description`）。

## 3. 实施清单

| 文件 | 改动 |
|------|------|
| `QuantNodes/research/quant_alpha/llm/parser.py` | +`_find_last_valid_json()` + 强化 `_validate_formula_translator()` |
| `QuantNodes/research/quant_alpha/workflow/alpha_gpt.py` | `_step_formula_translator` explanation post-process + idea-generator prompt `description`→`rationale` |
| `QuantNodes/research/quant_alpha/workflow/state.py` | IdeaRecord 兼容读取 `description` 或 `rationale`（可选） |
| `tests/quant_alpha/test_parser.py` | +3 P0 tests + 2 P2 tests |
| `tests/quant_alpha/test_alpha_gpt_workflow.py` | +1 E2E truncation recovery test |
| `tests/quant_alpha/run_4_logic_v6.py` | 复制 V5 + 应用 4 层防御 |

## 4. V6 验证

| 指标 | V5 (无修复) | V6 (有修复) | 期望 |
|------|-------------|-------------|------|
| mean_reversion 因子数 | 0 | ? | ≥2 |
| volatility 因子数 | 0 | ? | ≥2 |
| momentum 因子数 | 3 | ? | ≥3 |
| price_volume_divergence 因子数 | 0 | ? | ≥0 |
| **总因子** | **3** | ? | **≥7** |
| 最佳 \|IR\| | 0.1167 | ? | ≥0.1167 |

## 5. 风险评估

| 风险 | 缓解 |
|------|------|
| `_find_last_valid_json` 取错对象 | schema validator 过滤 + len >= 2 keys 过滤 |
| explanation 截断破坏可读性 | 200 chars 足够 + "..." 标记 |
| `rationale` 字段破坏旧 mock | 兼容读取 `d.get("rationale") or d.get("description", "")` |
| V6 失败（API quota 等）| 先跑 mock 测试验证修复逻辑 |

## 6. 后续

V6 通过后：
- 跑 V7 = V6 + 多轮迭代（3-5 轮）验证 IR 收敛
- 强化 MCTS LLM 通道使用 hypothesis（已在 search.py:_evaluate 接入）
- 跨逻辑共享 OpPrior（改 output_dir 为根目录）
