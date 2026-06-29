# V6 vs V5 对比报告

**版本对比**: V5 (Tier 1+2+4) vs V6 (V5 + 4 层防御修复)
**日期**: 2026-06-28
**作者**: LLM Pipeline
**分支**: `fix/explanation-truncation` (合并到 master)

---

## 实验设置

| 维度 | V5 (无修复) | V6 (有修复) |
|------|-------------|-------------|
| **Tier 1+2+4** | ✅ | ✅ |
| **P0: Robust JSON parser** | ❌ | ✅ Layer 4 找最后一个有效 JSON |
| **P1: explanation post-process** | ❌ | ✅ > 200 chars 强制截断 |
| **P2: Schema validator 强化** | ❌ | ✅ idea_id optional + 双重 truncate |
| **P3: 字段重命名** | ❌ | ✅ `description` → `rationale` |

完全相同的：
- 4 逻辑 × 3 ideas × MCTS 20
- LLM: MiniMax M3 via direct OpenAI API
- max_tokens=16384, timeout=300s

---

## 结果对比

### 总体

| 指标 | V4 baseline | V5 (Tier 1+2+4) | V6 (V5 + 4 修复) | V6 vs V5 |
|------|-------------|-------------------|-------------------|----------|
| 总因子数 | 9 | 3 | **9** | **+200%** ✅ |
| 最佳 \|IR\| | 0.1208 | 0.1167 | **0.1284** | **+10%** ✅ |
| 4 逻辑成功 | 3/4 | 1/4 | **3/4** | **+200%** ✅ |
| 总耗时 | 522s | 555s | 576s | +4% ≈ |

### 逐逻辑

| 逻辑 | V4 | V5 | V6 | V6 vs V5 |
|------|----|----|----|----------|
| **price_volume_divergence** | 0 | 0 | 0 | 0 (LLM 难生成) |
| **mean_reversion** | 3 | 0 | **3** | **+3** ✅ |
| **momentum** | 3 | 3 | **3** | 0 (但 IR 提升) |
| **volatility** | 3 | 0 | **3** | **+3** ✅ |

### 关键 IR 改善

**momentum**:
- V5: best |IR|=0.1167
- V6: best |IR|=**0.1284** (+10%)
- V6: 3 个因子均 > 0.06，V5 只有 1 个

**mean_reversion**:
- V5: 0 因子
- V6: 3 因子，best |IR|=0.0535

**volatility**:
- V5: 0 因子
- V6: 3 因子，best |IR|=0.1133

---

## 修复效果分析

### 根因（V5 失败）

LLM 在 formula-translator 阶段输出"截断 JSON + thinking + 重写 JSON"模式：

```
{
  "round": 1, ...   ← 第一次，被 max_tokens 截断
}

Actually, let me re-output:

```json
{"round": 1, "formulas": [...]}   ← 第二次完整
```
```

+ explanation 字段膨胀（444 / 406 / 522 字符 vs 期望 80 字符）。

V5 parser 用 `re.search(r"\{[\s\S]*\}")` 匹配从**第一个 `{` 到最后一个 `}`** → 跨过两个 JSON → 解析失败。

### V6 修复（4 层防御）

#### P0: Robust JSON parser（直接修复核心问题）

`parser.py:_find_last_valid_json()`：
- 扫描所有 JSON 起始位置
- 用 `json.JSONDecoder().raw_decode()` 逐个尝试解码
- 收集所有可解的 dict，过滤掉只含 1 个 key 的（避免误选元数据）
- 用 schema_validator 过滤（保证语义正确）
- **返回最后一个**（最可能是 LLM 重写的完整版本）

集成到 `parse_json_3layer()` 作为 **Layer 4**（在 truncated recovery 之前优先尝试）。

**实测**：V5 mean_reversion 真实响应（2261 chars）现在能恢复 3 个完整公式。

#### ~~P1: explanation post-process（预防性）~~ (refactor/smart-p2 后已删除)

`alpha_gpt.py:_step_formula_translator` 在 parse 成功后强制截断：

```python
for fd in formulas_data:
    if "explanation" in fd and len(fd["explanation"]) > 200:
        fd["explanation"] = fd["explanation"][:197] + "..."
```

**删除原因**（见 `refactor/smart-p2` 分支）：
- P1 与 P2 做完全相同的事（重复防御）
- V6 实测 P1 0 触发
- P2 已升级为智能拆分版，P1 完全冗余

#### P2: Schema validator 强化（智能拆分，refactor/smart-p2 后升级）

`_validate_formula_translator` 升级（v2 智能拆分）：
- `idea_id` 字段：optional（`setdefault("idea_id", "")`）
- `explanation` 字段 3 档处理：
  - 档 1：含结构化标记 → 拆分为 `explanation` (summary) + `explanation_detail` (detail)
  - 档 2：超长但无结构化 → 截断到 200 chars
  - 档 3：短小干净 → 保留原样
- `formula` 字段：保持 required（核心字段）

**关键改进**：信息不丢失（explanation_detail 完整保留结构化内容）

#### P3: 字段重命名（结构清晰化）

`_build_idea_prompt` 中：
- 旧: `"description": "经济直觉1-2句"`
- 新: `"rationale": "经济直觉1-2句（与 formula explanation 区分）"`

`IdeaRecord.from_dict()` 兼容读取：
```python
rationale_or_desc = d.get("rationale") or d.get("description", "")
```

向后兼容旧 mock LLM 和旧数据。

---

## 测试覆盖

| 测试 | 数量 | 状态 |
|------|------|------|
| P0: `_find_last_valid_json` | 5 | ✅ |
| P2: validator 强化 | 4 | ✅ |
| E2E: alpha_gpt workflow 恢复 | 1 | ✅ |
| 已有回归测试 | 103 | ✅ |
| **总测试** | **113** | **✅ 100%** |

新增测试文件：
- `tests/quant_alpha/test_parser.py`: TestFindLastValidJson (5) + TestValidateFormulaTranslatorP2 (4)
- `tests/quant_alpha/test_alpha_gpt_workflow.py`: test_formula_translator_recovers_from_truncated_pattern

---

## 关键文件

| 文件 | 改动 | 行数 |
|------|------|------|
| `QuantNodes/research/quant_alpha/llm/parser.py` | +`_find_last_valid_json()` + 强化 `_validate_formula_translator()` | +90 -10 |
| `QuantNodes/research/quant_alpha/workflow/alpha_gpt.py` | explanation post-process + prompt rationale | +12 -3 |
| `QuantNodes/research/quant_alpha/workflow/state.py` | `from_dict` 兼容 `rationale` | +5 -1 |
| `tests/quant_alpha/test_parser.py` | +P0 +P2 tests | +120 |
| `tests/quant_alpha/test_alpha_gpt_workflow.py` | +E2E test | +40 |
| `tests/quant_alpha/run_4_logic_v6.py` | 新脚本（复制 V5） | +250 |
| `docs/quant_alpha/explanation_truncation_fix.md` | 设计文档 | +200 |

**总代码量**: 717 行新增 + 14 行修改

---

## 后续

V6 解决了 V5 的回归问题。建议下一步：

1. **V7**: 跑多轮迭代（3-5 轮）验证 IR 收敛
2. **跨逻辑共享 OpPrior**：改 output_dir 为根目录
3. **强化 MCTS LLM 通道使用 hypothesis**：已在 search.py:_evaluate 接入
4. **wiki 页面展示 hypothesis**：把 IdeaRecord.hypothesis 写入 wiki/wiki/Factor/

---

## 结论

✅ **V6 完美修复 V5 回归**：
- 总因子 3 → 9（+200%）
- 最佳 \|IR\| 0.1167 → 0.1284（+10%）
- mean_reversion: 0 → 3 因子
- volatility: 0 → 3 因子
- momentum IR 进一步提升

4 层防御是相互补充的：P0 解决已经发生的截断，P1/P2 防止再次发生，P3 改善字段清晰度。
