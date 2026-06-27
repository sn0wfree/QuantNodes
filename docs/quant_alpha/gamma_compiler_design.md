# Γ 编译器设计文档

> **基于**: AlphaLogics 论文 (arXiv 2603.20247)
> **PR**: PR-2 (Logic→Γ Compiler)
> **状态**: 实现中
> **日期**: 2026-06-27

---

## 1. 概述

### 1.1 核心思想

AlphaLogics 论文的关键发现：**约束生成 > 自由生成**（Figure 3）。

Γ 编译器的作用是把**市场逻辑**（`WikiLogicStructured`）编译成**可执行约束**（`CompiledConstraint`），然后在因子生成过程中注入这些约束。

```
WikiLogicStructured → compile_to_constraint() → CompiledConstraint
                                                      ↓
                                              validate(formula)
                                              render_for_prompt()
```

### 1.2 为什么需要 Γ 编译器

| 问题 | 现状 | Γ 编译器解决 |
|------|------|-------------|
| 因子生成太自由 | LLM 在 162 个算子中随机选择 | 约束为 10-20 个算子白名单 |
| 参数范围无限制 | LLM 可能生成任意窗口 | 约束为 5-60 天范围 |
| 符号方向不确定 | LLM 可能生成正向或负向 | 约束为 +1 或 -1 |
| 变量使用无约束 | LLM 可能使用任意变量 | 约束为 close/volume/open 等 |

### 1.3 论文实证

- **Γ 约束生成 > 自由生成**（Figure 3, 3 个 LLM × 2 个市场均成立）
- **逻辑库越大，因子质量单调提升**（Figure 5）
- **约束生成的因子 IR 比自由生成高 30-50%**

---

## 2. 数据结构

### 2.1 WikiLogicStructured（逻辑结构化表示）

```python
@dataclass
class LogicCondition:
    """单条谓词 (v, op, θ, w)"""
    variable: str           # 市场变量名: "close", "volume", "open", "high", "low"
    op: str                 # 算子名: "ts_corr", "ts_mean", "rank"
    threshold: float        # 阈值 θ
    window: Optional[int]   # 时序算子的窗口 d
    weight: float = 1.0     # 权重 w
    second_variable: Optional[str] = None  # 双变量算子的第二个变量

@dataclass
class LogicBehavior:
    """ℬ = (y, d, h): 行为三元组"""
    target: str             # "forward_return_1" / "forward_return_5" / "forward_return_20"
    direction: int          # +1(信号方向与目标一致)/ -1(反向)
    horizon: int            # 持有期天数

@dataclass
class WikiLogicStructured:
    """H_struct: 规范化后的结构化逻辑"""
    predicates: List[LogicCondition]  # 条件谓词列表
    behavior: LogicBehavior           # 行为目标
    operator_whitelist: Optional[List[str]] = None  # 允许的算子族
    parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None  # 参数范围
    sign_constraint: Optional[int] = None  # +1 / -1 / None
```

### 2.2 CompiledConstraint（编译后的约束）

```python
@dataclass
class CompiledConstraint:
    """Γ: 编译后的可执行约束"""
    operator_whitelist: Set[str]           # 允许的算子名集合
    operator_blacklist: Set[str]           # 显式禁止的算子
    parameter_ranges: Dict[str, Tuple[float, float]]  # 参数范围
    sign_constraint: Optional[int]         # +1 / -1 / None
    variable_whitelist: Optional[Set[str]] # 允许的市场变量
    source_logic: Optional[str]            # 来源逻辑名（用于追溯）
```

---

## 3. 编译规则

### 3.1 从 WikiLogicStructured 到 CompiledConstraint

| 输入字段 | 编译结果 | 说明 |
|---------|---------|------|
| `predicates[].variable` | → `variable_whitelist` | 只允许使用的变量 |
| `predicates[].op` | → `operator_whitelist` | 只允许使用的算子 |
| `predicates[].window` | → `parameter_ranges[op]` | 固定窗口范围 |
| `behavior.direction` | → `sign_constraint` | 信号方向约束 |
| `behavior.horizon` | → 元数据 | 影响 forward_returns 选择 |
| `operator_whitelist`(显式) | → 覆盖默认 | 最终白名单 |
| `parameter_ranges`(显式) | → 合并 | 最终参数范围 |

### 3.2 编译示例

**输入**（市场逻辑）：
```
逻辑: 量价背离反转
𝒞: ts_corr(rank(open), rank(volume), 10) < -0.5
ℬ: (y=forward_return_5, d=-1, h=5)
operator_whitelist: [rank, ts_corr, sign]
parameter_ranges: {ts_corr: [5, 30]}
```

**输出**（Γ 约束）：
```python
CompiledConstraint(
    operator_whitelist={"rank", "ts_corr", "sign", "sub", "mul", "div"},
    operator_blacklist=set(),
    parameter_ranges={"ts_corr": (5, 30)},
    sign_constraint=-1,
    variable_whitelist={"open", "volume"},
)
```

---

## 4. 校验逻辑

### 4.1 validate() 方法

```python
def validate(self, formula: str) -> Tuple[bool, Optional[str]]:
    """校验 formula 是否满足所有约束"""
    
    # 1. 提取使用的算子
    used_ops = extract_operators(formula)
    
    # 2. 白名单检查
    for op in used_ops:
        if op not in self.operator_whitelist:
            return False, f"operator '{op}' not in whitelist"
    
    # 3. 提取使用的变量
    used_vars = extract_variables(formula)
    
    # 4. 变量白名单检查
    if self.variable_whitelist:
        for v in used_vars:
            if v not in self.variable_whitelist:
                return False, f"variable '{v}' not in whitelist"
    
    # 5. 参数范围检查
    for op, args in parse_op_args(formula):
        if op in self.parameter_ranges:
            for arg in args:
                lo, hi = self.parameter_ranges[op]
                if not (lo <= arg <= hi):
                    return False, f"{op} arg {arg} not in [{lo}, {hi}]"
    
    # 6. 符号方向检查（启发式）
    if self.sign_constraint is not None:
        if not check_sign_hint(formula, self.sign_constraint):
            return False, f"sign_constraint {self.sign_constraint} not satisfied"
    
    return True, None
```

### 4.2 辅助函数

- `extract_operators(formula)`: 从公式中提取使用的算子
- `extract_variables(formula)`: 从公式中提取使用的变量
- `parse_op_args(formula)`: 解析算子的参数
- `check_sign_hint(formula, direction)`: 检查符号方向

---

## 5. Prompt 注入

### 5.1 render_for_prompt() 方法

```python
def render_for_prompt(self) -> str:
    """生成可注入 LLM prompt 的人类可读描述"""
    lines = ["## Γ 约束（必须遵守）", ""]
    
    # 算子白名单
    if self.operator_whitelist:
        ops = ", ".join(sorted(self.operator_whitelist))
        lines.append(f"### 允许使用的算子")
        lines.append(f"只使用以下算子: {ops}")
        lines.append("")
    
    # 变量白名单
    if self.variable_whitelist:
        vars_ = ", ".join(sorted(self.variable_whitelist))
        lines.append(f"### 允许使用的变量")
        lines.append(f"只使用以下变量: {vars_}")
        lines.append("")
    
    # 参数范围
    if self.parameter_ranges:
        lines.append(f"### 参数范围")
        for op, (lo, hi) in self.parameter_ranges.items():
            lines.append(f"- {op}: [{lo}, {hi}]")
        lines.append("")
    
    # 符号约束
    if self.sign_constraint is not None:
        direction = "正向" if self.sign_constraint > 0 else "反向"
        lines.append(f"### 符号方向")
        lines.append(f"因子整体方向: {direction} ({self.sign_constraint:+d})")
        lines.append("")
    
    return "\n".join(lines)
```

### 5.2 注入到 formula-translator prompt

```python
def _build_formula_prompt(self, round_idx, ideas, available_ops, data_columns):
    prompt = f"..."
    
    # 注入 Γ 约束
    if self.config.gamma:
        prompt += f"\n\n{self.config.gamma.render_for_prompt()}"
    
    return prompt
```

---

## 6. 与现有系统的集成

### 6.1 AlphaGptConfig 扩展

```python
@dataclass
class AlphaGptConfig:
    # ... 原有字段 ...
    gamma: Optional[CompiledConstraint] = None  # Γ 约束
```

### 6.2 AlphaGptWorkflow 扩展

```python
class AlphaGptWorkflow:
    def _step_formula_translator(self, round_idx, ideas):
        formulas = self._generate_formulas(ideas)
        
        # Γ 约束校验
        if self.config.gamma:
            formulas = [f for f in formulas if self._check_gamma(f)]
        
        return formulas
    
    def _check_gamma(self, formula):
        passed, reason = self.config.gamma.validate(formula)
        if not passed:
            logger.info("Γ 校验失败: %s - %s", formula, reason)
        return passed
```

---

## 7. 文件结构

```
QuantNodes/research/quant_alpha/logic_mining/
    __init__.py
    compiler.py         # Γ 编译器（本 PR 核心）
    models.py           # WikiLogicStructured 等数据结构
    
tests/quant_alpha/
    test_logic_compiler.py  # 单元测试
```

---

## 8. 测试策略

### 8.1 单元测试

| 测试 | 说明 |
|------|------|
| `test_compile_basic` | 基本编译功能 |
| `test_validate_operator_whitelist` | 算子白名单校验 |
| `test_validate_variable_whitelist` | 变量白名单校验 |
| `test_validate_parameter_ranges` | 参数范围校验 |
| `test_validate_sign_constraint` | 符号方向校验 |
| `test_render_for_prompt` | Prompt 注入文本生成 |
| `test_integration_with_alphagpt` | 集成到 AlphaGptWorkflow |

### 8.2 论文示例验证

```python
# 论文示例: -TS_CORR(RANK(open), RANK(volume), 10)
logic = WikiLogicStructured(
    predicates=[
        LogicCondition(variable="open", op="rank", threshold=0, window=None),
        LogicCondition(variable="volume", op="rank", threshold=0, window=None),
        LogicCondition(variable="open", op="ts_corr", threshold=-0.5, window=10,
                       second_variable="volume"),
    ],
    behavior=LogicBehavior(target="forward_return_5", direction=-1, horizon=5),
    operator_whitelist=["rank", "ts_corr", "sign"],
    parameter_ranges={"ts_corr": (5, 30)},
    sign_constraint=-1,
)

gamma = compile_to_constraint(logic)

# 验证
assert gamma.validate("sign(-ts_corr(rank(open), rank(volume), 10))")[0] == True
assert gamma.validate("ts_argmax(close, 5)")[0] == False  # ts_argmax 不在白名单
```

---

## 9. 变更记录

| 日期 | 版本 | 变更 |
|------|------|------|
| 2026-06-27 | v1.0 | 初稿 |
