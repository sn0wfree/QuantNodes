# 添加 Engine A 规则

Engine A 是 YAML 驱动的规则引擎。添加新规则无需写 Python 代码。

## 规则结构

```yaml
- id: "lookahead.my_new_rule"      # 唯一标识 (category.name)
  description: "简短描述"
  pattern: "\\.shift\\(\\s*-[1-9]"   # 正则表达式
  severity: CRITICAL                # CRITICAL/HIGH/MEDIUM/LOW
  lesson: "L-20260720-1"            # 关联的 L-NNN
  category: "lookahead"             # 分类
  auto_checkable: "static"

  # 可选: 上下文排除
  skip_if_preceded_by: "\\.rolling"     # 同一行或前 N 行有此 pattern 则跳过
  skip_if_followed_by: "\\.where\\("    # 同一行或后 N 行有此 pattern 则跳过
  skip_window_lines: 3                  # 上下文窗口（行数，默认 3）

  # 可选: 上下文提示
  context_hint: ["return", "pct"]   # 仅当周围代码包含这些关键词时触发
```

## 编写流程

1. **确定 lesson**: 在 `lessons/` 中找到对应 L-NNN
2. **确定 pattern**: 从教训中提取代码模式
3. **测试正则**: 用 Python re 模块测试
4. **添加排除**: 避免误报（如 rolling 跳过 mean）
5. **测试**: 在 tests/engines/test_static_engine.py 添加测试

## 编写原则

1. **明确无歧义**: 只加 100% 确定的规则
2. **避免误报**: 用 skip_* 减少噪声
3. **完整测试**: 每个规则都要有测试用例
4. **关联 lesson**: 每个规则必须关联 L-NNN

## 示例

### 简单规则（无排除）

```yaml
- id: "lookahead.shift_negative"
  description: "使用 .shift(-N) 引用未来数据"
  pattern: "\\.shift\\(\\s*-([1-9]\\d*)\\s*\\)"
  severity: CRITICAL
  lesson: "L-20260720-1"
  category: "lookahead"
  auto_checkable: "static"
```

### 带 skip 的规则

```yaml
- id: "nan_safe.bare_pct_change"
  description: "裸 .pct_change() 可能不安全"
  pattern: "\\.pct_change\\(\\)"
  severity: HIGH
  lesson: "L-213"
  category: "nan_safe"
  auto_checkable: "static"
  skip_if_followed_by: "\\.where\\("   # 跳过 .pct_change().where(...) 模式
```

### 带 context_hint 的规则

```yaml
- id: "nan_safe.dropna_returns"
  description: ".dropna() 直接丢弃收益数据"
  pattern: "\\.dropna\\(\\)"
  severity: MEDIUM
  lesson: "L-213"
  category: "nan_safe"
  auto_checkable: "static"
  context_hint: ["return", "ret", "pct", "compute"]  # 仅当周围有这些词
```

## 调试

```python
from quantnodes_strategy_audit import StaticEngine
from pathlib import Path

engine = StaticEngine(Path("rules/simple_rules.yaml"))
warnings = engine.scan_file(Path("test.py"))
for w in warnings:
    print(f"[{w.severity.value}] {w.detector} @ {w.line}: {w.message}")
```

## 验证

```bash
pytest tests/engines/test_static_engine.py -v
ruff check src/ tests/
```