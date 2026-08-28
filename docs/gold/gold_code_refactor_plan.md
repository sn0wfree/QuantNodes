# 黄金策略代码整理计划

## 现状

- 核心策略代码: 10个 .py 文件, 共 2793 行
- 主策略 `gold_trend_bnh.py`: 1047行, 职责过多
- 正式测试: 仅 1 个文件 (test_gold_rollover.py, 16个用例)
- 研究脚本: 8 个 test_gold_v*.py, 大量重复

## 代码质量问题

| 严重程度 | 问题 | 影响 |
|---------|------|------|
| 高 | `load_gold_price_w()` 重复6份 | 6个文件 |
| 高 | 正式测试覆盖率0% (策略模块) | 10个文件 |
| 高 | `gold_trend_bnh.py` 1047行, 职责过多 | 1个文件 |
| 中 | `backtest_*()` / `get_weekly_returns()` 重复6份 | 6个文件 |
| 中 | test_gold_v2~v5 四文件几乎相同 | 4个文件 |
| 低 | 命名不统一 (DB/DB_PATH, 公开/私有) | 全部 |
| 低 | `fillna(method="ffill")` 已弃用 | 2个文件 |

## 整理计划

### Phase 1: 提取公共模块 (消除重复代码)

**新增**: `gold/gold_utils.py`

公共函数:
- `load_gold_price_d()` — 日频金价
- `load_gold_price_w()` — 周频金价
- `load_indicator_d(name, source)` — 日频指标
- `load_indicator_w(name, source)` — 周频指标
- `backtest_weekly(signal, cost_bps)` — 通用周频回测
- `get_weekly_returns(signal, start, end)` — 通用周频收益
- `DB_PATH` — 统一路径常量

**修改**: 6个单因子策略文件, 删除重复函数, 改为从 gold_utils 导入

### Phase 2: 拆分 gold_trend_bnh.py (1047行 → 4个文件)

| 新文件 | 原函数 | 行数估计 |
|--------|--------|---------|
| `gold_trend_bnh.py` | 基础趋势B&H + 追踪止损 | ~300行 |
| `gold_risk_overlay.py` | 三级风控覆盖层 | ~200行 |
| `gold_ensemble.py` | 自适应集成 + 分层集成 | ~250行 |
| `gold_macro_regime.py` | 宏观机制策略 | ~200行 |

### Phase 3: 添加单元测试 (全覆盖)

| 测试文件 | 测试内容 | 用例数 |
|---------|---------|--------|
| `test_gold_utils.py` | 公共函数 | ~15 |
| `test_gold_signals.py` | 6个单因子信号 | ~18 |
| `test_gold_trend_bnh.py` | 基础趋势B&H | ~10 |
| `test_gold_risk_overlay.py` | 三级风控 | ~12 |
| `test_gold_ensemble.py` | 集成策略 | ~10 |
| `test_gold_macro_regime.py` | 宏观机制 | ~8 |
| `test_gold_rollover.py` | 换月 (已有) | 16 |

总计: ~89个测试用例

### Phase 4: 合并重复测试脚本

**删除**: `test_gold_v2.py`, `test_gold_v3.py`, `test_gold_v4.py`, `test_gold_v5.py`

**保留**: `test_gold_strategies.py` (最完整的综合测试)

### Phase 5: 清理

- 修复 `fillna(method="ffill")` → `.ffill()`
- 统一命名: `DB` → `DB_PATH`
- 更新 `__init__.py` 导出

## 执行顺序

```
Phase 1 (提取公共模块)
    ↓
Phase 2 (拆分大文件)
    ↓
Phase 3 (添加测试) ← 确保重构不破坏功能
    ↓
Phase 4 (合并测试脚本)
    ↓
Phase 5 (清理)
```

## 风险控制

- 每个Phase完成后运行 `pytest tests/test_gold*.py -v`
- Phase 2 拆分后立即运行完整测试
- 任何测试失败立即修复

## 状态

- [ ] Phase 1: 提取公共模块 gold_utils.py
- [ ] Phase 2: 拆分 gold_trend_bnh.py
- [ ] Phase 3: 添加单元测试
- [ ] Phase 4: 合并重复测试脚本
- [ ] Phase 5: 清理
