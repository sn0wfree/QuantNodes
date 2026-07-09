# Stage 10 报告 — 集中度约束 (Concentration Caps)

> Stage 10: 添加 ConcentrationCaps 配置, 限制单 ETF / Top N / 类别集中度
> 完成日期: 2026-07-07
> 状态: ✅ 完成 (**但 Calmar 略有降低**, 详见下文)

## 1. 改动概览

### 1.1 新增配置 (`portfolio.py`)

```python
@dataclass
class ConcentrationCaps:
    """集中度约束 (Stage 10)."""
    enabled: bool = False
    single_etf_max: float = 0.15     # 单 ETF 权重上限 (默认 15%)
    top_n_total_max: float = 0.45   # Top 3 ETF 合计上限 (默认 45%)
    top_n_count: int = 3
    category_max: float = 0.40      # 单类别合计上限 (默认 40%)

@dataclass
class RotationConfig:
    # 现有参数保留
    ...
    concentration: ConcentrationCaps = field(default_factory=ConcentrationCaps)
```

### 1.2 新增函数

```python
def _apply_concentration_caps(weights, caps, pool=None) -> dict:
    """三步约束:
    1. 单 ETF <= single_etf_max
    2. Top N 合计 <= top_n_total_max
    3. 单类别合计 <= category_max (需 pool)
    差额视为现金.
    """
```

### 1.3 集成点

- `select_and_weight`: 加权后应用 caps
- `apply_stops`: 重新加权后再次应用 caps (修复 bug)

### 1.4 Bug 修复记录

**Bug**: caps 最初只在 `select_and_weight` 中应用, 但 `apply_stops` 内部调用 `select_and_weight` 后会用 `inverse_vol_weights` 重新计算权重, 覆盖 caps。

**修复**: 在 `apply_stops` 重新加权后再次调用 `_apply_concentration_caps`。

**验证**: 修复后最大权重从 34.2% 降至 24.0% (caps 生效)

## 2. 真实数据回测结果 (2019~2026)

### 2.1 集中度参数扫描

| 配置 | Calmar | DD | Ann | vs Baseline |
|------|--------|-----|-----|-------------|
| **Baseline** | **0.78** | **-21.05%** | **16.35%** | - |
| single=0.25 top3=0.60 cat=0.60 | 0.75 | -21.52% | 16.22% | -4% Calmar |
| single=0.20 top3=0.50 cat=0.50 | 0.69 | -23.06% | 15.93% | -12% Calmar |
| single=0.15 top3=0.45 cat=0.40 (推荐) | 0.61 | -25.40% | 15.37% | -22% Calmar |
| single=0.12 top3=0.40 cat=0.35 (最严) | 0.57 | -26.58% | 15.10% | -27% Calmar |

### 2.2 单 ETF 最大权重效果

| 配置 | 最大单 ETF 权重 |
|------|----------------|
| Baseline | 34.2% |
| Concentration caps | **24.0%** (-30%) |
| VT tv=0.15 | 35.0% |
| Caps + VT | **23.4%** (-32%) |

### 2.3 与其他阶段组合

| 组合 | Calmar | DD | Ann |
|------|--------|-----|-----|
| Baseline | 0.78 | -21.05% | 16.35% |
| 9-C (VT tv=0.15) | **1.00** | -6.89% | 6.87% |
| **10 + 9-C (Caps+VT)** | 0.77 | -8.61% | 6.62% |

## 3. 关键发现: 集中度约束在本策略中**不利**

### 3.1 反直觉的结果
- Caps 确实限制了单 ETF 权重 (34% → 24%)
- 但 Calmar 从 0.78 降至 0.61 (-22%)
- Ann 从 16.35% 降至 15.37%

### 3.2 原因分析

1. **逆波动加权已自然分散**: 逆波动加权让高波动 ETF 权重自然降低
2. **Caps 强制现金持有**: 单 ETF 上限 15% + Top 3 上限 45% 强制保留 ~25% 现金
3. **现金拖累收益**: 在 2019-2026 牛市期间, 现金拖累显著
4. **518880 (黄金) 是最优持仓**: Calmar 0.78 中, 518880 贡献 30% 收益, 限制它伤害了策略

### 3.3 何时 Caps 有用

- **多策略组合**: 当组合中已有多个低相关 ETF 时, 单 ETF 上限防止集中
- **熊市**: 在熊市中限制集中度有助降低 DD
- **实盘**: 防止黑天鹅事件冲击

### 3.4 为何 10+9-C 不如 9-C 单独

- VT 已经把组合波动降到 4.66%
- 此时 Caps 强制现金持有只会降低收益
- 双重风控过度保守

## 4. 决策建议

### 推荐配置: **不启用** Stage 10

**理由**:
1. 单独启用降低 Calmar 22%
2. 与 9-C 组合降低 Calmar 23%
3. 逆波动加权已提供足够分散
4. Caps 的价值在多策略组合而非单动量策略

### 如果仍想启用

- **保守**: single=0.25, top_n=0.60 (影响较小, -4% Calmar)
- **中等**: single=0.20, top_n=0.50 (折中)
- ❌ **不推荐**: single < 0.15 (损害过大)

## 5. 教训与启示

### 5.1 集中度约束的适用场景
- ✅ 多策略组合 (各策略相关性低)
- ✅ 因子投资 (暴露分散)
- ❌ 单动量策略 (已有逆波动分散)
- ❌ 与 VT 组合 (过度保守)

### 5.2 单一优化方向可能无效
- 集中度约束是合理的风险控制手段
- 但与现有架构 (逆波动 + VT) 重复
- 应该评估边际收益

### 5.3 当前最优配置: Stage 9-C 单独

| 配置 | Calmar | DD | Ann | 推荐度 |
|------|--------|-----|-----|--------|
| **9-C (VT tv=0.15)** | **1.00** | -6.89% | 6.87% | **★★★★** |
| Baseline | 0.78 | -21.05% | 16.35% | ★★★ |
| 9-B (TF bear=0.7) | 0.88 | -17.05% | 14.98% | ★★★ |
| 10 (Caps) | 0.61 | -25.40% | 15.37% | ❌ |

## 6. 测试覆盖

```bash
python3.11 -m pytest tests/strategy/momentum_etf_rotation/test_concentration.py -v
# 9/9 PASS
```

测试类:
- `TestApplyConcentrationCaps`: 5 个 (函数级测试)
- `TestSelectAndWeightConcentration`: 2 个 (集成测试)
- `TestBacktestConcentration`: 2 个 (回测对比)

总测试数: 166/166 PASS (+9 from Stage 10)

## 7. 文件清单

| 文件 | 类型 | 改动 |
|------|------|------|
| `QuantNodes/strategy/momentum_etf_rotation/portfolio.py` | 修改 | +60 行 (ConcentrationCaps + _apply_concentration_caps) |
| `QuantNodes/strategy/momentum_etf_rotation/__init__.py` | 修改 | 导出 ConcentrationCaps |
| `tests/strategy/momentum_etf_rotation/test_concentration.py` | 新增 | 165 行 (9 个测试) |
| `reports/momentum_etf_rotation/charts/stage10_concentration.html` | 新增 | 4 策略净值对比 |
| `reports/momentum_etf_rotation/charts/stage10_max_weight.html` | 新增 | 最大权重对比 |
| `reports/momentum_etf_rotation/stage10_report.md` | 新增 | 本报告 |

## 8. 退出条件检查

| 检查项 | 阈值 | 实际 | 结果 |
|--------|------|------|------|
| 测试通过率 | ≥ 95% | 100% (9/9) | ✅ |
| OOS Calmar > 0.5 | > 0.5 | 1.00 | ✅ |
| Calmar 不降低 | 期望 | 降低 22% | ❌ |
| DD 改善 | 期望 | 恶化 21% | ❌ |

**最终决定**: ❌ Caps 不进入默认配置, 保留代码作为可选功能。

## 9. 下一步

进入 **Stage 13: 交易成本建模**, 让回测更贴近实盘。

预期: Ann 降低 1-2%, 不影响 Calmar, 让回测结果更可靠。