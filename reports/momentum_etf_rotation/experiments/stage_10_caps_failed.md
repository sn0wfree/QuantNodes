# 实验失败: 集中度约束 (Stage 10)

> 阶段: Stage 10 (2026-07-07)
> 结论: ❌ NO-GO

## 假设

- H1: 限制单 ETF ≤ 15% 和 Top 3 ≤ 45% 可降低集中度风险
- H2: 在 2024 年 518880 (黄金) 占 31.5% 收益贡献的场景下, caps 应能改善 DD

## 实现

### 改动文件
- `QuantNodes/strategy/momentum_etf_rotation/portfolio.py` (修改, +60 行)
- `tests/strategy/momentum_etf_rotation/test_concentration.py` (新增, 165 行)
- `reports/momentum_etf_rotation/charts/stage10_*.html` (2 个图表)
- `reports/momentum_etf_rotation/v2/stage10_report.md` (详细报告)

### 代码量
- 代码: ~60 行
- 测试: 9 个 (9/9 PASS)
- 文档: 3 个文件

## 失败原因

### 主因: 风控过度叠加

当前策略已有的风控:
1. **逆波动加权** (自然分散: 高波动 ETF 权重低)
2. **类别 cap** (A 股 ≤ 3, 商品+海外必含)
3. **趋势过滤器** (Stage 9-B, 熊市转债券)
4. **波动率目标** (Stage 9-C, 高波动自动减仓)

**集中度约束 = 第 5 层风控, 过度保守**

### 次因: caps 强制现金持有

```python
# 修复后逻辑 (差异视为现金)
w["518880"] = min(w["518880"], 0.15)  # 强制上限
# 多出来的 0.10 变成现金 (不再投资)
```

在 2019-2026 牛市中:
- 518880 (黄金) 是**最优持仓** (贡献 30% 收益)
- caps 限制后, 黄金权重从 34% → 24%
- **错过最优收益**, 强制持有现金

### 第三因: 与 Stage 9-C (VolTargeting) 叠加时双重保守

- VT 已经把组合波动降到 4.66%
- caps 进一步限制高收益 ETF 的暴露
- 双重叠加, 收益降幅 > 单项

## 证据

| 配置 | Calmar | DD | Ann |
|------|--------|-----|-----|
| Baseline | 0.78 | -21.05% | 16.35% |
| Caps only | 0.61 | -25.40% | 15.37% |
| VT only (推荐) | **1.00** | -6.89% | 6.87% |
| Caps + VT | 0.77 | -8.61% | 6.62% |

### 单 ETF 最大权重效果 (修复后)
- Baseline: 34.2%
- With caps: **24.0%** (-30%)
- Caps 确实有效, 但**代价过高**

## 教训

1. **风控叠加的边际效用递减**: 每加一层风控, 边际收益越小, 边际代价越大
2. **逆波动已是天然分散**: 不需要额外 caps
3. **caps 适合的场景**:
   - ✅ 多策略组合 (各策略相关性低)
   - ✅ 因子投资 (因子暴露分散)
   - ❌ 单动量策略 (已有内部分散)

## 可能的复苏方向

### 条件
- 仅在**多策略组合**中启用 (例如本策略 + 等权 + 行业轮动)
- 或者作为**软约束** (软警告而非硬截断)

### 修改方案 (如果将来需要)
```python
@dataclass
class ConcentrationCaps:
    enabled: bool = False
    soft_cap: bool = True        # 改为软约束, 仅警告不截断
    single_etf_max: float = 0.30  # 放宽阈值
    top_n_total_max: float = 0.80
    # 不实施硬截断
```

### 验证要求
- 仅在多策略组合下测试
- 单策略下**不**启用

## 相关文档

- 详细报告: `../v2/stage10_report.md`
- 图表: `../charts/stage10_concentration.html`, `../charts/stage10_max_weight.html`
- 集成 bug: `apply_stops` 重新 `inverse_vol_weights` 时覆盖 caps
  (后续在 `portfolio.py` 中二次应用 caps 修复)

## 相关 Bug (开发过程发现)

1. **字段顺序问题**: 初始实现用位置参数, 导致 TrenderFilter 测试失败 (与本实验无关)
2. **集成顺序 bug**: caps 在 `apply_stops` 的 `inverse_vol_weights` 后被覆盖
   - 修复: 在 `apply_stops` 重新加权后二次应用 `_apply_concentration_caps`
3. **算法振荡问题**: 初始的水重分配算法会导致无限循环
   - 修复: 简化为直接截断, 差额视为现金

---

**失败归档完成** - 2026-07-07
