# 96. CTA 多因子策略现状总结

> 快照日期: 2026-08-12
> 关联: docs/94 (迭代记录) / docs/95 (策略规格)

## 一、策略体系

```
106 因子库 → 89 可用 (continuous_main 面板 OHLCV+OI) → 77 方向因子 (去重后)
  → 7 模块:
      趋势动量 22 / 均值回归 15 / 量能资金流 24 / 事件日历 6
      期限基差 2 / 宏观 3 / 产业链 5
    → 组内聚合: 因子级 TV-PR 学权重 (λ_tv=0.5, λ_l1=0 全维度保留,
       β 正负自由自动反用, expanding 每 21 天更新, 无前视)
      → 组间: 模块 inv_vol 波动率倒数加权 (滚动 252 天, 逐日归一化)
        → 信号动量过滤: 组合信号 60 日均值 < 0 → 空仓 (2022 回撤 -18%→-4%)
          → 执行层: sign → vol_target_positions → 市场级危机过滤
            (VolRiskFilter) → 杠杆硬限 (两遍法迭代) → 手数整数化
            → 方向变化才调仓 → run_cta_backtest (单一资金池 200 万)
```

品种池: **72 个有交易量品种** (2023-2025 日均量 ≥ 1 万手, 静态固定,
产业链 8 个配方品种全覆盖).

## 二、最优配置

```python
FactorModuleComboStrategy(config=ComboConfig(
    aggregation="tvpr",              # 组内 TV-PR 因子级学权重
    module_weighting="inv_vol",      # 组间模块波动率倒数加权
    momentum_filter_window=60,       # 信号动量过滤
    momentum_filter_coef=0.0,        # 动量为负 → 全空仓
    target_vol=0.30,                 # 2 倍杠杆
    max_leverage=3.0,                # vol_target 权重上限
    leverage_hard_cap=3.0,           # 组合总杠杆硬限
))
```

## 三、业绩表现 (72 品种, 2018-2025)

### 3.1 杠杆方案对比

| 配置 | Sharpe | 年化 | Vol | MaxDD | Calmar | OOS 年化 | 杠杆 max |
|---|---|---|---|---|---|---|---|
| 1x (0.15) 无硬限 | 1.813 | 13.1% | 7.2% | -10.6% | 1.234 | 11.8% | 3.36 |
| 2x (0.30) 无硬限 | 1.740 | 20.9% | 12.0% | -11.5% | 1.814 | 18.0% | **5.20** |
| **2x + 硬限 3.0** | **1.830** | **19.6%** | 10.7% | -14.1% | 1.388 | 15.2% | **3.21** |

OOS = 2022-2025 段 (全样本 NAV 切段统计).

### 3.2 稳健性指标 (2x+硬限配置)

- 换手率: 年化 ~3.5-4 倍 (低换手; 1bp 成本拖累 ~39% 收益)
- 杠杆: 中位 1.96 / p95 2.77 / max 3.21 (硬限生效)
- 尾部: VaR95 -0.67%, CVaR95 -1.06%, 最差单日 -2.58% (2025-04-07)
- 极端窗口: 2020 疫情 -5.7%, 2022 商品 -3.8% (动量过滤后), 2024 商品 -9.3%
- 最长回撤: 531 天 (2022 为主, 修复前 -18.7%)

## 四、过拟合验证 (全部完成)

| # | 检验 | 结果 | 判定 |
|---|---|---|---|
| 1 | 严格 OOS (仅 IS 调参, OOS 验证一次) | IS 最优 = 当前参数, OOS Sharpe 1.345 | ✅ |
| 2 | Bootstrap CV% (52 天块 × 200) | CV = 15.9% (Sharpe p5 1.29 / p95 2.30) | ✅ <50% |
| 3 | 起点依赖 (5 起点) | Sharpe 全正 (0.95-2.57); Calmar CV 64.9% | ⚠️ Calmar 红灯 |
| 4 | 5-fold walk-forward | OOS Sharpe 均值 1.788 全正 | ✅ |
| 5 | 因子跨段稳定性 (85 因子) | 54/85 一致; 8 被删 7 个剔除稳健 | ✅ |
| 6 | 75 因子无前视审计 | 仅 Seasonality/Weekday 命中 (已修) | ✅ |

**唯一红灯**: Calmar 对起点敏感 (回撤集中 2022 的位置效应),
Sharpe 全起点为正 → 收益能力不依赖起点, CTA 常见特性.

## 五、重大 bug 修复记录 (诚实清单)

1. **Seasonality/Weekday 前视偏差** (全样本统计量 → expanding 因果) —
   事件日历组真实 Sharpe 1.747→0.187
2. **max_leverage 非总杠杆硬限** (实际可超 5 倍 → 两遍法硬限 ≤3.2)
3. AmihudXS/MomentumXS/VolatilityXS/BasisXS 布尔 Series 索引 bug
4. divergence.py 布尔掩码 object dtype (NaN 品种列)
5. 分段回测品种池不一致 (固定 72 列)
6. TV-PR/动量过滤/杠杆硬限 均通过无前视一致性测试

## 六、交付物

| 类型 | 内容 |
|---|---|
| 策略代码 | `strategies/factor_modules.py` (7 模块+组合+过滤+硬限) / `tvpr_combo.py` |
| 数据管线 | `data_loader.py`: load_term_structure / load_macro_series / list_active_products |
| 脚本 | verify_round8_factors / cta_robustness_diag / cta_factor_stability / cta_overfitting_check |
| 文档 | docs/94 (迭代 10 章) + docs/95 (规格 7 章) + 本文 (现状快照) |
| 测试 | tests/strategy/cta/: 498 passed, 16 skipped, ruff clean |

## 七、残余风险与下一步

| # | 事项 | 状态 |
|---|---|---|
| 1 | Calmar 起点敏感 (64.9% CV) | 已缓解 (动量过滤), 可继续增强回撤保护 |
| 2 | near_basis 因子 (OOS +0.83) | 留独立样本前瞻验证, 不因样本内恢复 |
| 3 | 换手成本敏感 (10bp → Sharpe 0.857) | 实盘需评估冲击成本模型 |
| 4 | 事件日历组修复后 OOS 0.784 | 仍有效, 但依赖日历效应持续性 |
| 5 | 16 个外部数据因子 | 产业链/期限/宏观已纳入, 基差失效因子已剔除 |
