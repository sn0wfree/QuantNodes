# 价差套利子策略 — 基差+跨品种合并研究计划（2026-09-02）

> α来源方向：现货升贴水（同品种价差）+ 跨品种套利（26对）合并为**单一价差套利子策略**
> 方法论：IS/OOS方向一致性门槛（继承CFTC/库存/经纪持仓全部教训）

---

## 1. 架构决策（用户确认）

**合并为单一子策略**（Option C）：

```
价差套利子策略 (cta_spread_arbitrage)
├── 同品种基差因子 (55品种 × 4因子, spot_basis_daily)
├── 跨品种价差因子 (26对 × MR因子, cross_product_spread)
│   ↓
├── 统一验证: IC → 五分位分层 → IS/OOS方向一致性
├── 统一组合: 稳健子集等权 vol target 8%
└── 独立资金池 (与主CTA/锌/M并列)
```

**合并理由**：
- 经济逻辑统一（都是spread/价差交易，非方向性）
- 基差因子与跨品种因子在不同品种上自然正交（基差用CU单腿信息，跨品种用RB-I对腿）
- 一个资金池/一套风控/一篇文档
- 与主CTA方向性信号低相关（套利对冲特性）

## 2. 数据源

### spot_basis_daily（CTA DB, 83,230行, 55品种, 2018~2025-07-21）
| 字段 | 含义 |
|------|------|
| spot_price | 现货价 |
| near_basis_rate | (现货-近月)/近月 |
| dom_basis_rate | (现货-主力)/主力 |

⚠️ 数据断在2025-07-21（akshare中断）——先用现有数据验证α，数据修复另做。

### cross_product_spread（CTA DB, 140,472行, 26对）
| 字段 | 含义 |
|------|------|
| pair_name | 对名（如RB-I）|
| near/far_product | 两腿品种 |
| spread_raw / spread_bp | 价差绝对值/bp |

## 3. 因子设计

### 同品种基差（4因子，逻辑复用TermBasisModule已验证的形态，独立实现）
| 因子 | 定义 | 逻辑 |
|------|------|------|
| B1 期限斜率 | (F4-F1)/F1 z-score | 期限结构陡峭度（主CTA最强因子形态，Sharpe+0.81）|
| B2 基差水平 | near_basis_rate 60日均值 | 极端水平阈值 |
| B3 基差收敛 | near_basis_rate 60日z-score | \|z\|>2反转 |
| B4 主力基差收敛 | dom_basis_rate z-score | 同B3，不同腿 |

### 跨品种价差（2因子）
| 因子 | 定义 | 逻辑 |
|------|------|------|
| X1 价差z-score | spread_bp 60日z-score | \|z\|>2反转（CalendarSpreadMR同形态）|
| X2 价差动量 | spread_bp 20日变化 z-score | 价差趋势延续 |

**实现原则**：不import不修改TermBasisModule代码（别人的），因子逻辑在研究脚本内独立实现（形态已验证，实现简单）。

## 4. 方法论（固化门槛）

| 步骤 | 内容 | 止损线 |
|------|------|--------|
| 1 | 因子-品种/对组合 IC + 五分位分层 | - |
| 2 | IS(2018-2022)/OOS(2022-2025)方向一致性 | **一致率<60%止损** |
| 3 | IS定方向回测（IS价差>5%才入信号）| **OOS Sharpe<0.15止损** |
| 4 | 与主CTA/锌/M合并评估（后续统一做） | 合并Sharpe提升<0.05不并 |

## 5. 工程要求（用户新增）

1. **沉淀可复用组件**：本研究的IC/分层/IS-OOS/回测/统计函数抽到 `scripts/cta_shared/factor_research.py`，后续因子研究（CU/AU子策略等）直接复用
2. 不动别人代码（TermBasisModule/全天候模块等）
3. 一次性分析代码最少化

**组件库清单**（从已有4个研究脚本去重提取）：
| 组件 | 来源重复 |
|------|---------|
| `rolling_zscore()` | 4个脚本 |
| `compute_factor_ic()` | cftc/inventory/broker_holding |
| `quintile_spread()` | cftc/inventory/broker_holding |
| `is_oos_direction_consistency()` | inventory/broker_holding |
| `vol_target_backtest()` | cftc/inventory/broker_holding/zn |
| `period_stats()` (Sharpe/AnnRet/MaxDD/Calmar分期) | 全部 |
| `yearly_returns()` | 全部 |
| `load_cost_bps()` | 全部 |

## 6. 预期

- 55品种×4基差因子 + 26对×2价差因子 ≈ 272组合，预期5-10个方向稳健
- 子策略OOS Sharpe 0.3-0.5（参照锌0.48/M 0.58）
- 与主CTA相关性低（价差对冲 vs 方向性）
- 工作量：组件库半天 + 验证1天 + 回测半天 = 2天

## 7. 执行状态

| Step | 状态 |
|------|------|
| 1. 组件库沉淀 | 待做 |
| 2. 数据加载（基差+价差）| 待做 |
| 3. 因子工程 | 待做 |
| 4. IC+IS/OOS验证 | 待做 |
| 5. 稳健子集回测 | 待做 |
| 6. 文档+commit | 待做 |
