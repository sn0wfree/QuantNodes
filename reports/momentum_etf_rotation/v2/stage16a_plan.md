# Stage 16A: 多策略组合 (Multi-Strategy Combination)

> **创建日期**: 2026-07-09
> **优先级**: P2 战略级
> **前置**: Stage 14 (924 分析) 完成
> **状态**: 规划中

---

## 1. 目标与动机

### 1.1 核心问题

当前 v1.0 单一动量策略的特征:
- **过度依赖趋势市场**: Calmar 1.60 在 2019-2025 黄金/海外大涨中极强
- **震荡市回撤**: 2024 调整期表现欠佳
- **924 失分**: 18.87% 错过 A股反弹
- **同质化风险**: 动量+逆波动选出来的 10 只 ETF 高度集中 (黄金+海外>50%)

### 1.2 解决思路

引入**多策略组合** (Multi-Strategy Ensemble):
- 动量策略 (现有): 趋势跟随
- 均值反转策略: 抓反弹/超跌修复 (A 股尤其有效)
- 行业轮动策略: 抓 A 股行业轮动 (924 案例)

**互补优势**:
- 动量擅长趋势, 弱势震荡市回撤
- 均值反转擅长超跌反弹, 弱势趋势市
- 行业轮动擅长 A 股板块轮动, 弥补 924 类机会

### 1.3 预期收益

| 指标 | v1.0 当前 | 16A 目标 | 改善 |
|------|-----------|----------|------|
| Calmar | 1.60 | 1.70+ | +6% |
| DD | -3.93% | -3.0% | +24% |
| 924 期间涨幅 | 2.19% | 5~8% | +130~270% |
| 月度胜率 | ~58% | ~65% | +7pp |

**风险**: 调仓复杂度增加, 过拟合可能.

---

## 2. 技术方案

### 2.1 三个子策略

#### 2.1.1 动量策略 (现有 v1.0)
```python
# 不变
score = hybrid_momentum_score_v2(nav_df, lookback=144)
# + 逆波动加权 + VT + Cost
```

#### 2.1.2 均值反转策略 (新)
**思路**: 选过去 N 日跌幅最大但短期企稳的 ETF
**参数**:
- 跌幅窗口: 60 日 (中期回调)
- 企稳信号: 5 日均线由下穿 10 日均线
- 持仓数: 5 只 (与动量互补)
- 上限: A 股宽基+行业 ≤ 3

**信号公式**:
```
reversion_score(code) = -rank_pct(60d_return) + 0.3 × (ma5_above_ma10 ? 1 : 0)
```

**注意**: 配对权重时, 反转策略选择 0.4~0.6 配重 (避免过热)

#### 2.1.3 行业轮动策略 (新)
**思路**: A 股行业 ETF 内部轮动, 选行业动量最高者
**参数**:
- 行业池: 20 只 A 股行业 ETF
- 信号: 60 日动量
- 持仓数: 3 只
- 调仓频率: 周度 (加快对行业轮动反应)

**特殊处理**: 与动量策略合并到 A 股宽基 cap (a_share_total=3) 时, 行业轮动优先

### 2.2 组合方式

**方法**: 风险平价加权 (Risk Parity Weighting)
- 用 Ledoit-Wolf 协方差
- 目标: 各子策略对组合波动率贡献相等 (~33% each)
- 已有代码: `common/risk_parity.py::solve_risk_parity`

**备选**: 等权组合 (起步 baseline)
- 简单, 易理解
- 当子策略波动率差异大时, 风险贡献不均

### 2.3 集成点

在 `v2/backtest_v2.py` 新增 `run_multi_strategy_backtest`:
```python
def run_multi_strategy_backtest(
    etf_nav, pool, cfg,  # cfg 包含子策略列表
) -> MultiStrategyResult:
    # 1. 各子策略独立回测
    sub_results = {
        'momentum': run_rotation_backtest(etf_nav, pool, mom_cfg),
        'reversion': run_reversion_backtest(etf_nav, pool, rev_cfg),
        'rotation': run_industry_rotation(etf_nav, pool, rot_cfg),
    }

    # 2. 计算子策略权重 (风险平价 / 等权)
    sub_weights = compute_sub_strategy_weights(sub_results, method='risk_parity')

    # 3. 合并 NAV
    combined_nav = sum(sub_weights[s] * sub_results[s].nav for s in sub_results)
```

**关键不变量**:
- 子策略 1 (动量) 主调仓日 = 月末 (继承现有)
- 子策略 2 (反转) 主调仓日 = 半月 (15 日 + 月末)
- 子策略 3 (行业轮动) 主调仓日 = 周度
- 三个子策略独立跑, 最后在主调仓日合并

### 2.4 冲突解决

**问题**: 三个子策略可能选中同一只 ETF
- 解决: 合并去重 + 权重按子策略权重加权
- 示例: 动量 0.5 × A 股宽基 30% + 反转 0.3 × A 股宽基 20% = 总 21% (其中 6% 来自反转, 15% 来自动量)

**Cap 处理**:
- 子策略内部 cap 严格遵守
- 组合时 cap 继承最严限制 (max(sub1_cap, sub2_cap, sub3_cap))

---

## 3. 数据需求

| 数据 | 现有? | 补充? |
|------|-------|-------|
| ETF 日线 (close) | ✅ 已有 (44 只 × 2058 天) | - |
| ETF 日线 (high/low) | ❌ 缺失 | 不需要 (本 Stage 不用 RSRS) |
| 子策略权重 | 需新增 | 子策略输出 |
| 行业 ETF 池 | ✅ 已有 (20 只 A 股行业) | - |

**无新数据需求**.

---

## 4. 文件结构

```
v2/
├── backtest_v2.py             # 主回测 (扩展)
├── portfolio_v2.py            # 现有 (不改动)
├── momentum_v2.py             # 现有 (不改动)
├── strategy_versions_v2.py    # 现有 (不改动)
├── fi_plus_v2.py              # 现有 (不改动)
├── multi_strategy_v2.py       # 新增: 多策略主入口
├── reversion_v2.py            # 新增: 均值反转子策略
├── industry_rotation_v2.py    # 新增: 行业轮动子策略
└── sub_weighting_v2.py        # 新增: 子策略权重 (风险平价)

common/
├── sub_strategy.py            # 新增: 子策略抽象基类
└── (其他不变)
```

---

## 5. 实施步骤 (建议 5-7 天)

### 步骤 1: 子策略抽象层 (1 天)
- [ ] 创建 `common/sub_strategy.py`
- [ ] 定义 `SubStrategy` 抽象基类
- [ ] 实现 `select_and_weight()` 接口

### 步骤 2: 均值反转子策略 (1.5 天)
- [ ] 创建 `v2/reversion_v2.py`
- [ ] 实现 `reversion_score()`: -rank_pct(60d_return) + ma_crossover bonus
- [ ] 实现 `select_reversion_etfs()`: 选 5 只
- [ ] 单元测试: tests/strategy/momentum_etf_rotation/test_reversion_v2.py

### 步骤 3: 行业轮动子策略 (1.5 天)
- [ ] 创建 `v2/industry_rotation_v2.py`
- [ ] 实现 `industry_rotation_score()`: 60d momentum on 20 A-share sectors
- [ ] 实现 `weekly_rebalance()`: 周度调仓
- [ ] 单元测试

### 步骤 4: 子策略权重 (1 天)
- [ ] 创建 `v2/sub_weighting_v2.py`
- [ ] 实现 `risk_parity_sub_weights()`: 用 Ledoit-Wolf
- [ ] 实现 `equal_sub_weights()`: 简单 baseline
- [ ] 单元测试

### 步骤 5: 多策略主回测 (1 天)
- [ ] 创建 `v2/multi_strategy_v2.py`
- [ ] 实现 `run_multi_strategy_backtest()`
- [ ] 集成到 `v2/backtest_v2.py`
- [ ] 单元测试

### 步骤 6: 回测验证 (1 天)
- [ ] 全周期 (2019-2026) Calmar/DD/Sharpe
- [ ] 924 专项 (2024-09 ~ 2024-10) 验证
- [ ] 与 v1.0 (单策略) 对比

### 步骤 7: 文档与提交 (0.5 天)
- [ ] `reports/momentum_etf_rotation/v2/stage16a_multi_strategy.md`
- [ ] `reports/momentum_etf_rotation/charts/v2/stage16a_*.html`
- [ ] 更新 STAGE_SUMMARY.md
- [ ] git commit

---

## 6. 测试计划

### 6.1 单元测试
- `test_reversion_v2.py` (新): 反转得分, 选股, 加权
- `test_industry_rotation_v2.py` (新): 行业动量, 周度调仓
- `test_sub_weighting_v2.py` (新): 风险平价, 等权
- `test_multi_strategy_v2.py` (新): 主回测

### 6.2 集成测试
- 全周期: Calmar ≥ 1.60 (不退化), DD ≥ -5%
- 924 专项: A股宽基 + 行业 ETF 权重 ≥ 10%
- 多策略 vs 单策略: 风险贡献均等 (各 25-40%)

### 6.3 回归测试
- 132 个已有测试不破
- v1.0 单策略作为对照基线

---

## 7. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 过拟合 | 中 | 高 | 严格 2 年 OOS, 多周期切片验证 |
| 调仓冲突 | 高 | 中 | 子策略独立 + 主入口合并 |
| 换手率上升 | 高 | 中 | 引入 cost_model 严格扣减 |
| 代码复杂度 | 中 | 低 | 充分单元测试 |

---

## 8. 验收标准

| 指标 | 阈值 |
|------|------|
| 全周期 Calmar | ≥ 1.65 (优于 v1.0 1.60) |
| 全周期 DD | ≤ -4.0% (不显著退化) |
| 924 期间涨幅 | ≥ 5% (vs v1.0 2.19%) |
| 月度胜率 | ≥ 62% (vs v1.0 58%) |
| 测试通过率 | 100% (新增 + 回归) |
| 文档完整 | ✅ |

---

## 9. 后续 Stage 联动

- Stage 16B (RSRS): 行业轮动子策略可叠加 RSRS 调整行业权重
- Stage 16C (RL): 多策略权重可由 RL 学习 (代替风险平价)
- Stage 16D (实时): 多策略输出接入模拟盘

---

## 10. 文档与资产

完成后产出:
1. `reports/momentum_etf_rotation/v2/stage16a_multi_strategy.md` (详细报告)
2. `reports/momentum_etf_rotation/charts/v2/stage16a_*.html` (3-5 个图表)
3. `reports/momentum_etf_rotation/docs/STRATEGY_VERSIONS.md` 更新 (新增 v2.0)
4. 更新 `STAGE_SUMMARY.md` 记录 Stage 16A
5. 更新 `DECISION_LOG.md` 记录关键决策
