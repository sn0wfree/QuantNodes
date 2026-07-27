# v10 自上而下完备框架 — 最终规划 (用户确认版)

> 日期: 2026-07-23
> 状态: ✅ 用户确认, 开始编码
> 基于: v1-v9 全部策略核心因子/措施分析 (docs/54) + 敏感性测试 (Stage 30)

---

## 一、设计目标

v10 要做一个**自上而下的完备框架**,整合 v1-v9 所有优点:

- **可解释性**: 每个权重来自具体模块, 可逐层追溯
- **可回测**: 单一引擎, 周/月频可配, 5bp 成本
- **业绩优秀**: 目标 Sharpe **1.3-1.8**, Calmar **1.4-1.6**
- **集成优势**: 复用 v9 实证有效机制 (动态仓位 + 银河选股 + Jump Model)

## 二、用户决策 (2026-07-23)

| # | 议题 | 用户决策 | 实现 |
|---|------|----------|------|
| 1 | TV-PR (Layer 1) | **必加, 可配置** | 默认开启, 加 `use_tvpr: bool` 配置项 |
| 2 | Top-K 值 | **K=10** | 改 K=10 (Stage 30 实证 K=3 是 v4B 单策略, v10 多策略需 K=10) |
| 3 | Smart β 4 因子 vs 中信 5 因子 | **5 因子** | Layer 2C 用 v9 中信多因子 (mom/vol/qual/size/value_reversal) |
| 4 | Jump Model | **需要** | Layer 3 输出 bear_prob 乘到仓位 (0.5x 调整) |
| 5 | 调仓频率 | **周+月都测试** | `rebal_freq: 'W' | 'M'`, 出对比表 |
| 6 | 估值/基本面 | **v10.0 暂不加, v10.1 加** | v10.0 暂用代理 (已有 5 因子), v10.1 加 |

## 三、5 层架构 (基于实证调整)

```
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 5: 组合构建 (Portfolio Construction)                          │
│  ─ w_final = position × (base_rp × style_tilt × factor_tilt × sector_tilt) │
│  ─ 输出: 43 ETF 权重 + cash 比例                                     │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 4: 仓位缩放 (Dynamic Position)  ⭐ P0 #1                       │
│  ─ pos_t = (0.7 - 0.5·z_score).clip(0.2, 1.0)                       │
│  ─ pos *= (1 - bear_prob × 0.5)                                    │
│  ─ 借鉴: v9 银河方案 (Brinson 归因 71% alpha)                         │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 3: 风险控制 (Risk Management)  ⭐ P0 #2                       │
│  ─ Jump Model (v8): DD_10/Sortino_20/Sortino_60                     │
│  ─ 输出: bear_probability ∈ [0, 1]                                  │
│  ─ 借用: v8 jump_model_periodic_retrain (无未来函数版)              │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 2: 选股决策 (Stock Selection)                                 │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 2A: 行业轮动 (23 个行业 ETF)                                │    │
│  │   - 动量 + 反向波动率打分                                    │    │
│  │   - Regime 条件: bull→进攻型, bear→防御型                  │    │
│  │   - 相关性约束: corr > 0.7 剔除冗余 (Stage 30 实证无效, 关闭)│    │
│  │   - 输出: sector_tilt                                       │    │
│  └────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 2B: 风格轮动 (IC 驱动, 复用 v4 factor_timing_v4)            │    │
│  │   - 6 因子 IC: momentum/value/reversal/quality/size/low_vol│    │
│  │   - 因子特异性 FW + lag 平滑 + 阈值过滤                     │    │
│  │   - Regime-conditioned 权重                                 │    │
│  │   - 输出: style_weights (子策略权重)                        │    │
│  └────────────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ 2C: 因子选股 (43 ETF 横截面打分)  ⭐ P1 #3 + #4              │    │
│  │   - 5 因子: momentum/volatility/quality/size/value_reversal│    │
│  │   - Top-K=10 选优 + softmax 加权 (用户决策)                │    │
│  │   - candidate_pool 50% 权重 + 剩余 33 等权底仓              │    │
│  │   - 输出: factor_tilt (个股得分)                            │    │
│  └────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                              ▲
┌──────────────────────────────────────────────────────────────────────┐
│  Layer 1: 宏观择时 (Macro Regime)                                    │
│  ─ 5 宏观因子 z-score (growth/inflation/credit/fx/rate)            │
│  ─ 熵权法合成综合得分 (借鉴 v9 银河 factor_galaxy)                   │
│  ─ TV-PR 时变β (v7, 默认开启, 可配置)  ⭐ 用户决策                  │
│  ─ 输出: macro_signal ∈ [-1, +1], regime_state                      │
│  ─ 驱动: Layer 2 regime 条件 + Layer 4 动态仓位                     │
└──────────────────────────────────────────────────────────────────────┘
```

## 四、核心公式

```python
# === Layer 1: 宏观择时 ===
# 5 宏观因子 z-score
macro_z = {
    'growth':   z(宏观增长因子),
    'inflation': z(宏观通胀因子_生活端),
    'credit':   -z(信用利差因子),    # 反向
    'fx':       -z(宏观汇率因子),    # 反向
    'rate':     z(期限利差因子_债),  # 走阔利好股
}
# 熵权法
weights = entropy_weight(macro_z, window=104)
macro_score = Σ weights[k] × macro_z[k] for k in macro_z
# TV-PR (可选)
if use_tvpr:
    beta_tvpr = expanding_window_tvpr(Y=etf_returns, X=macro_features)
    macro_score = 0.5 × macro_score + 0.5 × z(beta_tvpr.mean())

regime_state = 'bull' if macro_score > 0.5 else 'bear' if macro_score < -0.5 else 'neutral'

# === Layer 2A: 行业轮动 ===
sector_score = z(momentum) - z(volatility)  # 仅行业 ETF
if regime == 'bull':
    sector_score[offensive_industries] *= 1.5
elif regime == 'bear':
    sector_score[defensive_industries] *= 1.5
sector_tilt = softmax(sector_score)  # Top-K=10 行业高配

# === Layer 2B: 风格轮动 (IC 驱动, 复用 v4 factor_timing_v4) ===
for style in [momentum, value, reversal, quality, size, low_vol]:
    ic[style] = rolling_ic(factor[style], forward_return, window=60)
ic_weight[style] = max(0, ic[style] + 0.05) ** 2
# Regime 条件
if regime == 'bull':
    ic_weight['momentum'] *= 1.3
elif regime == 'bear':
    ic_weight['value'] *= 1.5
    ic_weight['quality'] *= 1.5
ic_weight = ic_weight / sum(ic_weight)

# === Layer 2C: 因子选股 (5 因子, K=10) ===
factor_score = (
    z(momentum_52w_skip4) * 1.0
    - z(volatility_26w) * 1.0
    + z(quality_sharpe_26w) * 1.0
    - z(size_amplitude_4w) * 1.0
    + z(value_reversal_104w) * 1.0
)
# Top-K=10 选优
top_k_etfs = factor_score.nlargest(10).index
softmax_w = softmax(factor_score[top_k_etfs])
# candidate_pool 50% + 剩余 50% 等权
factor_tilt = candidate_pool_weight(50%) × softmax_w + rest_pool(50%) × 1/N_rest

# === Layer 3: 风险控制 (Jump Model) ===
features = compute_features(etf_returns)  # DD_10, Sortino_20, Sortino_60
states = jump_model_periodic_retrain(etf_returns, asset_type='equity')
bear_prob = states.rolling(60).mean()  # 60 日 bear 频率

# === Layer 4: 动态仓位 (双控) ===
z_score = (macro_score + sector_score.mean() + ic_weight_weighted_mean) / 3
position_size = (0.7 - 0.5 * z_score).clip(0.2, 1.0)
position_size *= (1 - bear_prob × 0.5)  # 用户决策: Jump Model 调整

# === Layer 5: 组合构建 ===
base_rp = inv_vol / sum(inv_vol)        # 风险平价基础
w_combined = base_rp × sector_tilt × factor_tilt
# style_weight 通过子策略权重影响 (v4 风格的 style_rotation 槽位)
w_final = position_size × w_combined
w_final = w_final / sum(w_final)
```

## 五、关键参数

| 参数 | 值 | 来源 | 实证依据 |
|------|-----|------|----------|
| **Top-K** | **10** | 用户决策 | v9 中信多因子 K=10 黄金点 |
| candidate_pool_weight | 0.50 | v9 中信多因子 | 50% 候选 + 50% 等权 |
| Softmax temperature | 1.0 | v9 中信多因子 | 默认 |
| factor_tilt_max_weight | 0.15 | v9 中信多因子 | 单 ETF 上限 |
| jump_penalty | 50 | v8 jump_model | 权益类参数 |
| train_window (Jump) | 1000 天 | v8 | 权益类参数 |
| retrain_every (Jump) | 30 天 | v8 | 权益类参数 |
| macro_entropy_window | 104 周 | v9 银河 | 2 年滚动 |
| ic_window (style) | 60 周 | v4 factor_timing | Stage 18 |
| ic_base | 0.05 | v4 factor_timing | Stage 18 |
| ic_power | 2.0 | v4 factor_timing | Stage 18 |
| position_z_score 范围 | [-1, 1] | v9 银河方案 | Brinson 归因验证 |
| position_size 范围 | [0.2, 1.0] | v9 银河方案 | 实证最优 |
| bear_prob_adjustment | 0.5 | 用户决策 | bear 概率降仓 50% |
| macro_score 阈值 | ±0.5 | v9 | bull/bear 切换 |
| industry_top_k | 5 | v9 中信行业轮动 | 23 行业里 Top-5 |
| sector_mult | 5.0 | v9 中信行业轮动 | Top-5 行业 5x 加权 |
| rebal_freq | W/M 都测试 | 用户决策 | 出对比表 |
| cost_bps | 5.0 | v4/v9 统一 | 单边 5bp |
| warmup_days | 252 | v4 | 1 年预热 |

## 六、与现有 v9/v4 的复用关系

| v10 Layer | 复用自 | 修改点 |
|-----------|--------|--------|
| Layer 1 宏观 (熵权) | v9 factor_galaxy.py | 直接复用 `entropy_weight()` |
| Layer 1 宏观 (TV-PR) | v7 tvpr_estimator.py | 复用 `expanding_window_tvpr()` |
| Layer 2A 行业 | v9 citic_rotation.py | 加 regime 条件 + 相关性约束开关 |
| Layer 2B 风格 | v4 factor_timing_v4.py | 复用 `compute_factor_weights_fusion()` |
| Layer 2C 因子 | v9 citic_multifactor.py | 改 K=10 (从 K=10 沿用) |
| Layer 3 风险 | v8 jump_model.py | 复用 `jump_model_periodic_retrain()` |
| Layer 4 仓位 | v9 position.py | pos 公式 + bear_prob 双控 |
| Layer 5 组合 | v4 multi_strategy_v4.py | 复用 `run_v4_backtest()` 框架 |
| 回测引擎 | v9 backtest.py | 复用 `compute_metrics()` + `run_backtest()` |

## 七、文件结构

> **历史注 (2026-07-27)**: 本设计 5 层架构代码已迁移到 `v11/`.
> `v10/` 现在只保留 4 策略 Vol-parity 主体.
> `scripts/v10/v10_backtest.py` 等 5 层脚本已迁移到 `scripts/v11/v11_backtest.py`.

```
QuantNodes/strategy/momentum_etf_rotation/v11/  (5 层架构 + ACT-1/2/3, 从 v10 迁移)
├── __init__.py
├── config_v11.py           # 配置中心 (所有可调参数)
├── macro_layer.py          # Layer 1: 宏观择时 (5 因子 + 熵权 + TV-PR)
├── industry_layer.py       # Layer 2A: 行业轮动
├── style_layer.py          # Layer 2B: 风格轮动 (IC 驱动)
├── factor_layer.py         # Layer 2C: 因子选股 (5 因子 + K=10)
├── risk_layer.py           # Layer 3: Jump Model 风险控制
├── risk_layer_v11.py       # ACT-2/3: Kelly 审计 + 回撤控制 (v11 新增)
├── position_layer.py       # Layer 4: 动态仓位
├── portfolio_layer.py      # Layer 5: 组合构建
├── v11_strategy.py         # 主入口: 5 层串联 + ACT-1/2/3
└── backtest_v11.py         # v11 回测引擎

scripts/v11/
└── v11_backtest.py         # v11 回测 + 对比 (吸收自 v10_backtest.py)
```

## 八、预期性能

| 指标 | v9 银河方案 | v9 中信多因子 | **v10 预期** |
|------|-------------|---------------|--------------|
| Sharpe | 1.23 | 0.62 | **1.3 - 1.8** |
| Calmar | 1.20 | 0.50 | **1.4 - 1.6** |
| MaxDD | -13.7% | -18.0% | **-8% ~ -11%** |
| 年化 | 16.4% | 9.0% | **18% - 25%** |

**提升来源**:
1. 动态仓位 (P0 #1) → +0.85
2. Jump Model 牛熊 (P0 #2) → +0.10-0.20
3. 多因子选股 5 因子 + K=10 (P1 #3) → +0.37
4. 风格轮动 IC 驱动 (P2 #8) → +0.10
5. 风险平价底仓 (P1 #4) → +0.05 (vol↓)
6. TV-PR 宏观择时 (P1 #5) → +0.15
7. 行业轮动 regime 条件 (P3 #10) → +0.03

## 九、实施顺序 (v10.0)

```
Step 1: 创建 v10 目录 + config_v10.py
Step 2: Layer 1 macro_layer.py (5 因子 + 熵权, TV-PR 可选)
Step 3: Layer 2A industry_layer.py (regime 条件)
Step 4: Layer 2B style_layer.py (复用 v4 factor_timing)
Step 5: Layer 2C factor_layer.py (5 因子 + K=10)
Step 6: Layer 3 risk_layer.py (Jump Model)
Step 7: Layer 4 position_layer.py (动态仓位 + bear_prob)
Step 8: Layer 5 portfolio_layer.py (RP × tilt × pos)
Step 9: v10_strategy.py 主入口 + backtest_v10.py 回测引擎
Step 10: scripts/v10/v10_backtest.py (周+月跑回测)
Step 11: scripts/v10/v10_compare.py (对比 v9 银河方案/中信多因子)
Step 12: 验证结果, 输出报告
```

## 十、与 docs/55 (讨论稿) 的差异

| 维度 | docs/55 (讨论稿) | 本方案 (基于实证 + 用户决策) |
|------|------------------|------------------------------|
| 顶层架构 | 4 层 | **5 层** (加 Layer 4 仓位) |
| 估值因子 | 行业轮动加估值 | **v10.1 再加, v10.0 暂不加** |
| Top-K | K=5-8 | **K=10** (用户决策) |
| 因子加权 | 5 风格因子 (中信) | **5 因子 + K=10** (用户决策) |
| TV-PR | 必加 | **默认开启 + 可配置** (用户决策) |
| Jump Model | Layer 3 + Layer 4 | **Layer 3 bear_prob × Layer 4 pos 双控** (用户决策) |
| 调仓频率 | 未定 | **周+月都测试** (用户决策) |

## 十一、产出文件清单

### 文档
- `docs/57-v10_final_design.md` (本文件, 用户确认版)

### 代码
- `QuantNodes/strategy/momentum_etf_rotation/v10/` (10 个文件)
- `scripts/v10/` (3 个脚本)

### 报告
- `reports/momentum_etf_rotation/v10/v10_backtest_results.csv` (周+月)
- `reports/momentum_etf_rotation/v10/v10_compare.md` (vs v9)
- `reports/momentum_etf_rotation/v10/v10_report.md`

## 十二、风险与回滚

1. **TV-PR 计算慢**: 加 `use_tvpr=False` 可关闭, 默认开启
2. **Jump Model 训练慢**: 用 v8 jump_model_periodic_retrain (已优化), 加 cache
3. **多策略叠加可能过拟合**: 用滚动 walk-forward 验证, 避免 look-ahead
4. **数据缺失**: 43 ETF 历史可能不足, 用 v9 实证可行的 2021-08-01 起始点