# v1-v9 策略全演进: 核心因子/措施汇总

> 日期: 2026-07-22
> 资产: 43 ETF (A 股宽基/行业/海外/商品)
> 数据: 2018-01 ~ 2026-05 (8.4 年)

## 一、版本演进路线图

```
v1/v2 (CICC动量轮动) ──→ v3/v4 (子策略+因子择时) ──→ v5/v6 (行业轮动)
                                                         │
v7 (TV-PR宏观) ──→ v8 (Jump Model) ──→ v9 (宏观周期+因子配置+中信4策略)
```

## 二、逐版本核心机制与贡献因子

| 版本 | 核心机制 | 关键因子/措施 | OOS Sharpe | 最大贡献 |
|------|----------|--------------|------------|----------|
| **v1** | 动量轮动 + 逆波动加权 | momentum (144d), inverse_vol, stops, 80/20 固收+ | 0.15 | 建立 ETF 轮动框架 |
| **v2** | v1 + 固收+ 优化 | 同 v1, 加 bond_weight 调参 | ~0.15 | 数据层 (Tencent API) |
| **v3** | 子策略组合 | industry_rotation, reversion, sub_strategy, multi_strategy | N/A | 子策略分解思路 |
| **v4** | IC 驱动因子择时 | factor_ic, factor_timing, regime_detector, 5因子(momentum/value/reversal/dividend/quality) | N/A | 因子特异性 forward_window, regime-conditioned |
| **v5/v6** | 行业轮动 | industry_factors, industry_rotation | N/A | 行业维度探索 |
| **v7** | TV-PR 时变预测回归 | tvpr_estimator, macro_substrategy, expanding_window, ADMM, 9 macro + 11 量价 | 0.44 | **时变 β 估计** (核心突破) |
| **v8** | Jump Model (统计跳跃) | jump_model (DD_10/Sortino_20/Sortino_60), 2状态(bull/bear), 动态规划 | **1.485** | **牛熊状态检测 + 仓位控制** |
| **v9** | 宏观周期 + 因子配置 | factor_allocator (17因子), factor_galaxy (熵权+风险预算), dynamic_risk_parity, citic_* (4策略) | **1.23** | **动态仓位 (71% alpha)** |

## 三、10 大可借鉴因子/措施 (按优先级)

| 优先级 | 因子/措施 | 来源版本 | Sharpe 贡献 | 机制说明 |
|--------|----------|----------|-------------|----------|
| **P0** | 动态仓位 (pos 0.2-1.0) | v9 银河方案 | **+0.85** | z_score → pos = (0.7-0.5·z).clip(0.2,1.0), 贡献 71% alpha |
| **P0** | Jump Model 牛熊检测 | v8 | **+1.0+** | DD_10/Sortino_20/Sortino_60 → bull/bear → 仓位调节 |
| **P1** | TV-PR 时变 β | v7 | +0.30 | expanding_window_tvpr 捕捉结构性变化 |
| **P1** | 5 风格因子横截面打分 | v9 中信多因子 | +0.37 | momentum/volatility/quality/size/value → Top-K |
| **P1** | 风险平价基础 | v9 基础 RP | +0.02 (但 vol↓15-20%) | inv_vol 加权, 降低波动 |
| **P2** | 熵权法综合得分 | v9 银河因子配置 | +0.14 | 17 因子 → 熵权 → 综合得分 |
| **P2** | IC 驱动因子择时 | v4 | +0.10 | 滚动 IC 监控因子有效性, 动态分配权重 |
| **P2** | 象限定位 (增长/通胀) | v9 中信里昂 | +0.10 | 4 象限 → 资产类别倾斜 |
| **P3** | 行业轮动 Top-K | v9 中信轮动 | +0.03 | 行业内动量+质量 → Top-K 高配 |
| **避免** | 60/40 固定比例 | — | -0.05 | A 股相关性失效 |

## 四、信号正交性矩阵 (可叠加性)

| 高 (完全正交) | 中 (部分重叠) | 低 (互斥) |
|--------------|-------------|-----------|
| 动态仓位 + 多因子选股 | 风险平价 + 象限定位 | 60/40 + 风险平价 |
| Jump Model + TV-PR | IC 择时 + 熵权法 | 行业轮动 + 多因子选股 |
| TV-PR + 多因子选股 | — | — |

## 五、v10 推荐方案

### 方案 1 (主推): E + C 组合 (动态仓位 + 多因子选股)

```
pos_t = (0.7 - 0.5·z_score).clip(0.2, 1.0)
score = z(mom) - z(vol) + z(qual) - z(size) + z(value_reversal)
w_final = pos_t × softmax(score)
```

预期 Sharpe: **1.0 - 1.3**

### 方案 2 (备选): E + A + C 组合 (动态仓位 + 风险平价 + 多因子)

- 用风险平价作底仓 (替等权 1/N)
- 预期 Sharpe: **1.0 - 1.2**, Calmar 更好

### 方案 3 (激进): Jump Model + TV-PR + 多因子

- v8 Jump Model 做牛熊检测
- v7 TV-PR 做时变 β 估计
- 多因子做横截面选股
- 预期 Sharpe: **1.5+** (但复杂度高)

## 六、关键发现

1. **动态仓位是 #1 alpha 源** (贡献 71%, Brinson 归因验证)
2. **Jump Model 牛熊检测是 #2** (Sharpe 1.485, 但需与选股结合)
3. **TV-PR 时变 β 是 #3** (捕捉结构性变化, 但计算复杂)
4. **多因子横截面选股是 #4** (Sharpe +0.37, 可与上述叠加)
5. **风险平价不增加 alpha, 但降低波动** (改善 Calmar)
6. **5 宏观因子(中信版)失效**, IC 多为负
7. **60/40 在 A 股不稳定**, 2021-2026 表现最差

## 七、v1-v9 详细对比表

| 版本 | 策略名称 | 核心算法 | 因子数 | 调仓频率 | 成本假设 | Sharpe | Calmar | MaxDD |
|------|----------|----------|--------|----------|----------|--------|--------|-------|
| v1 | CICC动量轮动 | momentum + inverse_vol | 1 | 月度 | 5bp | 0.15 | 0.15 | -36% |
| v2 | v1 + 固收+ | 同 v1 + bond | 1 | 月度 | 5bp | 0.15 | 1.07 | -4.7% |
| v3 | 子策略组合 | multi_strategy | 多 | 月度 | 5bp | N/A | N/A | N/A |
| v4 | IC因子择时 | factor_ic + regime | 5 | 周度 | 5bp | N/A | N/A | N/A |
| v5/v6 | 行业轮动 | industry_rotation | 多 | 月度 | 5bp | N/A | N/A | N/A |
| v7 | TV-PR宏观 | expanding_tvpr | 20 | 月度 | 5bp | 0.44 | 0.62 | -8.5% |
| v8 | Jump Model | jump_model | 3 | 日度 | 5bp | 1.485 | 1.467 | -14% |
| v9 | 银河方案-动态仓位 | galaxy + pos | 17 | 周度 | 5bp | **1.23** | **1.20** | **-13.7%** |

## 八、产出文件

- `docs/54-v1_v9_strategy_summary.md` (本报告)
- `docs/53-v9_strategy_factor_analysis.md` (v9 策略分析)
- `docs/52-v9_citic_strategies.md` (中信 4 策略)
- `docs/51-v9_brinson_attribution.md` (Brinson 归因)
- `docs/49-v9_cycle_timing.md` (v9 设计文档)
- `docs/46-v8_ml_design.md` (v8 设计文档)
- `docs/39-v7_6_tvpr.md` (v7.6 设计文档)
- `docs/38-v7_3_macro_only.md` (v7.3 设计文档)
- `docs/17-固收+动量ETF轮动.md` (v1/v2 设计文档)
