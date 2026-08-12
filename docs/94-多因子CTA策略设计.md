# 94. 多因子 CTA 策略设计 — 三模块子策略 + 组合

> 状态: 已实施 (2026-08-12), 验收通过
> 依据: 业界 CTA 多因子主流做法调研 (AQR/Man AHL/元盛 + 本项目 SOP docs/research_history/06 §5/§7)

## 一、背景与目标

第八轮因子库新增 11 个信号因子 (振荡/趋势进阶/均值回归), 现将其组织为 CTA 策略:

- 目标: 用 11 个因子构建可回测的多因子 CTA 策略
- 结构: **3 个模块子策略, 各自独立回测验证后, 再组合为单一资金池策略**
- 定位: 保守型中频策略, 默认 10 品种中池 (AU/RB/M/CU/IF/I/T/TA/MA/C)

## 二、业界做法对照 (设计依据)

| 维度 | 业界主流 | 本项目落地 |
|---|---|---|
| 信号形态 | 连续信号 (z-score/rank), 保留强度 | raw_value() 连续化 + 滚动 z-score |
| 子策略拆分 | 按逻辑类别 (趋势/均值回归/期限结构) | 3 模块: 振荡 / 趋势进阶 / 均值回归 |
| 组合方式 | 合并持仓单一资金池 (consolidated book) | 模块加权合成 → 单一 vol targeting |
| 权重 | 趋势主导 (CTA 收益主力), 反转补充 | 振荡 0.3 / 趋势 0.5 / 均值回归 0.2 |
| 风控 | 组合层波动率目标 + 敞口限制 | vol_target_positions + PositionLimitRisk |
| 分散闸门 | 子策略相关 r > 0.7 无分散价值 | 回测输出模块级 + 因子级相关矩阵 |

不采用: 二值投票 (丢弃强度)、独立资金池 (忽略信号相关性)、细粒度 1/N 等权 (V3 教训, SOP §7.1)。

## 三、架构 (三层)

```
信号层: 11 因子 raw_value → 滚动 252 天时序 z-score (按品种, clip ±3)
   └─ 事件型因子 (Chandelier/Squeeze) 保持 0/±1, 不强行连续化
合成层: 模块内等权求和 → 模块连续信号 (T,N)
   └─ 因子级归因: 每因子信号面板独立保存, 可单因子独立回测
执行层: 模块加权合成净信号 → sign → vol_target_positions (单一资金池)
   └─ 手数整数化 → 方向变化才调仓 → run_cta_backtest
```

### 3.1 raw_value 定义 (11 因子)

`compute()` 的 -1/0/+1 保持不动, 新增 `raw_value(prices|ohlc) -> DataFrame`:

| 因子 | raw_value | 类型 | 方向语义 |
|---|---|---|---|
| MACDFactor | macd − 信号线 | 连续 | 柱为正 → 多头 |
| StochasticFactor | k − d | 连续 | 交叉上穿 → 多头 |
| CCIFactor | cci 值 | 连续 | >0 强势 / <0 弱势 |
| WilliamsRFactor | %R 值 | 连续 | 值越高越接近超买 |
| AroonFactor | up − down | 连续 | 上趋势主导 → 多头 |
| VortexFactor | vi_plus − vi_minus | 连续 | 上涨动量主导 → 多头 |
| LinearSlopeFactor | 回归斜率 | 连续 | 斜率正 → 多头 |
| TRIXFactor | trix 值 | 连续 | >0 → 多头 |
| HalfLifeFactor | −半衰期 | 连续 | 短半衰期 (快回归) → 反转多头 |
| ChandelierExitFactor | 0/±1 保持 | 事件 | 破多止损 → 空 |
| SqueezeFactor | 0/±1 保持 | 事件 | 收缩中 → 多; 释放突破定方向 |

### 3.2 模块定义

| 子策略 | 因子 | 合成 |
|---|---|---|
| OscillatorModuleStrategy | MACD / Stochastic / CCI / WilliamsR | 等权和 |
| TrendAdvancedModuleStrategy | Aroon / Vortex / LinearSlope / TRIX / Chandelier | 等权和 |
| MeanReversionModuleStrategy | HalfLife / Squeeze | 等权和 |

注: 组内合成仍保留每因子独立信号面板 (归因用)。

### 3.3 组合规则

- 净信号 = 0.3×振荡 + 0.5×趋势进阶 + 0.2×均值回归 (z-score 后的模块信号)
- 方向 = sign(净信号), 0 平仓
- 仓位 = 方向 × vol_target_positions (target_vol=0.15, vol_lookback=20, max_leverage=3.0)
- 资金: 单一资金池, initial_cash=200 万
- 调仓: 仅方向变化时 (复用 trend_following._PanelStrategy 模式)
- 风控: PositionLimitRisk

## 四、实施阶段

1. **git first**: 提交第八轮因子工作 (已提交 8dc86ec)
2. **doc first**: 本文档 (已完成)
3. 阶段 1: 11 因子加 raw_value() + 测试
4. 阶段 2: scripts/verify_round8_factors.py 真实数据验证
5. 阶段 3: 执行层基类 + 三个子策略, 逐个测试 + 独立回测
6. 阶段 4: 组合策略 + 相关矩阵 + 指标对比
7. 阶段 5: 回归 + docs 更新

## 五、验收标准

- 因子 raw_value: 形状对齐、无前视、值域合理; 测试通过
- 子策略: 信号值域 ⊆ {-1,0,1} (sign 后)、手数整数、run() 出指标
- 组合: 10 品种回测跑通, 输出指标 + 模块相关矩阵 + 因子相关矩阵
- 闸门: 跨模块相关 r ≤ 0.7 (组内相关高为预期); 若信号过稀或 r 超限, 报告并调整
- ruff clean + cta 测试目录全过

## 六、实施结果 (2026-08-12)

### 6.1 交付物

- `QuantNodes/strategy/cta/strategies/factor_modules.py`: 执行层基类 `_ModuleStrategy` + 3 模块子策略 + `FactorModuleComboStrategy`
- 11 因子新增 `raw_value()` 连续化 (事件型 Chandelier/Squeeze 保持 0/±1)
- `scripts/verify_round8_factors.py`: 真实数据因子验证脚本
- `tests/strategy/cta/test_strategies_factor_modules.py`: 28 测试

### 6.2 10 品种回测 (2018-01-01 ~ 2025-12-31, 单一资金池 200 万)

| 策略 | AnnRet | Vol | Sharpe | Sortino | MaxDD | Calmar | WinRate |
|---|---|---|---|---|---|---|---|
| 振荡模块 | 0.049 | 0.066 | 0.732 | 1.109 | -0.128 | 0.380 | 0.51 |
| 趋势进阶模块 | 0.048 | 0.068 | 0.716 | 1.070 | -0.151 | 0.321 | 0.50 |
| 均值回归模块 | 0.061 | 0.170 | 0.362 | 0.486 | -0.308 | 0.200 | 0.50 |
| **组合 (0.3/0.5/0.2)** | **0.057** | **0.065** | **0.876** | **1.317** | **-0.138** | **0.414** | **0.51** |

组合 Sharpe/Calmar 超越全部单模块 → 分散有效。

### 6.3 相关性闸门

- 模块级: 振荡×趋势 **+0.54**, 振荡×均值回归 **-0.01**, 趋势×均值回归 **-0.03** → 跨组 r < 0.7 通过
- 因子级: 组内平均相关 0.584, 跨组平均 0.198; half_life/squeeze 与其余因子近乎零相关 (分散主力)

### 6.4 测试

- `tests/strategy/cta/` 全目录: **460 passed, 16 skipped**, ruff clean
- 因子 raw_value: 11 项新增测试 (round8: 27 passed)
- 策略: 28 passed (构造/面板/信号/手数/归因/相关矩阵/组合回测)
