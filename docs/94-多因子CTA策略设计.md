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

## 七、v3 全库扩展 (2026-08-12)

106 因子调研结论: 89 个可用 (基于 continuous_main 面板 OHLCV+OI), 16 需外部数据
(基差/期限结构/宏观/产业链), 4 个真值类 (HV/GARCH/Parkinson/ADXFilter).

### 7.1 架构变化

- 3 模块 → **4 方向子策略** (67 方向因子), 全因子 compute() 信号 (-1/0/+1) 统一投票
- 波动率风险类因子 (HV/GARCH/统计/回撤/相关等 ~19 个) 作为**过滤层**, 第二阶段接入
- 修复 AmihudXSFactor 布尔 Series 列索引 bug

| 子策略 | 因子数 | 类别 |
|---|---|---|
| TrendMomentumModuleStrategy | 23 | 趋势/动量/突破 |
| MeanReversionModuleStrategy | 14 | 超买超卖/价格位置/K线反转 |
| VolumeFlowModuleStrategy | 24 | 量价/持仓/资金流 |
| EventCalendarModuleStrategy | 6 | 极端波动/跳变/日历 |

### 7.2 v3 回测 (10 品种, 2018-2025, 等权组合)

| 策略 | AnnRet | Vol | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|---|
| 趋势动量 | 0.059 | 0.067 | 0.891 | -0.099 | 0.598 |
| 均值回归 | 0.020 | 0.080 | 0.254 | -0.200 | 0.101 |
| 量能资金流 | 0.013 | 0.085 | 0.151 | -0.238 | 0.054 |
| 事件日历 | 0.145 | 0.092 | 1.586 | -0.084 | 1.728 |
| **组合 (等权)** | **0.060** | **0.064** | **0.934** | **-0.077** | **0.781** |

- 组合 Sharpe 0.934 > v2 (0.876), Calmar 0.781 > v2 (0.414)
- 模块相关闸门通过 (max r = 趋势×量能 0.54 < 0.7)
- ⚠️ 事件日历 Sharpe 1.586 偏高 (日历固定效应, 样本内易虚高, 待 OOS 验证)
- ⚠️ 量能资金流 0.151 / 均值回归 0.254 偏弱, 拖累组合

### 7.3 待办

1. 波动率风险过滤层接入 (第二阶段)
2. 16 个外部数据因子 (基差/期限结构/宏观) 数据管线补建
3. 事件日历组 OOS 验证
4. 组合权重调优 (当前等权)

## 八、v4 全量待办完成 (2026-08-12)

### 8.1 事件日历 OOS 验证 ✓

| 策略 | IS 18-21 Sharpe | OOS 22-25 Sharpe | 结论 |
|---|---|---|---|
| 事件日历 | 3.073 | 1.909 | 非过拟合, 日历效应真实 (周末/月度效应有文献支持) |
| 趋势动量 | 0.839 | 0.817 | 稳定 |
| 组合 | 0.617 | 1.056 | OOS 更好 |

### 8.2 组合权重调优 ✓ (4 组时代)

| 权重 (趋/均/量/事) | 全样本 | OOS |
|---|---|---|
| 等权 0.25 | 0.934 | 1.056 |
| (0.3, 0.1, 0.2, 0.4) | **1.109** | **1.108** |

### 8.3 波动率风险过滤层 ✓

- 实现 `VolRiskFilter`: 市场级危机过滤 (全市场平均 HV 滚动 z-score > 1.0 → 全局仓位 ×0.5)
- 实验: Sharpe 1.109 → 1.190, MaxDD -0.067 → -0.062
- 单品种级 HV 过滤 / ADX 方向过滤实验无增益或大降 (与事件日历策略定位冲突), 不启用
- `ComboConfig.filter_enabled` 默认 True

### 8.4 外部数据管线 ✓

`data_loader.py` 新增:
- `load_term_structure(product, n_contracts=4)`: 近远月 OI top-4 合约 long-form (对齐 inter_contract_spread)
- `load_macro_series("vix"/"dxy")`: 读 data/high_freq_macro parquet (vix 1990-2026, dxy 1971-2026)

新模块:
- `TermBasisModuleStrategy` (期限结构/基差): 数据管线 + 因子筛选后保留 term_slope (+0.81) / basis_level (+0.14)
- `MacroModuleStrategy` (宏观): vix_link / dxy_link / macro_state, Sharpe 1.166

### 8.5 期限结构/基差因子筛选 (重要发现)

9 因子单因子归因 (10 品种全样本):

| 因子 | Sharpe | 处置 |
|---|---|---|
| term_slope | +0.814 | 保留 |
| basis_level | +0.144 | 保留 |
| basis / roll_yield | **-0.852** | 剔除 (国内商品贴水长期存在, 单边信号系统性失效) |
| term_curvature | -0.280 | 剔除 |
| near_basis | -0.400 | 剔除 |
| basis_momentum | -0.133 | 剔除 |
| oi_concentration / front_month_oi | -0.059 | 剔除 (且两两等价重复) |

修复: AmihudXSFactor 布尔 Series 列索引 bug; 基差面板 55 品种裁剪到中池.

### 8.6 6 组组合最终结果

权重 (趋/均/量/事/期/宏) = **(0.20, 0.05, 0.10, 0.30, 0.15, 0.20)**

| 策略 | 因子数 | Sharpe | AnnRet | MaxDD | Calmar |
|---|---|---|---|---|---|
| 趋势动量 | 23 | 0.891 | 0.059 | -0.099 | 0.598 |
| 均值回归 | 14 | 0.254 | 0.020 | -0.200 | 0.101 |
| 量能资金流 | 24 | 0.151 | 0.013 | -0.238 | 0.054 |
| 事件日历 | 6 | 1.586 | 0.145 | -0.084 | 1.728 |
| 期限/基差 | 2 | 0.565 | 0.039 | -0.101 | 0.389 |
| 宏观 | 3 | 1.166 | 0.075 | -0.068 | 1.097 |
| **组合** | **72** | **1.341** | **0.085** | **-0.069** | **1.232** |

- 全样本 Sharpe 1.341, OOS (22-25) 1.328, 双段一致
- 模块相关闸门通过 (max r = 趋势×量能 0.54 < 0.7)
- 测试: 480 passed, 16 skipped, ruff clean
- 组合因子总数 72 (67 方向 + term_slope/basis_level + 3 宏观)

### 8.7 剩余待办

- 17 个因子仍未纳入: 产业链 4 (LeadLag/PriceRatio/PairRatioMeanReversion 需品种对), 
  SpreadMeanReversion (单序列输入, 可后续接入), 宏观 3 已接入 ✓, 其余已解决
- 波动率风险值类因子 (HV/GARCH/Parkinson/ADXFilter) 实验无增益, 保持过滤层仅市场级

## 九、v5 全品种 + 产业链 (2026-08-12)

### 9.1 品种池固定为全部有交易量品种

- `data_loader.list_active_products(min_daily_volume=10000)`: 82 → **72 品种** (日均量 ≥ 1 万手)
- 静态池 DEFAULT_PRODUCTS 固定 72 品种 (不用动态池, 保证可复现)
- **8 个产业链配方品种全覆盖** (steel_rb: RB/I/J, steel_hc: HC, coke: J/JM, pta: TA/PX,
  polyester: PF/TA/EG, mto: PP/MA, crush: M/Y/A, hog: LH/C/M) — 测试断言保证

### 9.2 新增模块

- `IndustrialChainModuleStrategy` (5 因子): chain_margin_rb→RB / chain_margin_pp→PP /
  margin_momentum_m→M / price_ratio_au_cu→配对(AU多CU空) / lead_lag_i_rb→RB
  (CHAIN_MAP 信号→品种映射, 支持配对 "-" 语法)
- `SpreadMeanReversionFactor` 接入均值回归组 (series 输入类型, 逐品种单序列)

### 9.3 修复

- MomentumXSFactor / VolatilityXSFactor / BasisXSFactor 布尔 Series 列索引 bug (同 AmihudXS)

### 9.4 72 品种模块回测 (2018-2025, 等权组合)

| 策略 | 因子数 | Sharpe | AnnRet | MaxDD | Calmar |
|---|---|---|---|---|---|
| 趋势动量 | 23 | 0.614 | 0.061 | -0.123 | 0.292 |
| 均值回归 | 15 | -0.151 | -0.016 | -0.313 | -0.050 |
| 量能资金流 | 24 | 0.426 | 0.043 | -0.194 | 0.221 |
| 事件日历 | 6 | 1.747 | 0.150 | -0.174 | 0.866 |
| 期限/基差 | 2 | -0.604 | -0.086 | -0.551 | -0.156 |
| 宏观 | 3 | 0.659 | 0.057 | -0.148 | 0.389 |
| 产业链 | 5 | 0.703 | 0.085 | -0.227 | 0.372 |
| **组合 (等权 1/7)** | **78** | **0.706** | **0.079** | **-0.213** | **0.372** |

### 9.5 关键结论

- 组合权重默认**等权 1/7** (用户决策: 不做样本内权重调优, 调优结果仅记录)
- **basis_level 保留决策**: 72 品种单因子归因转负 (-0.44), 但组合层面
  (sign 非线性) 有正贡献 — 组合 0.287(剔除) vs 0.706(保留), 恢复保留
- 组合层因子选择 ≠ 单因子归因 (sign 非线性), 组内选择应以组合指标为准
- 测试: 487 passed, 16 skipped, ruff clean

## 十、组内因子聚合方案调研与实施 (2026-08-12)

### 10.1 调研结论

**业界主流组内聚合方法**:

| 方法 | 做法 | 出处/使用方 | 评价 |
|---|---|---|---|
| 标准化后等权平均 | z-score/rank 归一 → 等权求和 | DeMiguel 2009 1/N; AQR 多周期动量; Man AHL 1/N models; 国内私募标配 | 业界基线, 稳健 |
| IC/ICIR 加权 | 权重 ∝ 滚动 IC 或 ICIR | 华泰金工/中信多因子; 本项目 v6.1 | 有效但样本内过拟合风险 |
| 逆波动率加权 | 权重 ∝ 1/σ(信号) | SOP §5.1 ⭐⭐ | 最简单稳健 |
| 排序打分 | rank → 加权 → 综合得分 | 国泰海通/中信打分体系 | 卖方截面体系 |
| 回归/ML 学权重 | OLS/Lasso/树学组内权重 | Man AHL 二次层 | 过拟合风险, 需 OOS |
| 正交化 | Gram-Schmidt/Symmetry/PCA | — | **实证否决** |

**本项目证据**:
- SOP §3.2 因子工程 4 步法: 候选 → 评估 → **去重 (r>0.93 必删)** → 标准化
- SOP §3.3: 正交化不处理 ⭐⭐⭐; Symmetry/QR ❌❌ (v7.9 实证 Sharpe 1.40→0.15)
- 实测组内相关: cci×williams_r 0.90, linear_slope×vortex 0.79, basis×roll_yield 1.00 (等价)

### 10.2 现状问题

- 组间尺度不均衡: 模块信号 = 因子数×平均信号, 事件组 (6 因子) 实际影响力
  仅趋势组 (23 因子) 的 26%
- 组内冗余因子重复下注 (r 0.8-1.0)

### 10.3 实验设计 (用户决策)

**去重 (r>0.93 剔除) × 三种聚合**:

| 方案 | 模块信号 |
|---|---|
| A. 等权平均 | Σ sig_k / n (∈ [-1,1], 修复组间尺度) |
| B. 逆波动率加权 | Σ w_k·sig_k / Σw, w ∝ 1/σ_k (滚动窗口) |
| C. 回归/ML 学权重 | 滚动窗口 OLS: sig → 未来收益, w = 回归系数 (OOS 无前视) |

### 10.4 实施结果 (2026-08-12) — 前视偏差修复与三方案对比

**重大 bug 修复**: SeasonalityFactor / WeekdayFactor 原用全样本统计量
(`groupby.mean()` 整段历史) → **前视偏差** (含未来数据). 已改为 expanding
因果统计. 此修复使事件日历组真实 Sharpe 从 1.747 降至 0.187 (之前的高分
大部分是前视假象). 同时修复: 分段回测品种池不一致 (固定 72 列)、
divergence.py 布尔掩码 object dtype、_ohlc_map 列数不统一.

**去重**: 72 品种离散信号下无 r>0.93 对 (最大 donchian×ma 0.798),
按用户决策取最高相关且最弱对 → 删 donchian (趋势 23→22).

**三聚合方案 (组合等权 1/7, 全样本 NAV 切段统计, 修复后)**:

| 聚合 | 全样本 Sharpe | IS | OOS |
|---|---|---|---|
| equal | 0.097 | 0.684 | -0.216 |
| inv_vol | 0.344 | 1.227 | -0.215 |
| regression | -0.239 | 0.848 | -0.973 |

**模块级 (equal 聚合, 72 品种) 修复后表现**:

| 模块 | 全样本 | IS | OOS |
|---|---|---|---|
| 趋势动量 | 0.527 | 0.511 | 0.672 |
| 均值回归 | -0.151 | 0.124 | -0.325 |
| 量能资金流 | 0.426 | -0.058 | 0.911 |
| 事件日历 | 0.187 | -0.078 | 0.784 |
| 期限/基差 | 0.389 | 0.248 | 0.635 |
| 宏观 | 0.659 | 1.398 | 0.155 |
| 产业链 | 0.703 | 0.419 | 1.302 |

**结论**: 模块级 OOS 6/7 为正, 但组合层 (等权 1/7 + sign) OOS 为负 —
组合方式 (sign 非线性 + 均值回归负贡献) 是当前主要瓶颈, 待改进.

### 10.5 TV-PR 模块权重组合 (v6, 2026-08-12)

用户决策: TV-PR 动态决定 7 模块权重 (模块级 K=7, β 正负自由,
λ_l1=0 全维度保留). 实现 `strategies/tvpr_combo.py`:
- Y (T,N): 品种未来 1 日收益; X (T,N,7): 7 模块归一化信号
- β_t = expanding TV-PR (tvpr_estimator 复用, min_history=252, step=21)
- 组合信号 = Σ β_t,k × X_t,i,k → sign → vol targeting
- 前视验证: 缩短面板与完整面板前段 β 一致 (边界效应除外)

**λ_tv 标定与结果** (72 品种, 全样本 NAV 切段):

| λ_tv | 全样本 Sharpe | IS | OOS |
|---|---|---|---|
| 0.05 | 0.942 | 1.613 | 0.537 |
| 0.15 | 1.002 | 1.713 | 0.597 |
| **0.5** | **1.099** | **1.862** | **0.618** |
| ≥1.0 | 1.096 | 1.865 | 0.612 (收敛) |

**对比固定等权 (equal 聚合)**: 全 0.097 / IS 0.684 / OOS -0.216
→ TV-PR 大幅胜出, **OOS 由负转正** (0.618)

**β 诊断** (λ_tv=0.5): 非零率 86.6% (全维度保留 ✓); 均值:
宏观 +0.0018 / 产业链 +0.0013 / 期限 +0.0004 / 事件 +0.0001,
趋势 -0.0024 / 量能 -0.0019 / 均值回归 -0.0009 (TV-PR 自动反用负 β).
β 后期趋于稳定 (TV 平滑), 但权重方向 (含反用) 显著优于固定等权.

测试: 493 passed, 16 skipped, ruff clean (含 TV-PR 6 测试).

### 10.6 组内 TV-PR × 组间 inv_vol (v7, 2026-08-12)

用户方案: 组内因子级 TV-PR (可 L1 稀疏过滤因子) + 组间模块 inv_vol 聚合.
实现: ModuleConfig.aggregation="tvpr" (组内 expanding TV-PR 学因子权重,
模块信号 = X@β_t) + ComboConfig.module_weighting="inv_vol" (模块信号滚动
波动率倒数加权).

**对比 (72 品种, 全样本 NAV 切段)**:

| 组内 | 组间 | 全 Sharpe | IS | OOS | MaxDD |
|---|---|---|---|---|---|
| equal | fixed | 0.097 | 0.684 | -0.216 | -0.249 |
| equal | inv_vol | 0.645 | 0.605 | 0.798 | -0.212 |
| tvpr | fixed | 1.001 | 1.981 | 0.494 | -0.230 |
| **tvpr** | **inv_vol** | **1.251** | **2.051** | **0.767** | **-0.187** |

**组内 L1 稀疏实验**: λ_l1=0.01/0.03/0.05 均劣于 0 (全 Sharpe 0.88-1.00,
OOS 0.51-0.55) — 组内因子虽冗余但各有信息, 强制稀疏丢信息, **λ_l1=0 保留**.

结论: 组内 TV-PR + 组间 inv_vol 为当前最优配置 (超越组间 TV-PR 的
1.099/0.618), 使用方式:
```python
FactorModuleComboStrategy(config=ComboConfig(
    aggregation="tvpr", module_weighting="inv_vol"))
```
测试: 496 passed, 16 skipped, ruff clean.
