# CTA 因子库说明

`QuantNodes/strategy/cta/factors/` — 期货/期权 CTA 策略可用的量价因子。

## 分类概览

| 分类 | 模块 | 因子 |
|---|---|---|
| **趋势 (时序)** | `trend.py` | Donchian / MA / Momentum / TSMOM / RSI / Bollinger / Efficiency / ADX |
| **波动率 (时序)** | `volatility.py` | HV / Parkinson / VolRegime / VolSpread / GARCH / VolSkewness |
| **量价 (时序)** | `volume.py` | VolumeTrend / VolumeBreakout / OI / OBV / VolOIRatio / Amihud / VolumePriceCorr |
| **期限结构 (时序)** | `term_structure.py` | Basis / RollYield / TermSlope / TermCurvature / NearBasis |
| **统计特征** | `statistical.py` | Hurst / VarianceRatio / Autocorrelation / Skewness |
| **微观结构** | `microstructure.py` | Overnight / Intraday / Divergence / Gap / CloseLocation |
| **复合增强** | `composite.py` | MultiHorizonMomentum / SharpeMomentum / MomentumCrashFilter |
| **产业链** | `industrial_chain.py` | ChainMargin / PriceRatio / MarginMomentum / LeadLag |
| **基差动量** | `basis_momentum.py` | BasisMomentum / BasisLevel |
| **风险类** | `risk.py` | DownsideVol / MaxDrawdown / TailRisk |
| **宏观联动** | `macro_link.py` | VIXLink / DXYLink / MacroState |
| **事件/跳变** | `event.py` | ExtremeMove / GapStreak / LimitMove |
| **日历效应** | `calendar.py` | Seasonality / Weekday / MonthTurn |
| **资金面** | `capital.py` | CapitalInflow / Turnover |
| **持仓结构** | `position.py` | OIConcentration / FrontMonthOIShare |
| **量价背离** | `divergence.py` | PriceVolumeDivergence / VolumeConfirmation |
| **相关性结构** | `correlation.py` | AvgCorrelation / Dispersion |
| **横截面** | `momentum_xs.py` | MomentumXS / VolatilityXS / BasisXS |
| **技术指标** | `indicators.py` | RSI / ATR / ADX / Bollinger / EfficiencyRatio / Donchian / ZScore |

## 因子详细说明

### 一、时序趋势因子 (`trend.py`)

| 因子 | 信号逻辑 | 参数 | 经济逻辑 |
|---|---|---|---|
| `DonchianChannelFactor` | 价 > 前 N 日最高 → 多; 价 < 前 N 日最低 → 空 | entry_window=20, exit_window=10 | 海龟趋势突破 |
| `MAFactor` | 短均线 > 长均线 → 多; 反之空 | short=5, long=20 | 均线交叉 |
| `MomentumFactor` | N 期收益 > 0 → 多; < 0 → 空 | lookback=20 | 动量延续 |
| `TSMAFactor` | sign(N 期收益) | lookback=250 | AQR 时间序列动量 |
| `RSIFactor` | RSI < 30 → 多; > 70 → 空 | window=14, low=30, high=70 | 超买超卖 |
| `BollingerFactor` | %B > 1 → 多 (突破上轨); < 0 → 空 | window=20, n_std=2 | 波动突破 |
| `EfficiencyFactor` | 效率比 > 0.5 → 多 (强趋势); < 0.15 → 空 | window=10 | 考夫曼趋势强度 |
| `ADXFilterFactor` | 输出 ADX 值 (趋势强度过滤用) | window=14 | 趋势/震荡判别 |

### 二、波动率因子 (`volatility.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `HVFactor` | std(ln ret) × √252 | window=20 | 历史波动率 |
| `ParkinsonFactor` | high/low 区间波动率 (更高效) | window=20 | 含日内区间信息 |
| `VolRegimeFactor` | HV z-score 高低 regime | window=20, lookback=120 | 波动率状态切换 |
| `VolSpreadFactor` | 短波动 - 长波动 z-score | short=10, long=60 | 波动率期限结构 |
| `GARCHFactor` | GARCH(1,1) 条件波动率 (arch 库) | window=252 | 波动率聚集建模 |
| `VolSkewnessFactor` | \|ret\| 分布偏度 | window=60 | 波动率冲击不对称 |
| `vol_target_positions` | 仓位 ∝ 目标波动率/实际波动率 | target_vol=0.15 | 波动率目标仓位 |

### 三、量价因子 (`volume.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `VolumeTrendFactor` | 价方向 × 量方向 | window=20 | 量价配合确认 |
| `OIPriceDivergenceFactor` | 价与持仓量背离 (增仓涨/减仓跌) | window=5 | 持仓-价格背离 |
| `VolumeBreakoutFactor` | 量 > N 日均量 × 倍数 | multiplier=2 | 放量突破 |
| `OpenInterestTrendFactor` | sign(OI 变化) | window=5 | 持仓增减方向 |
| `OBVFactor` | sign(OBV 变化) | window=20 | 能量潮趋势 |
| `VolOIRatioFactor` | 量仓比 z-score | window=20 | 换手活跃度 |
| `AmihudFactor` | \|ret\|/vol z-score | window=20 | 非流动性 |
| `VolumePriceCorrFactor` | 滚动 corr(ret, vol) | window=20 | 量价同向性 |

### 四、期限结构因子 (`term_structure.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `BasisFactor` | 近月-次月价差 (spread_1_2) | — | 贴水/升水状态 |
| `RollYieldFactor` | 展期收益率年化 | days_gap=30 | 期限结构收益 |
| `TermSlopeFactor` | (F4-F1)/F1 | — | 期限结构斜率 |
| `TermCurvatureFactor` | F1-2F2+F3 | — | 期限结构曲率 |
| `NearBasisFactor` | (S-F1)/F1 | — | 近月基差率 |
| `basis_to_signal` | 价差 → 方向信号 | threshold=0 | 贴水做多/升水做空 |

### 五、统计特征因子 (`statistical.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `HurstExponentFactor` | H>0.55 趋势多 / H<0.45 回归空 | window=120 | 趋势/回归状态 |
| `VarianceRatioFactor` | VR>1.1 动量 / VR<0.9 反转 | window=120, k=5 | 动量持续性 |
| `AutocorrelationFactor` | 正自相关多 / 负自相关空 | window=60 | 动量/反转判别 |
| `SkewnessFactor` | 负偏多 / 正偏空 | window=60 | 偏度溢价 |

### 六、微观结构因子 (`microstructure.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `OvernightReturnFactor` | 隔夜收益 (开-前收) | window=5 | 隔夜信息 |
| `IntradayReturnFactor` | 日内收益 (收-开) | window=5 | 日内动能 |
| `OvernightIntradayDivergenceFactor` | 隔夜/日内背离 | window=5 | 分歧信号 |
| `GapFactor` | 跳空缺口 | window=5 | 缺口动能 |
| `CloseLocationFactor` | 收盘在区间位置 | window=5 | 日内强弱 |

### 七、复合增强因子 (`composite.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `MultiHorizonMomentumFactor` | 3/6/12月动量加权 | horizons, weights | 多周期动量 |
| `SharpeMomentumFactor` | 动量/波动率 | lookback=63 | 夏普动量 |
| `MomentumCrashFilterFactor` | 短负长正 → 崩溃规避 | short=20, long=252 | 动量崩溃过滤 |

### 八、产业链因子 (`industrial_chain.py`)

基于产业链上下游价格关系 (配方见 `CHAIN_FORMULAS` / docs/91):

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `ChainMarginFactor` | 盘面利润 z-score 回归 | formula, window=60 | 高利润空/低利润多 |
| `PriceRatioFactor` | 两品种比值 z-score | num/den_product | 金银比/铜铝比 |
| `MarginMomentumFactor` | 利润变化方向 | formula, window=10 | 利润趋势 |
| `LeadLagFactor` | 上游→下游滚动相关 | upstream/downstream | 成本传导 |

内置配方 (8 个): steel_profit_rb/hc, coke_profit, pta_fee, polyester_profit, mto_profit, crush_profit, hog_feed_cost

### 九、基差动量因子 (`basis_momentum.py`)

基于 spot_basis_daily 基差率 (近月/主力):

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `BasisMomentumFactor` | 基差率变化方向 | window=10 | 基差走强/走弱 |
| `BasisLevelFactor` | 基差率水平 (贴水/升水) | window=5, threshold=0.005 | 期现基差回归 |

### 十、风险类因子 (`risk.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `DownsideVolFactor` | 下行波动率 z-score | window=20 | 下跌风险 |
| `MaxDrawdownFactor` | 滚动最大回撤 z-score | window=60 | 回撤风险 |
| `TailRiskFactor` | 历史 VaR (5% 分位) z-score | window=60 | 尾部风险 |

### 十一、宏观联动因子 (`macro_link.py`)

基于 VIX/DXY 宏观指标 (data/high_freq_macro):

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `VIXLinkFactor` | VIX 大涨 → 商品空 (risk-off) | window=5 | 风险偏好 |
| `DXYLinkFactor` | 美元走强 → 商品空 | window=5 | 美元-商品跷跷板 |
| `MacroStateFactor` | VIX 高位+上升 → risk-off | vix_window=20 | 宏观状态择时 |

### 十二、事件/跳变因子 (`event.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `ExtremeMoveFactor` | 单日极端涨跌 | window=20, mult=3 | 极端事件 |
| `GapStreakFactor` | 连续同向跳空计数 | streak=3 | 跳空趋势 |
| `LimitMoveFactor` | 接近涨跌停 | limit_pct=0.05 | 涨跌停事件 |

### 十三、日历效应因子 (`calendar.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `SeasonalityFactor` | 月份历史平均收益方向 | min_years=2 | 商品季节性 |
| `WeekdayFactor` | 周内各日历史收益方向 | min_weeks=50 | 周内效应 |
| `MonthTurnFactor` | 月末/月初收益差 | window=3 | 调仓周期效应 |

### 十、资金面因子 (`capital.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `CapitalInflowFactor` | 沉淀资金 (OI×价) 变化 | window=20 | 资金流入趋势 |
| `TurnoverFactor` | 成交额放量/缩量 | window=20 | 市场活跃度 |

### 十一、持仓结构因子 (`position.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `OIConcentrationFactor` | 主力 OI/总 OI 集中度 | window=20 | 投机集中度 |
| `FrontMonthOIShareFactor` | 近月 OI 占比 | window=20 | 近月活跃/远月布局 |

### 十二、量价背离因子 (`divergence.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `PriceVolumeDivergenceFactor` | 价新高量不增 (顶背离) | window=20 | 反转预警 |
| `VolumeConfirmationFactor` | 放量+涨/跌 确认 | window=20 | 趋势确认 |

### 十三、相关性结构因子 (`correlation.py`)

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `AvgCorrelationFactor` | 品种间平均相关 | window=60 | 系统性风险 |
| `DispersionFactor` | 横截面收益离散度 | window=20 | 机会广度 |

### 十四、横截面因子 (`momentum_xs.py`)

在同一时点比较所有品种，做多强品种/做空弱品种。

| 因子 | 逻辑 | 参数 | 用途 |
|---|---|---|---|
| `MomentumXSFactor` | 收益 > q(80%) → 多; < q(20%) → 空 | lookback=60 | 横截面动量 |
| `VolatilityXSFactor` | 低波 → 多; 高波 → 空 | window=20 | 低波异象 |
| `BasisXSFactor` | 贴水深 → 多; 升水深 → 空 | top_pct=0.2 | 横截面期限结构 |

### 九、技术指标 (`indicators.py`)

纯 pandas 向量化实现，无 TA-Lib 依赖：
`rsi` / `atr` / `adx` / `bollinger_bands` / `bollinger_percent_b` / `efficiency_ratio` / `donchian` / `zscore`

## 输入约定

所有因子基于**主力连续面板** (`continuous_main_{product}_daily`):
- 时序因子: (T, N) 价格 DataFrame, index=DatetimeIndex, columns=产品代码
- 量价因子: 额外需要 volume / open_interest 面板
- OHLC 因子 (ADX/Parkinson/微观结构): 需 open/high/low/close 列
- 期限结构因子: inter_contract_spread / spot_basis_daily 派生表

## 使用示例

```python
from QuantNodes.strategy.cta.factors import (
    DonchianChannelFactor, MomentumXSFactor, RSIFactor,
    HurstExponentFactor, MultiHorizonMomentumFactor, vol_target_positions,
)

# 时序因子: 每品种独立
sig = DonchianChannelFactor(20, 10).compute(price_panel)   # (-1/0/+1)
sig = RSIFactor(14).compute(price_panel)
sig = HurstExponentFactor(120).compute(price_panel)         # 趋势/回归状态

# 微观结构 (OHLC)
sig = OvernightReturnFactor(5).compute(ohlc_panel)

# 横截面因子: 品种间比较
sig = MomentumXSFactor(60, 0.2).compute(price_panel)       # 强多/弱空

# 复合: 多周期动量
sig = MultiHorizonMomentumFactor().compute(price_panel)

# 仓位: 波动率目标
pos = vol_target_positions(sig, price_panel, target_vol=0.15)
```

## 测试

```bash
python -m pytest tests/strategy/cta/test_factors.py \
  tests/strategy/cta/test_factors_more.py \
  tests/strategy/cta/test_factors_research.py \
  tests/strategy/cta/test_factors_statistical.py -v
```

## 研报来源

| 因子 | 研报/文献 |
|---|---|
| TermSlope / TermCurvature | 华泰金工《商品期货期限结构因子》 |
| VolOIRatio | 中信期货 CTA 研报 |
| Amihud | Amihud (2002) "Illiquidity and stock returns" |
| VolumePriceCorr | 华泰金工量价因子系列 |
| VolSpread | AQR 波动率择时、兴业《波动率结构因子》 |
| Hurst / VarianceRatio | Hurst (1951)、Lo & MacKinlay (1988) |
| Skewness | Harvey et al. (2018)、Amaya et al. (2015) |
| Overnight/Intraday | 华泰金工隔夜收益系列 |
| MultiHorizon / SharpeMom | AQR、招商 CTA 因子体系 |
| MomentumCrash | 华泰金工动量崩溃研究 |
| TSMA / MomentumXS | AQR《A Century of Evidence on Trend-Following》、招商 CTA 研报 |


