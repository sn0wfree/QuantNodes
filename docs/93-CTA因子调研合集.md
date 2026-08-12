# CTA 因子调研合集 (89-92 合并)

> 四轮 CTA 因子调研合并: 研报/学术来源、因子全景、实现状态与待落地清单。
> 状态: 68 因子已落地, 289 测试通过, ruff 干净。

---

## 一、研报/文献来源

| 来源 | 类型 | 核心贡献 |
|---|---|---|
| 海龟交易法则 | 经典 | 唐奇安突破 + ATR 仓位/止损, CTA 趋势鼻祖 |
| AQR《A Century of Evidence on Trend-Following》 | 学术 | 时间序列动量百年证据, 多周期聚合 |
| Moskowitz et al. (2012) TSMOM | 学术 | 跨资产时序动量统一框架 |
| Baltas & Kosowski (2015) | 学术 | 动量与期限结构复合增强 |
| Amihud (2002) | 学术 | 非流动性因子 |
| Koijen et al. (2018) | 学术 | 展期收益、库存与商品定价 |
| Hurst (1951) R/S 分析 | 学术 | 趋势/均值回归判别 |
| Lo & MacKinlay (1988) | 学术 | 方差比检验 (动量持续性) |
| Harvey et al. (2018) | 学术 | 偏度溢价, 彩票偏好 |
| Amaya et al. (2015) | 学术 | 已实现偏度 |
| Duan (1995) | 学术 | GARCH 波动率 |
| 华泰金工 | 券商研报 | 期限结构斜率/曲率、量价因子、日历效应、动量崩溃 |
| 中信期货 | 券商研报 | 持仓量/资金面因子、量价背离、CTA 仓位管理 |
| 招商证券 | 券商研报 | 横截面动量、因子复合方法论 |
| 兴业证券 | 券商研报 | 波动率结构因子、波动率择时 |
| 国泰君安 | 券商研报 | 展期收益、持仓结构分析 |

---

## 二、因子全景 (按类别)

### 1. 时序动量 (trend.py)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 唐奇安通道突破 | 价突破前 N 日高/低 | 海龟: 突破入场, 反向通道出场 | ✅ `DonchianChannelFactor` |
| 双均线交叉 | MA_short > MA_long | 经典: 趋势确认 | ✅ `MAFactor` |
| N 期动量 | sign(P_t / P_{t-N} - 1) | AQR: 时序动量 | ✅ `MomentumFactor` |
| 时间序列动量 | sign(12月收益), 波动率归一 | Moskowitz 2012 | ✅ `TSMAFactor` |
| RSI 超买超卖 | RSI<30 多 / >70 空 | 经典 | ✅ `RSIFactor` |
| 布林带突破 | %B>1 多 / <0 空 | 区间突破 | ✅ `BollingerFactor` |
| 考夫曼效率比 | \|净变动\| / 路径长 | 华泰: 趋势质量 | ✅ `EfficiencyFactor` |
| ADX 趋势过滤 | ADX>阈值才开仓 | 经典: 趋势/震荡判别 | ✅ `ADXFilterFactor` |

### 2. 期限结构 (term_structure.py)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 近月-次月价差 | F1 - F2 | 华泰: 贴水/升水状态 | ✅ `BasisFactor` |
| 展期收益率 | (F1-F2)/F1 × 年化 | 中信建投: 期限结构收益 | ✅ `RollYieldFactor` |
| 期限结构斜率 | (F4-F1)/F1 | 华泰金工: 陡峭度 | ✅ `TermSlopeFactor` |
| 期限结构曲率 | F1 - 2F2 + F3 | 华泰金工: 二阶凸性 | ✅ `TermCurvatureFactor` |
| 近月基差率 | (S-F1)/F1 | 期现套利: 现货升贴水 | ✅ `NearBasisFactor` |

### 3. 波动率 (volatility.py)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 历史波动率 HV | std(ln ret) × √252 | 兴业: 波动率聚集 | ✅ `HVFactor` |
| Parkinson 区间波动 | ln(H/L) 估计 | 学术: 高效波动估计 | ✅ `ParkinsonFactor` |
| 波动率 regime | HV z-score 高低 | AQR: 波动率择时 | ✅ `VolRegimeFactor` |
| 短长波动率差 | HV_short - HV_long | 兴业: 波动率期限结构 | ✅ `VolSpreadFactor` |
| GARCH(1,1) 波动 | 条件波动率 | Duan 1995 | ✅ `GARCHFactor` |
| 波动率偏度 | \|ret\| 分布偏度 | 兴业: 波动率冲击不对称 | ✅ `VolSkewnessFactor` |
| IV-HV 风险溢价 | IV - HV | 期权 CTA: VRP | ⏳ 阶段3 (需 IV) |

### 4. 量价微观结构 (volume.py)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 量能趋势 | sign(ΔP) × sign(ΔV) | 华泰: 量价配合 | ✅ `VolumeTrendFactor` |
| 放量突破 | V > N日均量 × 倍数 | 经典 | ✅ `VolumeBreakoutFactor` |
| 持仓量变化 | sign(ΔOI) | 中信: 资金流入流出 | ✅ `OpenInterestTrendFactor` |
| 能量潮 OBV | Σ sign(ΔP) × V | 经典 | ✅ `OBVFactor` |
| 量仓比 | V / OI | 中信: 换手活跃度 | ✅ `VolOIRatioFactor` |
| Amihud 非流动性 | \|ret\| / V | Amihud 2002 | ✅ `AmihudFactor` |
| 量价相关性 | rolling corr(ret, V) | 华泰: 量价同向性 | ✅ `VolumePriceCorrFactor` |
| 持仓-价格背离 | 增仓涨/减仓跌 四象限 | 中信: 资金面 | ✅ `OIPriceDivergenceFactor` |

### 5. 横截面 (momentum_xs.py)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 横截面动量 | 收益排名前20%多/后20%空 | 招商 | ✅ `MomentumXSFactor` |
| 低波异象 | 低波做多/高波做空 | 学术 | ✅ `VolatilityXSFactor` |
| 横截面期限结构 | 贴水深做多/升水深做空 | 招商 | ✅ `BasisXSFactor` |

### 6. 统计特征 (statistical.py)

| 因子 | 定义 | 研报逻辑 | 状态 |
|---|---|---|---|
| Hurst 指数 | R/S 分析, H>0.5 趋势 / H<0.5 回归 | Hurst 1951 | ✅ `HurstExponentFactor` |
| 方差比 VR | Var(k期)/[k×Var(1期)], VR>1 趋势 | Lo & MacKinlay 1988 | ✅ `VarianceRatioFactor` |
| 收益自相关 | corr(ret, ret_{t-k}) | 华泰: 动量/反转判别 | ✅ `AutocorrelationFactor` |
| 收益偏度 | 滚动偏度 | Amaya 2015: 负偏度溢价 | ✅ `SkewnessFactor` |

### 7. 微观结构 (microstructure.py)

| 因子 | 定义 | 研报逻辑 | 状态 |
|---|---|---|---|
| 隔夜收益 | open/pre_close - 1 | 华泰: 隔夜信息 | ✅ `OvernightReturnFactor` |
| 日内收益 | close/open - 1 | 华泰: 日内动能 | ✅ `IntradayReturnFactor` |
| 隔夜-日内背离 | 两收益方向相反 | 华泰: 分歧信号 | ✅ `OvernightIntradayDivergenceFactor` |
| 跳空缺口 | open - prev_close | 经典 | ✅ `GapFactor` |
| 收盘位置 | (close-low)/(high-low) | 经典: 收盘强弱 | ✅ `CloseLocationFactor` |

### 8. 复合增强 (composite.py)

| 因子 | 定义 | 研报逻辑 | 状态 |
|---|---|---|---|
| 多周期动量聚合 | 3/6/12月信号加权 | AQR: 多周期稳健 | ✅ `MultiHorizonMomentumFactor` |
| 波动率调整动量 | momentum / vol | 招商: 夏普动量 | ✅ `SharpeMomentumFactor` |
| 动量崩溃过滤 | 短负长正 → 规避 | 华泰: 动量崩溃 | ✅ `MomentumCrashFilterFactor` |

### 9. 产业链 (industrial_chain.py)

| 因子 | 定义 | 研报逻辑 | 状态 |
|---|---|---|---|
| 盘面利润 | 下游 - 上游×系数 | 利润极值回归 | ✅ `ChainMarginFactor` |
| 比价因子 | A/B 比值 z-score | 比价均值回归 | ✅ `PriceRatioFactor` |
| 利润动量 | 利润变化方向 | 利润趋势持续 | ✅ `MarginMomentumFactor` |
| 上下游传导 | 上游收益对下游相关 | 成本传导时滞 | ✅ `LeadLagFactor` |

**产业链配方 (8 个)**:

| 配方名 | 公式 | 研报 |
|---|---|---|
| steel_profit_rb | RB - 1.6×I - 0.5×J | 黑色产业链研报 |
| steel_profit_hc | HC - 1.6×I - 0.5×J | 同上 |
| coke_profit | J - 1.33×JM | 焦化利润 |
| pta_fee | TA - 0.655×PX | PTA 加工费 |
| polyester_profit | PF - 0.855×TA - 0.13×EG | 聚酯利润 |
| mto_profit | PP - 3×MA | 甲醇制烯烃 |
| crush_profit | 0.78×M + 0.18×Y - A | 大豆压榨 |
| hog_feed | LH - 3.5×(0.7×C + 0.3×M) | 生猪养殖 |

### 10. 日历效应 (calendar.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 月份季节性 | 同月历史平均收益 | 商品季节性 (华泰) | ✅ `SeasonalityFactor` |
| 周内效应 | 周内各日平均收益 | 周一/周五效应 | ✅ `WeekdayFactor` |
| 月末效应 | 月末/月初收益差 | 调仓/交割周期 | ✅ `MonthTurnFactor` |

### 11. 资金面 (capital.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 资金沉淀 | OI×价格 变化 | 资金流入 → 趋势确认 (中信) | ✅ `CapitalInflowFactor` |
| 成交额趋势 | volume×price | 活跃度变化 | ✅ `TurnoverFactor` |

### 12. 持仓结构 (position.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 持仓集中度 | 主力 OI / 总 OI | 投机集中度 (国君) | ✅ `OIConcentrationFactor` |
| 近月持仓占比 | 近月 OI / 总 OI | 近月活跃/远月布局 | ✅ `FrontMonthOIShareFactor` |

### 13. 量价背离 (divergence.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 顶底背离 | 价新高量不增 / 价新低量不减 | 背离 → 反转 (华泰) | ✅ `PriceVolumeDivergenceFactor` |
| 量价确认 | 放量+涨/跌 确认 | 量价配合 | ✅ `VolumeConfirmationFactor` |

### 14. 相关性结构 (correlation.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 平均相关性 | 品种间平均相关 | 系统性风险 | ✅ `AvgCorrelationFactor` |
| 横截面离散度 | 品种间收益标准差 | 机会广度 | ✅ `DispersionFactor` |

### 15. 基差动量 (basis_momentum.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 基差动量 | 基差率变化方向 | 基差走强/走弱 (华泰) | ✅ `BasisMomentumFactor` |
| 基差水平 | 基差率水平 | 贴水/升水深浅 | ✅ `BasisLevelFactor` |

### 16. 风险类 (risk.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 下行波动率 | 仅负收益 std | 下跌风险 (Sortino) | ✅ `DownsideVolFactor` |
| 滚动最大回撤 | N 日回撤 | 回撤风险 | ✅ `MaxDrawdownFactor` |
| 尾部风险 VaR | 5% 分位收益 | 尾部风险 (VaR/ES) | ✅ `TailRiskFactor` |

### 17. 宏观联动 (macro_link.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| VIX 联动 | VIX 大涨 → 商品空 | 风险偏好 (risk-off) | ✅ `VIXLinkFactor` |
| 美元联动 | DXY 升 → 商品空 | 美元-商品跷跷板 | ✅ `DXYLinkFactor` |
| 宏观状态 | VIX 高位+上升 → risk-off | 宏观择时 | ✅ `MacroStateFactor` |

### 18. 事件/跳变 (event.py)

| 因子 | 定义 | 逻辑 | 状态 |
|---|---|---|---|
| 极端波动 | 单日 \|收益\| 超 M 倍均值 | 事件驱动 | ✅ `ExtremeMoveFactor` |
| 连续跳空 | 连续同向跳空天数 | 跳空缺口策略 | ✅ `GapStreakFactor` |
| 涨跌停近似 | 单日收益接近涨跌停 | 涨跌停事件效应 | ✅ `LimitMoveFactor` |

### 19. 技术指标 (indicators.py)

纯 pandas 向量化实现, 无 TA-Lib 依赖:
`rsi` / `atr` / `adx` / `bollinger_bands` / `bollinger_percent_b` / `efficiency_ratio` / `donchian` / `zscore`

---

## 三、已实现因子统计

- **68 个因子** + 8 个技术指标函数 + 4 个工具函数 (vol_target_positions / basis_to_signal / chain_margin_series / compute_atr)
- **19 大类**: 时序趋势 / 期限结构 / 波动率 / 量价 / 横截面 / 统计特征 / 微观结构 / 复合增强 / 产业链 / 日历效应 / 资金面 / 持仓结构 / 量价背离 / 相关性结构 / 基差动量 / 风险类 / 宏观联动 / 事件跳变 / 技术指标
- 全部基于现有 DuckDB 数据 (continuous_main / inter_contract_spread / spot_basis_daily / high_freq_macro)

| 模块 | 因子数 | 数据源 |
|---|---|---|
| trend.py | 8 | continuous_main |
| term_structure.py | 5 | inter_contract_spread / spot_basis_daily |
| volatility.py | 6 + vol_target | continuous_main |
| volume.py | 8 | continuous_main (含 volume/oi) |
| momentum_xs.py | 3 | continuous_main / 派生表 |
| statistical.py | 4 | continuous_main |
| microstructure.py | 5 | continuous_main (OHLC) |
| composite.py | 3 | continuous_main |
| industrial_chain.py | 4 | continuous_main |
| calendar.py | 3 | trading_calendar |
| capital.py | 2 | continuous_main |
| position.py | 2 | inter_contract_spread |
| divergence.py | 2 | continuous_main |
| correlation.py | 2 | continuous_main |
| basis_momentum.py | 2 | spot_basis_daily |
| risk.py | 3 | continuous_main |
| macro_link.py | 3 | high_freq_macro (VIX/DXY) |
| event.py | 3 | continuous_main |
| indicators.py | 8 函数 | — |

---

## 四、待落地 (需外部数据或阶段3)

| 因子 | 需求 | 状态 |
|---|---|---|
| IV-HV 风险溢价 (VRP) | 期权 IV 计算 | ⏳ 阶段3 gamma_scrapper |
| 期权 IV 偏斜 Skew | 期权 IV | ⏳ 阶段3 |
| 动量×期限结构双重确认 | 现有因子组合 | 可后续组合 |
| 行业中性横截面动量 | 产业链分组 | 可后续组合 |
| 库存/仓单因子 | 交易所库存数据 | ❌ 数据缺失 |
| 会员持仓排名 | 席位数据 | ⚠️ 仅 SHFE 可用 |
| 现货升贴水序列 | akshare | ⚠️ 止于 2025-07 |
| 资金流向 (沉淀明细) | 非标数据 | ❌ 无接口 |

---

## 五、验证

```bash
python -m pytest tests/strategy/cta/ -q
# 289 passed, 16 skipped (值类因子), ruff clean
```

真实数据验证:
- AU/RB/M 期限结构正确区分 contango (AU) 与 backwardation (M)
- 产业链利润序列合理 (螺纹利润~1200, 压榨利润~-600)
- VIX/DXY 联动、事件因子均产生有效信号

---

## 六、文档合并记录

- 原 docs/89 (一轮: 经典五类 27 因子)
- 原 docs/90 (二轮: 统计/复合/微观结构)
- 原 docs/91 (三轮: 产业链)
- 原 docs/92 (四轮: 日历/资金/持仓/背离/相关)
- 2026-08-12: 合并为本文档, 并纳入基差动量/风险类/宏观联动/事件类
