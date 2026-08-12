# 89-CTA因子研报调研与落地

> 调研业界 CTA 因子 (研报/学术文献), 评估基于现有 DuckDB 数据的可落地性。
> 先记录调研结论, 后按此实现因子。状态: 调研完成, 因子已落地。

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
| 华泰金工 | 券商研报 | 期限结构斜率/曲率、量价因子系列 |
| 中信期货 | 券商研报 | 持仓量/资金面因子, CTA 仓位管理 |
| 招商证券 | 券商研报 | 横截面动量、CTA 因子体系综述 |
| 兴业证券 | 券商研报 | 波动率结构因子、波动率择时 |
| 国泰君安 | 券商研报 | 展期收益、量仓分析 |

---

## 二、因子全景 (五大类)

### 1. 时序动量 (Trend)

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

### 2. 期限结构 (Term Structure)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 近月-次月价差 | F1 - F2 | 华泰: 贴水/升水状态 | ✅ `BasisFactor` |
| 展期收益率 | (F1-F2)/F1 × 年化 | 中信建投: 期限结构收益 | ✅ `RollYieldFactor` |
| 期限结构斜率 | (F4-F1)/F1 | 华泰金工: 陡峭度 | ✅ `TermSlopeFactor` |
| 期限结构曲率 | F1 - 2F2 + F3 | 华泰金工: 二阶凸性 | ✅ `TermCurvatureFactor` |
| 近月基差率 | (S-F1)/F1 | 期现套利: 现货升贴水 | ✅ `NearBasisFactor` |

### 3. 波动率 (Volatility)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 历史波动率 HV | std(ln ret) × √252 | 兴业: 波动率聚集 | ✅ `HVFactor` |
| Parkinson 区间波动 | ln(H/L) 估计 | 学术: 高效波动估计 | ✅ `ParkinsonFactor` |
| 波动率 regime | HV z-score 高低 | AQR: 波动率择时 | ✅ `VolRegimeFactor` |
| 短长波动率差 | HV_short - HV_long | 兴业: 波动率期限结构 | ✅ `VolSpreadFactor` |
| IV-HV 风险溢价 | IV - HV | 期权 CTA: VRP | ⏳ 阶段3 (需 IV) |

### 4. 量价微观结构 (Volume/Price)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 量能趋势 | sign(ΔP) × sign(ΔV) | 华泰: 量价配合 | ✅ `VolumeTrendFactor` |
| 放量突破 | V > N日均量 × 倍数 | 经典 | ✅ `VolumeBreakoutFactor` |
| 持仓量变化 | sign(ΔOI) | 中信: 资金流入流出 | ✅ `OpenInterestTrendFactor` |
| 能量潮 OBV | Σ sign(ΔP) × V | 经典 | ✅ `OBVFactor` |
| 量仓比 | V / OI | 中信: 换手活跃度 | ✅ `VolOIRatioFactor` |
| Amihud 非流动性 | \|ret\| / V | Amihud 2002 | ✅ `AmihudFactor` |
| 量价相关性 | rolling corr(ret, V) | 华泰: 量价同向性 | ✅ `VolumePriceCorrFactor` |

### 5. 横截面 (Cross-sectional)

| 因子 | 定义/公式 | 研报逻辑 | 状态 |
|---|---|---|---|
| 横截面动量 | 收益排名前20%多/后20%空 | 招商 | ✅ `MomentumXSFactor` |
| 低波异象 | 低波做多/高波做空 | 学术 | ✅ `VolatilityXSFactor` |
| 横截面期限结构 | 贴水深做多/升水深做空 | 招商 | ✅ `BasisXSFactor` |

---

## 三、研报中的复合因子 (下一阶段)

研报普遍指出单一因子有局限, 建议复合增强:

| 复合思路 | 研报 | 实现建议 |
|---|---|---|
| 动量 × 期限结构 | Baltas & Kosowski (2015) | 两信号方向一致才开仓 (双重确认) |
| 趋势 × 波动率择时 | AQR | 高波动降仓 (已实现 vol_target) |
| 横截面动量 × 行业中性 | 招商 | 产业链分组后组内排序 |
| 量价 × 持仓验证 | 中信 | 放量 + 增仓确认趋势 |
| 多周期动量聚合 | AQR | 3/6/12月信号加权 |

**建议**: 阶段2 结束后实现"动量×期限结构双重确认"与"多周期聚合"两个复合因子。

---

## 四、已实现因子统计

- **27 个因子** + 8 个技术指标函数 (indicators)
- 五大类全覆盖: 时序趋势 / 波动率 / 量价 / 期限结构 / 横截面
- 全部基于现有 DuckDB 数据

| 模块 | 因子数 | 数据源 |
|---|---|---|
| trend.py | 8 | continuous_main |
| volatility.py | 4 + vol_target | continuous_main |
| volume.py | 7 | continuous_main (含 volume/oi) |
| term_structure.py | 5 | inter_contract_spread / spot_basis_daily |
| momentum_xs.py | 3 | continuous_main / 派生表 |
| indicators.py | 8 函数 | — |

---

## 五、待落地 (需外部数据或阶段3)

| 因子 | 需求 | 状态 |
|---|---|---|
| IV-HV 风险溢价 (VRP) | 期权 IV 计算 | 阶段3 gamma_scrapper |
| 期权 IV 偏斜 Skew | 期权 IV | 阶段3 |
| 动量×期限结构复合 | 现有因子组合 | 本阶段后 |
| 多周期动量聚合 | 现有因子组合 | 本阶段后 |
| 库存/仓单因子 | 交易所库存数据 | ❌ 数据缺失 |
| 会员持仓排名 | 席位数据 | ⚠️ 仅 SHFE 可用 |
| 现货升贴水序列 | akshare | ⚠️ 止于 2025-07 |
| 资金沉淀/流向 | 非标数据 | ❌ 无接口 |

---

## 六、验证

```bash
python -m pytest tests/strategy/cta/test_factors.py \
  tests/strategy/cta/test_factors_more.py \
  tests/strategy/cta/test_factors_research.py -v
# 94 tests passed, ruff clean
```

真实数据验证: AU/RB/M 2020-2024 期限结构正确区分 contango (AU) 与 backwardation (M)。
