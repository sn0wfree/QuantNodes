# 92-CTA因子四轮调研 (日历/资金/持仓/背离/相关)

> 第四轮调研: 日历效应 / 资金沉淀 / 持仓集中度 / 量价背离 / 相关性结构类因子。

## 一、调研方向与研报来源

| 方向 | 研报/文献 | 数据支撑 |
|---|---|---|
| 日历效应 (月份/周内) | 华泰金工《商品期货日历效应》 | trading_calendar |
| 资金沉淀 | 中信期货、国泰君安 CTA 资金面研究 | continuous volume/oi |
| 持仓集中度 | 国泰君安《持仓结构分析》 | inter_contract_spread oi_near* |
| 量价背离 | 华泰金工量价背离系列 | continuous price/volume |
| 相关性结构 | 学术 (相关性与危机传染) | 连续面板 |

## 二、因子设计

### 1. 日历效应类 (calendar.py)
| 因子 | 定义 | 逻辑 |
|---|---|---|
| `SeasonalityFactor` | 同月历史平均收益 (逐月) | 商品季节性 (农产品/能源) |
| `WeekdayFactor` | 周内各日平均收益 | 周一/周五效应 |
| `MonthTurnFactor` | 月末/月初效应 | 调仓/交割周期 |

### 2. 资金面类 (capital.py)
| 因子 | 定义 | 逻辑 |
|---|---|---|
| `CapitalInflowFactor` | OI×价格 变化 (沉淀资金) | 资金流入 → 趋势确认 |
| `TurnoverFactor` | volume×price 成交额趋势 | 活跃度变化 |

### 3. 持仓结构类 (position.py)
| 因子 | 定义 | 逻辑 |
|---|---|---|
| `OIConcentrationFactor` | 主力合约 OI / 总 OI | 集中度变化 |
| `FrontMonthOIShareFactor` | 近月 OI 占比 | 近月活跃/远月布局 |

### 4. 量价背离类 (divergence.py)
| 因子 | 定义 | 逻辑 |
|---|---|---|
| `PriceVolumeDivergenceFactor` | 价创新高但量不创新高 | 背离 → 反转 |
| `VolumeConfirmationFactor` | 放量+涨 确认 / 放量+跌 反确认 | 量价配合 |

### 5. 相关性结构类 (correlation.py)
| 因子 | 定义 | 逻辑 |
|---|---|---|
| `DispersionFactor` | 品种间收益横截面离散度 | 分散度高 → 机会多 |
| `AvgCorrelationFactor` | 品种间平均相关 | 相关性高 → 系统性风险 |

## 四、实现状态

### ✅ 已实现 (12 因子)
- **calendar.py**: SeasonalityFactor / WeekdayFactor / MonthTurnFactor
- **capital.py**: CapitalInflowFactor / TurnoverFactor
- **position.py**: OIConcentrationFactor / FrontMonthOIShareFactor
- **divergence.py**: PriceVolumeDivergenceFactor / VolumeConfirmationFactor
- **correlation.py**: AvgCorrelationFactor / DispersionFactor

### 验证
- 真实数据 (8 品种 2018-2025): 全部因子产生信号 (AvgCorr 仅在危机时触发, 合理)
- 140 测试通过, ruff 干净

### 更新记录
- 2026-08-12: 创建调研 + 实现 12 因子 (140 测试通过)
