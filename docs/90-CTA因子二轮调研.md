# 90-CTA因子二轮调研 (统计/复合/微观结构)

> 第一轮调研 (docs/89) 覆盖经典五大类 27 因子。本轮深入调研**统计特征类 / 复合增强类 / 微观结构类**因子。

## 一、调研来源

| 来源 | 类型 | 贡献 |
|---|---|---|
| Hurst (1951) R/S 分析 | 学术 | 趋势/均值回归判别 |
| Lo & MacKinlay (1988) | 学术 | 方差比检验 (动量持续性) |
| Harvey et al. (2018) | 学术 | 偏度溢价, 彩票偏好 |
| Amaya et al. (2015) | 学术 | 已实现偏度 |
| Duan (1995) | 学术 | GARCH 波动率 |
| 华泰金工 | 券商研报 | 动量崩溃、长短期动量差 |
| 中信期货 | 券商研报 | 量价背离、资金流 |
| 招商证券 | 券商研报 | 因子复合方法论 |

## 二、可落地因子 (基于 continuous_main OHLC)

### 1. 统计特征类 (趋势持续性判别)

| 因子 | 定义 | 研报逻辑 | 数据需求 |
|---|---|---|---|
| Hurst 指数 | R/S 分析, H>0.5 趋势 / H<0.5 回归 | Hurst 1951: 判别市场状态 | 价格序列 |
| 方差比 VR | Var(k期)/[k×Var(1期)], VR>1 趋势 | Lo & MacKinlay: 随机游走检验 | 价格序列 |
| 收益自相关 | corr(ret, ret_{t-k}) | 华泰: 动量/反转判别 | 价格序列 |
| 收益偏度 | 滚动偏度 | Amaya: 负偏度溢价 | 价格序列 |

### 2. 复合增强类 (研报主流)

| 因子 | 定义 | 研报逻辑 | 数据需求 |
|---|---|---|---|
| 多周期动量聚合 | 3/6/12月信号加权 | AQR: 多周期稳健 | 价格序列 |
| 波动率调整动量 | momentum / vol | 招商: 夏普动量 | 价格+波动 |
| 动量×期限结构 | 双重确认 | Baltas & Kosowski | 价格+spread |
| 动量崩溃过滤 | 短长动量差 | 华泰: 规避崩溃 | 价格序列 |

### 3. 微观结构类 (OHLC 日内分解)

| 因子 | 定义 | 研报逻辑 | 数据需求 |
|---|---|---|---|
| 隔夜收益 | open/pre_close - 1 | 华泰: 隔夜信息 | open+prev_close |
| 日内收益 | close/open - 1 | 华泰: 日内动能 | open+close |
| 隔夜-日内背离 | 两收益方向相反 | 华泰: 分歧信号 | OHLC |
| 跳空缺口 | open - prev_close | 经典 | OHLC |
| 收盘位置 | (close-low)/(high-low) | 经典: 收盘强弱 | OHLC |

### 4. 波动率深化

| 因子 | 定义 | 研报逻辑 | 数据需求 |
|---|---|---|---|
| GARCH(1,1) 波动 | 条件波动率 | Duan 1995 | 价格序列 |
| 波动率偏度 | 波动的偏度 | 兴业 | 价格序列 |
| 下行波动率 | 仅负收益 std | 经典 | 价格序列 |

## 三、实现状态

### ✅ 已实现 (statistical.py)
1. **HurstExponentFactor** — H>0.55 趋势多 / H<0.45 回归空
2. **VarianceRatioFactor** — VR>1.1 动量 / VR<0.9 反转
3. **AutocorrelationFactor** — 正自相关多 / 负自相关空
4. **SkewnessFactor** — 负偏多 / 正偏空

### ✅ 已实现 (microstructure.py)
5. **OvernightReturnFactor** — 隔夜收益
6. **IntradayReturnFactor** — 日内收益
7. **OvernightIntradayDivergenceFactor** — 隔夜/日内背离
8. **GapFactor** — 跳空缺口
9. **CloseLocationFactor** — 收盘位置

### ✅ 已实现 (composite.py)
10. **MultiHorizonMomentumFactor** — 3/6/12月动量加权
11. **SharpeMomentumFactor** — 波动率调整动量
12. **MomentumCrashFilterFactor** — 短负长正 → 崩溃规避

### ⏳ 待实现 (需组合/复杂)
- ~~GARCH(1,1) 波动率~~ → ✅ 已实现 (arch 库, volatility.py)
- ~~波动率偏度~~ → ✅ 已实现 (volatility.py)
- 下行波动率 — 可后续补

## 四、文档更新记录

- 2026-08-12: 创建, 调研 12 个新因子, 优先级划分
- 2026-08-12: 实现 12 个因子, 110 测试通过, ruff 干净
- 2026-08-12: 补充实现 GARCH(1,1) 与波动率偏度, 116 测试通过
