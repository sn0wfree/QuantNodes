# 88-CTA 策略研发计划

> 期货 + 期权联合 CTA 策略研发路线图。目标在 `QuantNodes/strategy/cta/` 下构建完整研究框架，先期货因子策略、后期权 gamma 策略。

## 一、背景与目标

### 现状
- 数据层完备：`~/Public/DataCache/futures_options_daily.duckdb` — 88 期货 + 65 期权产品、2018~2026、8 张因子表（见 `docs/87-CTA数据准备现状.md`）
- `QuantNodes/strategy/cta/` 仅数据加载层（`data_loader.py` / `contract_specs.py`），**无任何策略逻辑**
- 已有通用回测引擎 `QuantNodes/backtest/`（节点式）+ 成熟轮动框架 `momentum_etf_rotation`（版本化管理）
- 期权 IV 计算仅覆盖 IO 一个品种（`scripts/v10_3/v10_3_compute_iv_skew.py`）

### 目标
1. **期货因子策略**：时序动量 / 期限结构 / 波动率等经典 CTA 因子 → 跑通完整回测
2. **期权联合层**：通用 IV/Greek 计算引擎（gamma scrapper）+ 期权 gamma 趋势策略
3. **Gamma Scalping**：long gamma + delta 对冲，赚已实现波动率
4. 复用现有回测引擎 + 版本化管理，先精选高流动性品种

## 二、业界 CTA 因子调研

### 四大经典因子家族

| 因子家族 | 代表性实现 | 经济逻辑 | 数据现状 |
|---|---|---|---|
| 时序动量 Trend | 海龟唐奇安通道、时间序列动量（AQR）、双均线 | 市场信息反应滞后，趋势延续 | ✅ 88 主力连续、跨期价差 |
| 期限结构 Term Structure | 展期收益率、Contango/Backwardation 状态 | 贴水/升水反映供需与库存 | ✅ inter_contract_spread、term_structure_daily |
| 横截面动量 Cross-sectional | 跨品种相对强弱排名、多空对冲 | 品种间动量扩散 | ✅ cross_product_spread |
| 波动率 Volatility | HV、IV-HV 风险溢价、波动率聚集 | 波动率集群、均值回归 | ✅ volatility_hv_daily；IV 需自建 |

### 增强型/另类因子

| 因子 | 说明 | 数据可用性 |
|---|---|---|
| 波动率风险溢价 VRP | IV 系统性高于 HV，卖出期权赚溢价 | ⚠️ 需自建 IV（期权数据可算） |
| 跨品种价差套利 | 黑色/油粕/能化/有色产业链套利对 | ✅ cross_product_spread 26 对 |
| 基本面/库存 | 现货库存、开工率、仓单 | ❌ 现货基差止于 2025-07，库存无 |
| 持仓结构 | 会员多空持仓比、净持仓变化 | ⚠️ 仅 SHFE 稳定 |
| 宏观因子 | VIX/DXY/利率/PMI | ✅ high_freq_macro |

### 期权端因子

| 因子 | 用途 |
|---|---|
| 隐含波动率 IV | 期权定价核心，ATM IV 序列 |
| IV 偏斜 Skew | 25-delta put - call IV 差，情绪/尾部风险 |
| PCR | 情绪反转指标 ✅ options_pcr_daily |
| IV-HV 价差 | 波动率风险溢价，期权价值判断 |

## 三、期权 gamma 趋势跟踪调研

### 核心机制：Gamma 的天然趋势跟随属性
- **期权多头（long gamma）**：价格任意方向移动，delta 随移动方向变化 → 上涨自动加多、下跌自动加空 = **天然趋势跟踪器**，无需预测方向
- 经典做法：**跨式（Straddle）/ 宽跨式（Strangle）买入** → 付 theta 时间损耗，换 gamma 方向性捕获
- 盈利条件：实际波动 > 期权定价波动（IV）→ "波动率兑现"
- 风险：纯买方胜率低（~30%）、时间价值衰减；需高频滚动

### Gamma Scrapper（两线并行）
1. **Gamma 数据引擎**：批量计算各品种逐日 IV/delta/gamma/theta/vega → 落 DuckDB 因子表
2. **Gamma Scalping 策略**：long gamma + 日频 delta 对冲，赚已实现波动率与 IV 之差（与趋势跟踪互补）

### 期权 symbol 格式（6 种，需统一解析）
```
CFFEX: IO2609-P-5400      SHFE: au2602P800
DCE:   m2608-P-2750       CZCE: TA703C5500
INE:   sc2605P510         GFEX: si2604-P-8500
```

## 四、目录结构

```
QuantNodes/strategy/cta/
├── __init__.py
├── data_loader.py            # 已有 (原子抽取)
├── contract_specs.py         # 已有
├── options/
│   ├── symbol_parser.py      # 6 种交易所期权 symbol 统一解析 (expiry/strike/type)
│   ├── greeks.py             # BS 定价 + IV/delta/gamma/theta/vega 通用计算
│   ├── gamma_scrapper.py     # Gamma 数据引擎: 逐产品批量算 IV/Greek → DuckDB
│   └── straddle.py           # 跨式/宽跨式选择、合成期货、期权仓位工具
├── factors/
│   ├── base.py               # Factor 基类 (可注入 loader)
│   ├── trend.py              # 时序动量: 唐奇安/双均线/动量强度
│   ├── term_structure.py     # 展期收益率、Contango/Backwardation
│   ├── volatility.py         # HV 排名、波动率调整目标
│   └── momentum_xs.py        # 横截面动量
├── signals/
│   ├── trend_signal.py       # 期货趋势信号 (先做)
│   └── gamma_signal.py       # 期权 gamma 趋势信号 (后做)
├── backtest/
│   ├── cta_broker.py         # 适配现有 BrokerNode → 期货/期权 (保证金/乘数/换月)
│   ├── cta_risk.py           # 风控节点 (仓位/保证金占用/单品种限额)
│   └── metrics.py            # CTA 指标 (复用 common/metrics.py)
├── strategies/
│   ├── trend_following.py    # 策略 v1: 多品种时序动量
│   ├── straddle_trend.py     # 策略 v2: 期权 long straddle 趋势
│   └── gamma_scalping.py     # 策略 v3: delta 对冲 gamma scalping
└── VERSION_TRACKING.md
```

## 五、实施阶段

### 阶段 0：基础设施
1. 期权 symbol 通用解析器（6 种格式）
2. 通用 greeks.py（复用 v10_3 BS 逻辑，全品种 + gamma/theta/vega）
3. 修正 `data_loader.py` 的 `V_ALL_OPTIONS_COLS`（声明了不存在的 `strike_yuan`/`symbol_raw`）
4. 建 `tests/strategy/cta/` 测试基线

### 阶段 1：CTA 回测适配
- `CtaBroker`：继承 `SimulatedBrokerNode`，支持期货合约乘数/保证金、合约换月（rollover）、多合约展期、代码键 `symbol+trade_date`（现有引擎用 `Code`/`date`/`Close` 大写）
- `CtaRisk`：保证金占用上限、单品种集中度、波动率目标（复用 risk_node.py 模式）
- CTA 指标（Sharpe/Calmar/MDD/胜率，复用 common/metrics.py）

### 阶段 2：期货因子策略 v1
- `trend_following.py`：唐奇安通道入场 + ATR 止损（海龟）+ 多品种组合，覆盖精选 20-30 高流动性品种
- `term_structure.py`：展期收益因子
- 回测验证：2018-2025 训练 / 2026 样本外，输出业绩报告

### 阶段 3：期权联合层 + Gamma 引擎
- `gamma_scrapper.py`：批量计算精选品种（IO/AU/M/TA/RB/CU/SI 等有深度期权的品种）逐日 IV/Greek → DuckDB `options_greeks_daily`
- 派生 IV 因子：ATM IV、IV-HV 价差、skew
- `straddle_trend.py`：策略 v2 — long straddle 趋势跟踪（低 IV + 趋势启动时买入跨式）

### 阶段 4：Gamma Scalping 策略 v3
- `gamma_scalping.py`：long gamma + 日频 delta 对冲，赚已实现波动率 vs IV 差

### 阶段 5：集成与验证
- 全品种组合回测、walk-forward、样本外验证
- 更新本计划文档

## 六、初始品种范围（精选高流动性）

- **期货趋势**：IF/IC/IH/IM + AU/AG/CU/AL/RB/HC/NI + M/Y/P/RM/OI + TA/MA/FG/SA + SC/I/J/JM 等 ~25 个
- **期权 gamma**：有深度期权的：IO/AU/M/TA/RB/CU/MA/FG/SI/SC ~10 个（验证 2025-01 流动性均充足）

## 七、复用边界

| 复用 | 说明 |
|---|---|
| `QuantNodes/backtest/` | `BrokerNode`/`RiskNode`/`BacktestResult` 抽象 → CTA 适配子类 |
| `common/metrics.py` | Sharpe/Calmar/MDD/胜率等指标 |
| `momentum_etf_rotation/VERSION_TRACKING.md` | 版本化管理模式 |
| `scripts/v10_3/v10_3_compute_iv_skew.py` | BS 定价/IV 反解逻辑（扩展为全品种） |

## 八、验证标准

- 每阶段 ruff 干净 + pytest 全过
- 期货策略训练窗口 Sharpe ≥ 0.8（多品种组合），2026 样本外不显著恶化
- 期权 gamma 策略明确展示"趋势段盈利来源"（分离 theta 损耗 vs gamma 收益）

## 九、技术差距清单

| 差距 | 说明 |
|---|---|
| CTA 策略逻辑 | 完全缺失，从零开发 |
| CTA 测试 | 0 个，需建立 |
| 期权希腊字母/IV | 不在数据中，需自建通用计算 |
| 文档与 schema 不一致 | `data_loader.py` 声明了不存在的列 |
| 回测引擎面向 A 股 | 需 CTA 适配：保证金/换月/多合约/期权 |
| 依赖声明缺口 | backtrader/empyrical/pyfolio 在环境未入 pyproject |
