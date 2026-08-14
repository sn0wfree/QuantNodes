# 98. CTA 期权对冲基础设施设计

> 状态: 规划定稿 (2026-08-14)
> 关联: docs/97 (因子到策略流程), docs/96 (现状总结)
> 代码: `QuantNodes/strategy/cta/options/` + `QuantNodes/strategy/cta/backtest/`
> 原则: **策略设计后延, 基础设施构建优先** — 本文只定义数据/工具/框架层,
>       对冲策略规则 (何时买/买多少) 后续在框架接口上实现

## 一、背景与目标

CTA fut_v1 (Sharpe 2.006) 尾部风险明确: 最差单日 -4.88% (2025-04-07),
2024 商品窗口 -11.1%, 回撤期净多头占比 91.9%. 计划用**期权尾部对冲**
(单品种虚值 Put) + **IV 分位择时**降低尾部风险, 并保留 gamma 增强空间.

本阶段只构建**基础设施** (数据面板/合约选择/对冲框架), 不实现具体
对冲策略. 策略在基础设施就绪后基于框架接口设计.

## 二、现状盘点

### 2.1 已有 ✅

| 层 | 组件 | 位置 |
|----|------|------|
| 数据 | 66 期权表 / 17.2M 行 / 2018-2026 | `futures_options_daily.duckdb` |
| 数据 | v_all_options 聚合视图 (65 产品) | DuckDB |
| 数据 | volatility_hv_daily (HV 20/60/120) | DuckDB |
| 数据 | options_pcr_daily (PCR 情绪) | DuckDB |
| 数据 | trading_calendar (交易日历) | DuckDB |
| 工具 | symbol_parser (CFFEX/DCE/SHFE/CZCE 4 格式) | `options/symbol_parser.py` |
| 工具 | greeks (BS/IV 反解/批量/跨式组合) | `options/greeks.py` |
| 回测 | 期权买方 MVP: 合约解析/调仓/到期结算 | `cta_contract.py`/`cta_broker.py`/`cta_engine.py` |

### 2.2 缺失 ❌ (本阶段构建)

| # | 缺口 | 说明 |
|---|------|------|
| 1 | **IV 面板** | 逐品种 ATM IV 时序 (市价反解), IV 分位择时基础 |
| 2 | **标的期货价格映射** | 期权 underlying → 对应期货合约每日价 |
| 3 | **到期日解析** | 期权真实到期日 (数据推断, 非月份近似) |
| 4 | **合约选择器** | 给定品种/日期/目标 delta → 期权合约 |
| 5 | **对冲层框架** | CTA 持仓 → 期权目标仓位接口 (策略逻辑留空) |
| 6 | **预算记账** | 权利金支出/年化预算统计 |

## 三、架构设计

```
┌─────────────────────────────────────────────────────────┐
│  数据层 (阶段 1)                                         │
│  ┌──────────────┐ ┌────────────────┐ ┌──────────────┐   │
│  │ iv_panel     │ │ underlying_map │ │ expiry       │   │
│  │ ATM IV 时序  │ │ 标的价格映射    │ │ 到期日推断   │   │
│  │ → DuckDB 缓存│ │                │ │              │   │
│  └──────────────┘ └────────────────┘ └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│  选择器层 (阶段 2)                                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ selector: select_put(product, date, target_delta)│   │
│  │          → 合约 symbol + 单价 + delta             │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  对冲框架层 (阶段 2, 策略逻辑后填)                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ hedge_layer: compute_hedge_targets(...)          │   │
│  │ HedgeConfig (预算/目标 delta/IV 分位阈值)          │   │
│  │ → 期权目标仓位 (策略规则后续实现)                 │   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  回测层 (已有 MVP, 增强)                                 │
│  run_cta_backtest(option_contracts, option_settle)      │
└─────────────────────────────────────────────────────────┘
```

## 四、阶段 1: 数据基础设施 (先做)

### 4.1 IV 面板 (`options/iv_panel.py`)

**目标**: 逐品种逐日 ATM 隐含波动率时序, 供 IV 分位择时与 gamma 分析.

**构建逻辑**:
1. `load_v_all_options` 加载品种期权行情 (symbol/strike/close/trade_date)
2. 每日选出 ATM 合约: |strike - 标的期货价| 最小
3. `compute_greeks_for_option` 反解 IV (市价 close)
4. 输出: `DataFrame (trade_date × product)`, 值 = ATM IV

**存储**: 预计算入 DuckDB 新表 `option_iv_daily`
(`product, exchange, trade_date, atm_iv, iv_20d, iv_60d`, 滚动统计一并预存)
回测直接读表, 避免实时反解 (72 品种 × 2000 天 × 多合约 计算量大).

**IV 分位**: `atm_iv` rolling 252 天 pct-rank → `iv_percentile`
(择时阈值: 分位 < 30% 低 IV, 30-70% 中, > 70% 高 IV).

### 4.2 标的期货价格映射 (`options/underlying_map.py`)

**目标**: `{underlying_symbol (rb2605): {trade_date: close}}`.

**构建逻辑**:
1. 从 `*_futures_daily` 加载各产品全部合约日线
2. 按 `symbol` 索引 → underlying 价格字典
3. CFFEX 指数期权特殊处理: IO→IF, MO→IM, HO→IH (已有 `map_cffex_underlying`)

**用途**: greeks 反解需要 underlying_price; 到期结算需标的结算价.

### 4.3 到期日解析 (`options/expiry.py`)

**目标**: `{option_symbol: expiry_date}`.

**构建逻辑** (数据推断, 无需外部日历):
1. 加载期权全历史 OI/volume
2. 每合约最后出现交易日 = 到期日 (OI 归零日)
3. 与 `trading_calendar` 交叉校验

**用途**: 回测到期结算 (`option_settle`), greeks 的 time_to_maturity.

## 五、阶段 2: 选择器与对冲框架

### 5.1 合约选择器 (`options/selector.py`)

```python
def select_put(
    product: str,           # 'RB'
    trade_date: str,        # 'YYYY-MM-DD'
    underlying_price: float,
    target_delta: float,    # 0.2-0.3 (虚值 Put)
    min_days: int = 30,     # 到期天数下限
    max_days: int = 60,     # 上限
) -> OptionSelection | None:
    """选虚值 Put: 候选到期月 → 各 strike 算 delta → |delta - target| 最小."""
```

`OptionSelection`: symbol / strike / delta / premium / expiry / 手数建议.

### 5.2 对冲层框架 (`options/hedge_layer.py`)

```python
@dataclass
class HedgeConfig:
    budget_pct: float = 0.005      # 年化权利金预算 (NAV 的 0.5%)
    target_delta: float = 0.25     # 目标 delta (虚值深度)
    hedge_ratio: float = 0.4       # 对冲净敞口比例
    iv_low_pct: float = 0.30       # IV 分位 < 30% 全额买
    iv_high_pct: float = 0.70      # IV 分位 > 70% 不买/平仓
    min_days: int = 30
    max_days: int = 60

def compute_hedge_targets(
    cta_positions: pd.DataFrame,   # CTA 逐日持仓
    iv_panel: pd.DataFrame,        # IV 面板 + 分位
    prices: pd.DataFrame,          # 期货价格面板
    config: HedgeConfig,
) -> dict[str, float]:
    """CTA 持仓 → 期权目标仓位. 策略规则后续实现, 本框架定义接口."""
```

**策略逻辑后延**: 框架接口先行, 具体规则 (哪些品种/何时买/预算分配)
在基础设施验证后基于真实 IV 面板设计.

## 六、阶段 3: 验证与固化

- 单测: IV 面板正确性 (vs 已知 IV 样例) / 选择器 delta 精度 / 到期推断
- 性能: IV 面板预计算耗时测量, 批量反解优化
- 文档: 本文档更新为"已实现"状态

## 七、关键设计决策 (已确认)

| 决策 | 选择 | 理由 |
|------|------|------|
| IV 面板存储 | 预计算入 DuckDB 新表 | 一次反解, 回测秒级读取 |
| 到期日来源 | 数据推断 (最后交易日) | 无需外部日历, 精确 |
| 对冲框架 | 接口先行, 策略后填 | 基础设施优先原则 |
| 对冲标的 | 单品种逐品种 | 已确认 |
| IV 择时 | 分位择时 (30/70 阈值) | 已确认 |
| 对冲结构 | 后延讨论 (纯 Put / 双尾) | 框架兼容 |

## 八、边界与风险

1. **期权上市时间差异**: RB 期权 2022-12 起, 部分品种 2023-2024 才上市
   → 2022 前回测对冲覆盖不全, 需注明
2. **IV 反解计算量**: 72 品种 × 2000 天 × 多合约, 需批量 + 缓存
3. **权利金成本**: 买方 theta 损耗持续, 需验证对冲收益 > 成本
4. **数据质量**: 部分期权合约流动性低 (OI 小), 需过滤 (如 OI > 1000)
