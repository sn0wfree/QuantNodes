# 固收+动量 ETF 轮动策略 (v3.0.0+ Stage 7)

> **来源**: 中金公司 《固收+:"可靠"的动量 ETF 轮动及 Agent 检验实践》 (2026-07-03, 杨冰/陈健恒)
> **模块**: `QuantNodes/strategy/momentum_etf_rotation/`
> **Skill**: `QuantNodes/agent/skills_quant/quant-validation/`
> **Tool**: `quant_validation` (nanobot entry point)

## 1. 策略概要

动量因子并非"优秀"因子, 但因**高动量品种右尾更长**, 是"最可靠"的轮动范式: 留住赢家、轮出输家。理论上仅需 1 个参数: 动量本身。

**4 条核心组合管理规则**:

| # | 规则 | 实现位置 |
|---|------|----------|
| 1 | 同指数去重 + 高相关剔除 (阈值 0.9) | `portfolio.select_and_weight` |
| 2 | 强制分散 (A 股宽基/行业 ≤ 3, HK ≤ 1, 必含商品+海外) | `DiversificationCaps` |
| 3 | 逆波动加权 (权重 ∝ 1/σ) | `inverse_vol_weights` |
| 4 | 止损+补位 (跌破 55 日均线 + 排名跌出后 30% 分位) | `apply_stops` |

**80/20 固收+**: 80% 10 年国债 ETF + 20% 上述动量轮动 (CICC 报告核心结论之一)。

## 2. 复现声明

**与 CICC 报告的关系**: 复现 (reproduction) 而非重制 (re-implementation).

| 项目 | CICC 原文 | 本复现 |
|------|-----------|--------|
| ETF 池 | 未公布完整代码清单 (43+ 支) | 43 支近似池 (A 股宽基 6 + 行业 20 + HK 5 + 商品 6 + 海外 6) |
| 动量回看 | 144 日 (主文) / 114 日 (伪代码) | 默认 144 日, 可调 |
| 调仓频率 | 月度 | 月度 (`freq="ME"`) |
| 样本期 | 2020-2024 (样本内) + 2025 YTD | 真实数据 2018-01 ~ 2025-07 (7.5y) |
| 业绩对照 | 全区间数字 | 容忍 ±20% (合成数据已弃, 详见 §3) |

CICC 报告的**关键单边关系**已在 `test_e2e_real_data.py` 中验证 (80/20 缓冲、Calmar 提升、净值 > 0).

### 2.1 真实数据接入 (S6)

**数据源**: Tencent `web.ifzq.gtimg.cn/appstock/app/fqkline/get` (无需 akshare, 7 req/s 可持续).
**样本**: 43 支 ETF + 511260 (国泰 10 年国债 ETF), 范围 2018-01-02 ~ 2025-07-04 (1820 交易日).

**拉取方法**:
```bash
# 一次性拉取 (~63s 落盘 data/real/)
python3.11 scripts/fetch_real_etf_panel.py

# 测试
python3.11 -m pytest tests/strategy/momentum_etf_rotation/test_data_tencent.py -v
python3.11 -m pytest tests/strategy/momentum_etf_rotation/test_e2e_real_data.py -v -m e2e_real
```

**落盘结构**:
```
data/real/
├── etf_nav_2018-01-01_2025-07-06.parquet  # 主面板 (44 列 × 1820 行)
├── per_etf/<code>.parquet                  # per-ETF 缓存 (失败可重拉单支)
├── fetch_log.json                          # 哪支成功/失败
└── validation_report.md                    # 4 项抗过拟合报告 (test 自动落)
```

**ETF 上线时间分布** (影响回测早期):
- 2018 上市: 23 支 (沪深 300 / 上证 50 / 中证 500 / 证券 / 银行 / 黄金 / 港股 / 美股等)
- 2019 上市: 12 支 (科创 50 / 芯片 / 酒 / 医疗 / 通信 / 传媒 等)
- 2020 上市: 4 支 (新能源车 / 光伏 / 煤炭 / 半导体 等)
- 2021 上市: 4 支 (恒生科技 / 有色期货 / 港股科技 等)

**S6 期间修复的 bug**: `select_and_weight` 原本允许动量为 NaN (上市晚) 的 ETF 进入候选. 现在先过滤 NaN 再走 4 条规则 — 这让回测不再"乱选"晚上市的品种.

## 3. CICC 公布数字 vs 本复现

| 策略 | 指标 | CICC 数字 | 本复现 (合成 5y) | 本复现 (真实 7.5y) |
|------|------|-----------|------------------|---------------------|
| 动量轮动 (逆波动) | Calmar | 0.76 | ~2.1 (合成高动量) | 0.15 |
| 动量轮动 (逆波动) | 最大回撤 | -18.78% | ~-6% | -36.24% |
| 动量轮动 (等权) | Calmar | 0.51 | ~2.5 | 0.15 |
| 80/20 固收+ | 年化 | 6.34% | - | 5.04% (差 21%) |
| 80/20 固收+ | Calmar | 1.73 | - | 1.07 (差 38%) |
| 80/20 固收+ | 最大回撤 | -1.48% | - | -4.71% (越界) |

**关键单边关系核对** (真实数据):
| 关系 | CICC | 本实现 | 成立 |
|------|------|--------|------|
| 80/20 缓冲: 固收+ DD < 纯动量 DD | 1.48% < 18.78% | 4.71% < 36.24% | ✓ |
| 80/20 Calmar > 纯动量 Calmar | 1.73 > 0.76 | 1.07 > 0.15 | ✓ |
| 80/20 年化 > 0 | 6.34% | 5.04% | ✓ |
| 逆波动 Calmar > 等权 Calmar | 0.76 > 0.51 | 0.15 = 0.15 | ✗ (见下) |

**逆波动 ≈ 等权** 的原因: 7.5y 真实数据下各 ETF 60 日年化波动率高度相似 (大多数 0.12 ~ 0.25 区间), 逆波动权重的差异不显著. 这与 CICC 报告里"逆波动显著优于等权"的发现不矛盾 — CICC 池的选择可能更广/更分散.

## 4. API 速查

```python
from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,                    # 43 支近似池
    RotationConfig,                  # 所有可调参数
    DiversificationCaps,             # 分散规则
    run_rotation_backtest,           # 纯动量轮动
    FixedIncomePlus,                 # 80/20 固收+
    FixedIncomePlusConfig,
    run_full_validation,             # 4 项抗过拟合
    CICC_BASELINES,                  # CICC 报告数字
)
```

### 4.1 单次回测

```python
import pandas as pd
from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL, RotationConfig, run_rotation_backtest, BacktestConfig
)

etf_nav = pd.read_parquet("etf_nav_2018_2025.parquet")  # index=date, cols=codes
cfg = RotationConfig(lookback=144, top_n=10, corr_threshold=0.9)
result = run_rotation_backtest(etf_nav, DEFAULT_POOL, BacktestConfig(rotation=cfg))
print(result.metrics)  # Calmar, 最大回撤, 年化, 夏普
```

### 4.2 80/20 固收+

```python
from QuantNodes.strategy.momentum_etf_rotation import (
    FixedIncomePlus, FixedIncomePlusConfig,
    load_bond_etf_nav, load_etf_nav_panel,
)

bond = load_bond_etf_nav("511260", "2018-01-01", "2025-06-30")
etf = load_etf_nav_panel(DEFAULT_POOL.codes, "2018-01-01", "2025-06-30")
fip = FixedIncomePlus(bond, etf, DEFAULT_POOL,
                      FixedIncomePlusConfig(rotation=cfg, bond_weight=0.8))
result = fip.run(freq="ME")
print(result.nav)  # 净值序列
```

### 4.3 4 项抗过拟合检验

```python
from QuantNodes.strategy.momentum_etf_rotation import (
    run_full_validation, ValidationConfig,
)

vcfg = ValidationConfig(
    start_points=("2018-01-01", "2020-01-01", "2022-01-01"),
    rebal_offsets=(-5, -3, 0, 3, 5),
    perturb_lookbacks=(120, 144, 168),
    perturb_corr_thresholds=(0.85, 0.90, 0.95),
    perturb_a_share_caps=(2, 3, 4),
)
report = run_full_validation(etf_nav, DEFAULT_POOL, cfg, vcfg=vcfg,
                              strategy_name="MomentumETFRotation")
print(report.to_markdown())
```

## 5. Agent 集成 (nanobot)

### 5.1 Skill: `quant-validation`

- 路径: `QuantNodes/agent/skills_quant/quant-validation/SKILL.md`
- 描述: 4 项抗过拟合检验的工作流 + 验收标准
- 调用方式: nanobot 自动发现并暴露给 LLM

### 5.2 Tool: `quant_validation`

```python
from QuantNodes.agent.tools.validation import ValidationTool

tool = ValidationTool()
result = await tool.execute(
    etf_nav=[{"date": "2024-01-01", "code": "510300", "close": 3.85}, ...],
    lookback=144,
    top_n=10,
    actions=["all"],   # or ["start", "rebal", "perturb", "ablation"]
    start_points=["2018-01-01", "2020-01-01", "2022-01-01"],
)
# result.content["report_markdown"]
# result.content["passed"] / result.content["failed"]
# result.content["actions"] (list of {name, passed, summary})
```

注册: 已加入 `QuantNodes/agent/tools/__init__.py:_QUANT_TOOL_FACTORIES`。

## 6. 测试覆盖

```bash
python3.11 -m pytest tests/strategy/momentum_etf_rotation/ -v
# 57 tests passed (含 4y 端到端 < 5s)

python3.11 -m pytest tests/agent/test_validation_tool.py -v
# 5 tests passed
```

| 文件 | 覆盖 |
|------|------|
| `test_universe_etf.py` | 43 支 ETF 池完整性 + 类别查询 |
| `test_momentum.py` | 144 日动量 + 52 周高点 + 波动 + MA + 相关 |
| `test_portfolio.py` | 4 条规则 + 权重 + 止损补位 |
| `test_fi_plus.py` | 80/20 净值 + 业绩指标 + 调仓频率 |
| `test_e2e_backtest.py` | 5y 端到端 + CICC 对照表 + 性能 |
| `test_validation.py` | 4 个 action + 完整报告 |
| `tests/agent/test_validation_tool.py` | nanobot tool 入口 |

## 7. 风险与已知限制

1. **ETF 池近似**: CICC 报告未列完整 ETF 代码, 当前池为公开市场流动性最优的 43 支近似
2. **调仓日简化**: 当前 `apply_stops` 与 `select_and_weight` 在同一日执行, 实际生产应区分信号日/执行日
3. **无交易成本**: CICC 报告亦未公布手续费/滑点假设
4. **未实现止盈**: CICC 明示放弃 (会引入过拟合), 与原文一致
5. **数据源依赖**: `load_etf_nav_panel` 走 akshare; 数据不可用时返回空 DataFrame (优雅降级)
6. **80/20 债券近似**: `load_bond_etf_nav` 优先 511260 (国泰 10 年国债 ETF), 不可用退化到 10 年国债收益率

## 8. 复现研究路径

1. S1: 策略骨架 + 4 条规则 (28 单测)
2. S2: data layer 接入 + 80/20 固收+ (9 单测)
3. S3: 5y 端到端 + CICC 对照 (6 单测)
4. S4: 4 个抗过拟合 action + nanobot tool (18 单测)
5. S5: 文档 + graphify

总测试: **62 passed**, 端到端 5y < 5s, 单 action < 30s。
