# 动量 ETF 轮动策略 — 复现材料

> CICC 2026-07-03 报告《动量 ETF 轮动 + 80/20 固收+》复现及增强
>
> **Stage 12A 完成** (2026-07): hybrid momentum + VolTargeting + CostModel — Calmar 1.60

## 目录结构

```
reports/momentum_etf_rotation/
├── README.md               # 本文件
├── experiments/            # 失败实验归档
├── v1/                     # CICC 原始复现验证
│   ├── validation_fix_report.md
├── v2/                     # 增强版 (Stage 9~13)
│   ├── stage*a~*d_report.md   # 各阶段详细报告
│   ├── stage*_report.md       # Stage 10~13
│   ├── sensitivity_*.json     # 敏感性分析
├── common/                 # 跨版本分析
│   ├── contribution_analysis.md
│   ├── covariance_research.md
│   ├── GAP_ANALYSIS.md
│   ├── extended_metrics.*
│   └── *.csv               # 贡献分解数据
├── docs/                   # 开发文档
│   ├── STAGE_SUMMARY.md
│   ├── STRATEGY_VERSIONS.md
│   ├── CHANGELOG.md
│   ├── DECISION_LOG.md
│   ├── DEV_WORKFLOW.md
│   └── CODE_REVIEW_CHECKLIST.md
└── charts/                 # HTML 图表
    ├── v1/                 # CICC 基线
    ├── v2/                 # Stage 9~13
    └── common/             # 跨版本
```

## 版本体系

| 版本 | 描述 | Calmar | DD |
|------|------|--------|-----|
| **v1** (CICC 复现) | price momentum, 纯 4 步组合 | 0.78 | -21% |
| **v2** (Stage 12A) | hybrid+VT+Cost, slope×R² | **1.60** | -3.93% |

## 关键指标 (v2.0, 2019~2026)

| 指标 | 值 |
|------|-----|
| 年化收益 | 8.60% |
| 年化波动 | 5.38% |
| Sharpe | 1.60 |
| 最大回撤 | -3.93% |
| Calmar | 1.60 |
| 盈利月占比 | 77.27% |
| 最大月亏损 | -4.80% |

## 最优配置

```python
RotationConfig(
    lookback=90, top_n=10,
    momentum_type="hybrid",          # price | slope_r2 | hybrid
    vol_targeting=VolTargeting(enabled=True, target_vol=0.15),
    cost_model=CostModel(enabled=True, commission=0.0005, slippage=0.0010),
)
```

## CICC 基线对照

| 策略 | Calmar | DD | Ann |
|------|--------|-----|-----|
| CICC 逆波动 (报告) | 0.76 | -18.78% | - |
| 本 v1 复现 (lb=90) | **0.78** (+2.6%) | -21.05% | 16.35% |
| v2 hybrid+VT+Cost | **1.60** (+110%) | -3.93% | 8.60% |
| CICC FI+ (报告) | 1.73 | -1.48% | 6.34% |

## 数据

- **数据源**: Tencent 行情 (web.ifzq.gtimg.cn)
- **范围**: 2018-01-02 ~ 2026-06-30, 2058 个交易日
- **池子**: 43 ETF + 511260 (国债)
- **落盘**: `data/real/etf_nav_2018-01-01_2026-06-30.parquet`

## 策略代码

- 模块: `QuantNodes/strategy/momentum_etf_rotation/`
- v1 复现: `v1/` (`momentum_v1`, `portfolio_v1`, `backtest_v1`)
- v2 增强: `v2/` (`momentum_v2`, `portfolio_v2`, `backtest_v2`)
- 共享: `common/`

## 快速查看

```bash
# v2 最新版本
python3.11 -c "
from QuantNodes.strategy.momentum_etf_rotation import get_version
cfg = get_version()  # v1.0 (hybrid+VT+Cost)
print(cfg)
"

# 完整报告索引
cat docs/STAGE_SUMMARY.md | head -50
cat docs/STRATEGY_VERSIONS.md
```
