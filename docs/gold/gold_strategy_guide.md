# 黄金生产策略 — 使用手册

> 适用: B3全执行栈 (策略规格见 `docs/gold/gold_strategy_v4.md`)
> 更新: 2026-09-01

---

## 1. 快速开始

```bash
# 一键执行: 数据刷新 + 查询最新仓位
python3 scripts/cmf/refresh_and_track.py

# 仅查询仓位 (跳过数据刷新)
python3 scripts/cmf/refresh_and_track.py --no-refresh
```

其他工具:
```bash
python3 scripts/cmf/gold_backtest.py --stack      # 策略回测 (锚点: 18.47%/1.56/-16.84%)
python3 scripts/cmf/gold_attribution.py --year 2026  # 各层贡献 + 调仓归因
python3 scripts/cmf/gold_track.py --year 2026     # 历史持仓CSV
```

---

## 2. 每日操作流程 (SOP)

```
15:00  沪金日盘收盘
15:30  python3 scripts/cmf/refresh_and_track.py
       → 自动完成: AU期货增量拉取 → FRED刷新 → 重建主力合约 → 重建adj价格 → 输出仓位
       → 读"今日动作": 持仓不变 / 加仓X% / 减仓X%
21:00  夜盘开盘按信号挂单成交 (限价或开盘瞬间市价)
       ⚠️ 必须当晚21:00执行, 拖到次日日盘在崩盘年代价~8pp (v4文档§9.3)
```

**track输出解读**:
- `当前仓位` = 明早前应持有的总仓位 (核心+网格库存)
- `风控状态`: 正常 / 波动压缩 / 软减仓 / 冷却期 / 硬止损 (五态)
- `今日动作`: 与前一交易日仓位的差

---

## 3. 数据管道 (按顺序, 全部幂等)

| 步骤 | 脚本 | 刷新内容 | 耗时 |
|---|---|---|---|
| 1 | `python3 scripts/cmf/fetch_gold_contracts.py` | 全合约日K (**增量模式**, 只拉活跃合约) | ~15s |
| 2 | `python3 scripts/cmf/1e_fetch_gold.py` | 主力连续K + vix/gvz/tips/dxy等 fundamentals | ~30s |
| 3 | `python3 scripts/cmf/detect_gold_main.py` | 主力合约识别+换月日 | ~10s |
| 4 | `python3 scripts/cmf/build_gold_smooth.py` | 换月调整连续价格 (生产依赖表) | ~10s |

- 全量重建 (每年/异常时): `fetch_gold_contracts.py --full`
- etf/gprd/dxy等非生产依赖: `refresh_gold_data.py` (可不跑, 不影响仓位)
- 数据窗口: 调整价格从**2018-06-20**起; 9-1的bar在当晚21:00夜盘开盘后才产生

**数据滞后检查**: `sqlite3`/duckdb查 `SELECT MAX(trade_date) FROM futures_main_close_au_adj`; 若落后于今天, 先跑管道再算信号。

---

## 4. 脚本清单

| 脚本 | 用途 | 关键参数 |
|---|---|---|
| `gold_track.py` | **每日跟踪** | `--year YYYY` 输出全年持仓CSV |
| `gold_backtest.py` | 回测分析 | `--stack`(B3) / `--compare` / `--fc --sc --hard`(自定义链) |
| `gold_attribution.py` | 归因分析 | `--year YYYY` 调仓日志 |
| `prepare_gold_overview_data.py` | 画图数据缓存 (report.html用) | 无 |
| `gen_html_report.py` / `plot_gold_overview.py` | HTML/图报告 | 无 |
| `research_ml_anomaly_v3.py` | ML异常检测 (独立辅助, 已暂缓) | `--l2l3`等 |

**生产代码**: `QuantNodes/strategy/commodity_fundamental/gold/gold_production.py`
```python
from QuantNodes.strategy.commodity_fundamental.gold import (
    PRODUCTION_PARAMS,          # 唯一参数源
    build_production_position,  # → pd.Series 日频仓位
    position_state,             # → 当日五态
    o2o_returns,                # → O2O收益序列
)
```

---

## 5. 输出文件字典

| 文件 | 内容 | 字段 |
|---|---|---|
| `reports/gold/positions_{year}.csv` | 全年逐日持仓 | date, position, pos_change, gold_close, open, daily_ret, state |
| `reports/gold/trades_{year}.csv` | 调仓归因日志 | date, pos, delta, core_delta, grid_delta, trigger(应急/链层/网格/平滑追赶), gold_ret |
| `reports/gold/attribution_ladder.csv` | 七级消融表 | level, cagr, sharpe, maxdd, calmar, y26, d_cagr, d_sharpe, d_maxdd |
| `reports/gold/overview_data.pkl` | HTML报告数据缓存 | pos_15/nav26/yearly/scenarios等 |

---

## 6. 参数修改指南

1. 改 `gold_production.py` 的 `PRODUCTION_PARAMS` (唯一入口, 勿改散件)
2. **必做回归锚点验证**:
   ```bash
   python3 scripts/cmf/gold_backtest.py --stack --compare
   ```
   当前生产锚点: `18.47% / 1.56 / -16.84%`。改参数后旧锚点会变, 属预期;
   但 `build_chain()` 默认链锚点 `15.42%/1.52/-13.97%` 不应受影响
3. 同步修改 `docs/gold/gold_strategy_v4.md` §3参数表
4. commit: `feat(gold): 生产参数调整 — <说明> + 新锚点`

**红线**: 不改 `o2o_returns` 的收益口径 / 不去掉 assert 互锁 / 不把shift(2)改小。

---

## 7. FAQ

**Q: 数据还停在昨天, 今天仓位怎么看?**
先跑§3管道。今晚21:00夜盘开盘后的bar次日才有, 因此今天的"信号"基于昨日收盘, 今晚执行。

**Q: 为什么空仓了?**
查 `gold_track.py` 的状态列: 硬止损 (单日<-5%应急/-20%回撤线) 或冷却期 (硬触发后5个交易日)。
冷却结束后需波动率回落才恢复 (compute_risk_overlay内置)。

**Q: 网格库存什么时候买入?**
周五收盘价较锚点每低2%买一档(0.0833), 最多3档(+0.25); 涨回锚点上方全部卖出; 偏离4档以上锚点重置。

**Q: 应急触发后多久恢复?**
应急压0后跟随平滑逐日收敛 (cap=0.20/天); 若二层硬触发另有5天冷却+VIX/GVZ双确认。

**Q: 回测口径是什么?**
收盘算信号 → 次日夜盘开盘(21:00)成交 → open-to-open + 5bps。
与"收盘即成交"仅差0.32pp/年 (口径等价); 与"拖到次日日盘成交"差0.92pp/年、崩盘年-8pp。
详见 v4文档 §9。

**Q: 纸面跟踪怎么做?**
B3已定为生产参数但实盘切换前, 每日:
① track输出B3仓位 → 手工记入对照表 ② 实盘若仍跑旧参数(-2/-3/-15)则双记
③ 3个月后对比 `gold_backtest.py --stack --compare` 与实盘成交价偏差再切换。

---

## 8. 故障排查

| 症状 | 排查 |
|---|---|
| track报错 KeyError/表不存在 | duckdb被另一个进程锁 → 关掉并发的回测进程 |
| 锚点对不上 (≠18.47%) | 数据是否刷新到位? `MAX(trade_date)`; gold_backtest与包版本是否同branch |
| fetch超时/空 | akshare限流 → 重跑 (增量模式失败不覆盖旧数据); 或等10分钟 |
| 换月期价格跳变 | 检查 `detect_gold_main.py` 是否已跑 (换月窗口依赖switches表) |

---

## 9. 相关文档

| 文档 | 内容 |
|---|---|
| **`gold_strategy_v4.md`** | 策略规格定版 (链路/参数/业绩/归因/时序) |
| `gold_final_strategy.md` | 研究编年史 (v1→v4迭代、旧生产参数、有效性检验) |
| `gold_rollover_plan.md` | 换月调整方法 |
| `ml_anomaly_research.md` 等 | ML异常检测研究归档 (已暂缓) |
