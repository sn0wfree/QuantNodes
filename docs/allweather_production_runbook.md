# 全天候 v3 — 生产运行手册 (Runbook)

> 登记: 2026-09-24 | 配置出处: `docs/allweather_production_candidate.md` v3

---

## 1. 配置摘要

| 项 | 值 |
|----|-----|
| 池 | REF41 (29 期货 + 27 ETF) |
| 引擎 | D v3 管线, **ME 月频**, cov=lw, bond_cap=0.40, window=252 |
| 叠加层 | D.1 负趋势(aggressive)+rapid / D.2 动量(β0.2) / P1-B(θ0.60,k1.0) / 共振门控(60/0.70/0.40/0.50,dd_floor-0.05,k0.9) |
| Alpha sleeve | **Carry 30%** (美元中性多空, 月频, 平滑 20d) |
| 杠杆 | D 内部 vol_target=15% (上限 2.5) |
| 成本 | 期货 1bp / ETF 5bp, 融资 2% |

**基准业绩 (IS 2019-01→2026-06)**: CAGR 10.10% / Sharpe 2.02 / MaxDD -5.20%.
**WF OOS (2022-2026)**: Sharpe ~1.70.
> ⚠️ Sharpe 目标 2.0 仅在 IS 达标; OOS ~1.7. 已文档化, 不掩盖.

## 2. 数据依赖与刷新

| 数据 | 路径 | 刷新命令 | 新鲜度 |
|------|------|---------|--------|
| 价格面板 | `data/allweather/panel_v5.parquet` | `python -m QuantNodes.strategy.allweather.data_etl.build_panel_v5` | 日 |
| Carry | `data/allweather/carry_daily.parquet` | `python3 scripts/extract_carry.py` | 日 |
| Cash proxy | `data/high_freq_macro/v7_index_v2.parquet` | `python -m ...data_etl.build_v7_indices` + `python3 scripts/extend_cash_proxy.py` | 周 |
| 量价(OI) | `data/allweather/volume_oi_cache.parquet` | DuckDB `~/Public/DataCache/futures_options_daily.duckdb` (外部, CTA 团队维护) | 日 |

> **量价新鲜度风险**: 若量价缓存落后于价格面板, `production.load_prod_data`
> 会 **ffill** 填充; 落后 > 5 交易日时监控告警. D.1(负趋势)/rapid 依赖量价,
> 长期落后会削弱该叠加层.

## 3. 日常运行

```bash
# 生成最新目标权重 (打印 + 落盘 data/allweather/signals/)
python3 scripts/allweather_signal.py

# 数据新鲜度 + 组合健康监控 (退出码 1 = 有告警)
python3 scripts/allweather_monitor.py

# 指定历史日期复现 (如对账)
python3 scripts/allweather_signal.py --as-of 2026-06-30
```

**输出**:
- `data/allweather/signals/latest_target.csv` — 最新目标权重快照 (asset, weight)
- `data/allweather/signals/signal_log.parquet` — 历史信号日志 (as_of + 各资产权重)

**运行时长**: 单次 ~40s (D 管线, 全历史重算). 月频调仓, 建议每月末后一交易日运行.

## 4. 告警阈值与处置

| 告警 | 阈值 | 处置 |
|------|------|------|
| 当前回撤 | < -8% | 复核信号; 确认是否共振门控已降杠杆; 接近 -10% 目标线需上报 |
| 数据落后 | > 5 交易日 | 运行对应刷新命令 (§2); 量价落后需联系 CTA 团队 |
| carry 近1年 Sharpe | < 0 | 复核 carry 因子健康 (IC/换手); 评估是否临时降权 |

**注意**: carry 30% 为固定权重 (经 §5.50 验证优于 ERC/数据驱动). **不**因短期
表现调整权重 (防过拟合).

## 5. 上线 checklist

- [x] 配置定版 + 文档 (`allweather_production_candidate.md` v3)
- [x] 生产运行层 (`production.py`) + 信号/监控 CLI
- [x] 数据依赖清单 + 刷新命令
- [x] 告警阈值定义
- [ ] **数据管道自动化** (panel/carry/v7 定时刷新; volume 依赖 CTA)
- [ ] **信号日志定时运行 + 告警通知接入** (邮件/IM)
- [ ] **交易执行对接** (目标权重 → 下单; 期货保证金/合约乘数换算)
- [ ] **实盘对账** (每日 NAV 与回测口径核对)
- [ ] **量价数据源补齐** (当前落后 47 交易日, 需先解决)

## 6. 已知局限 (诚实)

1. **OOS Sharpe ~1.7 < 2.0** — IS 2.02 部分为参数选择拟合; carry 是唯一稳健 alpha
2. **量价数据落后** — volume_oi_cache 截至 2026-07-21, 需补齐
3. **carry 数据依赖 ClickHouse** — `daily_quote` 库网络偶发超时 (需重试)
4. **月频调仓** — 信号仅在月末更新, 月中不调仓
5. **组合目标权重为"资本比例"** — 期货需按保证金/合约乘数换算为手数 (未实现)

## 7. 关键实验证据索引

| 主题 | 报告章节 |
|------|---------|
| 配置定版 | §5.37-§5.42 |
| 前视/过拟合验证 | §5.43 |
| Carry 真 alpha | §5.48 |
| 联合风险预算 (固定权重胜出) | §5.50 |
| carry 平滑 20d | §5.51 |
| IV/VRP (未通过) | §5.52 |
| IVTS + 到期日更正 (未采纳) | §5.53-§5.54 |
| 期限结构曲率 (未通过) | §5.55 |
