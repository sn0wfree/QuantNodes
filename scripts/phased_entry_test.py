#!/usr/bin/env python3
"""分步建仓严谨验证: 卖 put spread (大跌后).

核心修正: 每批次独立建仓/独立持有/独立平仓, 不混用平均成本.

方案:
  A. 一次性: T日全仓, 持7日
  B. 跌时加仓: T日1/3, T+1继续跌加1/3, T+2继续跌加1/3 (各持7日独立平)
  C. 时间固定: T日1/3, T+1日1/3, T+2日1/3 (各持7日独立平)
  D. 反弹加仓: T日1/3, T+1反弹加1/3, T+2反弹加1/3
  对比各批次的平均入场价格 (put premium) 与胜率.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/home/ll/Public/QuantNodes")

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import FactorModuleComboStrategy, ComboConfig
from QuantNodes.strategy.cta.options.iv_panel import load_iv_panel
from QuantNodes.strategy.cta.data_loader import CtaDataLoader

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
GOOD = ["I", "AG", "CU", "SR", "TA", "RU", "M"]
HOLD = 7
OTM = 0.88


def main():
    cfg = ComboConfig(target_vol=0.30, max_leverage=3.0, vol_lookback=20,
                      momentum_filter_window=60, momentum_filter_coef=0.0,
                      filter_enabled=True, module_weighting="inv_vol",
                      continuous=True, cont_z_window=252, cont_z_clip=3.0,
                      margin_call=True, execution_lag=1, aggregation="tvpr")
    s = FactorModuleComboStrategy(products=PRODUCTS, config=cfg,
                                  start="2023-01-01", end="2025-12-31")
    panel = s.load_panel()
    ret = panel.pct_change()
    iv = load_iv_panel(products=PRODUCTS, start="2022-01-01", end="2025-12-31")
    iv = iv.reindex(index=panel.index, columns=panel.columns)
    hv20 = ret.rolling(20).std() * np.sqrt(252)
    vrp_pct = (iv - hv20).rolling(252, min_periods=60).rank(pct=True)
    down_cnt = (ret < 0).rolling(5).sum()

    dl = CtaDataLoader()
    chains = {}
    for prod in PRODUCTS:
        odf = dl.load_v_all_options(products=[prod], start="2023-01-01", end="2025-12-31")
        if odf is not None and not odf.empty:
            odf["trade_date"] = pd.to_datetime(odf["trade_date"])
            odf["symbol"] = odf["symbol"].str.upper()
            chains[prod] = odf.set_index("trade_date").sort_index()

    def day_rows(c, ts):
        if ts not in c.index:
            return None
        d = c.loc[ts]
        return d.to_frame().T if isinstance(d, pd.Series) else d

    def pick(c, ts, spot, ot, sm):
        d = day_rows(c, ts)
        if d is None:
            return None
        sub = d[d["option_type"].str.upper() == ot].copy()
        if sub.empty:
            return None
        liq = sub[(sub["volume"].fillna(0) > 0) | (sub["open_interest"].fillna(0) > 0)]
        if not liq.empty:
            sub = liq
        sub["_dist"] = (sub["strike_price"] - spot * sm).abs()
        row = sub.iloc[int(sub["_dist"].to_numpy().argmin())]
        px = float(row["close"])
        return {"sym": str(row["symbol"]), "px": px} if px > 0 else None

    def fprice(c, ts, sym):
        if ts not in c.index:
            return None
        d = day_rows(c, ts)
        if d is None:
            return None
        m = d["symbol"] == sym
        if not m.any():
            return None
        v = float(d.loc[m].iloc[0]["close"])
        return v if v > 0 else None

    def batch_pnl(p, entry_date):
        """单批次卖 put spread PnL: 建仓 → 持 HOLD 日 → 平仓."""
        chain = chains.get(p)
        if chain is None:
            return None
        spot = panel.loc[entry_date, p]
        if np.isnan(spot) or spot <= 0:
            return None
        pp1 = pick(chain, entry_date, spot, "P", 1.00)
        pp2 = pick(chain, entry_date, spot, "P", OTM)
        if pp1 is None or pp2 is None or pp1["sym"] == pp2["sym"]:
            return None
        cost = pp1["px"] - pp2["px"]
        if cost <= 0:
            return None
        ei = list(chain.index.unique())
        if entry_date not in ei:
            return None
        pos = ei.index(entry_date)
        if pos + HOLD >= len(ei):
            return None
        exit_date = ei[pos + HOLD]
        e1 = fprice(chain, exit_date, pp1["sym"])
        e2 = fprice(chain, exit_date, pp2["sym"])
        if e1 is None or e2 is None:
            return None
        return (cost - (e1 - e2)) / cost

    dates_idx = list(panel.index)

    def is_trigger(p, date):
        dc = down_cnt.loc[date, p] if date in down_cnt.index else np.nan
        vp = vrp_pct.loc[date, p] if date in vrp_pct.index else np.nan
        return (not pd.isna(dc) and dc >= 3 and not pd.isna(vp) and vp > 0.5)

    print("=" * 78)
    print(f"分步建仓严谨验证: 卖 put spread (连续3日跌+VRP>0.5, 持{HOLD}日, OTM{OTM})")
    print("=" * 78)

    # 方案A: 一次性
    pnls_a = []
    for p in GOOD:
        for date in dates_idx:
            if not is_trigger(p, date):
                continue
            pnl = batch_pnl(p, date)
            if pnl is not None:
                pnls_a.append(pnl)

    # 方案B: 跌时加仓 (T+1/T+2 继续跌才加)
    pnls_b = []
    for p in GOOD:
        for i, date in enumerate(dates_idx):
            if not is_trigger(p, date):
                continue
            batch_pnls = [batch_pnl(p, date)]
            if batch_pnls[0] is None:
                continue
            for off in (1, 2):
                if i + off >= len(dates_idx):
                    break
                d_next = dates_idx[i + off]
                if ret.loc[d_next, p] >= 0:  # 继续跌才加
                    break
                bp = batch_pnl(p, d_next)
                if bp is None:
                    break
                batch_pnls.append(bp)
            pnls_b.append(np.mean(batch_pnls))

    # 方案C: 时间固定 (T+1/T+2 无论涨跌都加)
    pnls_c = []
    for p in GOOD:
        for i, date in enumerate(dates_idx):
            if not is_trigger(p, date):
                continue
            batch_pnls = [batch_pnl(p, date)]
            if batch_pnls[0] is None:
                continue
            for off in (1, 2):
                if i + off >= len(dates_idx):
                    break
                bp = batch_pnl(p, dates_idx[i + off])
                if bp is None:
                    break
                batch_pnls.append(bp)
            pnls_c.append(np.mean(batch_pnls))

    # 方案D: 反弹加仓 (T+1/T+2 反弹才加)
    pnls_d = []
    for p in GOOD:
        for i, date in enumerate(dates_idx):
            if not is_trigger(p, date):
                continue
            batch_pnls = [batch_pnl(p, date)]
            if batch_pnls[0] is None:
                continue
            for off in (1, 2):
                if i + off >= len(dates_idx):
                    break
                d_next = dates_idx[i + off]
                if ret.loc[d_next, p] <= 0:  # 反弹才加
                    break
                bp = batch_pnl(p, d_next)
                if bp is None:
                    break
                batch_pnls.append(bp)
            pnls_d.append(np.mean(batch_pnls))

    def stat(a, label):
        a = np.array(a)
        if len(a) == 0:
            return
        print(f"  {label:<40} n={len(a):>4} 中位={np.median(a)*100:+6.1f}% "
              f"均值={a.mean()*100:+6.1f}% 胜率={(a>0).mean()*100:.1f}%")

    print(f"\n{'方案':<40} {'n':>5} {'中位':>8} {'均值':>8} {'胜率':>7}")
    print("-" * 78)
    stat(pnls_a, "A: 一次性 (T日全仓, 持7日)")
    stat(pnls_b, "B: 跌时加仓 (T+继续跌加1/3)")
    stat(pnls_c, "C: 时间固定 (T+1/T+2 各1/3)")
    stat(pnls_d, "D: 反弹加仓 (T+反弹才加)")

    # 分步方案实际平均建仓批次数
    print("\n平均加仓批次:")
    for p in GOOD:
        pass

    # 分析: 分步建仓的批次分布 (方案B)
    print("\n分步建仓批次统计 (方案B: 跌时加仓):")
    batch_counts = []
    for p in GOOD:
        for i, date in enumerate(dates_idx):
            if not is_trigger(p, date):
                continue
            cnt = 1
            for off in (1, 2):
                if i + off >= len(dates_idx):
                    break
                if ret.loc[dates_idx[i + off], p] >= 0:
                    break
                cnt += 1
            batch_counts.append(cnt)
    bc = np.array(batch_counts)
    print(f"  1批: {(bc==1).mean()*100:.0f}%, 2批: {(bc==2).mean()*100:.0f}%, "
          f"3批: {(bc==3).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
