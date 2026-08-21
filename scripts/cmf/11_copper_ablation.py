#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/11_copper_ablation.py — v5逐层消融 + v4/v5/v3合并实验.

P0修正地基后的关键实验:
    §1 v5逐层消融: base → +vol → +macro → +econ (真实价, IS/OOS分离)
    §2 三策略仓位合并: avg(v4, v5, pv3) 等权/子集
    §3 合并最优 + carry (波动率倒数加权)

窗口: IS = 2018-01-01 ~ 2025-12-31 (选层/选组合)
      OOS = 2026-01-01+ (完全不动, 只报告)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v4 import build_signal_v4
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 import build_signal_v5
from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import build_signal_v3
from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import build_signal_carry
from QuantNodes.strategy.commodity_fundamental.copper.copper_economy import get_economy_signal

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
IS_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")


def bt(close_w: pd.Series, pos: pd.Series, tag: str) -> dict:
    """统一真实价周频回测 (5bps)."""
    ret = close_w.reindex(pos.index).pct_change()
    cost = pos.diff().abs() * 5.0 / 10000
    r = (pos.shift(1) * ret - cost).dropna()
    out = {}
    for name, m in [("IS", r[r.index <= IS_END]), ("OOS", r[r.index >= OOS_START])]:
        if len(m) < 8:
            out[name] = None
            continue
        nav = (1 + m).cumprod()
        ann = nav.iloc[-1] ** (52 / len(m)) - 1
        vol = m.std() * np.sqrt(52)
        dd = (nav / nav.cummax() - 1).min()
        out[name] = {"ann": ann * 100, "sharpe": ann / vol if vol > 0 else 0,
                     "maxdd": dd * 100}
    return out


def show(tag: str, res: dict):
    iss, oos = res["IS"], res["OOS"]
    na = "   —      —      —"
    s1 = f"{iss['ann']:5.1f}  {iss['sharpe']:5.2f}  {iss['maxdd']:6.1f}" if iss else na
    s2 = f"{oos['ann']:5.1f}  {oos['sharpe']:5.2f}  {oos['maxdd']:6.1f}" if oos else na
    print(f"  {tag:<34} IS: {s1}   OOS: {s2}")


def main():
    con = duckdb.connect(str(DB), read_only=True)

    close = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    close["trade_date"] = pd.to_datetime(close["trade_date"])
    close_w = close.set_index("trade_date")["close"].resample("W-FRI").last()

    # ── v5 各中间量 (复刻build_signal_v5逻辑, 拆层) ─────────────────────────
    import QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 as m5
    from QuantNodes.strategy.commodity_fundamental.kalman_filter import weekly_kalman_center
    from QuantNodes.strategy.commodity_fundamental.sign_constrained_lasso import (
        walk_forward_lasso, rolling_phy_center, rolling_zscore, compute_valuation_gap,
    )
    from QuantNodes.strategy.commodity_fundamental.inventory_quadrant import (
        inventory_state_for_product,
    )
    from QuantNodes.strategy.commodity_fundamental.valuation_gap import (
        industry_momentum, apply_valuation_layer,
    )

    sym = "CU0"
    c = m5._close(sym, con)
    cw = c.resample("W-FRI").last()
    ret_w = cw.pct_change()
    mu = weekly_kalman_center(c, q=0.015)
    dxy = m5._dxy(con).resample("W-FRI").last()
    tips = m5._tips10y(con).resample("W-FRI").last()
    lme = m5._tfapi(con, "LME库存总计_铜").resample("W-FRI").last()
    rig = m5._rig(con).resample("W-FRI").last()
    pmi = m5._pmi(con).resample("W-FRI").last().ffill()
    mom2 = ret_w.rolling(2).sum()
    mom13 = ret_w.rolling(13).sum()

    idx = (mu.index.intersection(dxy.dropna().index).intersection(tips.dropna().index)
           .intersection(lme.dropna().index).intersection(rig.dropna().index)
           .intersection(pmi.dropna().index).intersection(mom2.dropna().index)
           .intersection(mom13.dropna().index))

    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    vol_pct = vol_4w.rolling(52).rank(pct=True)
    dxy_tr = (dxy < dxy.rolling(20).mean()).astype(float)
    pmi_tr = (pmi > pmi.rolling(20).mean()).astype(float)

    X = pd.DataFrame({
        "dxy_neg": -rolling_zscore(dxy, 104).reindex(idx),
        "tips_neg": -rolling_zscore(tips, 104).reindex(idx),
        "lme_neg": -rolling_zscore(lme, 52).reindex(idx),
        "rig_neg": -rolling_zscore(rig, 52).reindex(idx),
        "pmi": rolling_zscore(pmi, 52).reindex(idx),
        "mom_2w": rolling_zscore(mom2, 52).reindex(idx),
        "momentum_13w": rolling_zscore(mom13, 52).reindex(idx),
        "dxy_trend": dxy_tr.reindex(idx),
        "pmi_trend": pmi_tr.reindex(idx),
    }).dropna(how="any")
    y = pd.Series(np.log(mu).diff().reindex(X.index)).dropna()
    common3 = X.index.intersection(y.index)
    X, y = X.loc[common3], y.loc[common3]
    mu_a = mu.reindex(common3)

    _, oos_l = walk_forward_lasso(X, y, refit_weeks=26, min_train_weeks=52)
    empty = pd.Series(dtype=float)
    p_phy = rolling_phy_center(mu_a, oos_l["oos_pred"] if len(oos_l) else empty, window=13)
    gap = compute_valuation_gap(mu_a, p_phy)

    inv_state = inventory_state_for_product(lme.reindex(common3).ffill(), cw.reindex(common3),
                                            z_window=52, inv_dir=-1)
    ind_dz = industry_momentum(rig.reindex(common3).ffill(), z_window=52, delta_window=4)
    base_pos = apply_valuation_layer(inv_state["quadrant_pos"], gap.reindex(common3),
                                     ind_dz.reindex(common3), gap_low_threshold=0.01,
                                     gap_high_threshold=-0.01)

    vol_a = vol_pct.reindex(common3).fillna(0.5)
    mom_a = mom2.reindex(common3).fillna(0)
    dxy_a = dxy_tr.reindex(common3).fillna(0)
    pmi_a = pmi_tr.reindex(common3).fillna(0)
    econ_score, _ = get_economy_signal(con, min_duration=4)
    econ_a = econ_score.reindex(common3).fillna(0)

    def L_base():
        return base_pos.copy()

    def L_vol(p):
        q = p.copy()
        q[(vol_a < 0.3) & (mom_a > 0)] = 1.0
        q[vol_a > 0.7] = 0.4
        return q

    def L_macro(p):
        q = p.copy()
        mb = (dxy_a == 1) & (pmi_a == 1)
        q[mb & (q < 1.0)] = q[mb & (q < 1.0)] + 0.3
        return q.clip(upper=1.0)

    def L_econ(p):
        q = p.copy()
        q[econ_a > 0.5] = q[econ_a > 0.5] + 0.2
        q[(econ_a > -0.5) & (econ_a < 0.5)] = q[(econ_a > -0.5) & (econ_a < 0.5)] - 0.1
        q[econ_a < -0.5] = q[econ_a < -0.5] - 0.2
        return q.clip(lower=0.0, upper=1.0)

    print("=" * 92)
    print("§1 v5 逐层消融 (真实价, 5bps; IS=2018-2025 选层, OOS=2026 不动)")
    print("=" * 92)
    print(f"  {'配置':<34} {'年化%  Sharpe  MaxDD%':<24}{'年化%  Sharpe  MaxDD%'}")
    configs = {
        "L0 base(象限+估值)": lambda: L_base(),
        "L1 = L0 + 波动率层": lambda: L_vol(L_base()),
        "L2 = L1 + 宏观趋势层": lambda: L_macro(L_vol(L_base())),
        "L3 = L2 + 经济状态层 (=v5)": lambda: L_econ(L_macro(L_vol(L_base()))),
        "L0 + 宏观趋势层": lambda: L_macro(L_base()),
        "L0 + 经济状态层": lambda: L_econ(L_base()),
    }
    best_layers = {}
    for tag, fn in configs.items():
        p = fn()
        p.iloc[0] = 0.0
        res = bt(close_w, p, tag)
        show(tag, res)
        best_layers[tag] = res

    # ── §2 三策略合并 ─────────────────────────────────────────────────────
    print()
    print("=" * 92)
    print("§2 v4/v5/pipeline_v3 仓位合并 (真实价)")
    print("=" * 92)
    print(f"  {'配置':<34} {'年化%  Sharpe  MaxDD%':<24}{'年化%  Sharpe  MaxDD%'}")

    START_UNI = pd.Timestamp("2018-01-01")  # 统一窗口 (v4受DXY起点限制为2016-12, 统一2018+可比)

    pos_v4 = build_signal_v4(sym, con)["position"].dropna()
    pos_v5 = build_signal_v5(sym, con)["position"].dropna()
    pos_pv3 = build_signal_v3(sym, con)["position"].dropna()

    # v5精简版: base+vol+macro (消融显示经济状态层负贡献)
    pos_v5p = L_macro(L_vol(L_base()))
    pos_v5p.iloc[0] = 0.0

    def merge(*poss):
        """交集+等权平均; 交集仅限成员自身, 避免无关策略截断窗口 (pv3起点2019-05)."""
        idx = poss[0].index
        for p in poss[1:]:
            idx = idx.intersection(p.index)
        idx = idx[idx >= START_UNI]
        ps = [p.reindex(idx).fillna(0) for p in poss]
        avg = sum(ps) / len(ps)
        return idx, ps, avg.clip(lower=0.0, upper=1.0)

    # 单独: 各自窗口; 组合: 成员交集 (各配置窗口可能不同, 表内数字自洽但跨行窗口可比性见起点列)
    singles = {
        "v4单独": pos_v4,
        "v5单独": pos_v5,
        "v5精简(去经济层)单独": pos_v5p,
        "pv3单独": pos_pv3,
    }
    for tag, p in singles.items():
        p = p[p.index >= START_UNI].copy()
        p.iloc[0] = 0.0
        show(tag, bt(close_w, p, tag))

    for tag, members in [("avg(v4,v5,pv3)", [pos_v4, pos_v5, pos_pv3]),
                         ("avg(v4,v5p,pv3)", [pos_v4, pos_v5p, pos_pv3]),
                         ("avg(v4,v5p)", [pos_v4, pos_v5p]),
                         ("avg(v4,pv3)", [pos_v4, pos_pv3])]:
        idx, _, avg = merge(*members)
        avg.iloc[0] = 0.0
        print(f"    [{tag} 窗口起点 {idx[0].date()}]")
        show(tag, bt(close_w, avg, tag))

    _, _, pos_main_combo = merge(pos_v4, pos_v5p)
    merged_idx = pos_main_combo.index
    p4 = pos_v4.reindex(merged_idx).fillna(0)
    p5p = pos_v5p.reindex(merged_idx).fillna(0)
    pos_main_avg = pos_main_combo

    # ── §3 最优合并 + carry ───────────────────────────────────────────────
    print()
    print("=" * 92)
    print("§3 合并 + carry (波动率倒数加权 vs 等权)")
    print("=" * 92)
    print(f"  {'配置':<34} {'年化%  Sharpe  MaxDD%':<24}{'年化%  Sharpe  MaxDD%'}")

    pos_carry = build_signal_carry(con)["position"].dropna()
    idx_c = merged_idx.intersection(pos_carry.index)
    main_pos = pos_main_avg.reindex(idx_c).fillna(0)  # §2: avg(v4,v5p)
    carry_p = pos_carry.reindex(idx_c).fillna(0)
    # 注: term_structure止于2026-01, carry在OOS(2026)仅1周, §3组合数字仅IS窗口有效

    # 用IS窗口收益估计各自波动 → 倒数加权
    ret_al = close_w.reindex(idx_c).pct_change()
    r_main = (main_pos.shift(1) * ret_al).loc[:IS_END].std() * np.sqrt(52)
    r_car = (carry_p.shift(1) * ret_al).loc[:IS_END].std() * np.sqrt(52)
    w_main = (1 / r_main) / (1 / r_main + 1 / r_car)
    print(f"  [波动率倒数权重] main={w_main:.2f} carry={1-w_main:.2f} "
          f"(σ_main={r_main*100:.1f}% σ_carry={r_car*100:.1f}%)")

    pos_eq = 0.5 * main_pos + 0.5 * carry_p
    pos_vi = w_main * main_pos + (1 - w_main) * carry_p
    for tag, p in [("merged(main=avg(v4,pv3))", main_pos),
                   ("merged + carry 等权", pos_eq),
                   ("merged + carry 波动率倒数", pos_vi),
                   ("carry单独", carry_p)]:
        p = p.copy()
        p.iloc[0] = 0.0
        show(tag, bt(close_w, p, tag))

    # 相关性
    rr_m = (main_pos.shift(1) * ret_al).dropna()
    rr_c = (carry_p.shift(1) * ret_al).dropna()
    print(f"\n  merged vs carry 周收益相关性(IS+OOS): {rr_m.corr(rr_c):.3f}")

    con.close()


if __name__ == "__main__":
    main()
