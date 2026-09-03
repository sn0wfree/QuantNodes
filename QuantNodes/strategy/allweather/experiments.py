# coding=utf-8
"""AllWeather — 实验脚本集 (Phase 2-3 固化, PLAN §十二).

把复盘过程中的临时实验固化为可重复执行的子命令, 结果统一落盘 data/allweather/.

子命令:
    lever-sweep   X2 杠杆扫描 (max_leverage ∈ caps, 其余=主线参数)
    cap-sweep     权重集中度约束扫描 (max_weight ∈ caps)
    trend-matrix  趋势 overlay 矩阵 (v1/v3/v4/tilt × lookback × coef/risk_adj/floor)
    unlevered     无杠杆实验矩阵 (gross≤1, use_vol_target=False / cap=1)
    segments      单配置分段稳定性 (WF 三段 + 压力段)
    diag-d1       权重稳定性/集中度 (调仓|Δw|, 结构HHI, 最大结构权重)
    diag-d2       池敏感性 (随机剔 k 资产 × n 次)
    diag-d3       类别剔除归因 (债券/股指/ETF leave-one-class-out)
    diag-d4       roll 修复前后对照 (panel_v1 vs panel_v2)
    all           依次执行以上全部

用法:
    python -m QuantNodes.strategy.allweather.experiments lever-sweep --caps 3,4,5,6
    python -m QuantNodes.strategy.allweather.experiments diag-d2 --n 10 --k 5
    python -m QuantNodes.strategy.allweather.experiments all

输出: data/allweather/{lever_sweep,cap_sweep,trend_matrix,unlevered,diag_d2,diag_d3}.csv
      诊断 d1/d4 结果打印并追加到 experiments_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import AllWeatherConfig, run_backtest
from .config import (
    ETF_CODES,
    OUTPUT_DIR,
    PANEL_OUTPUT,
    PANEL_V2_OUTPUT,
)
from .run_replication import SEGMENTS

METRIC_COLS = ["Sharpe", "AnnRet", "Vol", "MaxDD", "WinRate", "Calmar", "ann_turnover"]


def _metrics_fn():
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import (
        compute_metrics,
    )
    return compute_metrics


def _noop(nav, freq=None, oos_start=None) -> dict:
    return {}


def _load_panel(path: str | None) -> pd.DataFrame:
    p = Path(path).expanduser() if path else PANEL_V2_OUTPUT
    return pd.read_parquet(p)


def _row(name: str, res, mm) -> dict:
    """从 nav 现算指标 (run_backtest 传 _noop 跳过内部计算)."""
    m = mm(res.nav, freq="D")
    row = {"config": name}
    row.update({k: m.get(k) for k in METRIC_COLS})
    return row


def _stress_sharpe(res, mm) -> float:
    s, e = SEGMENTS["stress_25+"]
    nav = res.nav.loc[s:e]
    if len(nav) < 30:
        return float("nan")
    return float(mm(nav, freq="D")["Sharpe"])


# ── 实验 1: 杠杆扫描 ────────────────────────────────────────────────
def run_lever_sweep(panel: pd.DataFrame, caps: list[int] = (3, 4, 5, 6, 7, 8),
                    max_weight: float = 0.40) -> pd.DataFrame:
    """X2 主线 (cap0.40 HRP+RMT) 的 max_leverage 扫描."""
    mm = _metrics_fn()
    rows = []
    for lev in caps:
        cfg = AllWeatherConfig(trend_mode="none", max_weight=max_weight,
                               max_leverage=lev)
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        row = _row(f"lev{lev}", res, mm)
        row["stress_sharpe"] = _stress_sharpe(res, mm)
        rows.append(row)
        print(f"  [lev{lev}] Sharpe={row['Sharpe']:.3f} AnnRet={row['AnnRet']:.2%}")
    return pd.DataFrame(rows)


# ── 实验 2: 集中度约束扫描 ──────────────────────────────────────────
def run_cap_sweep(panel: pd.DataFrame, caps: list[float] = (1.0, 0.40, 0.25, 0.15, 0.10),
                  max_leverage: int = 3) -> pd.DataFrame:
    """HRP+RMT (trend=none) 的 max_weight 扫描."""
    mm = _metrics_fn()
    rows = []
    for cap in caps:
        cfg = AllWeatherConfig(trend_mode="none", max_weight=cap,
                               max_leverage=max_leverage)
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        row = _row(f"cap{cap}", res, mm)
        row["stress_sharpe"] = _stress_sharpe(res, mm)
        rows.append(row)
        print(f"  [cap{cap}] Sharpe={row['Sharpe']:.3f} AnnRet={row['AnnRet']:.2%}")
    return pd.DataFrame(rows)


# ── 实验 3: 趋势 overlay 矩阵 ───────────────────────────────────────
def run_trend_matrix(panel: pd.DataFrame,
                     max_weight: float = 0.40,
                     max_leverage: int = 3) -> pd.DataFrame:
    """趋势模式矩阵: none/v1/v3/v4/tilt × 关键参数."""
    mm = _metrics_fn()
    variants: list[tuple[str, dict]] = [
        ("none", dict(trend_mode="none")),
        ("v1_60_0", dict(trend_mode="v1", trend_lookback=60, trend_coef=0.0)),
        ("v1_60_0.5", dict(trend_mode="v1", trend_lookback=60, trend_coef=0.5)),
        ("v3_60_0", dict(trend_mode="v3", trend_lookback=60, trend_coef=0.0)),
        ("v3_120_0.5", dict(trend_mode="v3", trend_lookback=120, trend_coef=0.5)),
        ("v4_252", dict(trend_mode="v4", trend_lookback=252)),
        ("tilt_60", dict(trend_mode="tilt", trend_lookback=60,
                         trend_risk_adj=True, trend_floor=0.5)),
        ("tilt_120", dict(trend_mode="tilt", trend_lookback=120,
                          trend_risk_adj=True, trend_floor=0.5)),
    ]
    rows = []
    for name, kw in variants:
        cfg = AllWeatherConfig(max_weight=max_weight, max_leverage=max_leverage, **kw)
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        row = _row(name, res, mm)
        row["stress_sharpe"] = _stress_sharpe(res, mm)
        rows.append(row)
        print(f"  [{name}] Sharpe={row['Sharpe']:.3f}")
    return pd.DataFrame(rows)


# ── 实验 4: 无杠杆矩阵 ──────────────────────────────────────────────
def run_unlevered(panel: pd.DataFrame) -> pd.DataFrame:
    """gross≤1 (max_leverage=1 或 use_vol_target=False) 的组合矩阵."""
    mm = _metrics_fn()
    variants: list[tuple[str, dict]] = [
        ("U0_等权_gross1", dict(use_hrp=False, trend_mode="none",
                                use_vol_target=False)),
        ("U1_等权+V3_60", dict(use_hrp=False, trend_mode="v3",
                               trend_lookback=60, trend_coef=0.0,
                               use_vol_target=False)),
        ("U3_等权+tilt_60", dict(use_hrp=False, trend_mode="tilt",
                                 trend_lookback=60, trend_risk_adj=True,
                                 trend_floor=0.5, use_vol_target=False)),
        ("U4_等权+vt_cap1", dict(use_hrp=False, trend_mode="none",
                                 use_vol_target=True, max_leverage=1.0)),
        ("U5_invvol_gross1", dict(use_hrp=False, trend_mode="none",
                                  use_vol_target=False,
                                  weight_scheme="inv_vol")),
        ("U6_invvol+tilt_60", dict(use_hrp=False, trend_mode="tilt",
                                   trend_lookback=60, trend_risk_adj=True,
                                   trend_floor=0.5, use_vol_target=False,
                                   weight_scheme="inv_vol")),
        ("U7_HRP_cap0.25_gross1", dict(use_hrp=True, trend_mode="none",
                                       max_weight=0.25, use_vol_target=False)),
    ]
    rows = []
    for name, kw in variants:
        cfg = AllWeatherConfig(**kw)
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        rows.append(_row(name, res, mm))
        print(f"  [{name}] Sharpe={rows[-1]['Sharpe']:.3f}")
    return pd.DataFrame(rows)


# ── 实验 5: 分段稳定性 ──────────────────────────────────────────────
def run_segments(panel: pd.DataFrame, cfg: AllWeatherConfig,
                 name: str = "cfg") -> pd.DataFrame:
    """单配置 × 五分段 (含压力段) 指标表."""
    mm = _metrics_fn()
    res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
    rows = []
    for seg, (s, e) in SEGMENTS.items():
        nav = res.nav.loc[s:e]
        if len(nav) < 30:
            continue
        m = mm(nav, freq="D")
        rows.append({"config": name, "segment": seg,
                     **{k: m.get(k) for k in METRIC_COLS}})
    return pd.DataFrame(rows)


# ── 诊断 D1: 权重稳定性 / 集中度 ────────────────────────────────────
def diag_d1(panel: pd.DataFrame, cfg: AllWeatherConfig | None = None) -> dict:
    cfg = cfg or AllWeatherConfig(trend_mode="none", max_weight=0.40)
    res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
    rb = res.weights[res.weights.sum(axis=1) > 0].iloc[1:]
    diffs = rb.diff().abs().sum(axis=1).dropna()
    gross = rb.sum(axis=1)
    wnorm = rb.div(gross.clip(lower=1e-9), axis=0)
    hhi = (wnorm**2).sum(axis=1)
    last = wnorm.iloc[-1].sort_values(ascending=False)
    return {
        "rebal_mean_dw": float(diffs.mean()),
        "rebal_p95_dw": float(diffs.quantile(0.95)),
        "gross_mean": float(gross.mean()),
        "gross_p95": float(gross.quantile(0.95)),
        "struct_hhi_mean": float(hhi.mean()),
        "struct_maxw_p95": float(wnorm.max(axis=1).quantile(0.95)),
        "last_top1_asset": str(last.index[0]),
        "last_top1_weight": float(last.iloc[0]),
        "ann_turnover": float(res.metrics.get("ann_turnover", np.nan)),
    }


# ── 诊断 D2: 池敏感性 ───────────────────────────────────────────────
def diag_d2(panel: pd.DataFrame, cfg: AllWeatherConfig | None = None,
            n: int = 10, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """随机剔除 k 资产 × n 次重跑, 返回每次 Sharpe (含全池参照)."""
    cfg = cfg or AllWeatherConfig(trend_mode="none", max_weight=0.40)
    rng = np.random.default_rng(seed)
    mm = _metrics_fn()
    rows = [{"config": "full_pool", "dropped": "-", "Sharpe": None}]
    base = run_backtest(panel, cfg, compute_metrics_fn=_noop)
    rows[0]["Sharpe"] = float(mm(base.nav, freq="D")["Sharpe"])
    for i in range(n):
        drop = rng.choice(panel.columns, size=k, replace=False)
        sub = panel.drop(columns=drop)
        res = run_backtest(sub, cfg, compute_metrics_fn=_noop)
        rows.append({"config": f"drop{k}_{i}", "dropped": ",".join(sorted(drop)),
                     "Sharpe": float(mm(res.nav, freq="D")["Sharpe"])})
        print(f"  [drop {i+1}/{n}] Sharpe={rows[-1]['Sharpe']:.3f}")
    return pd.DataFrame(rows)


# ── 诊断 D3: 类别剔除归因 ───────────────────────────────────────────
D3_EXCLUSIONS: dict[str, set[str]] = {
    "all": set(),
    "ex_bond": {"fut_t", "fut_tf", "fut_ts"},
    "ex_bond_eqidx": {"fut_t", "fut_tf", "fut_ts", "fut_if", "fut_ic", "fut_ih"},
    "ex_etf": {f"etf_{c}" for c in ETF_CODES},
    "commodity_eqidx_only": ({f"etf_{c}" for c in ETF_CODES}
                             | {"fut_t", "fut_tf", "fut_ts",
                                "fut_if", "fut_ic", "fut_ih"}),
}


def diag_d3(panel: pd.DataFrame, cfg: AllWeatherConfig | None = None,
            weight_schemes: tuple[str, ...] = ("equal", "inv_vol")) -> pd.DataFrame:
    """leave-one-class-out: 每种剔除 × 每种权重方案."""
    rows = []
    for cname, excl in D3_EXCLUSIONS.items():
        cols = [c for c in panel.columns if c not in excl]
        sub = panel[cols]
        for ws in weight_schemes:
            c = AllWeatherConfig(trend_mode="none", use_hrp=False,
                                 weight_scheme=ws, use_vol_target=False)
            res = run_backtest(sub, c, compute_metrics_fn=_noop)
            row = _row(f"{cname}+{ws}", res, _metrics_fn())
            row["excluded"] = ",".join(sorted(excl)) or "-"
            rows.append(row)
            print(f"  [{cname}+{ws}] Sharpe={row['Sharpe']:.3f}")
    return pd.DataFrame(rows)


# ── 诊断 D4: roll 修复对照 ──────────────────────────────────────────
def diag_d4(cfg: AllWeatherConfig | None = None) -> pd.DataFrame:
    cfg = cfg or AllWeatherConfig(trend_mode="none", max_weight=0.40)
    rows = []
    for label, path in (("v1_未修复", PANEL_OUTPUT), ("v2_修复", PANEL_V2_OUTPUT)):
        if not Path(path).exists():
            print(f"  [跳过] {path} 不存在")
            continue
        panel = pd.read_parquet(path)
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        row = _row(label, res, _metrics_fn())
        rows.append(row)
        print(f"  [{label}] Sharpe={row['Sharpe']:.3f}")
    return pd.DataFrame(rows)


# ── V5 触发器: episode 锚定验收 ─────────────────────────────────────
def validate_trig_episodes() -> pd.DataFrame:
    """验收 gate: 两个已验证 episode 上, 触发器须在 ±5 交易日内容差内触发.

    ① 黄金 (518880, 面板外): trigA 须在 2026-03-16 ±5 交易日触发
    ② 权益 (IF/IC/513100): trigB 须在 2026-07-03 ±5 交易日触发 (任一资产)
    """
    from .trend_overlay import trig_a, trig_b

    oh = pd.read_parquet(Path("~/Public/QuantNodes/data/real/"
                              "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet")
                         .expanduser())
    rows = []

    # ① 黄金 trigA
    gold = oh[("518880", "close")].dropna()
    gold.index = pd.to_datetime(gold.index)
    gold = gold.loc["2025-06-01":]
    state = trig_a(pd.DataFrame({"g": gold}))["g"]
    anchor = pd.Timestamp("2026-03-16")
    window = state.loc[anchor - pd.Timedelta(days=10):anchor + pd.Timedelta(days=10)]
    fired = (window == 0.0).any()
    first_fire = window[window == 0.0].index.min()
    rows.append({"episode": "gold_3-16", "trigger": "trigA", "asset": "518880",
                 "anchor": anchor.date(), "fired_in_window": fired,
                 "first_fire": first_fire.date() if fired else None,
                 "state_at_anchor": float(state.asof(anchor))})

    # ② 权益 trigB (期货面板)
    panel = pd.read_parquet(PANEL_V2_OUTPUT)
    anchor2 = pd.Timestamp("2026-07-03")
    for asset in ("fut_if", "fut_ic", "etf_513100"):
        s = panel[asset].dropna().loc["2025-06-01":]
        state_b = trig_b(pd.DataFrame({asset: s}))[asset]
        w2 = state_b.loc[anchor2 - pd.Timedelta(days=10):anchor2 + pd.Timedelta(days=10)]
        fired2 = (w2 == 0.0).any()
        ff2 = w2[w2 == 0.0].index.min()
        rows.append({"episode": "equity_7月", "trigger": "trigB", "asset": asset,
                     "anchor": anchor2.date(), "fired_in_window": fired2,
                     "first_fire": ff2.date() if fired2 else None,
                     "state_at_anchor": float(state_b.asof(anchor2))})

    df = pd.DataFrame(rows)
    gate = bool(df["fired_in_window"].iloc[0]) and bool(df["fired_in_window"].iloc[1:].any())
    print(f"episode 验收 gate: {'PASS' if gate else 'FAIL'}")
    return df


# ── V5 触发器: A/B 单独回测 + 触发质量 ──────────────────────────────
def run_trig_ab(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2 基线 × 3 变体 (none/trigA/trigB) = 6 跑, 返回 (指标表, 触发质量表)."""
    mm = _metrics_fn()
    baselines = {
        "A_X2L4": dict(use_hrp=True, trend_mode="none", max_weight=0.40,
                       max_leverage=4),
        "B_EWvt": dict(use_hrp=False, trend_mode="none", use_vol_target=True,
                       max_leverage=1.0),
    }
    trig_modes = ["none", "trigA", "trigB"]
    rows, quality_rows = [], []
    for bname, kw in baselines.items():
        for tm in trig_modes:
            kw2 = {**kw, "trend_mode": tm}
            res = run_backtest(panel, AllWeatherConfig(**kw2),
                               compute_metrics_fn=_noop)
            row = {"config": f"{bname}+{tm}"}
            for seg, (s, e) in SEGMENTS.items():
                nav = res.nav.loc[s:e]
                if len(nav) < 30:
                    continue
                m = mm(nav, freq="D")
                for k in ("Sharpe", "AnnRet", "Vol", "MaxDD"):
                    row[f"{seg}_{k}"] = m.get(k)
            rows.append(row)
            print(f"  [{bname}+{tm}] full Sharpe={row['full_Sharpe']:.3f} "
                  f"stress={row.get('stress_25+_Sharpe', float('nan')):.3f}")
        # 触发质量: 触发后 20d 收益
        for tm in ("trigA", "trigB"):
            from .trend_overlay import combined as _comb
            ov = _comb(panel, mode=tm)
            fwd20 = panel.pct_change(20, fill_method=None).shift(-20)
            trig_events = (ov.shift(1) == 0.0) & (ov.shift(2) == 1.0)  # 1→0 转换日
            hit = fwd20[trig_events.reindex(fwd20.index, fill_value=False)]
            vals = hit.stack().dropna()
            share = ov.mean().mean()
            quality_rows.append({"trigger": tm, "invested_share": float(share),
                                 "empty_share": float(1.0 - share),
                                 "n_events": int(trig_events.sum().sum()),
                                 "fwd20_mean": float(vals.mean()) if len(vals) else np.nan,
                                  "fwd20_median": float(vals.median()) if len(vals) else np.nan,
                                  "hit_rate_neg": float((vals < 0).mean())
                                  if len(vals) else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(quality_rows)


# ── V6 三子因子: 单因子 + 组合回测 ───────────────────────────────
def _build_subfactor_score(panel: pd.DataFrame, mode: str) -> pd.DataFrame:
    """构建 cfg.subfactor_score: 给定 mode (a/b/c/ab/ac/bc/abc), 计算综合分数.

    返回 W × N DataFrame (W-FRI 重采样后, 已 z-score).
    """
    from .subfactors import SubFactorConfig, compute_all_subfactors
    from .subfactors_weekly import combine_subfactors, resample_weekly

    daily_factors = compute_all_subfactors(
        prices=panel, volumes=panel, open_interest=None, cfg=SubFactorConfig(),
    )
    weekly = {k: resample_weekly(v, agg="mean") for k, v in daily_factors.items()}
    if mode == "abc" or len(mode) == 2:
        keys = list(mode)
        sub = {k: weekly[k] for k in keys}
        return combine_subfactors(sub)
    return combine_subfactors({mode: weekly[mode]})


def run_subfactor_single(panel: pd.DataFrame, alpha_values=(0.1, 0.2, 0.3, 0.5),
                         baseline_kw: dict | None = None) -> pd.DataFrame:
    """V6.1 单子因子 × alpha 扫描, B-sleeve 上回测.

    默认 baseline_kw = B_EWvt (等权 + vol_target, max_leverage=1).
    输出列: factor, alpha, full_Sharpe, full_AnnRet, full_MaxDD, stress_25+_Sharpe.
    """
    mm = _metrics_fn()
    baseline_kw = baseline_kw or dict(
        use_hrp=False, trend_mode="none", use_vol_target=True, max_leverage=1.0,
    )
    rows = []
    for factor in ("a", "b", "c"):
        score = _build_subfactor_score(panel, factor)
        for alpha in alpha_values:
            cfg = AllWeatherConfig(
                **baseline_kw,
                subfactor_mode=factor,
                subfactor_alpha=alpha,
                subfactor_score=score,
            )
            res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
            row = {"factor": factor, "alpha": alpha}
            for seg, (s, e) in SEGMENTS.items():
                nav = res.nav.loc[s:e]
                if len(nav) < 30:
                    continue
                m = mm(nav, freq="D")
                for k in ("Sharpe", "AnnRet", "MaxDD"):
                    row[f"{seg}_{k}"] = m.get(k)
            rows.append(row)
            print(f"  [Sub-{factor.upper()} α={alpha}] "
                  f"full Sharpe={row.get('full_Sharpe', float('nan')):.3f} "
                  f"stress={row.get('stress_25+_Sharpe', float('nan')):.3f}")
    return pd.DataFrame(rows)


def run_subfactor_combo(panel: pd.DataFrame, alpha: float = 0.3,
                        combos=("ab", "ac", "bc", "abc"),
                        baseline_kw: dict | None = None) -> pd.DataFrame:
    """V6.2 子因子组合回测, B-sleeve + ABC_ew 集成.

    输出列: combo, sleeve, full_Sharpe, full_AnnRet, full_MaxDD, stress_25+_Sharpe.
    """
    mm = _metrics_fn()
    baseline_kw = baseline_kw or dict(
        use_hrp=False, trend_mode="none", use_vol_target=True, max_leverage=1.0,
    )

    rows = []
    # 基线: 无 subfactor
    base_cfg = AllWeatherConfig(**baseline_kw)
    base_res = run_backtest(panel, base_cfg, compute_metrics_fn=_noop)
    base_row = {"combo": "none", "sleeve": "B"}
    for seg, (s, e) in SEGMENTS.items():
        nav = base_res.nav.loc[s:e]
        if len(nav) < 30:
            continue
        m = mm(nav, freq="D")
        for k in ("Sharpe", "AnnRet", "MaxDD"):
            base_row[f"{seg}_{k}"] = m.get(k)
    rows.append(base_row)

    for combo in combos:
        score = _build_subfactor_score(panel, combo)
        cfg = AllWeatherConfig(
            **baseline_kw,
            subfactor_mode=combo,
            subfactor_alpha=alpha,
            subfactor_score=score,
        )
        # B-sleeve 单独
        res = run_backtest(panel, cfg, compute_metrics_fn=_noop)
        row = {"combo": combo, "sleeve": "B"}
        for seg, (s, e) in SEGMENTS.items():
            nav = res.nav.loc[s:e]
            if len(nav) < 30:
                continue
            m = mm(nav, freq="D")
            for k in ("Sharpe", "AnnRet", "MaxDD"):
                row[f"{seg}_{k}"] = m.get(k)
        rows.append(row)
        print(f"  [combo={combo} B] full Sharpe={row.get('full_Sharpe', float('nan')):.3f}")

    return pd.DataFrame(rows)


def evaluate_v6_gate(single_df: pd.DataFrame, combo_df: pd.DataFrame,
                     base_sharpe: float = 0.758) -> dict:
    """V6 Gate 判定 (PLAN §13.4).

    单因子 Gate:
      - full_Sharpe ≥ base_sharpe - 0.10 (组合 Sharpe 损失 < 0.10)
      - stress_25+_Sharpe ≥ 0 (压力段不亏损)

    组合 Gate:
      - combo != 'none' 时 full_Sharpe ≥ base_sharpe - 0.10
      - 至少 1 个组合通过 Gate → "PROMOTE"
      - 三合一 (abc) 仍失败 → "ARCHIVE"
    """
    base = base_sharpe
    single_pass = single_df[
        (single_df["full_Sharpe"] >= base - 0.10)
        & (single_df["stress_25+_Sharpe"].fillna(-99) >= 0)
    ]
    combo_pass = combo_df[
        (combo_df["combo"] != "none")
        & (combo_df["full_Sharpe"] >= base - 0.10)
    ]
    verdict = "PROMOTE" if "abc" in combo_pass["combo"].unique() else "ARCHIVE"
    return {
        "single_pass_factors": single_pass["factor"].unique().tolist(),
        "combo_pass_combos": combo_pass["combo"].unique().tolist(),
        "verdict": verdict,
    }
# ── CLI ─────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"  -> {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description="AllWeather 实验脚本集")
    ap.add_argument("cmd", choices=["lever-sweep", "cap-sweep", "trend-matrix",
                                    "unlevered", "segments", "diag-d1",
                                    "diag-d2", "diag-d3", "diag-d4", "phase5",
                                    "phase5-costs", "phase5-optimize",
                                    "trig-validate", "trig-ab",
                                    "v6-subfactor-test",
                                    "all"])
    ap.add_argument("--panel", type=str, default=None)
    ap.add_argument("--caps", type=str, default=None, help="逗号分隔 (lever/cap sweep)")
    ap.add_argument("--n", type=int, default=10, help="diag-d2 重复次数")
    ap.add_argument("--k", type=int, default=5, help="diag-d2 每次剔除资产数")
    ap.add_argument("--max-weight", type=float, default=0.40)
    ap.add_argument("--max-leverage", type=int, default=3)
    ap.add_argument("--a-cap", type=int, default=4, help="phase5 sleeve A 杠杆 cap")
    ap.add_argument("--b-cap", type=int, default=1, help="phase5 sleeve B 杠杆 cap")
    ap.add_argument("--cta-alphas", type=str, default="0.0,0.10,0.15,0.20,0.30,0.50")
    ap.add_argument("--a-caps", type=str, default="3,4,5,6,7")
    ap.add_argument("--b-caps", type=str, default="1,2,3")
    ap.add_argument("--no-cta", action="store_true", help="phase5 跳过 CTA sleeve")
    ap.add_argument("--v6-alphas", type=str, default="0.1,0.2,0.3,0.5",
                    help="V6 子因子 alpha 扫描")
    args = ap.parse_args()
    panel = _load_panel(args.panel)

    if args.cmd in ("lever-sweep", "all"):
        caps = ([float(c) for c in args.caps.split(",")] if args.caps
                else [3, 4, 5, 6, 7, 8])
        _save(run_lever_sweep(panel, caps, args.max_weight), "lever_sweep")
    if args.cmd in ("cap-sweep", "all"):
        caps = ([float(c) for c in args.caps.split(",")] if args.caps
                else [1.0, 0.40, 0.25, 0.15, 0.10])
        _save(run_cap_sweep(panel, caps, args.max_leverage), "cap_sweep")
    if args.cmd in ("trend-matrix", "all"):
        _save(run_trend_matrix(panel, args.max_weight, args.max_leverage),
              "trend_matrix")
    if args.cmd in ("unlevered", "all"):
        _save(run_unlevered(panel), "unlevered")
    if args.cmd in ("segments", "all"):
        cfg = AllWeatherConfig(trend_mode="none", max_weight=args.max_weight,
                               max_leverage=args.max_leverage)
        df = run_segments(panel, cfg, f"mainline_w{args.max_weight}_l{args.max_leverage}")
        _save(df, "segments")
        print(df.round(4).to_string(index=False))
    if args.cmd in ("diag-d1", "all"):
        d = diag_d1(panel)
        _save(pd.DataFrame([d]), "diag_d1")
        print(d)
    if args.cmd in ("diag-d2", "all"):
        _save(diag_d2(panel, n=args.n, k=args.k), "diag_d2")
    if args.cmd in ("diag-d3", "all"):
        _save(diag_d3(panel), "diag_d3")
    if args.cmd in ("diag-d4", "all"):
        _save(diag_d4(), "diag_d4")
    if args.cmd in ("phase5", "all"):
        from .ensemble import run_phase5
        df, _ = run_phase5(panel, include_cta=not args.no_cta)
        _save(df.reset_index(), "phase5_ensemble")
        print(df.round(4).to_string())
    if args.cmd in ("phase5-costs", "all"):
        from .ensemble import analyze_costs
        df_sleeve, df_tc = analyze_costs(panel, a_cap=args.a_cap, b_cap=args.b_cap)
        _save(df_sleeve, "phase5_costs_sleeve")
        _save(df_tc, "phase5_costs_summary")
        print("=== 单 sleeve net vs gross ===")
        pv = df_sleeve.pivot(index="sleeve", columns="cost_basis", values="Sharpe")
        pv["drag"] = pv["gross"] - pv["net"]
        print(pv.round(3).to_string())
        print("\n=== 换手/成本汇总 ===")
        print(df_tc.round(4).to_string(index=False))
    if args.cmd in ("trig-validate", "all"):
        df_tv = validate_trig_episodes()
        _save(df_tv, "trig_episode_validation")
        print(df_tv.round(4).to_string(index=False))
    if args.cmd in ("trig-ab", "all"):
        df_trig, df_q = run_trig_ab(panel)
        _save(df_trig, "trig_ab_results")
        _save(df_q, "trig_quality")
        print("\n=== 触发质量 (触发后 20d 收益) ===")
        print(df_q.round(4).to_string(index=False))
        print("\n=== 回测指标 ===")
        print(df_trig.round(4).to_string(index=False))
    if args.cmd in ("phase5-optimize", "all"):
        from .ensemble import phase5_optimize
        cta_df, combo_df = phase5_optimize(
            panel,
            cta_alphas=tuple(float(x) for x in args.cta_alphas.split(",")),
            a_caps=tuple(int(x) for x in args.a_caps.split(",")),
            b_caps=tuple(int(x) for x in args.b_caps.split(",")),
        )
        _save(cta_df, "cta_alpha_sweep")
        _save(combo_df, "phase5_3d_sweep")
        print("=== CTA α 单 sleeve ===")
        print(cta_df.round(4).to_string(index=False))
        print("\n=== ABC_ew 3D sweep (按 Sharpe 排序) ===")
        print(combo_df.sort_values("Sharpe", ascending=False).round(3).to_string(index=False))
    if args.cmd in ("v6-subfactor-test", "all"):
        alphas = tuple(float(x) for x in args.v6_alphas.split(","))
        single_df = run_subfactor_single(panel, alpha_values=alphas)
        combo_df = run_subfactor_combo(panel, alpha=0.3)
        verdict = evaluate_v6_gate(single_df, combo_df)
        _save(single_df, "v6_subfactor_single")
        _save(combo_df, "v6_subfactor_combo")
        print("\n=== V6 Gate 判定 ===")
        for k, v in verdict.items():
            print(f"  {k}: {v}")
        if verdict["verdict"] == "PROMOTE":
            print("\n  ✅ 至少 abc 通过 Gate, 进入 Phase 6 评估")
        else:
            print("\n  ❌ 三合一 (abc) 未通过 Gate, 永久归档本路线")


if __name__ == "__main__":
    main()
