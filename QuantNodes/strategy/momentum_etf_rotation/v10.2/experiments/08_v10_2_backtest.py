"""Real v10 + v10.2 backtest comparison.

Calls the actual v10 dynamic_weight_schemes.scheme_e_hybrid and overlays
the CA-GCP risk filter, then compares Sharpe / MaxDD / Calmar.

Loads calibrated CAGCPConfig from data/results/best_params.json.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (  # noqa: E402
    compute_nav,
    load_navs,
    scheme_e_hybrid,
)

_V102_INIT = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "__init__.py"
_spec = _ilu.spec_from_file_location("v10_2_module", _V102_INIT)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CAGCPipeline = _mod.CAGCPipeline
RiskFilterRules = _mod.RiskFilterRules
ca_gcp_risk_filter = _mod.ca_gcp_risk_filter
load_calibrated_config = _mod.load_calibrated_config

from integration import experimental_rules as _exp_rules  # noqa: E402

OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def nav_metrics(nav: pd.Series) -> dict:
    if len(nav) < 20:
        return {}
    rets = nav.pct_change().fillna(0.0)
    ann_ret = (1 + rets).prod() ** (252 / len(rets)) - 1
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar),
    }


def load_etf_returns() -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "high_freq_macro" / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)
    return df


def run_with_rules(
    test_navs: pd.DataFrame,
    weights_e: pd.DataFrame,
    intervals: dict,
    rules: RiskFilterRules,
) -> tuple[pd.Series, pd.DataFrame, list]:
    adjusted_weights_rows = []
    diag_rows = []
    history_hw = None
    for t in test_navs.index:
        if t not in weights_e.index:
            continue
        w_today = weights_e.loc[t]
        if w_today.isna().all():
            continue
        idx_t = intervals["half_width"].index.get_loc(t) if t in intervals["half_width"].index else None
        if idx_t is None:
            adjusted_weights_rows.append(w_today.values)
            diag_rows.append({"date": t, "alert_level": "green", "applied_scale": 1.0})
            continue
        intervals_t = {
            "lower": intervals["lower"].iloc[[idx_t]],
            "upper": intervals["upper"].iloc[[idx_t]],
            "half_width": intervals["half_width"].iloc[[idx_t]],
            "stress": intervals["stress"].iloc[[idx_t]],
        }
        w_adj, diag = ca_gcp_risk_filter(w_today, intervals_t, rules, today=t, history=history_hw)
        adjusted_weights_rows.append(w_adj.values)
        diag_rows.append({"date": t, **diag})
        if history_hw is None:
            history_hw = intervals["half_width"].iloc[: idx_t + 1].copy()
        else:
            history_hw = pd.concat([history_hw, intervals["half_width"].iloc[[idx_t]]])

    adjusted_weights = pd.DataFrame(
        adjusted_weights_rows,
        index=test_navs.index[: len(adjusted_weights_rows)],
        columns=test_navs.columns,
    )
    nav = compute_nav(test_navs, adjusted_weights, cost_bp=10)
    return nav, adjusted_weights, diag_rows


def main() -> None:
    navs_all = load_navs()
    print(f"Loaded v10 NAVs: {navs_all.shape}, {navs_all.index[0].date()} ~ {navs_all.index[-1].date()}")

    test_start = pd.Timestamp("2022-04-12")
    test_end = pd.Timestamp("2023-04-21")

    test_navs = navs_all.loc[test_start:test_end]
    print(f"Test window: {test_navs.shape[0]} days, {test_navs.index[0].date()} ~ {test_navs.index[-1].date()}")

    etf_returns = load_etf_returns()
    etf_returns_test = etf_returns.reindex(test_navs.index, method="ffill").fillna(0.0)
    etf_returns_train = etf_returns.loc[: test_start - pd.Timedelta(days=1)].iloc[-600:]

    cfg = load_calibrated_config()
    print(f"Using calibrated CAGCPConfig: k={cfg.k}, eta={cfg.sensitivity_eta}, tau={cfg.recency_tau}")

    pipe = CAGCPipeline(cfg)
    pipe.fit(etf_returns_train)
    calib_window = etf_returns.loc[test_start - pd.Timedelta(days=300) : test_start - pd.Timedelta(days=1)]
    print(f"Fitting CA-GCP on {etf_returns_train.shape}, predicting on test={etf_returns_test.shape}")
    intervals = pipe.predict_fast(calib_window, etf_returns_test)

    weights_e = scheme_e_hybrid(navs_all)
    weights_e = weights_e.reindex(test_navs.index)
    weights_e = weights_e.fillna(method="ffill").fillna(0.25)

    nav_v10 = compute_nav(test_navs, weights_e.reindex(test_navs.index), cost_bp=10)
    m_v10 = nav_metrics(nav_v10)

    results = [{"strategy": "v10 (scheme_e_hybrid)", "rules": "n/a", **m_v10}]

    nav_v10_2_default, _, diag_default = run_with_rules(test_navs, weights_e, intervals, RiskFilterRules())
    results.append({"strategy": "v10.2 (v10 + CA-GCP risk)", "rules": "default", **nav_metrics(nav_v10_2_default)})

    nav_v10_2_exp, _, diag_exp = run_with_rules(test_navs, weights_e, intervals, _exp_rules())
    results.append({"strategy": "v10.2 (v10 + CA-GCP risk)", "rules": "experimental", **nav_metrics(nav_v10_2_exp)})

    yellow_only_rules = RiskFilterRules(
        width_z_yellow=99.0,
        width_z_red=99.0,
        stress_yellow=99.0,
        stress_red=99.0,
    )
    nav_v10_2_none, _, _ = run_with_rules(test_navs, weights_e, intervals, yellow_only_rules)
    results.append({"strategy": "v10.2 (warning only, no scale)", "rules": "none", **nav_metrics(nav_v10_2_none)})

    metrics_df = pd.DataFrame(results)
    print("\n=== Real v10 vs v10.2 (Test: 2022-04-12 ~ 2023-04-21) ===")
    print(metrics_df.round(4).to_string(index=False))
    metrics_df.to_csv(OUT_DIR / "v10_2_real_comparison.csv", index=False)

    nav_v10.to_csv(OUT_DIR / "nav_v10_real.csv", index=True)
    nav_v10_2_default.to_csv(OUT_DIR / "nav_v10_2_real_default.csv", index=True)
    nav_v10_2_exp.to_csv(OUT_DIR / "nav_v10_2_real_experimental.csv", index=True)

    if diag_exp:
        pd.DataFrame(diag_exp).to_csv(OUT_DIR / "v10_2_real_diagnostics.csv", index=False)
        fired = [r["date"] for r in diag_exp if r["alert_level"] != "green"]
        print(f"\nExperimental rules: alerts fired on {len(fired)} days")
        for d in fired[:5]:
            print(f"  {pd.Timestamp(d).date()}")


if __name__ == "__main__":
    main()