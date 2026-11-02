# coding: utf-8
"""E1 启用评估 (复用 cta_engine.run() 完整 NAV).

复用 FactorModuleComboStrategy.run() (含 cta_engine 完整功能: 手续费/换月/资金分配/
position_ema/leverage_hard_cap/dd_filter) 而非简化 NAV 算法.

对比主 CTA baseline (ZK off) vs ZK vol_adj (ZK on).
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.data_loader import CtaDataLoader  # noqa: E402
from QuantNodes.strategy.cta.strategies.base import ZK_VOL_ADJ_CFG  # noqa: E402
from QuantNodes.strategy.cta.strategies.factor_modules import (  # noqa: E402
    FactorModuleComboStrategy,
)

OUT_DIR = Path("/home/ll/Public/QuantNodes/reports/cta/zookeeper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = logging.getLogger("zookeeper.e1_v2")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _run(enabled: bool, start: str = "2019-01-01", end: str = "2025-07-21",
         loader: CtaDataLoader | None = None) -> dict:
    """跑一次主 CTA 回测 (含 cta_engine.run() 全功能), 返回 NAV/positions/metrics."""
    ZK_VOL_ADJ_CFG["enabled"] = enabled
    if loader is None:
        loader = CtaDataLoader()
    strat = FactorModuleComboStrategy(data_loader=loader, start=start, end=end)
    result = strat.run()  # CtaBacktestResult 含完整 cta_engine 跑出来的 NAV
    nav = result.nav
    # positions 是 dict[product, list[float]] per date
    pos_df = pd.DataFrame(result.positions, index=pd.DatetimeIndex(result.dates))
    return {"nav": nav, "positions": pos_df, "trades": result.trades,
            "metrics": result.metrics}


def _compute_extra(nav: pd.Series, positions: pd.DataFrame,
                   loader: CtaDataLoader) -> dict:
    """基于 cta_engine 输出计算额外指标: 杠杆率, 仓位总规模.

    杠杆 = Σ|手数 × close × contract_multiplier| / equity (元)
    """
    if nav.empty or positions.empty:
        return {"leverage_avg": np.nan, "leverage_max": np.nan,
                "total_lots_avg": np.nan}
    common_idx = positions.index.intersection(nav.index)
    if len(common_idx) == 0:
        return {"leverage_avg": np.nan, "leverage_max": np.nan,
                "total_lots_avg": np.nan}
    pos = positions.loc[common_idx]
    nav_a = nav.reindex(common_idx)

    # 合约乘数
    try:
        con = loader._connect()
        try:
            spec_rows = con.execute(
                "SELECT product, contract_multiplier FROM contract_specs"
            ).fetchall()
            mult_map = {p: m for p, m in spec_rows}
        finally:
            con.close()
    except Exception as e:
        LOG.warning(f"contract_multiplier 获取失败: {e}")
        mult_map = {}

    # 价格: 用 smoothed continuous (策略内部用同源) - 但平滑价有轻微前视偏差
    # 改用 continuous_main (逐步换月, 无前视) 计算杠杆
    prices_dict = {}
    for p in pos.columns:
        try:
            df = loader.load_continuous_main(p, start=str(common_idx[0].date()),
                                              end=str(common_idx[-1].date()))
            if df is None or df.empty:
                continue
            s = pd.Series(df["close"].values,
                          index=pd.DatetimeIndex(pd.to_datetime(df["trade_date"]).date))
            prices_dict[p] = s.reindex(common_idx.date).ffill().values
        except Exception:
            continue

    if not prices_dict:
        return {"leverage_avg": np.nan, "leverage_max": np.nan,
                "total_lots_avg": np.nan}

    prices_df = pd.DataFrame(prices_dict, index=common_idx)
    # 名义敞口 (元)
    notional = pd.DataFrame(0.0, index=common_idx, columns=pos.columns)
    for p in pos.columns:
        if p in prices_df.columns and p in mult_map:
            notional[p] = pos[p].abs() * prices_df[p].fillna(method="ffill") * mult_map[p]
    total_notional = notional.sum(axis=1)
    # 杠杆率
    leverage = total_notional / nav_a.replace(0, np.nan)
    return {
        "leverage_avg": float(leverage.mean()),
        "leverage_max": float(leverage.max()),
        "total_lots_avg": float(pos.abs().sum(axis=1).mean()),
        "notional_avg_yuan": float(total_notional.mean()),
    }


def main() -> int:
    # FINAL_LUMP_CFG 沿用 cta_final_baseline.py (保证口径与 production 一致)
    from dataclasses import fields
    from QuantNodes.strategy.cta.strategies.combo import ComboConfig

    FINAL_LUMP_CFG = dict(
        aggregation="tvpr", module_weighting="inv_vol",
        continuous=True, cont_z_window=252, cont_z_clip=3.0,
        momentum_filter_window=60, momentum_filter_coef=0.0,
        target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
        margin_call=True, position_ema_alpha=0.15,
        roll_cost_bps=10.0, roll_model="lump",
    )
    # 加载 calibrated 成本 (per-product commission/slippage)
    cost_model_path = Path("/home/ll/Public/QuantNodes/reports/cta/cost_model.json")
    cfg_kw = dict(FINAL_LUMP_CFG)
    if cost_model_path.exists():
        table = json.loads(cost_model_path.read_text())
        cfg_kw["commission_rate"] = {
            p: v["commission_bps"] / 10000 for p, v in table.items()
        }
        cfg_kw["slippage_bps"] = {p: v["slippage_bps"] for p, v in table.items()}
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})

    # Baseline
    LOG.info("=== Baseline (FINAL_LUMP_CFG, ZK disabled, cta_engine.run) ===")
    ZK_VOL_ADJ_CFG["enabled"] = False
    loader = CtaDataLoader()
    strat = FactorModuleComboStrategy(data_loader=loader, config=cfg,
                                       start="2019-01-01", end="2025-07-21")
    result = strat.run()
    m_base = result.metrics
    nav_base = result.nav
    pos_base = pd.DataFrame(result.positions, index=pd.DatetimeIndex(result.dates))
    extra_base = _compute_extra(nav_base, pos_base, loader)
    LOG.info(f"  Sharpe={m_base.get('Sharpe', np.nan):.3f} "
             f"Calmar={m_base.get('Calmar', np.nan):.2f} "
             f"DD={m_base.get('MaxDD', np.nan):.2%} "
             f"AnnRet={m_base.get('AnnRet', np.nan):.2%} "
             f"Lev(avg/max)={extra_base['leverage_avg']:.2f}/{extra_base['leverage_max']:.2f}")

    # ZK vol_adj (同样的 FINAL_LUMP_CFG)
    LOG.info("=== ZK vol_adj enabled (FINAL_LUMP_CFG, cta_engine.run) ===")
    ZK_VOL_ADJ_CFG["enabled"] = True
    strat2 = FactorModuleComboStrategy(data_loader=loader, config=cfg,
                                        start="2019-01-01", end="2025-07-21")
    result2 = strat2.run()
    m_zk = result2.metrics
    nav_zk = result2.nav
    pos_zk = pd.DataFrame(result2.positions, index=pd.DatetimeIndex(result2.dates))
    extra_zk = _compute_extra(nav_zk, pos_zk, loader)
    LOG.info(f"  Sharpe={m_zk.get('Sharpe', np.nan):.3f} "
             f"Calmar={m_zk.get('Calmar', np.nan):.2f} "
             f"DD={m_zk.get('MaxDD', np.nan):.2%} "
             f"AnnRet={m_zk.get('AnnRet', np.nan):.2%} "
             f"Lev(avg/max)={extra_zk['leverage_avg']:.2f}/{extra_zk['leverage_max']:.2f}")

    # 增量
    d_sharpe = m_zk.get('Sharpe', 0) - m_base.get('Sharpe', 0)
    d_calmar = m_zk.get('Calmar', 0) - m_base.get('Calmar', 0)
    d_dd = m_zk.get('MaxDD', 0) - m_base.get('MaxDD', 0)
    d_ann = m_zk.get('AnnRet', 0) - m_base.get('AnnRet', 0)
    d_lev = extra_zk['leverage_avg'] - extra_base['leverage_avg']
    LOG.info(f"=== 增量: ΔSharpe={d_sharpe:+.3f} ΔCalmar={d_calmar:+.2f} "
             f"ΔDD={d_dd:+.2%} ΔAnnRet={d_ann:+.2%} ΔLev={d_lev:+.2f} ===")

    # 导出
    out = pd.DataFrame({
        "version": ["Baseline (FINAL_LUMP, ZK off)",
                     "ZK vol_adj (FINAL_LUMP, on)"],
        "annual_return": [m_base.get("AnnRet", np.nan),
                          m_zk.get("AnnRet", np.nan)],
        "annual_vol": [m_base.get("Vol", np.nan),
                        m_zk.get("Vol", np.nan)],
        "sharpe": [m_base.get("Sharpe", np.nan),
                   m_zk.get("Sharpe", np.nan)],
        "max_dd": [m_base.get("MaxDD", np.nan),
                   m_zk.get("MaxDD", np.nan)],
        "calmar": [m_base.get("Calmar", np.nan),
                   m_zk.get("Calmar", np.nan)],
        "leverage_avg": [extra_base["leverage_avg"],
                         extra_zk["leverage_avg"]],
        "leverage_max": [extra_base["leverage_max"],
                         extra_zk["leverage_max"]],
        "total_lots_avg": [extra_base["total_lots_avg"],
                           extra_zk["total_lots_avg"]],
    })
    out.to_csv(OUT_DIR / "e1_main_cta_v2_evaluation.csv", index=False)
    nav_base.to_csv(OUT_DIR / "e1_v2_nav_baseline.csv", header=["nav"])
    nav_zk.to_csv(OUT_DIR / "e1_v2_nav_zk.csv", header=["nav"])
    LOG.info(f"输出: {OUT_DIR}/e1_*_v2.csv (3 文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
