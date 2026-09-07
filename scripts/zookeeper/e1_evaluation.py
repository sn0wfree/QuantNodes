# coding: utf-8
"""E1 启用评估: 直接跑 v11 FactorModuleComboStrategy (ZK_VOL_ADJ_CFG.enabled=True).

对比主 CTA baseline (无Z K) vs ZK vol_adj 版.

⚠️ 临时评估脚本, 不修改 v11 引擎本身. 改 ZK_VOL_ADJ_CFG.enabled 切换.
产出: reports/cta/zookeeper/e1_main_cta_evaluation.csv (含baseline + ZK 两个版本)
"""
from __future__ import annotations

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
LOG = __import__("logging").getLogger("zookeeper.e1_eval")


def _run_strategy(enabled: bool, start: str = "2019-01-01",
                  end: str = "2025-07-21") -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """跑主 CTA 策略, 返回 (日频 NAV, 日频 position 矩阵, 名义敞口矩阵).

    算法 (简化版, 仅供 vol_target 调节对比):
      pnl_{t,p} = position_{t,p} × notional_per_lot_{t,p} × price_return_{t,p}
      daily_pnl_t = Σ_p pnl_{t,p}
      NAV_t = initial_cash + cumulative_daily_pnl
    """
    ZK_VOL_ADJ_CFG["enabled"] = enabled
    loader = CtaDataLoader()
    strat = FactorModuleComboStrategy(data_loader=loader, start=start, end=end)
    strat.load_panel()
    positions = strat._positions()
    prices = strat._price_panel
    cfg = strat._config
    initial_cash = cfg.initial_cash  # 2_000_000

    # notional per lot (元) per product
    notional_per_lot = pd.DataFrame(index=prices.index, columns=prices.columns)
    for p in positions.columns:
        spec = strat._registry.get_or_default(p)
        notional_per_lot[p] = prices[p] * spec.contract_multiplier

    # 各品种日收益 (%)
    rets = prices.pct_change(fill_method=None)
    # 各品种日盈亏 (元) = 手数 × 名义 × 收益率
    pnl_per_product = positions * notional_per_lot * rets
    daily_pnl = pnl_per_product.sum(axis=1).fillna(0)
    # NAV = 期初资金 + 累计盈亏 (下限 0.01 防止除零)
    nav = (initial_cash + daily_pnl.cumsum()).clip(lower=initial_cash * 0.01)
    # 名义敞口 = Σ|手数 × 名义/手| (单位元)
    notional = (positions.abs() * notional_per_lot).fillna(0)
    return nav, positions, notional


def _metrics(nav: pd.Series, notional: pd.DataFrame | None = None) -> dict:
    r = nav.pct_change(fill_method=None).dropna()
    if r.empty:
        return {"annual": np.nan, "sharpe": np.nan, "dd": np.nan,
                "vol": np.nan, "calmar": np.nan, "n_days": 0,
                "leverage_avg": np.nan, "leverage_max": np.nan}
    # 移除极端异常值 (>1σ 4倍) 避免极值污染 Sharpe/Vol
    r_clean = r.where(r.abs() < r.abs().mean() * 10)
    r_clean = r_clean.dropna()
    if len(r_clean) < 100:
        r_clean = r  # 极端清洗后样本不足, 用原始
    ann = r_clean.mean() * 252
    shp = r_clean.mean() / r_clean.std() * np.sqrt(252) if r_clean.std() > 0 else np.nan
    vol = r_clean.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    calmar = -ann / dd if dd < 0 and ann > 0 else np.nan
    # 杠杆率 = 名义敞口 / NAV
    if notional is not None and not notional.empty:
        total_notional = notional.sum(axis=1)
        leverage = total_notional / nav.replace(0, np.nan)
        lev_avg = leverage.mean()
        lev_max = leverage.max()
    else:
        lev_avg = lev_max = np.nan
    return {"annual": float(ann), "sharpe": float(shp), "dd": float(dd),
            "vol": float(vol), "calmar": float(calmar), "n_days": int(len(r_clean)),
            "leverage_avg": float(lev_avg), "leverage_max": float(lev_max)}


def main() -> int:
    # Baseline (无 ZK)
    LOG.info("=== Baseline (ZK disabled) ===")
    nav_base, pos_base, notional_base = _run_strategy(enabled=False)
    m_base = _metrics(nav_base, notional_base)
    LOG.info(f"  Baseline: Sharpe={m_base['sharpe']:.3f} DD={m_base['dd']:.2%} "
             f"Annual={m_base['annual']:.2%} Calmar={m_base['calmar']:.2f} "
             f"Lev(avg/max)={m_base['leverage_avg']:.2f}/{m_base['leverage_max']:.2f}")

    # ZK vol_adj 启用
    LOG.info("=== ZK vol_adj enabled ===")
    nav_zk, pos_zk, notional_zk = _run_strategy(enabled=True)
    m_zk = _metrics(nav_zk, notional_zk)
    LOG.info(f"  ZK vol_adj: Sharpe={m_zk['sharpe']:.3f} DD={m_zk['dd']:.2%} "
             f"Annual={m_zk['annual']:.2%} Calmar={m_zk['calmar']:.2f} "
             f"Lev(avg/max)={m_zk['leverage_avg']:.2f}/{m_zk['leverage_max']:.2f}")

    # 增量
    delta_sharpe = m_zk["sharpe"] - m_base["sharpe"]
    delta_dd = m_zk["dd"] - m_base["dd"]
    delta_annual = m_zk["annual"] - m_base["annual"]
    delta_lev = m_zk["leverage_avg"] - m_base["leverage_avg"]
    LOG.info(f"=== 增量: ΔSharpe={delta_sharpe:+.3f} ΔDD={delta_dd:+.2%} "
             f"ΔAnnual={delta_annual:+.2%} ΔLev(avg)={delta_lev:+.2f} ===")

    # 仓位总规模对比 (验证 ZK 确实调节了仓位)
    total_pos_base = pos_base.abs().sum(axis=1).mean()
    total_pos_zk = pos_zk.abs().sum(axis=1).mean()
    LOG.info(f"=== 仓位总规模: Baseline={total_pos_base:.1f}手, "
             f"ZK={total_pos_zk:.1f}手, Δ={total_pos_zk-total_pos_base:+.1f}手 "
             f"({(total_pos_zk-total_pos_base)/total_pos_base:+.1%}) ===")

    # 导出
    out = pd.DataFrame({
        "version": ["Baseline (ZK off)", "ZK vol_adj (on)"],
        "annual_return": [m_base["annual"], m_zk["annual"]],
        "annual_vol": [m_base["vol"], m_zk["vol"]],
        "sharpe": [m_base["sharpe"], m_zk["sharpe"]],
        "max_dd": [m_base["dd"], m_zk["dd"]],
        "calmar": [m_base["calmar"], m_zk["calmar"]],
        "n_days": [m_base["n_days"], m_zk["n_days"]],
        "leverage_avg": [m_base["leverage_avg"], m_zk["leverage_avg"]],
        "leverage_max": [m_base["leverage_max"], m_zk["leverage_max"]],
        "total_lots_avg": [total_pos_base, total_pos_zk],
    })
    out.to_csv(OUT_DIR / "e1_main_cta_evaluation.csv", index=False)
    nav_base.to_csv(OUT_DIR / "e1_nav_baseline.csv", header=["nav"])
    nav_zk.to_csv(OUT_DIR / "e1_nav_zk.csv", header=["nav"])
    LOG.info(f"输出: {OUT_DIR}/e1_*.csv (3 文件)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
