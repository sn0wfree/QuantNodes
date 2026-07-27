# coding=utf-8
"""v1 组合管理: 4 步组合管理 (原始CICC复现, Stage 8).

v1 = 纯CICC报告复现, 不含 v2 增强:
  - 无 momentum_type (固定 "price")
  - 无 VolTargeting
  - 无 CostModel
  - 无 ConcentrationCaps
  - 无 CovEstimator

如需这些功能, 请使用 v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import pandas as pd

from ..common.universe import ETFPool
from .momentum_v1 import (
    rank_pctl_v1,
    realized_vol_v1,
    below_ma_v1,
    pairwise_corr_v1,
)


@dataclass
class DiversificationCaps_v1:
    """v1 分散度约束 (原始CICC默认)."""
    a_share_broad: int = 2
    a_share_sector: int = 2
    hk: int = 1
    require_commodity: bool = True
    require_overseas: bool = True
    a_share: int = 3  # A股宽基+行业合计上限

    def cap_for(self, category: str) -> int:
        if category == "a_broad":
            return self.a_share_broad
        if category == "a_sector":
            return self.a_share_sector
        if category == "hk":
            return self.hk
        return 99


@dataclass
class RotationConfig_v1:
    """v1 策略参数 (Stage 8 baseline)."""
    lookback: int = 144
    top_n: int = 10
    corr_threshold: float = 0.9
    corr_window: int = 60
    ma_window: int = 55
    rank_cutoff: float = 0.30
    diversification: DiversificationCaps_v1 = field(default_factory=DiversificationCaps_v1)
    weight_method: str = "inv_vol"  # "inv_vol" | "equal"
    vol_window: int = 21
    weight_floor: float = 1e-4
    min_history: int = 144


@dataclass
class PortfolioState_v1:
    date: pd.Timestamp
    ranked: list[str]
    chosen: list[str]
    weights: dict[str, float]
    skipped_dedup: list[str] = field(default_factory=list)
    skipped_corr: list[str] = field(default_factory=list)
    skipped_div: list[str] = field(default_factory=list)
    stopped: list[str] = field(default_factory=list)
    replaced: dict[str, str] = field(default_factory=dict)


def inverse_vol_weights_v1(
    nav_df: pd.DataFrame,
    codes: Sequence[str],
    as_of: pd.Timestamp,
    vol_window: int = 21,
    floor: float = 1e-4,
) -> dict[str, float]:
    """v1: 逆波动加权 (CICC 伪代码 21日窗口)."""
    if not codes:
        return {}
    vols = realized_vol_v1(nav_df, as_of=as_of, window=vol_window).reindex(list(codes))
    vols = vols.fillna(vols.median() if not vols.empty and vols.median() > 0 else 1.0)
    inv = 1.0 / vols
    inv[inv < floor] = 0.0
    total = inv.sum()
    if total <= 0:
        return equal_weights_v1(codes)
    return (inv / total).to_dict()


def equal_weights_v1(codes: Sequence[str]) -> dict[str, float]:
    if not codes:
        return {}
    w = 1.0 / len(codes)
    return {c: w for c in codes}


def _compute_best_per_index_v1(pool: ETFPool, blacklist: set) -> dict[str, str]:
    """v1: 同指数只保留流动性最好 (liquidity_rank 最小)."""
    best = {}
    for m in pool.members:
        if m.code in blacklist:
            continue
        cur = best.get(m.index_code)
        if cur is None or m.liquidity_rank < pool.liquidity_rank_of(cur):
            best[m.index_code] = m.code
    return best


def _maybe_inject_required_v1(
    state: PortfolioState_v1,
    ranked: list[str],
    pctl: pd.Series,
    pool: ETFPool,
    cfg: RotationConfig_v1,
) -> None:
    """v1: 必含商品和海外 (替换最弱非商品/海外)."""
    div = cfg.diversification
    chosen = state.chosen

    for category in ("commodity", "overseas"):
        required = (div.require_commodity and category == "commodity") or \
                   (div.require_overseas and category == "overseas")
        if not required:
            continue
        if any(pool.category_of(c).value == category for c in chosen):
            continue
        for code in ranked:
            if (code in pool.codes
                and pool.category_of(code).value == category
                and code not in chosen
                and code not in state.skipped_dedup
                and code not in state.skipped_div):
                # 检查 pctl (防御重复索引)
                try:
                    pctl_val = pctl.loc[code]
                except KeyError:
                    continue
                if isinstance(pctl_val, pd.Series):
                    pctl_val = pctl_val.iloc[0]
                if pd.isna(pctl_val):
                    continue
                replaced = False
                for i in range(len(chosen) - 1, -1, -1):
                    cn = chosen[i]
                    if pool.category_of(cn).value not in ("commodity", "overseas"):
                        chosen[i] = code
                        state.skipped_div.append(cn)
                        replaced = True
                        break
                if not replaced and len(chosen) < cfg.top_n:
                    chosen.append(code)
                break


def select_and_weight_v1(
    nav_df: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig_v1,
    as_of: pd.Timestamp,
    blacklist: Sequence[str] = (),
) -> PortfolioState_v1:
    """v1 单次调仓 (严格按 CICC 伪代码 5 步)."""
    pctl = rank_pctl_v1(nav_df, cfg.lookback, as_of)
    ranked = pctl.sort_values(ascending=False).index.tolist()
    blacklist_set = set(blacklist)

    state = PortfolioState_v1(date=as_of, ranked=ranked, chosen=[], weights={})

    # 1. 预去重
    best_per_index = _compute_best_per_index_v1(pool, blacklist_set)
    deduped_ranked: list[str] = []
    for code in ranked:
        if code not in pool.codes:
            continue
        if code in blacklist_set:
            state.skipped_dedup.append(code)
            continue
        try:
            pctl_val = pctl.loc[code]
        except KeyError:
            state.skipped_dedup.append(code)
            continue
        if isinstance(pctl_val, pd.Series):
            pctl_val = pctl_val.iloc[0]
        if pd.isna(pctl_val):
            state.skipped_dedup.append(code)
            continue
        idx = pool.index_of(code)
        if best_per_index.get(idx) != code:
            state.skipped_dedup.append(code)
            continue
        deduped_ranked.append(code)

    # 2. 主循环: caps → corr → append
    chosen: list[str] = []
    chosen_cat_count: dict[str, int] = {}
    for code in deduped_ranked:
        if len(chosen) >= cfg.top_n:
            break
        cat_name = pool.category_of(code).value
        if cat_name in ("a_broad", "a_sector"):
            current_a_share = (chosen_cat_count.get("a_broad", 0) +
                               chosen_cat_count.get("a_sector", 0))
            if current_a_share >= cfg.diversification.a_share:
                state.skipped_div.append(code)
                continue
        cap = cfg.diversification.cap_for(cat_name)
        if chosen_cat_count.get(cat_name, 0) >= cap:
            state.skipped_div.append(code)
            continue
        if chosen:
            corr = pairwise_corr_v1(nav_df, [code] + chosen, as_of, cfg.corr_window)
            cc = corr.loc[code, chosen]
            if isinstance(cc, pd.DataFrame):
                cc = cc.iloc[:, 0]
            if isinstance(cc, pd.Series):
                if (cc > cfg.corr_threshold).any():
                    state.skipped_corr.append(code)
                    continue
            elif cc > cfg.corr_threshold:
                state.skipped_corr.append(code)
                continue
        chosen.append(code)
        chosen_cat_count[cat_name] = chosen_cat_count.get(cat_name, 0) + 1

    state.chosen = chosen

    # 3. 必含注入
    _maybe_inject_required_v1(state, ranked, pctl, pool, cfg)

    # 4. 加权
    if cfg.weight_method == "inv_vol":
        state.weights = inverse_vol_weights_v1(
            nav_df, state.chosen, as_of,
            vol_window=cfg.vol_window, floor=cfg.weight_floor,
        )
    else:
        state.weights = equal_weights_v1(state.chosen)

    return state


def apply_stops_v1(
    nav_df: pd.DataFrame,
    pool: ETFPool,
    cfg: RotationConfig_v1,
    prev_weights: Mapping[str, float],
    as_of: pd.Timestamp,
) -> PortfolioState_v1:
    """v1: 对已有持仓做止损+补位."""
    pctl_series = rank_pctl_v1(nav_df, cfg.lookback, as_of)
    to_stop: list[str] = []
    for code, w in prev_weights.items():
        if w <= 0:
            continue
        if code not in pool.codes:
            continue
        if not below_ma_v1(nav_df, code, cfg.ma_window, as_of):
            continue
        if code not in pctl_series.index:
            continue
        pctl_val = pctl_series[code]
        if isinstance(pctl_val, pd.Series):
            pctl_val = pctl_val.iloc[0]
        if pctl_val < cfg.rank_cutoff:
            to_stop.append(code)

    prev_chosen = [c for c, w in prev_weights.items()
                   if w > 0 and c in pool.codes and c not in to_stop]
    stopped = list(to_stop)
    replaced: dict[str, str] = {}

    if not stopped:
        state = PortfolioState_v1(
            date=as_of, ranked=pctl_series.sort_values(ascending=False).index.tolist(),
            chosen=prev_chosen, weights=dict(prev_weights),
        )
        return state

    base_cats: dict[str, int] = {}
    for c in prev_chosen:
        cat = pool.category_of(c).value
        base_cats[cat] = base_cats.get(cat, 0) + 1

    state = select_and_weight_v1(nav_df, pool, cfg, as_of, blacklist=stopped)

    available = [c for c in state.chosen if c not in prev_chosen]
    for s in stopped:
        if not available:
            break
        new_code = available.pop(0)
        prev_chosen.append(new_code)
        replaced[s] = new_code

    state.chosen = prev_chosen
    if cfg.weight_method == "inv_vol":
        state.weights = inverse_vol_weights_v1(
            nav_df, prev_chosen, as_of, vol_window=cfg.vol_window, floor=cfg.weight_floor
        )
    else:
        state.weights = equal_weights_v1(prev_chosen)
    state.stopped = stopped
    state.replaced = replaced
    return state


__all__ = [
    "DiversificationCaps_v1",
    "RotationConfig_v1",
    "PortfolioState_v1",
    "inverse_vol_weights_v1",
    "equal_weights_v1",
    "select_and_weight_v1",
    "apply_stops_v1",
]
