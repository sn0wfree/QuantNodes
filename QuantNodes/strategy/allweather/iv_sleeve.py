#!/usr/bin/env python3
# coding=utf-8
"""IVTS (波动率期限结构斜率) 多空 sleeve (§5.53).

信号 = iv_far − iv_near (同品种不同到期 ATM 隐含波动率之差).
经济含义: 波动率期限结构斜率 (陡=远端 IV 高). 与 carry 相关 0.15,
与 D 组合相关 0.005 (正交), 与 carry 合成后 OOS Sharpe +0.19.

数据: scripts/extract_iv_surface.py → data/allweather/iv_surface/<col>.parquet
      (每产品含 iv_near / iv_far / iv_put / iv_call 列).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from QuantNodes.strategy.allweather.carry_sleeve import build_factor_sleeve

DEFAULT_DIR = "data/allweather/iv_surface"


def load_ivts(surface_dir: str | Path = DEFAULT_DIR) -> pd.DataFrame:
    """加载各产品 IV 曲面 → IVTS = iv_far − iv_near (产品 × 日期)."""
    surface_dir = Path(surface_dir)
    near, far = {}, {}
    for cp in sorted(surface_dir.glob("fut_*.parquet")):
        s = pd.read_parquet(cp)
        s.index = pd.to_datetime(s.index)
        if "iv_near" in s.columns and "iv_far" in s.columns:
            near[cp.stem] = s["iv_near"]
            far[cp.stem] = s["iv_far"]
    if not near:
        return pd.DataFrame()
    return pd.DataFrame(far).sort_index() - pd.DataFrame(near).sort_index()


def build_ivts_sleeve(panel: pd.DataFrame, ivts: pd.DataFrame, **kw
                      ) -> tuple[pd.Series, pd.DataFrame]:
    """IVTS 多空 sleeve (见 carry_sleeve.build_factor_sleeve)."""
    kw.setdefault("smooth", 1)
    port, w = build_factor_sleeve(panel, ivts, **kw)
    port.name = "ivts_sleeve"
    return port, w
