# coding=utf-8
"""主力连续数据扩展加载器: 拼接 export 2026 库的原始合约数据.

背景:
    现有 DataCache (futures_options_daily.duckdb) 的 continuous_main 表
    仅更新至 2026-07-21; export/futures_market_2026.duckdb 含 2026 年
    原始合约日线 (2026-01-05 ~ 2026-09-04).

方案:
    主力连续 = 现有 continuous_main (至 2026-07-21)
             + export 原始表 2026-07-22 起按 OI 最大合约拼接

    主力选择规则与现有管道一致 (OI 最大; 并列时取成交量更大).

用法:
    from QuantNodes.strategy.cta.data_loader_ext import load_continuous_main_ext
    df = load_continuous_main_ext('AU', start='2020-01-01', end='2026-09-04')
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .data_loader import CtaDataLoader

EXPORT_2026_DUCKDB = (
    "/home/ll/Public/QuantNodes/export/futures_market_2026.duckdb"
)

# 品种 → export 库表名 (exchange_product_futures_daily)
_PRODUCT_TABLE_2026: dict[str, str] = {
    "IF": "cffex_if_futures_daily", "IC": "cffex_ic_futures_daily",
    "IH": "cffex_ih_futures_daily", "T": "cffex_t_futures_daily",
    "TF": "cffex_tf_futures_daily", "TS": "cffex_ts_futures_daily",
    "AU": "shfe_au_futures_daily", "AG": "shfe_ag_futures_daily",
    "CU": "shfe_cu_futures_daily", "AL": "shfe_al_futures_daily",
    "ZN": "shfe_zn_futures_daily", "NI": "shfe_ni_futures_daily",
    "RB": "shfe_rb_futures_daily", "I": "dce_i_futures_daily",
    "SC": "ine_sc_futures_daily", "MA": "czce_ma_futures_daily",
    "TA": "czce_ta_futures_daily", "M": "dce_m_futures_daily",
    "Y": "dce_y_futures_daily", "P": "dce_p_futures_daily",
    "SR": "czce_sr_futures_daily",
    # Phase A2 扩展品种
    "pb": "shfe_pb_futures_daily", "sn": "shfe_sn_futures_daily",
    "hc": "shfe_hc_futures_daily", "sf": "czce_sf_futures_daily",
    "sm": "czce_sm_futures_daily", "j": "dce_j_futures_daily",
    "jm": "dce_jm_futures_daily",
    "bu": "shfe_bu_futures_daily", "eg": "dce_eg_futures_daily",
    "eb": "dce_eb_futures_daily", "v": "dce_v_futures_daily",
    "l": "dce_l_futures_daily", "pp": "dce_pp_futures_daily",
    "cf": "czce_cf_futures_daily", "rm": "czce_rm_futures_daily",
    "oi": "czce_oi_futures_daily",
}


def load_continuous_main_ext(
    product: str,
    start: str = "2020-01-01",
    end: str = "2026-09-04",
    loader: CtaDataLoader | None = None,
) -> pd.DataFrame | None:
    """加载主力连续 (现有缓存 + export 2026 原始表拼接).

    Args:
        product: 品种代码 (需在 _PRODUCT_TABLE_2026 中)
        start / end: 日期区间
        loader: CtaDataLoader (默认新建)

    Returns:
        DataFrame, 列 [trade_date, open, high, low, close, volume,
        open_interest], 索引或列与 load_continuous_main 一致.
        export 库无该品种或无新数据时返回现有 continuous_main (可能截断).
    """
    loader = loader or CtaDataLoader()
    base = loader.load_continuous_main(product, start=start, end=end)
    if base is None or base.empty:
        return base

    table = _PRODUCT_TABLE_2026.get(product)
    if table is None:
        return base

    # 现有主力的最后日期
    base_dates = pd.to_datetime(base["trade_date"])
    last_date = base_dates.max()

    con = duckdb.connect(EXPORT_2026_DUCKDB, read_only=True)
    try:
        # 2026 段: 每交易日选 OI 最大合约 (并列取成交量)
        tail = con.execute(
            f"""
            SELECT trade_date, symbol, open, high, low, close,
                   volume, open_interest
            FROM (
                SELECT trade_date, symbol, open, high, low, close,
                       volume, open_interest,
                       ROW_NUMBER() OVER (
                           PARTITION BY trade_date
                           ORDER BY open_interest DESC, volume DESC
                       ) AS rn
                FROM {table}
                WHERE trade_date > ?
                  AND open_interest IS NOT NULL
                  AND open_interest > 0
            )
            WHERE rn = 1
            ORDER BY trade_date
            """,
            [str(last_date.date())],
        ).fetchall()
    finally:
        con.close()

    if not tail:
        return base

    tail_df = pd.DataFrame(
        tail,
        columns=["trade_date", "symbol", "open", "high", "low",
                 "close", "volume", "open_interest"],
    )
    # base 中只保留 <= last_date (理论上 base 已满足)
    base_keep = base[pd.to_datetime(base["trade_date"]) <= last_date].copy()
    tail_df = tail_df.copy()
    # 统一日期类型
    base_keep["trade_date"] = pd.to_datetime(base_keep["trade_date"])
    tail_df["trade_date"] = pd.to_datetime(tail_df["trade_date"])
    merged = pd.concat([base_keep, tail_df], ignore_index=True)
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    # 过滤 end
    merged = merged[pd.to_datetime(merged["trade_date"]) <= pd.Timestamp(end)]
    return merged
