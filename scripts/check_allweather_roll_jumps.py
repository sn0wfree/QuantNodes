# coding=utf-8
"""AllWeather Step 0 — 期货主连 roll jump 检查 (PLAN §十二 Step 0 / D4).

方法:
    主力合约映射表定位换月日 (symbol 变化日) →
    对比 continuous_main 收益率在 换月日 vs 非换月日 的分布 →
    大跳变日是否集中在换月日 → 判定 roll 污染程度。

判定规则 (PLAN §十二 Step 0):
    - |ret| > 3% 的日子里 roll 日占比高 → 污染显著 → data_loader 需加 roll 处理
    - roll 日收益分布与非 roll 日接近 → 低污染, 直接用 panel_v1
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.config import DUCKDB_PATH, FUTURES_PRODUCTS

JUMP_THRESHOLD = 0.03  # |单日收益| > 3% 视为跳变候选


def check_product(con: duckdb.DuckDBPyConnection, product: str) -> dict:
    """单品种 roll jump 统计."""
    mapping = con.execute(
        f"SELECT trade_date, symbol FROM main_contract_mapping "
        f"WHERE product = '{product}' ORDER BY trade_date"
    ).df()
    if mapping.empty:
        return {"product": product, "error": "no mapping"}

    mapping["trade_date"] = pd.to_datetime(mapping["trade_date"])
    roll_dates = set(
        mapping.loc[mapping["symbol"] != mapping["symbol"].shift(1), "trade_date"]
    ) - {mapping["trade_date"].iloc[0]}

    cont = con.execute(
        f"SELECT trade_date, close FROM continuous_main_{product.lower()}_daily ORDER BY trade_date"
    ).df()
    cont["trade_date"] = pd.to_datetime(cont["trade_date"])
    cont = cont.set_index("trade_date")["close"]
    ret = cont.pct_change().dropna()

    is_roll = pd.Series([d in roll_dates for d in ret.index], index=ret.index)
    jump = ret.abs() > JUMP_THRESHOLD

    roll_rets = ret[is_roll]
    non_roll_rets = ret[~is_roll]

    n_roll_jumps = int((jump & is_roll).sum())
    n_jumps = int(jump.sum())
    return {
        "product": product,
        "n_days": len(ret),
        "n_rolls": len(roll_dates),
        "roll_ret_std": float(roll_rets.std()) if len(roll_rets) else np.nan,
        "nonroll_ret_std": float(non_roll_rets.std()) if len(non_roll_rets) else np.nan,
        "n_jump_gt3pct": n_jumps,
        "n_jump_on_roll": n_roll_jumps,
        "jump_on_roll_pct": (n_roll_jumps / n_jumps) if n_jumps else 0.0,
        "max_abs_ret": float(ret.abs().max()),
        "max_abs_ret_is_roll": bool(ret.abs().idxmax() in roll_dates),
    }


def main() -> pd.DataFrame:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        rows = [check_product(con, p) for p in FUTURES_PRODUCTS]
    finally:
        con.close()
    df = pd.DataFrame(rows).set_index("product")
    df = df.drop(columns=["error"], errors="ignore")

    print("=" * 90)
    print("Step 0 / D4: roll jump 检查 (|ret|>3% 为跳变候选)")
    print("=" * 90)
    print(df.round(4).to_string())
    print()

    total_jumps = df["n_jump_gt3pct"].sum()
    total_roll_jumps = df["n_jump_on_roll"].sum()
    overall_pct = total_roll_jumps / total_jumps if total_jumps else 0.0
    print(
        f"全池: 跳变日 {total_jumps} 个, "
        f"其中落在换月日 {total_roll_jumps} 个 ({overall_pct:.1%})"
    )
    print()

    # 判定
    contaminated = df[df["jump_on_roll_pct"] > 0.5]
    if overall_pct > 0.3 or len(contaminated) > 5:
        print("判定: roll 污染显著 → data_loader 需加 roll 处理 (双价拼接或换月日剔除)")
    else:
        print("判定: roll 污染低 → 直接使用 panel_v1")
    if len(contaminated):
        print(f"污染品种 (>50% 跳变在换月日): {list(contaminated.index)}")
    return df


if __name__ == "__main__":
    main()
