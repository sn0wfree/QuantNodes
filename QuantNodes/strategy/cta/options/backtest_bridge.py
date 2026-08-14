# coding=utf-8
"""回测桥接 — 从 期权工作平台/FastRunBackTest.py 移植.

策略权重 → 交易记录 → P&L 计算
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .match_engine import DequeMatcher


def status_to_transactions(
    dd: pd.DataFrame,
    commission_rate: float = 0.000432,
) -> pd.DataFrame:
    """策略权重 DataFrame → 交易记录格式.

    Args:
        dd: 含 dt/contract_code/fee/weight 列
        commission_rate: 手续费率

    Returns:
        交易记录 DataFrame (broker 格式)
    """
    col_rename = {"dt": "报单日期", "contract_code": "委托合约", "fee": "成交均价"}
    dd = dd.rename(columns=col_rename)
    dd["买卖"] = ((dd["weight"] > 0) * 1).replace(1, "买").replace(0, "卖")
    dd["开平"] = "开"
    dd["手数"] = np.int64(dd["weight"].abs())
    dd["挂单状态"] = "全部成交"
    dd["报单时间"] = dd["报单日期"]
    dd["详细状态"] = "全部成交"
    dd["盈亏"] = 0
    dd["手续费"] = dd["手数"] * dd["成交均价"] * 100 * commission_rate
    dd["成交号"] = "hedge1"
    dd["未成交"] = 0
    dd["remarks"] = 0
    dd["报单日期"] = pd.to_datetime(dd["报单日期"]).dt.strftime("%Y-%m-%d")

    target_cols = [
        "报单日期", "委托合约", "手数", "未成交", "买卖", "开平",
        "成交均价", "挂单状态", "报单时间", "详细状态", "盈亏", "手续费", "成交号",
    ]
    return dd[target_cols]


def strategy_result_to_transactions(
    strategy_result: pd.DataFrame,
    quote: pd.DataFrame,
    commission_rate: float = 0.000432,
) -> pd.DataFrame:
    """策略权重 → 开仓+平仓交易记录.

    Args:
        strategy_result: 含 dt/contract_code/fee/weight 列
        quote: 行情数据 (含下一日价格)
        commission_rate: 手续费率

    Returns:
        完整交易记录
    """
    if strategy_result.empty:
        return pd.DataFrame()

    # 开仓记录
    open_records = status_to_transactions(strategy_result, commission_rate)

    # 平仓记录: 下一个交易日以收盘价平仓
    close_records = _create_close_records(strategy_result, quote, commission_rate)

    return pd.concat([open_records, close_records], ignore_index=True)


def _create_close_records(
    strategy_result: pd.DataFrame,
    quote: pd.DataFrame,
    commission_rate: float,
) -> pd.DataFrame:
    """创建平仓记录."""
    records = []
    for dt, sub in strategy_result.groupby("dt"):
        # 找下一个交易日
        next_dates = quote.index[quote.index > dt]
        if next_dates.empty:
            continue
        next_dt = next_dates[0]

        for _, row in sub.iterrows():
            code = row["contract_code"]
            if code in quote.columns:
                close_price = quote.loc[next_dt, code]
                if pd.isna(close_price):
                    continue
            else:
                continue

            direction = "卖" if row["weight"] > 0 else "买"
            records.append({
                "报单日期": str(next_dt.date()) if hasattr(next_dt, "date") else str(next_dt),
                "委托合约": code,
                "手数": int(abs(row["weight"])),
                "未成交": 0,
                "买卖": direction,
                "开平": "平",
                "成交均价": float(close_price),
                "挂单状态": "全部成交",
                "报单时间": str(next_dt.date()) if hasattr(next_dt, "date") else str(next_dt),
                "详细状态": "全部成交",
                "盈亏": 0,
                "手续费": abs(row["weight"]) * close_price * 100 * commission_rate,
                "成交号": "hedge1",
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


def compute_pnl_from_transactions(
    transactions: pd.DataFrame,
    quote: pd.DataFrame,
    commission_rate: float = 0.000432,
) -> pd.DataFrame:
    """从交易记录计算 P&L.

    使用 DequeMatcher FIFO 匹配开仓→平仓对.

    Args:
        transactions: 交易记录 (broker 格式)
        quote: 行情数据

    Returns:
        每日 P&L DataFrame
    """
    if transactions.empty:
        return pd.DataFrame()

    # 标记买卖方向
    transactions = transactions.copy()
    transactions["trade_type"] = transactions["买卖"].map({"买": 1, "卖": -1})
    transactions["日期"] = pd.to_datetime(transactions["报单日期"])

    # 按合约分组匹配
    pnl_records = []
    for contract, sub in transactions.groupby("委托合约"):
        long_mask = sub["trade_type"] == 1
        short_mask = sub["trade_type"] == -1

        if not long_mask.any() and not short_mask.any():
            continue

        # FIFO 匹配
        buy_records = sub[long_mask].to_dict("records")
        sell_records = sub[short_mask].to_dict("records")

        buy_deque = DequeMatcher(k=buy_records)
        sell_deque = DequeMatcher(k=sell_records)

        try:
            avg_price, matched = DequeMatcher._recur_match_core(
                buy_deque, sell_deque, [],
                share_col="手数", price_col="成交均价",
                trade_type="trade_type", dt_col="日期",
                method="FIFO",
            )
        except ValueError:
            continue

        for buy, sell in matched:
            buy_dt = buy["日期"]
            sell_dt = sell["日期"]
            shares = sell["手数"]
            pnl = (sell["成交均价"] - buy["成交均价"]) * shares * 100
            pnl_records.append({
                "dt": sell_dt,
                "contract": contract,
                "shares": shares,
                "buy_price": buy["成交均价"],
                "sell_price": sell["成交均价"],
                "pnl": pnl,
            })

    return pd.DataFrame(pnl_records)
