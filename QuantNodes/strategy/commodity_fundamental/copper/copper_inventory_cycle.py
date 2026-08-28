# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_inventory_cycle.py

铜库存周期四象限策略

基于库存周期理论（基钦周期），将库存变化与经济周期结合：
    - 被动去库（经济↑ + 库存↓）→ 做多（最强信号）
    - 主动补库（经济↑ + 库存↑）→ 做多
    - 被动补库（经济↓ + 库存↑）→ 做空
    - 主动去库（经济↓ + 库存↓）→ 观望/做空

核心逻辑:
    1. 经济增长方向: 铜价20周动量 > 0
    2. 库存变化方向: 4周变化率 > 0
    3. 四象限组合信号

数据: inventory_daily (LME铜库存, 2014+), futures_main_close_cu (价格)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_lme_inventory(con=None) -> pd.Series:
    """加载LME铜库存数据.

    返回: 日频库存序列 (index=trade_date)
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value AS inventory "
        "FROM inventory_daily "
        "WHERE value_col='铜-库存' "
        "ORDER BY trade_date"
    ).df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["inventory"]


def compute_inventory_cycle_quadrant(
    con=None,
    econ_window: int = 52,
    inv_change_window: int = 13,
    econ_threshold: float = 0.0,
    inv_threshold: float = 0.0,
    use_short: bool = False,
) -> pd.DataFrame:
    """计算库存周期四象限信号.

    参数:
        con: 数据库连接
        econ_window: 经济增长方向判断窗口 (周)
        inv_change_window: 库存变化窗口 (周)
        econ_threshold: 经济增长方向阈值
        inv_threshold: 库存变化方向阈值
        use_short: 是否允许做空 (被动补库时)

    四象限逻辑:
        经济↑ + 库存↓ = 被动去库 → 做多 (最强)
        经济↑ + 库存↑ = 主动补库 → 做多
        经济↓ + 库存↑ = 被动补库 → 做空 (可选)
        经济↓ + 库存↓ = 主动去库 → 观望

    返回: DataFrame with columns:
        - close_w: 周频收盘价
        - inventory: 周频库存
        - econ_momentum: 经济增长动量
        - inv_change: 库存变化率
        - quadrant: 四象限分类 (1-4)
        - position: 仓位 [-1, 0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格数据
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 加载库存数据
    inv = load_lme_inventory(con)
    inv_w = inv.resample("W-FRI").last()

    # 对齐数据
    common = close_w.index.intersection(inv_w.index)
    close_w = close_w.reindex(common)
    inv_w = inv_w.reindex(common)

    # 计算经济增长方向: 铜价动量 > 0
    econ_momentum = close_w.pct_change(econ_window)
    econ_up = econ_momentum > econ_threshold

    # 计算库存变化方向: 4周变化率 > 0
    inv_change = inv_w.pct_change(inv_change_window)
    inv_up = inv_change > inv_threshold

    # 四象限分类
    quadrant = pd.Series(0, index=common)
    quadrant[econ_up & ~inv_up] = 1    # 被动去库: 做多 (最强)
    quadrant[econ_up & inv_up] = 2     # 主动补库: 做多
    quadrant[~econ_up & inv_up] = 3    # 被动补库: 做空
    quadrant[~econ_up & ~inv_up] = 4   # 主动去库: 观望

    # 仓位信号
    position = pd.Series(0.0, index=common)
    position[quadrant == 1] = 1.0      # 被动去库: 做多
    position[quadrant == 2] = 1.0      # 主动补库: 做多
    if use_short:
        position[quadrant == 3] = -1.0  # 被动补库: 做空
    else:
        position[quadrant == 3] = 0.0   # 被动补库: 空仓
    position[quadrant == 4] = 0.0       # 主动去库: 观望

    return pd.DataFrame({
        "close_w": close_w,
        "inventory": inv_w,
        "econ_momentum": econ_momentum,
        "inv_change": inv_change,
        "quadrant": quadrant,
        "position": position,
    })


def compute_inventory_cycle_v2(
    con=None,
    econ_window: int = 52,
    inv_change_window: int = 13,
    econ_threshold: float = 0.0,
    inv_threshold: float = 0.0,
    use_short: bool = False,
    use_momentum_confirm: bool = True,
    mom_window: int = 4,
) -> pd.DataFrame:
    """四象限策略v2 - 加入动量确认.

    优化点:
        1. 被动去库时加入动量确认 (铜价4周动量>0)
        2. 主动补库时加入动量确认 (铜价4周动量>0)
        3. 减少震荡市的误判

    参数:
        con: 数据库连接
        econ_window: 经济增长方向判断窗口 (周)
        inv_change_window: 库存变化窗口 (周)
        econ_threshold: 经济增长方向阈值
        inv_threshold: 库存变化方向阈值
        use_short: 是否允许做空 (被动补库时)
        use_momentum_confirm: 是否使用动量确认
        mom_window: 动量确认窗口 (周)

    返回: DataFrame with columns:
        - close_w: 周频收盘价
        - inventory: 周频库存
        - econ_momentum: 经济增长动量
        - inv_change: 库存变化率
        - quadrant: 四象限分类 (1-4)
        - position: 仓位 [-1, 0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格数据
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    close_w = df_price.set_index("trade_date")["close"].resample("W-FRI").last()

    # 加载库存数据
    inv = load_lme_inventory(con)
    inv_w = inv.resample("W-FRI").last()

    # 对齐数据
    common = close_w.index.intersection(inv_w.index)
    close_w = close_w.reindex(common)
    inv_w = inv_w.reindex(common)

    # 计算经济增长方向: 铜价动量 > 0
    econ_momentum = close_w.pct_change(econ_window)
    econ_up = econ_momentum > econ_threshold

    # 计算库存变化方向: 4周变化率 > 0
    inv_change = inv_w.pct_change(inv_change_window)
    inv_up = inv_change > inv_threshold

    # 四象限分类
    quadrant = pd.Series(0, index=common)
    quadrant[econ_up & ~inv_up] = 1    # 被动去库: 做多 (最强)
    quadrant[econ_up & inv_up] = 2     # 主动补库: 做多
    quadrant[~econ_up & inv_up] = 3    # 被动补库: 做空
    quadrant[~econ_up & ~inv_up] = 4   # 主动去库: 观望

    # 动量确认
    if use_momentum_confirm:
        mom = close_w.pct_change(mom_window)
        # 被动去库时, 需要动量>0才做多
        quadrant[(quadrant == 1) & (mom <= 0)] = 4  # 降级为观望
        # 主动补库时, 需要动量>0才做多
        quadrant[(quadrant == 2) & (mom <= 0)] = 4  # 降级为观望

    # 仓位信号
    position = pd.Series(0.0, index=common)
    position[quadrant == 1] = 1.0      # 被动去库: 做多
    position[quadrant == 2] = 1.0      # 主动补库: 做多
    if use_short:
        position[quadrant == 3] = -1.0  # 被动补库: 做空
    else:
        position[quadrant == 3] = 0.0   # 被动补库: 空仓
    position[quadrant == 4] = 0.0       # 主动去库: 观望

    return pd.DataFrame({
        "close_w": close_w,
        "inventory": inv_w,
        "econ_momentum": econ_momentum,
        "inv_change": inv_change,
        "quadrant": quadrant,
        "position": position,
    })


def backtest_inventory_cycle(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.10,
    target_vol: float = 0.12,
) -> dict:
    """回测库存周期四象限策略.

    参数:
        signal: compute_inventory_cycle_quadrant输出
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
        target_vol: 目标波动率

    返回: 回测结果字典
    """
    close_w = signal["close_w"]
    pos = signal["position"]

    # 波动率目标化
    ret = close_w.pct_change()
    vol_4w = ret.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(-1, 1)

    # 回测
    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损: 回撤超阈值时清仓
    nav = (1 + strat_ret).cumprod()
    for i in range(1, len(nav)):
        current_dd = nav.iloc[i] / nav.iloc[:i+1].max() - 1
        if current_dd < dd_threshold:
            strat_ret.iloc[i] = 0

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def analyze_quadrant_performance(signal: pd.DataFrame) -> dict:
    """分析各象限表现.

    参数:
        signal: compute_inventory_cycle_quadrant输出

    返回: 各象限表现统计
    """
    close_w = signal["close_w"]
    ret = close_w.pct_change()

    results = {}
    for q in [1, 2, 3, 4]:
        mask = signal["quadrant"] == q
        if mask.sum() == 0:
            continue

        q_ret = ret[mask]
        q_pos = signal.loc[mask, "position"]

        # 策略收益
        strat_ret = q_pos.shift(1) * q_ret
        strat_ret = strat_ret.dropna()

        if len(strat_ret) == 0:
            continue

        nav = (1 + strat_ret).cumprod()
        total_ret = nav.iloc[-1] / nav.iloc[0] - 1
        n_years = len(strat_ret) / 52
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
        ann_vol = strat_ret.std() * np.sqrt(52)
        max_dd = (nav / nav.cummax() - 1).min()

        quadrant_names = {
            1: "被动去库 (经济↑+库存↓)",
            2: "主动补库 (经济↑+库存↑)",
            3: "被动补库 (经济↓+库存↑)",
            4: "主动去库 (经济↓+库存↓)",
        }

        results[q] = {
            "name": quadrant_names[q],
            "n_weeks": len(strat_ret),
            "ann_ret": ann_ret,
            "ann_vol": ann_vol,
            "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
            "max_dd": max_dd,
            "win_rate": (strat_ret > 0).mean(),
        }

    return results


def compute_inventory_cycle_optimized(
    con=None,
    econ_window: int = 52,
    inv_change_window: int = 13,
    inv_ema_span: int = 5,
    use_ema: bool = True,
    econ_threshold: float = 0.0,
    inv_threshold: float = 0.0,
    use_short: bool = False,
) -> pd.DataFrame:
    """优化版四象限信号.

    优化点:
        1. 库存数据: EMA平滑，保留日内趋势
        2. 价格数据: 使用收益率序列，避免换月跳价影响

    参数:
        con: 数据库连接
        econ_window: 经济增长方向判断窗口 (周)
        inv_change_window: 库存变化窗口 (周)
        inv_ema_span: 库存EMA平滑窗口 (天)
        use_ema: 是否使用EMA
        econ_threshold: 经济增长方向阈值
        inv_threshold: 库存变化方向阈值
        use_short: 是否允许做空

    返回: DataFrame with columns:
        - weekly_ret: 周频收益率
        - inventory: 周频库存
        - econ_momentum: 经济增长动量
        - inv_change: 库存变化率
        - quadrant: 四象限分类 (1-4)
        - position: 仓位 [-1, 0, 1]
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载价格数据 - 使用收益率序列
    df_price = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
    df_price = df_price.set_index("trade_date")

    # 计算日收益率，再转周频
    daily_ret = df_price["close"].pct_change()
    weekly_ret = daily_ret.resample("W-FRI").sum()  # 周收益率 = 日收益率之和

    # 加载库存数据
    inv = load_lme_inventory(con)

    # 库存数据处理: EMA平滑
    if use_ema:
        inv_smooth = inv.ewm(span=inv_ema_span).mean()
    else:
        inv_smooth = inv

    inv_w = inv_smooth.resample("W-FRI").last()

    # 对齐数据
    common = weekly_ret.index.intersection(inv_w.index)
    weekly_ret = weekly_ret.reindex(common)
    inv_w = inv_w.reindex(common)

    # 计算经济增长方向: 使用累积收益率
    cum_ret = (1 + weekly_ret).cumprod()
    econ_momentum = cum_ret.pct_change(econ_window)
    econ_up = econ_momentum > econ_threshold

    # 计算库存变化方向
    inv_change = inv_w.pct_change(inv_change_window)
    inv_up = inv_change > inv_threshold

    # 四象限分类
    quadrant = pd.Series(0, index=common)
    quadrant[econ_up & ~inv_up] = 1    # 被动去库: 做多 (最强)
    quadrant[econ_up & inv_up] = 2     # 主动补库: 做多
    quadrant[~econ_up & inv_up] = 3    # 被动补库: 做空
    quadrant[~econ_up & ~inv_up] = 4   # 主动去库: 观望

    # 仓位信号
    position = pd.Series(0.0, index=common)
    position[quadrant == 1] = 1.0      # 被动去库: 做多
    position[quadrant == 2] = 1.0      # 主动补库: 做多
    if use_short:
        position[quadrant == 3] = -1.0  # 被动补库: 做空
    else:
        position[quadrant == 3] = 0.0   # 被动补库: 空仓
    position[quadrant == 4] = 0.0       # 主动去库: 观望

    return pd.DataFrame({
        "weekly_ret": weekly_ret,
        "inventory": inv_w,
        "econ_momentum": econ_momentum,
        "inv_change": inv_change,
        "quadrant": quadrant,
        "position": position,
    })


def backtest_inventory_cycle_optimized(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.10,
    target_vol: float = 0.12,
) -> dict:
    """回测优化版四象限策略.

    参数:
        signal: compute_inventory_cycle_optimized输出
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
        target_vol: 目标波动率
    """
    weekly_ret = signal["weekly_ret"]
    pos = signal["position"]

    # 波动率目标化
    vol_4w = weekly_ret.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(-1, 1)

    # 回测
    strat_ret = pos.shift(1) * weekly_ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损: 回撤超阈值时清仓
    nav = (1 + strat_ret).cumprod()
    for i in range(1, len(nav)):
        current_dd = nav.iloc[i] / nav.iloc[:i+1].max() - 1
        if current_dd < dd_threshold:
            strat_ret.iloc[i] = 0

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


def compute_inventory_cycle_with_atr_stop(
    con=None,
    econ_window: int = 52,
    inv_change_window: int = 13,
    inv_ema_span: int = 5,
    use_ema: bool = True,
    atr_multiple: float = 3.0,
    atr_window: int = 4,
    use_weekly: bool = True,
) -> pd.DataFrame:
    """四象限策略 + ATR止损.

    参数:
        con: 数据库连接
        econ_window: 经济增长方向判断窗口 (周)
        inv_change_window: 库存变化窗口 (周)
        inv_ema_span: 库存EMA平滑窗口 (天)
        use_ema: 是否使用EMA
        atr_multiple: ATR倍数 (默认3.0)
        atr_window: ATR窗口 (默认4周)
        use_weekly: 是否使用周频止损 (推荐)

    返回: DataFrame with columns:
        - weekly_ret: 周频收益率
        - inventory: 周频库存
        - econ_momentum: 经济增长动量
        - inv_change: 库存变化率
        - quadrant: 四象限分类 (1-4)
        - position: 原始仓位
        - position_with_stop: 止损后仓位
    """
    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 计算四象限信号
    signal = compute_inventory_cycle_optimized(
        con=con,
        econ_window=econ_window,
        inv_change_window=inv_change_window,
        inv_ema_span=inv_ema_span,
        use_ema=use_ema,
    )

    if use_weekly:
        # 周频止损
        weekly_ret = signal["weekly_ret"]
        pos = signal["position"]

        # 使用收益率波动率作为ATR
        ret_vol = weekly_ret.rolling(atr_window).std() * np.sqrt(52)

        # 应用止损
        new_pos = pos.copy()
        current_stop = np.nan

        for i in range(1, len(signal)):
            # 信号变化时重置止损
            if pos.iloc[i] != pos.iloc[i-1]:
                if pos.iloc[i] > 0:
                    current_stop = weekly_ret.iloc[i] - atr_multiple * ret_vol.iloc[i]
                elif pos.iloc[i] < 0:
                    current_stop = weekly_ret.iloc[i] + atr_multiple * ret_vol.iloc[i]
                else:
                    current_stop = np.nan

            # 更新止损线
            if pos.iloc[i] > 0 and not np.isnan(ret_vol.iloc[i]):
                new_stop = weekly_ret.iloc[i] - atr_multiple * ret_vol.iloc[i]
                current_stop = max(current_stop, new_stop)
            elif pos.iloc[i] < 0 and not np.isnan(ret_vol.iloc[i]):
                new_stop = weekly_ret.iloc[i] + atr_multiple * ret_vol.iloc[i]
                current_stop = min(current_stop, new_stop)

            # 检查触发
            if pos.iloc[i] > 0 and not np.isnan(current_stop):
                if weekly_ret.iloc[i] <= current_stop:
                    new_pos.iloc[i] = 0.0
                    current_stop = np.nan
            elif pos.iloc[i] < 0 and not np.isnan(current_stop):
                if weekly_ret.iloc[i] >= current_stop:
                    new_pos.iloc[i] = 0.0
                    current_stop = np.nan

        signal["position_with_stop"] = new_pos
    else:
        # 日频止损
        from .copper_daily_stop import apply_daily_atr_stop

        # 加载日频价格数据
        df_price = con.execute(
            "SELECT trade_date, open, high, low, close FROM futures_main_close_cu ORDER BY trade_date"
        ).df()
        df_price["trade_date"] = pd.to_datetime(df_price["trade_date"])
        df_price = df_price.set_index("trade_date")

        # 准备日频OHLC数据
        daily_ohlc = df_price[["open", "high", "low", "close"]]

        # 应用止损
        daily_position = apply_daily_atr_stop(
            weekly_positions=signal["position"],
            daily_ohlc=daily_ohlc,
            atr_multiple=atr_multiple,
            atr_window=14,  # 日频ATR窗口
        )

        # 转回周频
        signal["position_with_stop"] = daily_position.resample("W-FRI").last().reindex(signal.index, method="ffill").fillna(0)

    return signal


def backtest_inventory_cycle_with_atr_stop(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    dd_threshold: float = -0.10,
    target_vol: float = 0.12,
) -> dict:
    """回测四象限策略 + ATR止损.

    参数:
        signal: compute_inventory_cycle_with_atr_stop输出
        cost_bps: 交易成本 (基点)
        dd_threshold: 回撤止损阈值
        target_vol: 目标波动率
    """
    weekly_ret = signal["weekly_ret"]
    pos = signal["position_with_stop"]

    # 波动率目标化
    vol_4w = weekly_ret.rolling(4).std() * np.sqrt(52)
    scale = (target_vol / vol_4w).fillna(1.0).clip(upper=1.0)
    pos = (pos * scale).clip(-1, 1)

    # 回测
    strat_ret = pos.shift(1) * weekly_ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    # 止损: 回撤超阈值时清仓
    nav = (1 + strat_ret).cumprod()
    for i in range(1, len(nav)):
        current_dd = nav.iloc[i] / nav.iloc[:i+1].max() - 1
        if current_dd < dd_threshold:
            strat_ret.iloc[i] = 0

    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_weeks": 0,
            "ann_ret": 0,
            "ann_vol": 0,
            "sharpe": 0,
            "max_dd": 0,
            "win_rate": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


__all__ = [
    "load_lme_inventory",
    "compute_inventory_cycle_quadrant",
    "compute_inventory_cycle_v2",
    "compute_inventory_cycle_optimized",
    "compute_inventory_cycle_with_atr_stop",
    "backtest_inventory_cycle",
    "backtest_inventory_cycle_optimized",
    "backtest_inventory_cycle_with_atr_stop",
    "analyze_quadrant_performance",
]
