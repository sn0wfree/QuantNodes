# coding=utf-8
"""AllWeather — 38 资产统一面板加载.

数据源:
    - 期货 29: DuckDB continuous_main_{product}_daily → close 序列
      roll 调整版: main_contract_mapping + v_all_futures 品种内收益拼接
      (Step 0/D4 结论: 13 品种 roll 日波动 2-5x, 未复权拼接假收益 1-3%)
    - ETF 9: parquet (主面板 + smartbeta 合并) → NAV 序列

输出: data/allweather/panel_v1.parquet (未调整) / panel_v2.parquet (roll 调整)
    - columns: 38 资产 (futures_{P} 前缀 + etf_{code} 前缀, 避免重名)
    - index: 交易日 (DatetimeIndex, 无时区)
    - values: close/NAV 价格

设计要点:
    - 期货和 ETF 用前缀区分, 防止重名
    - 起始日取交集, 池内资产按其可用日期渐入池 (HRP 天然支持)
    - 终点取最早停止更新日
    - roll 调整: 换月日收益 = 持有合约(旧主约)同日真实收益, 消除拼接假收益;
      价格指数 = 从起始 close 累乘 (1+ret)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .config import (
    DUCKDB_PATH,
    ETF_CODES,
    ETF_NAV_PATH,
    ETF_OHLCV_ADJ_PATH,
    ETF_SMARTBETA_PATH,
    END_DATE,
    FUTURES_PRODUCTS,
    OUTPUT_DIR,
    PANEL_OUTPUT,
    START_DATE,
)

warnings.filterwarnings("ignore")

PANEL_V2_OUTPUT = OUTPUT_DIR / "panel_v2.parquet"


# 列名加前缀避免冲突
def _fcol(p: str) -> str:
    return f"fut_{p.lower()}"


def _ecol(c: str) -> str:
    return f"etf_{c}"


def load_futures_panel(products: list[str], duckdb_path: Path) -> pd.DataFrame:
    """从 DuckDB continuous_main_*_daily 加载 29 期货 close 序列.

    Args:
        products: 期货产品代码列表, e.g. ['AU','AG','CU',...]
        duckdb_path: DuckDB 文件路径

    Returns:
        DataFrame (T x 29), columns=fut_{product}, values=close
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        frames: dict[str, pd.Series] = {}
        for p in products:
            table = f"continuous_main_{p.lower()}_daily"
            sql = (
                f"SELECT trade_date, close FROM {table} "
                f"WHERE trade_date >= '{START_DATE}' AND trade_date <= '{END_DATE}' "
                f"ORDER BY trade_date"
            )
            df = con.execute(sql).df()
            if df.empty:
                warnings.warn(f"[{p}] DuckDB 表为空, 跳过", stacklevel=2)
                continue
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            s = df.set_index("trade_date")["close"].rename(_fcol(p))
            frames[_fcol(p)] = s
        panel = pd.concat(frames.values(), axis=1)
        panel.index.name = "date"
        return panel
    finally:
        con.close()


def _same_day_return(close_mat: pd.DataFrame, d: pd.Timestamp, sym: str,
                     max_lookback: int = 5) -> float | None:
    """合约 sym 在交易日 d 的真实日收益 close(d)/close(上一有数据交易日) - 1.

    用 close_mat 的交易日索引回看 (非日历日), 最多回看 max_lookback 个交易日;
    数据不足返回 None (调用方决定回退策略).
    """
    if sym not in close_mat.columns or d not in close_mat.index:
        return None
    pos = close_mat.index.get_loc(d)
    if pos == 0:
        return None
    c_t = close_mat.iloc[pos][sym]
    if pd.isna(c_t) or c_t == 0:
        return None
    for back in range(1, max_lookback + 1):
        if pos - back < 0:
            return None
        c_prev = close_mat.iloc[pos - back][sym]
        if not pd.isna(c_prev) and c_prev != 0:
            return float(c_t / c_prev - 1)
    return None


def load_futures_returns_roll_adjusted(
    products: list[str], duckdb_path: Path
) -> pd.DataFrame:
    """构建 roll 调整的期货收益序列 → 合成价格面板.

    方法 (收益拼接):
        - 非换月日: ret = 连续主连当日收益 (同一合约, 即真实收益)
        - 换月日:   ret = 旧主约同日收益 close_old(t)/close_old(t-1) - 1
          (交易者持有旧约到换月, 换约本身不产生 PnL, 拼接假收益被消除)
        - 旧约 t 或 t-1 收盘缺失时, 回退新约同日收益
        - 价格 = 首日真实 close × cumprod(1+ret)

    Returns:
        DataFrame (T x 29), columns=fut_{product}, values=调整后价格
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        frames: dict[str, pd.Series] = {}
        for product in products:
            mapping = con.execute(
                f"SELECT trade_date, symbol FROM main_contract_mapping "
                f"WHERE product = '{product}' ORDER BY trade_date"
            ).df()
            if mapping.empty:
                warnings.warn(f"[{product}] 无主力映射, 跳过", stacklevel=2)
                continue
            mapping["trade_date"] = pd.to_datetime(mapping["trade_date"])
            mapping = mapping.set_index("trade_date")["symbol"]

            symbols = sorted(mapping.unique())
            sym_list = ",".join(f"'{s}'" for s in symbols)
            px = con.execute(
                f"SELECT symbol, trade_date, close FROM v_all_futures "
                f"WHERE product = '{product}' AND symbol IN ({sym_list}) "
                f"ORDER BY trade_date"
            ).df()
            if px.empty:
                warnings.warn(f"[{product}] v_all_futures 无数据, 跳过", stacklevel=2)
                continue
            px["trade_date"] = pd.to_datetime(px["trade_date"])
            close_mat = px.pivot_table(index="trade_date", columns="symbol",
                                       values="close", aggfunc="last").sort_index()

            # 主连 close (与 continuous_main 同源) 与换月日标记
            rows_idx = close_mat.index.get_indexer(mapping.index)
            cols_idx = close_mat.columns.get_indexer(mapping.values)
            vals = close_mat.to_numpy()[rows_idx, cols_idx]
            vals[rows_idx < 0] = np.nan  # get_indexer 未命中返回 -1, 防静默取末行
            main_close = pd.Series(vals, index=mapping.index)
            main_close.index.name = None
            is_roll = mapping != mapping.shift(1)
            is_roll.iloc[0] = False

            ret = main_close.pct_change()
            old_sym = mapping.shift(1)
            for d in mapping.index[is_roll]:
                old = old_sym.loc[d]
                new = mapping.loc[d]
                # 换月日真实收益: 持有合约(旧约)同日收益; 缺失回退新约
                r = _same_day_return(close_mat, d, old)
                if r is None:
                    r = _same_day_return(close_mat, d, new)
                if r is not None:
                    ret.loc[d] = r

            # 合成价格指数: 首日真实 close 累乘
            ret = ret.fillna(0.0)
            price = main_close.iloc[0] * (1 + ret).cumprod()
            price.iloc[0] = main_close.iloc[0]
            frames[_fcol(product)] = price.rename(_fcol(product))

        panel = pd.concat(frames.values(), axis=1)
        panel.index.name = "date"
        return panel
    finally:
        con.close()


def load_etf_panel(
    codes: list[str],
    nav_path: Path,
    smartbeta_path: Path,
    ohlcv_path: Path,
) -> pd.DataFrame:
    """从 parquet 加载 9 只 ETF NAV.

    512890 在主面板不在, 从 smartbeta 面板补;
    其余 8 只从主面板取.

    收益级交叉验证 (2026-09-02 新增, 修复 ETF 数据毛刺):
        - NAV 面板 512690 有 18 个坏日 (份额折算未同步, |ret|>10.5% 超涨跌停)
        - OHLCV 复权面板有 5 个拆分日毛刺 (NAV 正常), 但其余日期干净且到 2026-07-08
        - 合并规则: ret_nav 为主; |ret_nav|>10.5% → 用 ret_ohlcv;
          两者都坏 → 拆分比修正 (k=prev/cur 取整, ret=cur*k/prev-1), 仍坏则 0
        - NAV 结束后 (06-30 → 07-08) 用 OHLCV 收益延长期限

    Args:
        codes: ETF 代码列表
        nav_path: 主 NAV 面板路径
        smartbeta_path: smartbeta 面板路径
        ohlcv_path: OHLCV 复权面板路径 (交叉验证源)

    Returns:
        DataFrame (T x 9), columns=etf_{code}, values=合成价格
    """
    nav_main = pd.read_parquet(nav_path)
    nav_smart = pd.read_parquet(smartbeta_path)
    ohlcv = pd.read_parquet(ohlcv_path)  # MultiIndex (code, field)

    if not isinstance(nav_main.index, pd.DatetimeIndex):
        nav_main.index = pd.to_datetime(nav_main.index)
    if not isinstance(nav_smart.index, pd.DatetimeIndex):
        nav_smart.index = pd.to_datetime(nav_smart.index)

    nav_main = nav_main.loc[START_DATE:]
    nav_smart = nav_smart.loc[START_DATE:]

    frames: dict[str, pd.Series] = {}
    for c in codes:
        if c in nav_main.columns:
            nav = nav_main[c].dropna()
        elif c in nav_smart.columns:
            nav = nav_smart[c].dropna()
        else:
            warnings.warn(f"[{c}] 两表均无, 跳过", stacklevel=2)
            continue

        if (c, "close") in ohlcv.columns:
            oh = ohlcv[(c, "close")].dropna()
            oh.index = pd.to_datetime(oh.index)
            oh = oh.sort_index()
        else:
            oh = None

        ret_nav = nav.pct_change()
        if oh is None:
            frames[_ecol(c)] = nav.rename(_ecol(c))
            continue
        ret_oh = oh.pct_change()

        # 逐 ETF 选主源: 坏日 (|ret|>5%, ETF 不可能的高频大波动) 少者为主
        # (512690 NAV 有份额折算污染 ~18 天>10.5% + 大量 5-9% 隐藏坏日;
        #  OHLCV 只有拆分日毛刺 → 512690 选 OHLCV, 其余选 NAV)
        bad_nav = int((ret_nav.abs() > 0.05).sum())
        bad_oh = int((ret_oh.abs() > 0.05).sum())
        if bad_oh < bad_nav:
            primary, secondary = ret_oh, ret_nav
            base_level = oh.iloc[0]
        else:
            primary, secondary = ret_nav, ret_oh
            base_level = nav.iloc[0]

        # 合并: 主源为主; 超涨跌停日 (>10.5%) 换副源; 双坏用拆分比修正
        ret = primary.copy()
        bad = ret.abs() > 0.105
        for d in ret.index[bad.fillna(False)]:
            r_sec = secondary.get(d)
            if r_sec is not None and not pd.isna(r_sec) and abs(r_sec) <= 0.105:
                ret.loc[d] = r_sec
                continue
            pos = oh.index.get_indexer([d])[0]
            if pos > 0:
                prev, cur = oh.iloc[pos - 1], oh.iloc[pos]
                if prev > 0 and cur > 0:
                    k = prev / cur
                    k_round = round(k * 2) / 2  # 整数或 x.5
                    if k > 1.02 and abs(k - k_round) / k < 0.02:
                        ret.loc[d] = cur * k_round / prev - 1

        # 主源结束后用副源收益延长 (NAV 06-30 → OHLCV 07-08)
        sec_by_idx = secondary.dropna()
        tail = sec_by_idx[sec_by_idx.index > ret.index[-1]]
        tail = tail[tail.index <= pd.Timestamp(END_DATE)]
        for d, r_sec in tail.items():
            if abs(r_sec) <= 0.105:
                ret.loc[d] = r_sec

        ret = ret.loc[ret.first_valid_index():]
        price = base_level * (1 + ret.fillna(0.0)).cumprod()
        price.iloc[0] = base_level
        frames[_ecol(c)] = price.rename(_ecol(c))

    if not frames:
        raise RuntimeError("未加载到任何 ETF 数据")

    panel = pd.concat(frames.values(), axis=1)
    panel.index.name = "date"
    return panel


def build_unified_panel(
    futures_products: list[str] = FUTURES_PRODUCTS,
    etf_codes: list[str] = ETF_CODES,
    duckdb_path: Path = DUCKDB_PATH,
    etf_nav_path: Path = ETF_NAV_PATH,
    etf_smartbeta_path: Path = ETF_SMARTBETA_PATH,
    etf_ohlcv_path: Path = ETF_OHLCV_ADJ_PATH,
    start: str = START_DATE,
    end: str = END_DATE,
    roll_adjust: bool = True,
) -> pd.DataFrame:
    """构建 38 资产统一面板.

    Args:
        roll_adjust: True → 期货用收益拼接消除 roll 假收益 (panel_v2, 主线);
                     False → 原始 continuous_main close (panel_v1, 对照).

    Returns:
        DataFrame (T x 38), 期货前缀 fut_ + ETF 前缀 etf_
    """
    if roll_adjust:
        fut_panel = load_futures_returns_roll_adjusted(futures_products, duckdb_path)
    else:
        fut_panel = load_futures_panel(futures_products, duckdb_path)
    etf_panel = load_etf_panel(etf_codes, etf_nav_path, etf_smartbeta_path,
                               etf_ohlcv_path)

    panel = pd.concat([fut_panel, etf_panel], axis=1)
    panel = panel.loc[start:end]
    panel = panel.sort_index()
    return panel


def _integrity_report(panel: pd.DataFrame, label: str, path: Path):
    coverage = panel.notna().mean()
    print(f"[AllWeather {label}] shape={panel.shape}")
    print(f"  交易日数: {panel.shape[0]}, 日期范围: "
          f"{panel.index.min().date()} ~ {panel.index.max().date()}")
    print(f"  整体非空率: {coverage.mean():.2%}, 落盘: {path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # v2 (roll 调整, 主线)
    panel_v2 = build_unified_panel(roll_adjust=True)
    panel_v2.to_parquet(PANEL_V2_OUTPUT)
    _integrity_report(panel_v2, "panel_v2 (roll调整)", PANEL_V2_OUTPUT)

    # v1 (未调整, 对照) — 已存在则跳过重算
    if not PANEL_OUTPUT.exists():
        panel_v1 = build_unified_panel(roll_adjust=False)
        panel_v1.to_parquet(PANEL_OUTPUT)
        _integrity_report(panel_v1, "panel_v1 (未调整)", PANEL_OUTPUT)

    # v1 vs v2 差异抽查: roll 日收益变化幅度
    v1 = pd.read_parquet(PANEL_OUTPUT)
    ret1 = v1.pct_change(fill_method=None)
    ret2 = panel_v2.pct_change(fill_method=None)
    diffs = (ret1 - ret2).abs()
    print()
    print("roll 调整效果 (|v1_ret - v2_ret| 分位数, 仅期货列):")
    fut_cols = [c for c in v1.columns if c.startswith("fut_")]
    d = diffs[fut_cols].stack()
    print(f"  mean={d.mean():.5f}  p95={d.quantile(0.95):.5f}  "
          f"p99={d.quantile(0.99):.5f}  max={d.max():.5f}")
    print()
    print("各资产起始日 (HRP 渐入池支持):")
    for c in panel_v2.columns:
        first = panel_v2[c].first_valid_index()
        if first is not None:
            n_valid = panel_v2[c].notna().sum()
            pct = n_valid / len(panel_v2)
            print(f"  {c:<20} 起始={first.date()}  非空={n_valid}/{len(panel_v2)} ({pct:.1%})")


if __name__ == "__main__":
    main()
