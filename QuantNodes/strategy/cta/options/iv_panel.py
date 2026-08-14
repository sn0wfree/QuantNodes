# coding=utf-8
"""期权 IV 面板: 逐品种逐日 ATM 隐含波动率时序 + 历史分位.

构建逻辑:
    1. 从 v_all_options 加载品种期权行情
    2. 每日期权 underlying 价格 (UnderlyingMap)
    3. 每日选 ATM 合约: |strike - underlying| 最小 (优先近月 30-60 天)
    4. compute_greeks_for_option 反解 IV
    5. 输出 (trade_date × product) ATM IV 面板 + rolling 分位

预计算存储: DuckDB 新表 option_iv_daily (回测直接读, 避免实时反解).
"""
from __future__ import annotations

import pandas as pd

from ..data_loader import CtaDataLoader
from .greeks import map_cffex_underlying
from .underlying_map import UnderlyingMap
from .expiry import OptionExpiry

IV_TABLE = "option_iv_daily"
IV_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS option_iv_daily (
    product VARCHAR,
    exchange VARCHAR,
    trade_date DATE,
    atm_iv DOUBLE,
    iv_percentile DOUBLE,       -- rolling 252 天 pct-rank
    n_contracts INTEGER,
    PRIMARY KEY (product, trade_date)
)
"""


def _nearest_strike_contract(day_df: pd.DataFrame, underlying_price: float,
                             min_days: int, max_days: int) -> pd.DataFrame | None:
    """当日选 ATM 合约 (近月 30-60 天): |strike - underlying| 最小且流动性可.

    Returns:
        单行 DataFrame 或 None (无合适合约)
    """
    if day_df.empty or underlying_price is None or underlying_price <= 0:
        return None
    # 过滤流动性 (成交量或持仓量 > 0)
    liq = day_df[(day_df["volume"].fillna(0) > 0) | (day_df["open_interest"].fillna(0) > 0)]
    if liq.empty:
        liq = day_df
    ttm = liq.get("_ttm")
    if ttm is not None:
        in_window = liq[ttm.between(min_days / 365, max_days / 365)]
        if not in_window.empty:
            liq = in_window
    liq = liq.assign(_dist=(liq["strike_price"] - underlying_price).abs())
    # 优先 ATM (距离最小), 同距离取 call (流动性通常更好)
    best = liq.sort_values(["_dist", "option_type"], ascending=[True, True]).iloc[:1]
    return best


def _load_iv_for_product(
    product: str,
    loader: CtaDataLoader,
    um: UnderlyingMap,
    oe: OptionExpiry,
    min_days: int = 30,
    max_days: int = 60,
) -> pd.DataFrame:
    """构建单产品 ATM IV 面板 (trade_date × atm_iv)."""
    df = loader.load_options_daily(product)
    if df is None or df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # underlying 价格: 按日查 (用唯一 underlying 集合)
    underlyings = df["underlying"].dropna().unique().tolist()
    price_frames = {}
    for u in underlyings:
        mapped = map_cffex_underlying(u, product)
        ser = um.get(mapped)
        if not ser.empty:
            price_frames[u] = ser
    # 合并 underlying 价格 (long form)
    parts = []
    for u in underlyings:
        ser = price_frames.get(u)
        if ser is None:
            continue
        tmp = ser.rename("u_price").reset_index().rename(columns={"index": "trade_date"})
        tmp["underlying"] = u
        parts.append(tmp)
    if not parts:
        return pd.DataFrame()
    up = pd.concat(parts, ignore_index=True)
    df = df.merge(up, on=["trade_date", "underlying"], how="left")

    # 到期日
    syms = df["symbol"].unique().tolist()
    exp_frame = oe.get_frame(syms)
    exp_map = dict(zip(exp_frame["symbol"], exp_frame["expiry_date"]))
    df["_expiry"] = df["symbol"].str.upper().map(exp_map)
    df["_expiry"] = pd.to_datetime(df["_expiry"], errors="coerce")
    df["_ttm"] = (df["_expiry"] - df["trade_date"]).dt.days

    # 逐日选 ATM 合约 + 反解 IV
    rows = []
    for day, day_df in df.groupby("trade_date"):
        u_price = day_df["u_price"].dropna()
        if u_price.empty:
            continue
        price = float(u_price.iloc[0])
        best = _nearest_strike_contract(day_df, price, min_days, max_days)
        if best is None or best.empty:
            continue
        row = best.iloc[0]
        ttm = row["_ttm"] / 365.0 if pd.notna(row["_ttm"]) and row["_ttm"] > 0 else None
        if ttm is None:
            continue
        from .greeks import compute_greeks_for_option
        g = compute_greeks_for_option(
            symbol=row["symbol"],
            option_type=row["option_type"],
            strike_price=row["strike_price"],
            underlying_price=price,
            market_price=row["close"],
            time_to_maturity=ttm,
        )
        if g.valid:
            rows.append({
                "product": product,
                "exchange": row.get("exchange", ""),
                "trade_date": day,
                "atm_iv": g.iv,
                "n_contracts": len(day_df),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("trade_date").sort_index()


def build_iv_panel(
    products: list[str] | None = None,
    loader: CtaDataLoader | None = None,
    min_days: int = 30,
    max_days: int = 60,
    verbose: bool = True,
) -> pd.DataFrame:
    """构建多产品 ATM IV 面板 → DataFrame (trade_date × product).

    列: product / exchange / trade_date / atm_iv / n_contracts
    """
    loader = loader or CtaDataLoader()
    um = UnderlyingMap(loader)
    oe = OptionExpiry(loader)
    if products is None:
        products = sorted(loader.list_option_products())
    frames = []
    for p in products:
        if verbose:
            print(f"  IV[{p}] ...", flush=True)
        try:
            f = _load_iv_for_product(p, loader, um, oe, min_days, max_days)
        except Exception as e:
            if verbose:
                print(f"  IV[{p}] FAILED: {e}")
            continue
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames)
    panel = panel.pivot_table(
        index="trade_date", columns="product", values="atm_iv",
    ).sort_index()
    return panel


def add_percentile(panel: pd.DataFrame, window: int = 252,
                   min_periods: int = 60) -> pd.DataFrame:
    """面板加 rolling 252 天 pct-rank 分位列 (iv_percentile 后缀列).

    Returns:
        DataFrame (trade_date × product), 值为 IV 分位 [0,1].
    """
    return panel.rolling(window, min_periods=min_periods).rank(pct=True)


def save_iv_panel(panel: pd.DataFrame, duckdb_path: str | None = None) -> None:
    """预计算 IV 面板入库 option_iv_daily (含分位)."""
    from pathlib import Path
    import duckdb

    path = duckdb_path or str(Path.home() / "Public" / "DataCache"
                              / "futures_options_daily.duckdb")
    long = panel.reset_index().melt(id_vars="trade_date", var_name="product",
                                    value_name="atm_iv").dropna()
    if long.empty:
        return
    pct = add_percentile(panel)
    pct_long = pct.reset_index().melt(id_vars="trade_date", var_name="product",
                                      value_name="iv_percentile")
    merged = long.merge(pct_long, on=["trade_date", "product"])
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.date
    con = duckdb.connect(path)
    try:
        con.execute(IV_TABLE_SCHEMA)
        con.execute(f"DELETE FROM {IV_TABLE}")
        con.execute(
            f"INSERT INTO {IV_TABLE} (product, trade_date, atm_iv, iv_percentile) "
            f"SELECT product, trade_date, atm_iv, iv_percentile FROM merged"
        )
        n = con.execute(f"SELECT COUNT(*) FROM {IV_TABLE}").fetchone()[0]
        print(f"option_iv_daily: {n:,} 行已入库")
    finally:
        con.close()


def load_iv_panel(
    products: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    duckdb_path: str | None = None,
) -> pd.DataFrame:
    """从 option_iv_daily 读取 IV 面板 (trade_date × product)."""
    from pathlib import Path
    import duckdb

    path = duckdb_path or str(Path.home() / "Public" / "DataCache"
                              / "futures_options_daily.duckdb")
    conds = []
    if products:
        plist = ", ".join(f"'{p.upper()}'" for p in products)
        conds.append(f"product IN ({plist})")
    if start:
        conds.append(f"trade_date >= DATE '{start}'")
    if end:
        conds.append(f"trade_date <= DATE '{end}'")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    con = duckdb.connect(path, read_only=True)
    try:
        df = con.execute(f"SELECT * FROM {IV_TABLE}{where} ORDER BY product, trade_date").df()
    finally:
        con.close()
    if df.empty:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    panel = df.pivot_table(index="trade_date", columns="product",
                           values="atm_iv").sort_index()
    return panel
