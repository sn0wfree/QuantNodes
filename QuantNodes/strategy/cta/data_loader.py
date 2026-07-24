# coding=utf-8
"""CTA 数据加载器 — DuckDB 原子抽取.

设计原则:
    - 只暴露原始抽取函数, 不做面板聚合 (策略代码自行决定)
    - 复用 contract_specs 作为产品合法集合
    - 路径默认 ~/Public/DataCache/futures_options_daily.duckdb

DuckDB 表结构:
    - per-product: {exchange_lowercase}_{product_lowercase}_{kind}_daily
      e.g. cffex_if_futures_daily, shfe_au_options_daily
    - v_all_futures: 1.3M 跨所聚合, 含 symbol_raw / symbol
    - v_all_options: 17M 跨所聚合, 含 strike_yuan / combo_components
    - contract_specs: 91 行, exchange/product/multiplier/tick/price_unit
    - trading_calendar: 交易日历 (由 build_derived_tables.py 生成)
    - main_contract_mapping: 主力合约映射 (每日每产品 OI 最大)
    - continuous_main_{PRODUCT}_daily: 主力连续面板 (92 张表)

公共 API:
    CtaDataLoader(path=...)         → DuckDB read-only connector
    .list_products(exchange=None)   → [product_code]
    .list_option_products()         → [product_code]
    .list_exchanges()               → [CFFEX, SHFE, ...]
    .info()                         → 元数据摘要 (≈119 产品 + 总行数)
    .load_futures_daily(product, start=, end=)     → long df
    .load_options_daily(product, start=, end=)     → long df + strike
    .load_contract_specs()          → 91 行 DataFrame
    .load_v_all_futures(start=, end=)              → 跨所聚合
    .load_v_all_options(start=, end=)              → 跨所聚合
    .load_trading_calendar()        → 交易日历 DataFrame
    .load_main_contract_mapping(product=)  → 主力映射 DataFrame
    .load_continuous_main(product)  → 主力连续面板 DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()

# 交易所 ↔ 表名约定
KNOWN_EXCHANGES: tuple[str, ...] = ("CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX")

# v_all_* 视图的固定列 (供外部引用)
V_ALL_FUTURES_COLS: tuple[str, ...] = (
    "exchange", "product", "symbol_raw", "symbol",
    "trade_date", "open", "high", "low", "close",
    "volume", "turnover", "open_interest", "pre_settlement", "pre_close",
)
V_ALL_OPTIONS_COLS: tuple[str, ...] = (
    "exchange", "product", "symbol_raw", "symbol",
    "underlying", "option_type", "strike_price", "strike_yuan",
    "trade_date", "open", "high", "low", "close", "volume",
)


class ProductKind(str, Enum):
    """产品表类型."""

    FUTURES = "futures"
    OPTIONS = "options"


@dataclass(frozen=True, slots=True)
class ProductTable:
    """(exchange, product, kind) 三元组, 用于定位表名."""

    exchange: str   # 'CFFEX'
    product: str    # 'IF'
    kind: ProductKind

    @property
    def table_name(self) -> str:
        """DuckDB 表名: 'cffex_if_futures_daily'."""
        return f"{self.exchange.lower()}_{self.product.lower()}_{self.kind.value}_daily"

    @property
    def exchange_key(self) -> str:
        """'CFFEX.IF'."""
        return f"{self.exchange}.{self.product}"


class CtaDataLoader:
    """DuckDB CTA 数据加载器 (read-only).

    Example:
        >>> dl = CtaDataLoader()
        >>> print(dl.info())
        >>> df = dl.load_futures_daily("IF", start="2024-01-01", end="2024-12-31")
        >>> print(df.head())
    """

    def __init__(self, duckdb_path: Path | str | None = None):
        self._path = Path(duckdb_path).expanduser() if duckdb_path else DEFAULT_DUCKDB_PATH
        if not self._path.exists():
            raise FileNotFoundError(
                f"DuckDB not found: {self._path}\n"
                f"Expecting ~/Public/DataCache/futures_options_daily.duckdb"
            )

    def __repr__(self) -> str:
        return f"CtaDataLoader(path={self._path})"

    def _connect(self):
        """DuckDB 只读连接 (调用方负责 close)."""
        import duckdb
        return duckdb.connect(str(self._path), read_only=True)

    @staticmethod
    def _list_tables(duckdb_path: Path) -> dict[str, list[str]]:
        """扫描 DuckDB, 返回 {exchange_lower: [product, ...]} 按 _futures_ 表整理.

        跳过 v_all_* / contract_specs / default_ip(s) 等非标准表.
        """
        import duckdb
        con = duckdb.connect(str(duckdb_path), read_only=True)
        try:
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = [r[0] for r in rows]
        finally:
            con.close()

        out: dict[str, list[str]] = {}
        for n in names:
            # 匹配 "xxxx_yyyyyy_futures_daily" / "xxxx_yyyyyy_options_daily"
            if "_futures_daily" not in n and "_options_daily" not in n:
                continue
            kind = "futures" if "_futures_daily" in n else "options"
            # 切分 exchange_low / product_low
            base = n.replace(f"_{kind}_daily", "")
            # base 形如 'cffex_if' / 'cffex_io' / 'default_ips'
            parts = base.split("_", 1)
            if len(parts) != 2 or parts[0] not in (e.lower() for e in KNOWN_EXCHANGES):
                continue
            exchange = parts[0].upper()
            product = parts[1].upper()
            out.setdefault((exchange, kind), []).append(product)
        # 转成 {exchange: {kind: [product]}}
        nested: dict[str, dict[str, list[str]]] = {}
        for (exch, kind), prods in out.items():
            nested.setdefault(exch, {}).setdefault(kind, []).extend(prods)
        for v in nested.values():
            for kk in v:
                v[kk] = sorted(set(v[kk]))
        # 拍平成 dict[str, list[str]] —— 返回按 exchange 索引, kind 来自 _tables() 辅助方法
        # 但为了对外易用, 这里再分两层: {exchange: {kind: [products]}}
        return nested  # type: ignore[return-value]

    def list_exchanges(self) -> list[str]:
        """列出 DuckDB 中可用的交易所 (uppercase)."""
        nested = self._list_tables(self._path)
        return sorted(nested.keys())

    def list_products(
        self,
        exchange: str | None = None,
        kind: ProductKind | str | None = ProductKind.FUTURES,
    ) -> list[str]:
        """列出可用产品代码 (e.g. ['IF','IC','IH','IM','AU','RB',...]).

        Args:
            exchange: 可选, 过滤到单一交易所 ('CFFEX' / 'SHFE' ...).
                      None 表示全部.
            kind: ProductKind.FUTURES (默认) / ProductKind.OPTIONS.
        """
        if isinstance(kind, ProductKind):
            kind_str = kind.value
        elif isinstance(kind, str):
            kind_str = kind
        else:
            kind_str = ProductKind.FUTURES.value

        nested = self._list_tables(self._path)
        if exchange is None:
            products: list[str] = []
            for ex_dict in nested.values():
                products.extend(ex_dict.get(kind_str, []))
            return sorted(set(products))
        ex = exchange.upper()
        return sorted(set(nested.get(ex, {}).get(kind_str, [])))

    def list_option_products(self, exchange: str | None = None) -> list[str]:
        """列出有期权表的产品代码."""
        return self.list_products(exchange=exchange, kind=ProductKind.OPTIONS)

    def _table_name(self, product: str, kind: ProductKind) -> str:
        """从 product + kind 解析表名 (用 contract_specs 反查 exchange)."""
        specs_table = self._contract_specs_view()
        ex_match = specs_table[specs_table["product"] == product.upper()]
        if ex_match.empty:
            raise KeyError(f"product '{product}' not in contract_specs")
        exchange = ex_match["exchange"].iloc[0]

        nested = self._list_tables(self._path)
        kind_str = kind.value
        table_name = f"{exchange.lower()}_{product.lower()}_{kind_str}_daily"
        available = nested.get(exchange, {}).get(kind_str, [])
        if product.upper() not in available:
            raise KeyError(
                f"table '{table_name}' not found for product={product} kind={kind_str}; "
                f"available for {exchange}: {available[:10]}"
            )
        return table_name

    def _date_filter(self, start: str | None, end: str | None) -> str:
        """构造 trade_date BETWEEN 子句."""
        conds: list[str] = []
        if start:
            conds.append(f"trade_date >= DATE '{start}'")
        if end:
            conds.append(f"trade_date <= DATE '{end}'")
        return (" WHERE " + " AND ".join(conds)) if conds else ""

    def _contract_specs_view(self) -> pd.DataFrame:
        """用于反查 product→exchange. 复用 DuckDB 内 contract_specs 表."""
        return self.load_contract_specs()

    def load_contract_specs(self) -> pd.DataFrame:
        """读 contract_specs 全表."""
        con = self._connect()
        try:
            return con.execute(
                "SELECT exchange, product, contract_multiplier, price_unit, tick_size "
                "FROM contract_specs ORDER BY exchange, product"
            ).df()
        finally:
            con.close()

    def load_futures_daily(
        self,
        product: str,
        start: str | None = None,
        end: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读单产品期货日线 (long form, 含所有具体合约).

        Args:
            product: 产品代码 'IF' / 'AU' / 'RB' (来自 contract_specs.product)
            start / end: 'YYYY-MM-DD', 可选
            columns: 投影列, 默认全列
                exchange / product / symbol / trade_date / open/high/low/close
                / volume / turnover / open_interest / pre_settlement / pre_close / pre_open_interest

        Returns:
            pd.DataFrame, 按 trade_date ASC 排序
        """
        tbl = self._table_name(product, ProductKind.FUTURES)
        proj = ", ".join(columns) if columns else "*"
        sql = f"SELECT {proj} FROM {tbl}{self._date_filter(start, end)} ORDER BY trade_date, symbol"
        con = self._connect()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def load_options_daily(
        self,
        product: str,
        start: str | None = None,
        end: str | None = None,
        columns: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读单产品期权日线 (含 strike_price / option_type / underlying)."""
        tbl = self._table_name(product, ProductKind.OPTIONS)
        proj = ", ".join(columns) if columns else "*"
        sql = f"SELECT {proj} FROM {tbl}{self._date_filter(start, end)} ORDER BY trade_date, symbol"
        con = self._connect()
        try:
            return con.execute(sql).df()
        finally:
            con.close()

    def load_v_all_futures(
        self,
        start: str | None = None,
        end: str | None = None,
        products: Iterable[str] | None = None,
        exchanges: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读 v_all_futures 统一视图 (1.3M rows, 全所聚合).

        可选过滤: products (e.g. ['IF','AU']) / exchanges (e.g. ['CFFEX','SHFE'])
        """
        conds: list[str] = []
        if start:
            conds.append(f"trade_date >= DATE '{start}'")
        if end:
            conds.append(f"trade_date <= DATE '{end}'")
        if products:
            plist = ", ".join(f"'{p.upper()}'" for p in products)
            conds.append(f"product IN ({plist})")
        if exchanges:
            elist = ", ".join(f"'{e.upper()}'" for e in exchanges)
            conds.append(f"exchange IN ({elist})")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        con = self._connect()
        try:
            return con.execute(
                f"SELECT * FROM v_all_futures{where} ORDER BY trade_date, product, symbol"
            ).df()
        finally:
            con.close()

    def load_v_all_options(
        self,
        start: str | None = None,
        end: str | None = None,
        products: Iterable[str] | None = None,
        exchanges: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """读 v_all_options 统一视图 (17M rows, 含 strike_yuan / combo_components)."""
        conds: list[str] = []
        if start:
            conds.append(f"trade_date >= DATE '{start}'")
        if end:
            conds.append(f"trade_date <= DATE '{end}'")
        if products:
            plist = ", ".join(f"'{p.upper()}'" for p in products)
            conds.append(f"product IN ({plist})")
        if exchanges:
            elist = ", ".join(f"'{e.upper()}'" for e in exchanges)
            conds.append(f"exchange IN ({elist})")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        con = self._connect()
        try:
            return con.execute(
                f"SELECT * FROM v_all_options{where} ORDER BY trade_date, product, symbol"
            ).df()
        finally:
            con.close()

    def load_trading_calendar(self) -> pd.DataFrame:
        """读交易日历表.

        Returns:
            pd.DataFrame, 列: trade_date, year, month, day_of_week,
                          is_first_day_of_month, is_last_day_of_month
        """
        con = self._connect()
        try:
            return con.execute("SELECT * FROM trading_calendar ORDER BY trade_date").df()
        finally:
            con.close()

    def load_main_contract_mapping(
        self,
        product: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读主力合约映射表 (每日每产品 OI 最大合约).

        Args:
            product: 可选, 过滤到单一产品
            start / end: 'YYYY-MM-DD', 可选

        Returns:
            pd.DataFrame, 列: product, trade_date, symbol, open_interest, volume, close
        """
        conds: list[str] = []
        if product:
            conds.append(f"product = '{product.upper()}'")
        if start:
            conds.append(f"trade_date >= DATE '{start}'")
        if end:
            conds.append(f"trade_date <= DATE '{end}'")
        where = (" WHERE " + " AND ".join(conds)) if conds else ""
        con = self._connect()
        try:
            return con.execute(
                f"SELECT * FROM main_contract_mapping{where} ORDER BY product, trade_date"
            ).df()
        finally:
            con.close()

    def load_continuous_main(
        self,
        product: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """读主力连续面板 (单产品, 按主力映射拼接).

        Args:
            product: 产品代码 'IF' / 'AU' / 'RB'
            start / end: 'YYYY-MM-DD', 可选

        Returns:
            pd.DataFrame, 列: trade_date, open, high, low, close, volume, open_interest
        """
        table_name = f"continuous_main_{product.lower()}_daily"
        where = self._date_filter(start, end)
        con = self._connect()
        try:
            # 检查表是否存在
            exists = con.execute(
                f"SELECT count(*) FROM information_schema.tables "
                f"WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            if not exists:
                raise KeyError(
                    f"Table '{table_name}' not found. "
                    f"Run scripts/build_derived_tables.py first."
                )
            return con.execute(
                f"SELECT * FROM {table_name}{where} ORDER BY trade_date"
            ).df()
        finally:
            con.close()

    def list_continuous_products(self) -> list[str]:
        """列出已有主力连续面板的产品."""
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'continuous_main_%_daily'"
            ).fetchall()
            products = []
            for r in rows:
                name = r[0]
                # continuous_main_IF_daily → IF
                parts = name.replace("continuous_main_", "").replace("_daily", "")
                products.append(parts.upper())
            return sorted(products)
        finally:
            con.close()

    def info(self) -> dict:
        """元数据摘要: 总 products / options / 行数 / 日期范围.

        Returns:
            dict 含 keys:
              - 'path'
              - 'exchanges'             : list[dict{exchange, n_futures, n_options, rows}]
              - 'futures_products'      : 全部期货产品 (list[str])
              - 'option_products'       : 全部期权产品 (list[str])
              - 'v_all_futures_rows'    : int
              - 'v_all_options_rows'    : int
              - 'v_all_futures_range'   : (min_date, max_date)
              - 'v_all_options_range'   : (min_date, max_date)
              - 'trading_calendar_rows' : int
              - 'main_contract_rows'    : int
              - 'continuous_products'   : list[str]
        """
        nested = self._list_tables(self._path)
        con = self._connect()
        try:
            exchanges_info = []
            for ex in sorted(nested.keys()):
                fut = nested.get(ex, {}).get(ProductKind.FUTURES.value, [])
                opt = nested.get(ex, {}).get(ProductKind.OPTIONS.value, [])
                rows_fut = 0
                if fut:
                    tables = ",".join(f"{ex.lower()}_{p.lower()}_futures_daily" for p in fut)
                    rows_fut = con.execute(
                        f"SELECT sum(n_rows) FROM ("
                        f"SELECT count() AS n_rows FROM {tables.split(',')[0]}"
                        + "".join(f" UNION ALL SELECT count() FROM {t}" for t in tables.split(",")[1:])
                        + ")"
                    ).fetchone()[0] or 0
                exchanges_info.append({
                    "exchange": ex,
                    "n_futures": len(fut),
                    "n_options": len(opt),
                    "rows_futures": int(rows_fut),
                })
            all_fut = sorted({p for ex_d in nested.values() for p in ex_d.get(ProductKind.FUTURES.value, [])})
            all_opt = sorted({p for ex_d in nested.values() for p in ex_d.get(ProductKind.OPTIONS.value, [])})

            v_fut = con.execute(
                "SELECT count(), CAST(min(trade_date) AS VARCHAR), CAST(max(trade_date) AS VARCHAR) "
                "FROM v_all_futures"
            ).fetchone()
            v_opt = con.execute(
                "SELECT count(), CAST(min(trade_date) AS VARCHAR), CAST(max(trade_date) AS VARCHAR) "
                "FROM v_all_options"
            ).fetchone()

            # 派生表信息
            cal_rows = 0
            try:
                cal_rows = con.execute("SELECT count() FROM trading_calendar").fetchone()[0]
            except Exception:
                pass

            mapping_rows = 0
            try:
                mapping_rows = con.execute("SELECT count() FROM main_contract_mapping").fetchone()[0]
            except Exception:
                pass

            continuous_prods = self.list_continuous_products()

            return {
                "path": str(self._path),
                "exchanges": exchanges_info,
                "futures_products": all_fut,
                "option_products": all_opt,
                "n_futures_products": len(all_fut),
                "n_option_products": len(all_opt),
                "v_all_futures_rows": int(v_fut[0]),
                "v_all_options_rows": int(v_opt[0]),
                "v_all_futures_range": (v_fut[1], v_fut[2]),
                "v_all_options_range": (v_opt[1], v_opt[2]),
                "trading_calendar_rows": cal_rows,
                "main_contract_rows": mapping_rows,
                "continuous_products": continuous_prods,
            }
        finally:
            con.close()
