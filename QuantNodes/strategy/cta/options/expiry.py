# coding=utf-8
"""期权到期日解析: 从期权数据推断每合约最后交易日 (数据驱动, 无需外部日历).

推断逻辑:
    - 每个期权合约的最后交易日 = 该合约在数据中最后出现的日期
      (交易/OI 归零日); 用 volume/OI 双条件过滤尾部噪声.
    - 提供 {option_symbol: expiry_date} 映射 + 批量查询.
"""
from __future__ import annotations

import pandas as pd

from ..data_loader import CtaDataLoader


class OptionExpiry:
    """期权到期日映射 (按需加载, 缓存).

    用法:
        oe = OptionExpiry()
        oe.get('rb2605P600')     # → pd.Timestamp 到期日 (或 None)
        oe.get_frame(['rb2605P600', 'm2608P2750'])
    """

    def __init__(self, loader: CtaDataLoader | None = None,
                 min_volume: int = 0, min_oi: int = 0):
        self._loader = loader or CtaDataLoader()
        self._min_volume = min_volume
        self._min_oi = min_oi
        self._expiry: dict[str, pd.Timestamp] = {}
        self._products: set[str] | None = None

    def _product_codes(self) -> set[str]:
        if self._products is None:
            try:
                self._products = {
                    p.upper() for p in self._loader.list_option_products()
                }
            except Exception:
                self._products = set()
        return self._products

    def _load_product(self, product: str) -> None:
        """加载单产品全部期权合约的最后交易日."""
        try:
            df = self._loader.load_options_daily(product)
        except Exception:
            return
        if df is None or df.empty:
            return
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        # 过滤低流动性 (可选): volume > min_volume 或 OI > min_oi 的天才算有效
        mask = pd.Series(True, index=df.index)
        if self._min_volume > 0:
            mask &= (df["volume"].fillna(0) >= self._min_volume)
        if self._min_oi > 0:
            mask &= (df["open_interest"].fillna(0) >= self._min_oi)
        valid = df[mask]
        last = valid.groupby("symbol")["trade_date"].max()
        for sym, dt in last.items():
            self._expiry[sym.upper()] = pd.Timestamp(dt)

    def _ensure(self, symbol: str) -> None:
        sym = symbol.upper()
        if sym in self._expiry:
            return
        # 从 symbol 推断产品 (前缀字母段, 到首个数字为止):
        #   m2608-P-2750 → M;  rb2303C3250 → RB;  IO2502-C-3700 → IO
        head = sym.split("-")[0]
        product = ""
        for ch in head:
            if ch.isalpha():
                product += ch
            else:
                break
        codes = self._product_codes()
        if product and product in codes:
            self._load_product(product)
        else:
            self._expiry[sym] = pd.Timestamp(pd.NaT)

    def get(self, symbol: str) -> pd.Timestamp | None:
        """返回期权到期日 (Timestamp) 或 None."""
        self._ensure(symbol)
        dt = self._expiry.get(symbol.upper())
        if dt is None or pd.isna(dt):
            return None
        return dt

    def get_frame(self, symbols: list[str]) -> pd.DataFrame:
        """批量 → DataFrame (symbol, expiry_date)."""
        rows = []
        for s in symbols:
            dt = self.get(s)
            rows.append({"symbol": s.upper(), "expiry_date": dt})
        return pd.DataFrame(rows)
