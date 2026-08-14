# coding=utf-8
"""标的期货价格映射: 期权 underlying (rb2605) → 对应期货合约每日收盘价.

来源: v_all_futures (全部具体合约日线).
输出: {underlying_symbol: pd.Series(close, index=trade_date)}.
"""
from __future__ import annotations

import pandas as pd

from ..data_loader import CtaDataLoader

# 期权合约乘数参考 (与标的期货一致, 从 contract_specs 读)
MULTIPLIER_OVERRIDE = {}


class UnderlyingMap:
    """期权标的期货价格映射 (按需加载, 缓存).

    用法:
        um = UnderlyingMap()
        um.get('rb2605')              # → Series (close, index=trade_date)
        um.get_frame(['rb2605','M2609'])  # → DataFrame (date × underlying)
    """

    def __init__(self, loader: CtaDataLoader | None = None):
        self._loader = loader or CtaDataLoader()
        self._series: dict[str, pd.Series] = {}

    def _load_symbol(self, symbol: str) -> pd.Series | None:
        """从 v_all_futures 加载单一合约价格序列."""
        sym = symbol.upper()
        if sym in self._series:
            return self._series[sym]
        # 产品推断: 前缀字母段 (m2608 → M, rb2303 → RB)
        product = ""
        for ch in sym:
            if ch.isalpha():
                product += ch
            else:
                break
        try:
            df = self._loader.load_v_all_futures(products=[product] if product else None)
        except Exception:
            df = pd.DataFrame()
        if df is None or df.empty:
            self._series[sym] = pd.Series(dtype=float)
            return None
        sub = df[df["symbol"].str.upper() == sym]
        if sub.empty:
            self._series[sym] = pd.Series(dtype=float)
            return None
        s = sub.set_index("trade_date")["close"].sort_index()
        s.index = pd.DatetimeIndex(s.index)
        self._series[sym] = s
        return s

    def get(self, symbol: str) -> pd.Series:
        """获取单一标的期货价格序列 (空 Series 表示无数据)."""
        ser = self._load_symbol(symbol)
        return ser if ser is not None else pd.Series(dtype=float)

    def get_frame(self, symbols: list[str]) -> pd.DataFrame:
        """批量获取 → DataFrame (trade_date × underlying_symbol)."""
        out = {}
        for s in symbols:
            ser = self._load_symbol(s)
            if ser is not None and not ser.empty:
                out[s.upper()] = ser
        if not out:
            return pd.DataFrame()
        frame = pd.DataFrame(out).sort_index()
        return frame

    def get_at(self, symbol: str, trade_date: str | pd.Timestamp) -> float | None:
        """获取标的在指定日期的收盘价 (无则 None)."""
        ser = self.get(symbol)
        if ser.empty:
            return None
        ts = pd.Timestamp(trade_date)
        # 允许 ±5 天查找最近有效价 (停牌/数据缺失容错)
        window = ser[ser.index <= ts].tail(6)
        if window.empty:
            return None
        return float(window.iloc[-1])
