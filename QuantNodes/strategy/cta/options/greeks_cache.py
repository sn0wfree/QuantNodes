# coding=utf-8
"""期权 Greeks 预计算缓存 + 元数据速查.

用法 (docs/103 对角线策略):
    - build_option_greeks_cache.py  并行全量预计算 → data/cache/option_greeks.parquet
    - GreeksCache(PARQUET) 加载全量 → O(1) 查 (product, symbol, trade_date)

吞吐: 2.43M option-days 载入内存 ~2-4s, 查询为 dict 命中原生开销.
回测命中率 ~95%+ (策略选中的 liquid call 均在缓存内).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET = Path("/home/ll/Public/QuantNodes/data/cache/option_greeks.parquet")


class GreeksCache:
    """按 (product, symbol, date) 提供 delta/gamma/vega/theta/iv + 元数据.

    数据源优先级:
        1. parquet 全量缓存 (build_option_greeks_cache.py 产出)
        2. 未命中时按需反解 (graceful fallback, 不回写 parquet)
        3. pickle 增量缓存 (可选, 回测中途持久化命中结果)
    """

    def __init__(self, parquet_path: str | Path | None = None,
                 pickle_path: str | Path | None = None):
        self._data: dict[tuple, dict] = {}
        self._hits = 0
        self._misses = 0
        self._pickle_path = pickle_path

        parquet = Path(parquet_path) if parquet_path else DEFAULT_PARQUET
        if parquet.exists():
            self.load_parquet(parquet)
        elif pickle_path is not None and Path(pickle_path).exists():
            try:
                with Path(pickle_path).open("rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                self._data = {}

    def load_parquet(self, path: str | Path):
        """载入 parquet → dict[(product, symbol, trade_date)] = greeks dict."""
        import pyarrow.parquet as pq

        tbl = pq.read_table(str(path), columns=[
            "product", "symbol", "trade_date", "strike_price", "option_type",
            "underlying", "expiry_date", "spot", "delta", "gamma",
            "vega", "theta", "iv", "valid",
        ])
        df = tbl.to_pandas()
        df = df[df["valid"]]
        if df.empty:
            self._data = {}
            return
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], errors="coerce")
        for row in df.itertuples(index=False):
            self._data[(row.product, str(row.symbol).upper(), row.trade_date)] = {
                "delta": row.delta, "gamma": row.gamma, "vega": row.vega,
                "theta": row.theta, "iv": row.iv,
                "strike": row.strike_price, "expiry": row.expiry_date,
                "spot": row.spot,
            }

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def stats(self) -> dict:
        return {"size": len(self._data), "hits": self._hits, "misses": self._misses}

    def lookup(self, product: str, symbol: str, date: pd.Timestamp) -> dict | None:
        """查全量缓存 (O(1))."""
        key = (product, str(symbol).upper(), pd.Timestamp(date))
        rec = self._data.get(key)
        if rec is not None:
            self._hits += 1
        else:
            self._misses += 1
        return rec

    def get_or_compute(
        self,
        product: str,
        date: pd.Timestamp,
        symbol: str,
        strike: float,
        price: float,
        spot: float,
        expiry: pd.Timestamp | None,
        option_type: str = "C",
    ) -> dict | None:
        """查缓存, 未命中按需反解 (fallback)."""
        ts = pd.Timestamp(date)
        key = (product, str(symbol).upper(), ts)
        hit = self._data.get(key)
        if hit is not None:
            self._hits += 1
            return hit
        self._misses += 1
        if expiry is None or price <= 0 or spot <= 0:
            return None
        ttm = (pd.Timestamp(expiry) - ts).days / 365.0
        if ttm <= 0:
            return None
        try:
            from .greeks import compute_greeks_for_option
            g = compute_greeks_for_option(
                symbol=symbol, option_type=option_type,
                strike_price=strike, underlying_price=spot,
                market_price=price, time_to_maturity=ttm,
            )
            if not g.valid:
                return None
            rec = {
                "delta": float(g.delta), "gamma": float(g.gamma),
                "vega": float(g.vega), "theta": float(g.theta),
                "strike": float(strike), "expiry": str(expiry),
                "spot": float(spot), "iv": float(g.iv),
            }
            self._data[key] = rec
            return rec
        except Exception:
            return None

    def save_pickle(self, path: str | Path | None = None):
        path = Path(path) if path else self._pickle_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._data, f)
