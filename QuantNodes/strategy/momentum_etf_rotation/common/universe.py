# coding=utf-8
"""ETF 池定义 (Stage 7: 43+1 = 44 ETFs, 与 data/real/ 对齐).

类别:
    A_BROAD    A 股宽基 (6)
    A_SECTOR   A 股行业 (20)
    HK         港股 (5)
    COMMODITY  商品 (6)
    OVERSEAS   海外 (6)
    BOND       固收 (1, 511260 10年国债ETF, 仅用于 80/20 固收+)

代码与 data/real/fetch_log.json 完全一致.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    A_BROAD = "a_broad"
    A_SECTOR = "a_sector"
    HK = "hk"
    COMMODITY = "commodity"
    OVERSEAS = "overseas"
    BOND = "bond"


@dataclass(frozen=True)
class ETFMeta:
    code: str
    name: str
    category: Category
    index_code: str = ""
    liquidity_rank: int = 1


# 完整 44 ETF 池 (与 data/real/etf_nav_*.parquet 列对齐)
_DEFAULT_ETFS: list[tuple[str, Category, str, int]] = [
    # A 股宽基 (6) — broad-market index ETFs
    ("510300", Category.A_BROAD, "HS300", 1),
    ("510500", Category.A_BROAD, "CSI500", 1),
    ("510050", Category.A_BROAD, "SSE50", 1),
    ("159915", Category.A_BROAD, "ChiNext", 1),
    ("588000", Category.A_BROAD, "STAR50", 1),
    ("159901", Category.A_BROAD, "SZSE100", 1),
    # A 股行业 (20) — sector ETFs
    ("512760", Category.A_SECTOR, "半导体", 2),
    ("512480", Category.A_SECTOR, "半导体", 1),
    ("515030", Category.A_SECTOR, "新能源车", 1),
    ("515790", Category.A_SECTOR, "光伏", 1),
    ("512690", Category.A_SECTOR, "酒", 1),
    ("512170", Category.A_SECTOR, "医药", 2),
    ("512010", Category.A_SECTOR, "医药", 1),
    ("515050", Category.A_SECTOR, "5G通信", 1),
    ("159928", Category.A_SECTOR, "消费", 1),
    ("512880", Category.A_SECTOR, "证券", 1),
    ("512000", Category.A_SECTOR, "券商", 2),
    ("512800", Category.A_SECTOR, "银行", 1),
    ("515220", Category.A_SECTOR, "煤炭", 1),
    ("512200", Category.A_SECTOR, "地产", 1),
    ("512400", Category.A_SECTOR, "有色金属", 1),
    ("512660", Category.A_SECTOR, "军工", 1),
    ("512980", Category.A_SECTOR, "传媒", 1),
    ("515880", Category.A_SECTOR, "通信", 1),
    ("159996", Category.A_SECTOR, "家电", 1),
    ("512120", Category.A_SECTOR, "化工", 1),
    # 港股 (5) — Hong Kong ETFs
    ("510900", Category.HK, "HSI", 1),
    ("159920", Category.HK, "HSI", 2),
    ("513010", Category.HK, "HKTech", 1),
    ("513050", Category.HK, "中概互联", 1),
    ("159740", Category.HK, "恒生科技", 2),
    # 商品 (6) — commodity ETFs
    ("518880", Category.COMMODITY, "Au", 1),
    ("518800", Category.COMMODITY, "Au", 2),
    ("159985", Category.COMMODITY, "豆粕", 1),
    ("161226", Category.COMMODITY, "Ag", 1),
    ("159981", Category.COMMODITY, "能源化工", 1),
    ("159766", Category.COMMODITY, "有色期货", 1),
    # 海外 (6) — overseas index ETFs
    ("513100", Category.OVERSEAS, "NDX", 1),
    ("513300", Category.OVERSEAS, "NDX", 2),
    ("513500", Category.OVERSEAS, "SPX", 1),
    ("513520", Category.OVERSEAS, "N225", 1),
    ("513880", Category.OVERSEAS, "N225", 2),
    ("159941", Category.OVERSEAS, "SPX", 2),
    # 固收 (1) — bond ETF (80/20 固收+)
    ("511260", Category.BOND, "10Y国债", 1),
]


def _make_default_pool() -> "ETFPool":
    members = tuple(
        ETFMeta(code=c, name=c, category=cat, index_code=idx, liquidity_rank=lr)
        for c, cat, idx, lr in _DEFAULT_ETFS
    )
    return ETFPool(members=members)


class ETFCategorizer:
    """按类别 / 同指数对 ETF code 分类 (CICC 同指数去重 依赖此).

    Usage:
        cat = ETFCategorizer(DEFAULT_POOL)
        best = cat.best_per_index()  # {index_code: [code1, code2, ...]} 按 liquidity_rank 排序
    """

    def __init__(self, pool: "ETFPool") -> None:
        self.pool = pool

    def best_per_index(self) -> dict[str, list[str]]:
        """对每个 index_code, 按 liquidity_rank 升序返回 code 列表."""
        from collections import defaultdict
        idx_codes: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for m in self.pool.members:
            if m.index_code:
                idx_codes[m.index_code].append((m.liquidity_rank, m.code))
        result = {}
        for idx, codes in idx_codes.items():
            codes.sort(key=lambda x: x[0])
            result[idx] = [c for _, c in codes]
        return result

    def categories(self) -> dict[str, list[str]]:
        """返回 {Category.value: [code, ...]}."""
        from collections import defaultdict
        cat_codes: dict[str, list[str]] = defaultdict(list)
        for m in self.pool.members:
            cat_codes[m.category.value].append(m.code)
        return dict(cat_codes)


class ETFPool:
    """静态 ETF 池 + 类别查询."""

    def __init__(self, members: tuple[ETFMeta, ...] = ()) -> None:
        self.members = members
        self._by_code: dict[str, ETFMeta] = {m.code: m for m in members}
        seen = set()
        for m in self.members:
            if m.code in seen:
                raise ValueError(f"ETF code 重复: {m.code}")
            seen.add(m.code)

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(m.code for m in self.members)

    def category_of(self, code: str) -> Category:
        m = self._by_code.get(code)
        if m is None:
            raise KeyError(f"未找到 ETF code: {code}")
        return m.category

    def get(self, code: str) -> ETFMeta:
        """按 code 获取 ETFMeta (KeyError if missing)."""
        m = self._by_code.get(code)
        if m is None:
            raise KeyError(f"未找到 ETF code: {code}")
        return m

    def liquidity_rank_of(self, code: str) -> int:
        """按 code 获取流动性排名 (默认 1)."""
        m = self._by_code.get(code)
        return m.liquidity_rank if m else 1

    def by_category(self, cat: Category) -> tuple[ETFMeta, ...]:
        return tuple(m for m in self.members if m.category == cat)

    def by_index(self, index_code: str) -> tuple[ETFMeta, ...]:
        """按 index_code 获取 ETF 列表 (按 liquidity_rank 排序)."""
        items = [m for m in self.members if m.index_code == index_code]
        items.sort(key=lambda m: m.liquidity_rank)
        return tuple(items)

    def index_of(self, code: str) -> str:
        """根据 ETF code 返回其 index_code."""
        m = self._by_code.get(code)
        if m is None:
            raise KeyError(f"未知 ETF: {code}")
        return m.index_code

    def __len__(self) -> int:
        return len(self.members)

    def __contains__(self, code: str) -> bool:
        return code in self._by_code


DEFAULT_POOL: ETFPool = _make_default_pool()


__all__ = [
    "Category",
    "ETFMeta",
    "ETFCategorizer",
    "ETFPool",
    "DEFAULT_POOL",
]
