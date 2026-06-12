# coding: utf-8
"""常量定义 / Constants

H13-H16: 行业/指数/天数支持外部 JSON 覆盖 (默认内置)。
用法:
    from QuantNodes.research.factor_test.utils.constants import load_overrides
    overrides = load_overrides(Path("./my_industry_map.json"))
    ind_map = overrides.get("INDUSTRY_MAP", DEFAULT_INDUSTRY_MAP)
"""

from pathlib import Path
from typing import Any, Optional

import json

# 指数映射 (默认; SZ50 id_50 路由不在 ifind_database 中, 仅作占位)
INDEX_MAPPING = {
    'HS300': ('stk_daily.h5', 'id_300'),
    'ZZ500': ('stk_daily.h5', 'id_500'),
}

# 指数收盘价映射
INDEX_CP_MAPPING = {
    'HS300': '000300.SH',
    'ZZ500': '000905.SH',
}

# 中信行业映射
INDUSTRY_MAPPING = {
    'id_citic1A': 'ind_name_CITIC_1A',
    'id_citic1': 'ind_name_CITIC_1',
}

# 年化天数 (A 股 250; 美股/港股 252; 24h 市场 365)
ANNUAL_DAYS = 250


def load_overrides(path: Optional[Path | str] = None) -> dict[str, Any]:
    """加载外部 JSON 覆盖 (H13-H16)。

    支持覆盖字段:
        - INDUSTRY_MAP: dict[行业 key, 显示名 key]
        - INDEX_MAPPING: dict[指数名, (h5, key)]
        - INDEX_CP_MAPPING: dict[指数名, 代码]
        - ANNUAL_DAYS: int

    Args:
        path: JSON 路径, None 或文件不存在 → 返回空 dict (用全部默认)

    Returns:
        dict: 覆盖项 (子集, 只含实际有 override 的字段)
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def resolve_industry_map(overrides: dict[str, Any] | None = None) -> dict[str, str]:
    """解析 INDUSTRY_MAP (含 override 合并)。"""
    base = dict(INDUSTRY_MAPPING)
    if overrides and "INDUSTRY_MAP" in overrides:
        base.update(overrides["INDUSTRY_MAP"])
    return base


def resolve_index_mapping(overrides: dict[str, Any] | None = None) -> dict[str, tuple]:
    """解析 INDEX_MAPPING (含 override 合并)。"""
    base = {k: tuple(v) for k, v in INDEX_MAPPING.items()}
    if overrides and "INDEX_MAPPING" in overrides:
        for k, v in overrides["INDEX_MAPPING"].items():
            base[k] = tuple(v)
    return base


def resolve_annual_days(overrides: dict[str, Any] | None = None) -> int:
    """解析 ANNUAL_DAYS (含 override)。"""
    if overrides and "ANNUAL_DAYS" in overrides:
        return int(overrides["ANNUAL_DAYS"])
    return ANNUAL_DAYS
