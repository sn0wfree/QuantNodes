# coding=utf-8
"""v4 ETF 池 + 风格组定义 (Stage 17).

v4 = 风格轮动 + Smart β + 因子择时.

风格组 (5): 大盘/中盘/成长/科创/红利.
Smart β 工具 (7): 红利低波/低波/质量/价值/现金流/红利100/红利低波100.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Mapping


class StyleGroup(Enum):
    """风格组分类 (5 个)."""
    LARGE_CAP = "large_cap"        # 大盘
    MID_CAP = "mid_cap"            # 中盘
    GROWTH = "growth"              # 成长 (创业板)
    TECH = "tech"                  # 科创
    DIVIDEND = "dividend"          # 红利


class SmartBetaFactor(Enum):
    """Smart β 因子分类 (7 个工具)."""
    DIV_LOW_VOL = "div_low_vol"    # 红利低波 (512890)
    LOW_VOL = "low_vol"            # 300 低波 (512260)
    QUALITY = "quality"            # 质量 (515900)
    VALUE = "value"                # 价值 (512040)
    CASHFLOW = "cashflow"          # 现金流 (159786)
    DIV_100 = "div_100"            # 红利 100 (515080)
    DIV_LOW_VOL_100 = "div_low_vol_100"  # 红利低波 100 (515100)


# 风格组 → 代表 ETF
STYLE_GROUP_CODES: dict[StyleGroup, tuple[str, ...]] = {
    StyleGroup.LARGE_CAP: ("510300",),       # HS300
    StyleGroup.MID_CAP: ("510500",),         # CSI500
    StyleGroup.GROWTH: ("159915",),          # 创业板
    StyleGroup.TECH: ("588000",),            # 科创 50
    StyleGroup.DIVIDEND: ("510880",),        # 华泰柏瑞红利
}


# Smart β 工具 → ETF
SMART_BETA_CODES: dict[SmartBetaFactor, str] = {
    SmartBetaFactor.DIV_LOW_VOL: "512890",
    SmartBetaFactor.LOW_VOL: "512260",
    SmartBetaFactor.QUALITY: "515900",
    SmartBetaFactor.VALUE: "512040",
    SmartBetaFactor.CASHFLOW: "159786",
    SmartBetaFactor.DIV_100: "515080",
    SmartBetaFactor.DIV_LOW_VOL_100: "515100",
}


# Smart β 因子分类: 防御/进攻/价值
SMART_BETA_FACTOR_TYPE: dict[SmartBetaFactor, str] = {
    SmartBetaFactor.DIV_LOW_VOL: "defensive",       # 红利低波
    SmartBetaFactor.LOW_VOL: "defensive",           # 低波
    SmartBetaFactor.QUALITY: "defensive",           # 质量
    SmartBetaFactor.VALUE: "value",                 # 价值
    SmartBetaFactor.CASHFLOW: "value",              # 现金流
    SmartBetaFactor.DIV_100: "defensive",           # 红利 100
    SmartBetaFactor.DIV_LOW_VOL_100: "defensive",   # 红利低波 100
}


@dataclass(frozen=True)
class StyleGroupMeta:
    """风格组元数据."""
    group: StyleGroup
    codes: tuple[str, ...]
    name_cn: str


@dataclass(frozen=True)
class SmartBetaMeta:
    """Smart β 元数据."""
    factor: SmartBetaFactor
    code: str
    name_cn: str
    factor_type: str  # "defensive" | "value" | "growth" | "offensive"


# 风格组元数据
STYLE_GROUP_METAS: dict[StyleGroup, StyleGroupMeta] = {
    StyleGroup.LARGE_CAP: StyleGroupMeta(
        group=StyleGroup.LARGE_CAP,
        codes=("510300",),
        name_cn="大盘 (HS300)",
    ),
    StyleGroup.MID_CAP: StyleGroupMeta(
        group=StyleGroup.MID_CAP,
        codes=("510500",),
        name_cn="中盘 (CSI500)",
    ),
    StyleGroup.GROWTH: StyleGroupMeta(
        group=StyleGroup.GROWTH,
        codes=("159915",),
        name_cn="成长 (创业板)",
    ),
    StyleGroup.TECH: StyleGroupMeta(
        group=StyleGroup.TECH,
        codes=("588000",),
        name_cn="科创 (科创50)",
    ),
    StyleGroup.DIVIDEND: StyleGroupMeta(
        group=StyleGroup.DIVIDEND,
        codes=("510880",),
        name_cn="红利",
    ),
}


# Smart β 元数据
SMART_BETA_METAS: dict[SmartBetaFactor, SmartBetaMeta] = {
    SmartBetaFactor.DIV_LOW_VOL: SmartBetaMeta(
        factor=SmartBetaFactor.DIV_LOW_VOL, code="512890",
        name_cn="红利低波", factor_type="defensive",
    ),
    SmartBetaFactor.LOW_VOL: SmartBetaMeta(
        factor=SmartBetaFactor.LOW_VOL, code="512260",
        name_cn="300 低波", factor_type="defensive",
    ),
    SmartBetaFactor.QUALITY: SmartBetaMeta(
        factor=SmartBetaFactor.QUALITY, code="515900",
        name_cn="中证质量", factor_type="defensive",
    ),
    SmartBetaFactor.VALUE: SmartBetaMeta(
        factor=SmartBetaFactor.VALUE, code="512040",
        name_cn="国泰价值", factor_type="value",
    ),
    SmartBetaFactor.CASHFLOW: SmartBetaMeta(
        factor=SmartBetaFactor.CASHFLOW, code="159786",
        name_cn="现金流", factor_type="value",
    ),
    SmartBetaFactor.DIV_100: SmartBetaMeta(
        factor=SmartBetaFactor.DIV_100, code="515080",
        name_cn="中信红利", factor_type="defensive",
    ),
    SmartBetaFactor.DIV_LOW_VOL_100: SmartBetaMeta(
        factor=SmartBetaFactor.DIV_LOW_VOL_100, code="515100",
        name_cn="红利低波 100", factor_type="defensive",
    ),
}


# 所有 v4 关注的 ETF
ALL_V4_CODES: tuple[str, ...] = (
    "510300", "510500", "159915", "588000", "510880",  # 风格组
    "512890", "512260", "515900", "512040", "159786", "515080", "515100",  # Smart β
)


def all_style_codes() -> tuple[str, ...]:
    """所有风格组 ETF code."""
    out: list[str] = []
    for meta in STYLE_GROUP_METAS.values():
        out.extend(meta.codes)
    return tuple(out)


def all_smart_beta_codes() -> tuple[str, ...]:
    """所有 Smart β 工具 ETF code."""
    return tuple(m.code for m in SMART_BETA_METAS.values())


def style_group_of(code: str) -> StyleGroup | None:
    """反查: ETF code 属于哪个风格组."""
    for group, meta in STYLE_GROUP_METAS.items():
        if code in meta.codes:
            return group
    return None


def smart_beta_of(code: str) -> SmartBetaFactor | None:
    """反查: ETF code 属于哪个 Smart β 因子."""
    for factor, meta in SMART_BETA_METAS.items():
        if meta.code == code:
            return factor
    return None


def load_smartbeta_panel(path: str | Path | None = None) -> "pd.DataFrame":  # type: ignore[name-defined]
    """加载 Smart β ETF 净值面板.

    Args:
        path: parquet 路径. 缺省 = data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet
    """
    import pandas as pd
    if path is None:
        path = Path("data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Smart β 面板未找到: {path}\n"
            f"请先运行: python3.11 scripts/fetch_smartbeta_panel.py"
        )
    df = pd.read_parquet(path)
    return df


def export_style_groups(path: str | Path) -> None:
    """导出风格组定义为 JSON (供脚本加载)."""
    out: dict = {
        "style_groups": {
            g.value: {"name_cn": m.name_cn, "codes": list(m.codes)}
            for g, m in STYLE_GROUP_METAS.items()
        },
        "smart_beta": {
            f.value: {
                "name_cn": m.name_cn,
                "code": m.code,
                "factor_type": m.factor_type,
            }
            for f, m in SMART_BETA_METAS.items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


__all__ = [
    "StyleGroup",
    "SmartBetaFactor",
    "STYLE_GROUP_CODES",
    "SMART_BETA_CODES",
    "SMART_BETA_FACTOR_TYPE",
    "STYLE_GROUP_METAS",
    "SMART_BETA_METAS",
    "ALL_V4_CODES",
    "all_style_codes",
    "all_smart_beta_codes",
    "style_group_of",
    "smart_beta_of",
    "load_smartbeta_panel",
    "export_style_groups",
]
