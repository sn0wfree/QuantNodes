# coding=utf-8
"""v9 CPD — Cycle Position Diagnosis 周期诊断模块.

CPD 回答核心问题:
    1. 当前经济处于哪个周期 (美林时钟 4 阶段)
    2. 当前市场在 10 年周期中的位置 (Pring 季节性)
    3. 多周期叠加的当前相位 (Kitchin/Juglar/Kuznets/Kondratieff)
    4. 综合评分与大盘择时信号 (0-100 + 0/1)

模块结构:
    merrill_clock    美林时钟 4 阶段识别 (Recovery/Overheat/Stagflation/Recession)
    pring_cycles     Pring 10 年周期 + 多周期叠加
    cycle_position   综合定位 (CycleState 数据类)
    diagnose         报告生成 (Markdown + HTML 仪表盘)

设计文档:
    docs/49a-v9_cycle_diagnosis.md  (CPD 框架)
    docs/49-v9_cycle_timing.md      (主文档)
    docs/50-v9_current_cycle_state.md (当前状态报告模板)
"""
from .merrill_clock import (
    detect_merrill_phase,
    detect_merrill_phase_with_confidence,
    MERRILL_PHASE_NAMES,
    MERRILL_PHASE_NAMES_CN,
    MERRILL_PHASE_NUM,
)
from .pring_cycles import (
    pring_decennial_position,
    pring_decennial_seasonality,
    multi_cycle_position,
    PRING_SEASONALITY,
)
from .cycle_position import (
    CycleState,
    diagnose_current_state,
)
from .diagnose import (
    generate_markdown_report,
    generate_html_dashboard,
)

__all__ = [
    "detect_merrill_phase",
    "detect_merrill_phase_with_confidence",
    "MERRILL_PHASE_NAMES",
    "MERRILL_PHASE_NAMES_CN",
    "MERRILL_PHASE_NUM",
    "pring_decennial_position",
    "pring_decennial_seasonality",
    "multi_cycle_position",
    "PRING_SEASONALITY",
    "CycleState",
    "diagnose_current_state",
    "generate_markdown_report",
    "generate_html_dashboard",
]
