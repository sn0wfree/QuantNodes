"""QuantNodes 核心常量 — 跨模块统一的常量定义。

修复 4 路常量漂移 (H7+H8):
  - BASE_FEATURE_NAMES: feedback/channels.py 与 quality_gate/complexity.py 漂移
  - METRIC_KEYS: feedback/dataclass + trajectory/entry + trajectory/pool 3 处定义
  - PARQUET_COLUMNS: trajectory/pool 14 字段, 复用 METRIC_KEYS + 6 业务字段
"""
from __future__ import annotations

from typing import FrozenSet, Tuple


# ============================================================================
# H7: 基础特征名 (统一 feedback + quality_gate)
# ============================================================================
# 包含 13 个核心字段 + 2 个扩展字段 (industry, cap) = 15 个
# 注: 此前 feedback/channels.py 缺 industry/cap, quality_gate/complexity.py 含全部 15
BASE_FEATURE_NAMES: FrozenSet[str] = frozenset({
    "open", "high", "low", "close", "volume", "amount",
    "vwap", "turnover", "mv_float", "total_mv", "circ_mv",
    "returns", "vwap_adj",
    "industry", "cap",  # 扩展字段
})


# ============================================================================
# H8: Metric 键 (统一 4 处定义)
# ============================================================================
# 6 个标准指标 — 出现在 Parquet schema + entry.metrics 序列化
METRIC_KEYS: Tuple[str, ...] = (
    "ic_mean", "rank_ic_mean", "sharpe", "arr", "mdd", "calmar",
)

# 9 个扩展指标 — 出现在 FactorFeedback.metadata (来自 result dict)
EXTENDED_METRIC_KEYS: Tuple[str, ...] = (
    "ic", "rank_ic", "sharpe", "arr", "mdd", "calmar",
    "turnover", "win_rate", "ic_ir",
)


# ============================================================================
# TrajectoryPool Parquet 业务列 (与 METRIC_KEYS 集成)
# ============================================================================
PARQUET_COLUMNS: Tuple[str, ...] = (
    "entry_id", "round_idx", "operation", "parent_ids",
    "decision", "duration_ms", "timestamp", "factor_name", "summary",
) + METRIC_KEYS
