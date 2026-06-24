# coding=utf-8
"""
因子挖掘器 - 模板枚举 + 公式生成

基于预定义模板库，系统性生成候选因子公式。
支持4大类因子族：动量、均值回归、波动率、量价。

⚠️ DeprecationWarning (v2.7.0+, since 2026-06-23):
    本模块进入 deprecation 周期。新代码请迁移到
    `QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab`。

    迁移理由：
    - 模板硬编码 10 个算子 → 162 算子动态查询
    - 公式生成器与评估器解耦更清晰
    - 支持 LLM 友好的元数据（Alpha-GPT 路线需要）

    Phase 时间表：
    - Phase A (current): 本文件仍可用，行为完全兼容
    - Phase B (v2.9+): 本类变 thin wrapper
    - Phase C (v3.0): 归档到 _legacy_3c/
"""

from __future__ import annotations

import hashlib
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from QuantNodes.research.wiki import FactorCategory

warnings.warn(
    "QuantNodes.research.factor_miner 已弃用 (DeprecationWarning)。"
    "请迁移到 QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab。",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class FactorCandidate:
    """候选因子"""
    name: str
    formula: str
    description: str
    operators_used: List[str]
    category: FactorCategory
    template_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateEntry:
    """模板条目"""
    formula_pattern: str
    description_pattern: str
    required_ops: List[str]
    n_cols: int  # 需要的输入列数 (1 或 2)


# 模板库
TEMPLATES: Dict[str, Dict[str, Any]] = {
    "momentum": {
        "category": FactorCategory.MOMENTUM,
        "description": "动量因子",
        "entries": [
            TemplateEntry(
                formula_pattern="ts_delta({col}, {w})",
                description_pattern="{col} 的 {w} 期动量",
                required_ops=["ts_delta"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="ts_pct_change({col}, {w})",
                description_pattern="{col} 的 {w} 期涨跌幅",
                required_ops=["ts_pct_change"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="ts_mean({col}, {w}) / ts_std({col}, {w})",
                description_pattern="{col} 的 {w} 期夏普比",
                required_ops=["ts_mean", "ts_std"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="{col} / ts_lag({col}, {w}) - 1",
                description_pattern="{col} 的 {w} 期收益率",
                required_ops=["ts_lag"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="rank(ts_delta({col}, {w}))",
                description_pattern="{col} 的 {w} 期动量排名",
                required_ops=["rank", "ts_delta"],
                n_cols=1,
            ),
        ],
    },
    "mean_reversion": {
        "category": FactorCategory.OTHER,
        "description": "均值回归因子",
        "entries": [
            TemplateEntry(
                formula_pattern="({col} - ts_mean({col}, {w})) / ts_std({col}, {w})",
                description_pattern="{col} 的 {w} 期 Z-Score",
                required_ops=["ts_mean", "ts_std"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="{col} / ts_mean({col}, {w}) - 1",
                description_pattern="{col} 相对 {w} 期均值偏离",
                required_ops=["ts_mean"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="rank({col} / ts_mean({col}, {w}) - 1)",
                description_pattern="{col} 的均值回归排名",
                required_ops=["rank", "ts_mean"],
                n_cols=1,
            ),
        ],
    },
    "volatility": {
        "category": FactorCategory.VOLATILITY,
        "description": "波动率因子",
        "entries": [
            TemplateEntry(
                formula_pattern="ts_std({col}, {w})",
                description_pattern="{col} 的 {w} 期波动率",
                required_ops=["ts_std"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="ts_std({col}, {w}) / ts_mean({col}, {w})",
                description_pattern="{col} 的 {w} 期变异系数",
                required_ops=["ts_std", "ts_mean"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="ts_max({col}, {w}) - ts_min({col}, {w})",
                description_pattern="{col} 的 {w} 期振幅",
                required_ops=["ts_max", "ts_min"],
                n_cols=1,
            ),
            TemplateEntry(
                formula_pattern="rank(ts_std({col}, {w}))",
                description_pattern="{col} 的波动率排名",
                required_ops=["rank", "ts_std"],
                n_cols=1,
            ),
        ],
    },
    "volume_price": {
        "category": FactorCategory.OTHER,
        "description": "量价因子",
        "entries": [
            TemplateEntry(
                formula_pattern="ts_corr({col1}, {col2}, {w})",
                description_pattern="{col1} 与 {col2} 的 {w} 期相关性",
                required_ops=["ts_corr"],
                n_cols=2,
            ),
            TemplateEntry(
                formula_pattern="ts_cov({col1}, {col2}, {w})",
                description_pattern="{col1} 与 {col2} 的 {w} 期协方差",
                required_ops=["ts_cov"],
                n_cols=2,
            ),
            TemplateEntry(
                formula_pattern="rank(ts_corr({col1}, {col2}, {w}))",
                description_pattern="{col1} 与 {col2} 的相关性排名",
                required_ops=["rank", "ts_corr"],
                n_cols=2,
            ),
        ],
    },
}

# 单列输入组合
SINGLE_COL_COMBOS = [
    (["close"],),
    (["open"],),
    (["high"],),
    (["low"],),
    (["vol"],),
]

# 双列输入组合
DUAL_COL_COMBOS = [
    (["close", "vol"],),
    (["close", "open"],),
    (["high", "low"],),
    (["close", "high"],),
    (["vol", "close"],),
]

# 默认窗口期
DEFAULT_WINDOWS = [5, 10, 20, 60]


def _make_factor_name(formula: str) -> str:
    """根据公式生成确定性因子名"""
    h = hashlib.md5(formula.encode()).hexdigest()[:8]
    return f"auto_{h}"


class FactorMiner:
    """模板因子挖掘器"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate(
        self,
        available_columns: List[str],
        config: Any = None,
    ) -> List[FactorCandidate]:
        """生成候选因子列表

        Args:
            available_columns: 数据中可用的列名
            config: MiningConfig (可选)

        Returns:
            候选因子列表
        """
        windows = getattr(config, "windows", DEFAULT_WINDOWS) if config else DEFAULT_WINDOWS
        categories = getattr(config, "template_categories", None) if config else None
        max_factors = getattr(config, "max_factors", 100) if config else 100

        candidates = []

        for template_name, template_group in TEMPLATES.items():
            if categories and template_name not in categories:
                continue

            category = template_group["category"]
            entries = template_group["entries"]

            for entry in entries:
                col_combos = (
                    SINGLE_COL_COMBOS if entry.n_cols == 1 else DUAL_COL_COMBOS
                )

                for combo in col_combos:
                    cols = combo[0]

                    # 检查列是否可用
                    if not all(c in available_columns for c in cols):
                        continue

                    for w in windows:
                        formula, desc = self._fill_template(
                            entry, cols, w
                        )

                        candidate = FactorCandidate(
                            name=_make_factor_name(formula),
                            formula=formula,
                            description=desc,
                            operators_used=list(entry.required_ops),
                            category=category,
                            template_name=template_name,
                            metadata={"window": w, "columns": cols},
                        )
                        candidates.append(candidate)

        # 去重 (同一公式可能通过不同路径生成)
        seen = set()
        unique = []
        for c in candidates:
            if c.formula not in seen:
                seen.add(c.formula)
                unique.append(c)

        # 限制数量
        if len(unique) > max_factors:
            unique = self.rng.sample(unique, max_factors)

        return unique

    def _fill_template(
        self, entry: TemplateEntry, cols: List[str], window: int
    ) -> Tuple[str, str]:
        """填充模板"""
        placeholders = {
            "col": cols[0],
            "w": str(window),
        }
        if entry.n_cols == 2 and len(cols) >= 2:
            placeholders["col1"] = cols[0]
            placeholders["col2"] = cols[1]

        formula = entry.formula_pattern
        desc = entry.description_pattern
        for k, v in placeholders.items():
            formula = formula.replace("{" + k + "}", v)
            desc = desc.replace("{" + k + "}", v)

        return formula, desc

    def generate_single(
        self,
        formula: str,
        description: str = "",
        operators_used: Optional[List[str]] = None,
        category: FactorCategory = FactorCategory.OTHER,
    ) -> FactorCandidate:
        """手动创建单个候选因子"""
        return FactorCandidate(
            name=_make_factor_name(formula),
            formula=formula,
            description=description or f"手动因子: {formula}",
            operators_used=operators_used or [],
            category=category,
            template_name="manual",
        )
