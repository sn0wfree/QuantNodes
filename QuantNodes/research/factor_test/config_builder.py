# coding: utf-8
"""SingleFactorTestConfig 流式构造器 / Config Builder (Phase 3.4)

``SingleFactorTestConfig`` 是深度嵌套的 pydantic 模型 (factor / preprocess /
analysis{ic,group,longshort,score,risk_corr} / output / feedback /
quality_gate / evolution)。手工构造冗长易错 (e.g.
``run_evolution_e2e._build_config`` 37 行嵌套)。

``SingleFactorTestConfigBuilder`` 提供流式 API 扁平化常用字段:

    cfg = (
        SingleFactorTestConfigBuilder()
        .factor("mom20", "mom20.h5", hypothesis="momentum")
        .dates(20240101, 20241231)
        .preprocess(extreme="median", norm="zscore")
        .groups(5, direction=1)
        .output("./out/", fmt=["json"])
        .data_path("./data/")
        .build()
    )

设计要点:
  - 每个 setter 返回 ``self`` 支持链式
  - 默认值全部委托各 ``*Setting`` 的 pydantic 默认 (单一真值源, 不重复)
  - ``build()`` 时统一构造 pydantic 模型并触发校验 (缺必填如 factor 即报错)
  - 不改 ``SingleFactorTestConfig`` / ``config.py``; 现有直接构造方式不变
"""
from __future__ import annotations

from typing import Any, Optional

from QuantNodes.research.factor_test.config import (
    AnalysisSetting,
    EvolutionConfig,
    FactorSetting,
    FeedbackSetting,
    GroupSetting,
    ICSetting,
    LongShortSetting,
    OutputSetting,
    PreprocessSetting,
    QualityGateConfig,
    RiskCorrelationSetting,
    ScoreSetting,
    SingleFactorTestConfig,
    TradableSetting,
)


def _clean(kwargs: dict) -> dict:
    """剔除值为 None 的项, 让 pydantic 默认值生效 (单一真值源)。"""
    return {k: v for k, v in kwargs.items() if v is not None}


class SingleFactorTestConfigBuilder:
    """流式构造 ``SingleFactorTestConfig``。

    所有 setter 仅记录非 None 字段; ``build()`` 时聚合为 pydantic 模型,
    缺省字段沿用各 ``*Setting`` 的默认值。
    """

    def __init__(self) -> None:
        self._factor: Optional[dict] = None
        self._preprocess: dict = {}
        self._tradable: Optional[dict] = None
        self._ic: dict = {}
        self._group: dict = {}
        self._longshort: dict = {}
        self._score: dict = {}
        self._risk_corr: dict = {}
        self._output: dict = {}
        self._feedback: dict = {}
        self._quality_gate: dict = {}
        self._evolution: dict = {}
        self._top: dict = {}

    # ── factor ────────────────────────────────────────────────
    def factor(
        self,
        name: str,
        factor_dir: str,
        *,
        factor_key: Optional[str] = None,
        fmt: Optional[str] = None,
        hypothesis: Optional[str] = None,
        description: Optional[str] = None,
        expression: Optional[str] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._factor = _clean({
            "name": name,
            "factor_dir": factor_dir,
            "factor_key": factor_key,
            "format": fmt,
            "hypothesis": hypothesis,
            "description": description,
            "expression": expression,
        })
        return self

    # ── preprocess ────────────────────────────────────────────
    def dates(self, beg: int, end: int) -> "SingleFactorTestConfigBuilder":
        self._preprocess["adj_date_beg"] = beg
        self._preprocess["adj_date_end"] = end
        return self

    def adj_mode(self, mode: list) -> "SingleFactorTestConfigBuilder":
        self._preprocess["adj_mode"] = mode
        return self

    def sample(
        self,
        index: Optional[str] = None,
        industry: Optional[str] = None,
        *,
        index_customdir: Optional[tuple] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._preprocess.update(_clean({
            "sample_index": index,
            "sample_industry": industry,
            "sample_index_customdir": index_customdir,
        }))
        return self

    def preprocess(
        self,
        *,
        missing: Optional[str] = None,
        extreme: Optional[str] = None,
        norm: Optional[str] = None,
        mad_n: Optional[float] = None,
        pct_low: Optional[float] = None,
        pct_high: Optional[float] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._preprocess.update(_clean({
            "missing": missing,
            "extreme": extreme,
            "norm": norm,
            "mad_n": mad_n,
            "pct_low": pct_low,
            "pct_high": pct_high,
        }))
        return self

    def neutralize(
        self,
        *,
        industry: Optional[bool] = None,
        risk: Optional[bool] = None,
        risk_factors: Optional[list] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._preprocess.update(_clean({
            "industry_neutral": industry,
            "risk_neutral": risk,
            "risk_factors": risk_factors,
        }))
        return self

    def tradable(self, **kwargs: Any) -> "SingleFactorTestConfigBuilder":
        self._tradable = dict(kwargs)
        return self

    # ── analysis ──────────────────────────────────────────────
    def ic(self, *, min_group_size: Optional[int] = None) -> "SingleFactorTestConfigBuilder":
        self._ic.update(_clean({"min_group_size": min_group_size}))
        return self

    def groups(
        self,
        n: Optional[int] = None,
        *,
        direction: Optional[int] = None,
        floor_mode: Optional[str] = None,
        hedge: Optional[str] = None,
        hedge_path: Optional[str] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._group.update(_clean({
            "groups": n,
            "factor_direction": direction,
            "floor_mode": floor_mode,
            "hedge": hedge,
            "hedge_path": hedge_path,
        }))
        return self

    def longshort(self, *, direction: Optional[int] = None) -> "SingleFactorTestConfigBuilder":
        self._longshort.update(_clean({"factor_direction": direction}))
        return self

    def score(
        self,
        *,
        enabled: Optional[bool] = None,
        n_industries: Optional[int] = None,
        n_size_groups: Optional[int] = None,
        n_quantile_groups: Optional[int] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._score.update(_clean({
            "enabled": enabled,
            "n_industries": n_industries,
            "n_size_groups": n_size_groups,
            "n_quantile_groups": n_quantile_groups,
        }))
        return self

    def risk_corr(self, factors: Any = None) -> "SingleFactorTestConfigBuilder":
        if factors is not None:
            self._risk_corr["factors"] = factors
        return self

    # ── output / feedback / quality_gate / evolution ─────────
    def output(
        self,
        dir: Optional[str] = None,
        *,
        fmt: Optional[list] = None,
    ) -> "SingleFactorTestConfigBuilder":
        self._output.update(_clean({"dir": dir, "format": fmt}))
        return self

    def feedback(self, **kwargs: Any) -> "SingleFactorTestConfigBuilder":
        self._feedback.update(kwargs)
        return self

    def quality_gate(self, **kwargs: Any) -> "SingleFactorTestConfigBuilder":
        self._quality_gate.update(kwargs)
        return self

    def evolution(self, **kwargs: Any) -> "SingleFactorTestConfigBuilder":
        self._evolution.update(kwargs)
        return self

    # ── top-level ─────────────────────────────────────────────
    def data_path(self, path: str) -> "SingleFactorTestConfigBuilder":
        self._top["data_path"] = path
        return self

    def load_keys(self, keys: list) -> "SingleFactorTestConfigBuilder":
        self._top["load_keys"] = keys
        return self

    # ── build ─────────────────────────────────────────────────
    def build(self) -> SingleFactorTestConfig:
        """聚合并构造 SingleFactorTestConfig (触发 pydantic 校验)。

        Raises:
            ValueError: 未设置 factor (name/factor_dir 必填)。
            pydantic.ValidationError: 字段校验失败。
        """
        if self._factor is None:
            raise ValueError(
                "factor is required: call .factor(name, factor_dir) before build()"
            )

        preprocess = dict(self._preprocess)
        if self._tradable is not None:
            preprocess["tradable"] = TradableSetting(**self._tradable)

        analysis = AnalysisSetting(
            ic=ICSetting(**self._ic),
            group=GroupSetting(**self._group),
            longshort=LongShortSetting(**self._longshort),
            score=ScoreSetting(**self._score),
            risk_corr=RiskCorrelationSetting(**self._risk_corr),
        )

        kwargs: dict = {
            "factor": FactorSetting(**self._factor),
            "preprocess": PreprocessSetting(**preprocess),
            "analysis": analysis,
            "output": OutputSetting(**self._output),
            "feedback": FeedbackSetting(**self._feedback),
            "quality_gate": QualityGateConfig(**self._quality_gate),
            "evolution": EvolutionConfig(**self._evolution),
        }
        kwargs.update(self._top)
        return SingleFactorTestConfig(**kwargs)
