# coding: utf-8
"""12 节点 Pydantic 配置模型 / Node Configuration Models

Phase 3.1 T0: 12 节点的 __init__ 改 Union 接受 (dict / *Config / None),
内部 model_validate() 校验, 保留 self._xxx 实例属性 (向后兼容 5 处测试).

拼写错 (extra="forbid") 立即 ValidationError, 跨进程 model_dump() → dict 链稳定.

Phase R3-A (2026-06-19): Schema 收敛
- 8 节点 Config 直接继承 config.py 子模型 (避免字段重复定义)
- 4 节点 (LoadData/AdjustDate/SamplePool/Report) 字段语义不同, 保留独立定义
- 单一真值源: 改默认值只改 config.py 一处即可同步到 nodes/configs.py
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from QuantNodes.research.factor_test.config import (
    FactorSetting,
    GroupSetting,
    ICSetting,
    LongShortSetting,
    PreprocessSetting,
    RiskCorrelationSetting,
    ScoreSetting,
    TradableSetting,
)


_FORBID = ConfigDict(extra="forbid")


class _NodeBase(BaseModel):
    """12 节点 Config 公共基类
    - extra="forbid": 拼写错立即 ValidationError (新防线)
    """
    model_config = _FORBID


class LoadDataNodeConfig(_NodeBase):
    """Node 1: LoadDataNode 配置

    P-2: data_path 改 Field(...) 必填, 启动时报 ValidationError (None/缺字段).
    现有 test_data_loader_edges.py:309-316 已显式提供 load_keys 但缺 data_path,
    需更新为提供 data_path (load_keys 测试不变).
    """
    data_path: str = Field(..., description="数据根目录 (P-2 必填, 启动校验)")
    load_keys: list = Field(
        default_factory=lambda: [
            "stklist", "trade_dt", "cp", "id_citic1", "mv_float",
            "st", "suspend", "ud_limit", "ipo_days",
        ],
        description="需要加载的数据 key 列表 (M7 默认含 tradability 必需键)",
    )
    factor: Optional[FactorSetting] = Field(
        default=None, description="因子配置 (None=不加载因子)",
    )


class SamplePoolNodeConfig(_NodeBase):
    """Node 2: SamplePoolFilterNode 配置

    M9: index_mapping 可自定义，合并全局默认 INDEX_MAPPING。
    M12: i18n_name_map 可自定义行业代码→名称映射，覆盖全局默认 INDUSTRY_MAPPING。

    字段语义独立于 PreprocessSetting (sample_index/sample_industry 是节点专属),
    保留独立定义.
    """
    sample_index: str = Field(
        default="all", description="样本池: all/HS300/ZZ500/ZZ800/custom",
    )
    sample_industry: str = Field(
        default="all", description="行业筛选: all/中信行业名",
    )
    sample_index_customdir: Optional[tuple] = Field(
        default=None, description="自定义样本池路径 (sample_index=custom 时必填)",
    )
    index_mapping: Optional[dict[str, tuple[str, str]]] = Field(
        default=None, description="自定义指数映射 (覆盖全局默认 INDEX_MAPPING)",
    )
    i18n_name_map: Optional[dict[str, str]] = Field(
        default=None, description="自定义行业代码→名称映射 (覆盖全局默认 INDUSTRY_MAPPING)",
    )


class TradabilityNodeConfig(_NodeBase):
    """Node 3: TradabilityFilterNode 配置"""
    tradable: TradableSetting = Field(
        default_factory=TradableSetting,
        description="可交易性配置 (no_st/no_suspended/no_up_down_limit/min_ipo_days/trace)",
    )


class AdjustDateNodeConfig(_NodeBase):
    """Node 4: AdjustDateNode 配置

    H10 兼容: adj_date_beg/end 默认 None → _execute 启动校验抛 ValueError,
    避免静默跑废日期范围.

    字段语义独立 (adj_date_beg/end 在 PreprocessSetting 中是预处理日期窗口,
    在节点中是调仓日生成范围), 保留独立定义.
    """
    adj_date_beg: Optional[int] = Field(
        default=None, description="起始日期 yyyymmdd (None → 启动报错)",
    )
    adj_date_end: Optional[int] = Field(
        default=None, description="截止日期 yyyymmdd (None → 启动报错)",
    )
    adj_mode: list = Field(
        default_factory=lambda: ["M", "end"],
        description="调仓模式: [mode, position], mode=M/W/Q/D, position=end/start",
    )


class PreprocessNodeConfig(BaseModel):
    """Node 5: FactorPreprocessNode 配置 (R3-A: 继承 config.py::PreprocessSetting 子集字段).

    字段全部从 PreprocessSetting 继承 (missing/extreme/norm/mad_n/pct_low/pct_high/i18n_name_map),
    通过 model_dump(include={...}) 切片传入。改 mad_n 默认值只需改 PreprocessSetting 一处.
    """
    model_config = _FORBID
    missing: str = PreprocessSetting.model_fields["missing"]
    extreme: str = PreprocessSetting.model_fields["extreme"]
    norm: str = PreprocessSetting.model_fields["norm"]
    mad_n: float = PreprocessSetting.model_fields["mad_n"]
    pct_low: float = PreprocessSetting.model_fields["pct_low"]
    pct_high: float = PreprocessSetting.model_fields["pct_high"]
    i18n_name_map: Optional[dict[str, str]] = PreprocessSetting.model_fields["i18n_name_map"]


class NeutralizeNodeConfig(BaseModel):
    """Node 6: FactorNeutralizeNode 配置 (R3-A: 继承 PreprocessSetting 中性化字段)."""
    model_config = _FORBID
    industry_neutral: bool = PreprocessSetting.model_fields["industry_neutral"]
    risk_neutral: bool = PreprocessSetting.model_fields["risk_neutral"]
    risk_factors: list = PreprocessSetting.model_fields["risk_factors"]


class ICAnalyzerNodeConfig(ICSetting):
    """Node 7: ICAnalyzerNode 配置 (R3-A: 继承 ICSetting)."""
    model_config = _FORBID


class GroupAnalyzerNodeConfig(GroupSetting):
    """Node 8: GroupAnalyzerNode 配置 (R3-A: 继承 GroupSetting)."""
    model_config = _FORBID


class LongShortNodeConfig(LongShortSetting):
    """Node 9: LongShortNode 配置 (R3-A: 继承 LongShortSetting)."""
    model_config = _FORBID


class ScoreNodeConfig(ScoreSetting):
    """Node 10: FactorScoreNode 配置 (R3-A: 继承 ScoreSetting).

    T0-2: 4 字段 (enabled/n_industries=29/n_size_groups=3/n_quantile_groups=5)
    全部继承自 ScoreSetting.
    """
    model_config = _FORBID


class RiskCorrelationNodeConfig(RiskCorrelationSetting):
    """Node 11: RiskCorrelationNode 配置 (R3-A: 继承 RiskCorrelationSetting)."""
    model_config = _FORBID


class ReportNodeConfig(_NodeBase):
    """Node 12: FactorTestReportNode 配置

    P-1: dir 路径优先级 env QUANTNODES_OUTPUT_DIR > expanduser > default='./output/'.

    字段语义独立 (OutputSetting.dir vs ReportNodeConfig.dir 同名但生命周期不同),
    保留独立定义.
    """
    dir: str = Field(
        default="./output/", description="输出目录 (P-1: env > expanduser > default)",
    )
    format: list = Field(
        default_factory=lambda: ["parquet", "json"],
        description="输出格式",
    )


# ── 节点名 → Config Schema 路由表 ────────────────────────────
NODE_CONFIG_SCHEMAS: dict[str, type[BaseModel]] = {
    "LoadData": LoadDataNodeConfig,
    "SamplePoolFilter": SamplePoolNodeConfig,
    "TradabilityFilter": TradabilityNodeConfig,
    "AdjustDate": AdjustDateNodeConfig,
    "FactorPreprocess": PreprocessNodeConfig,
    "FactorNeutralize": NeutralizeNodeConfig,
    "ICAnalyzer": ICAnalyzerNodeConfig,
    "GroupAnalyzer": GroupAnalyzerNodeConfig,
    "LongShort": LongShortNodeConfig,
    "FactorScore": ScoreNodeConfig,
    "RiskCorrelation": RiskCorrelationNodeConfig,
    "FactorTestReport": ReportNodeConfig,
}
