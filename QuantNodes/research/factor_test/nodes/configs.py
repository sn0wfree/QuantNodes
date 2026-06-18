# coding: utf-8
"""12 节点 Pydantic 配置模型 / Node Configuration Models

Phase 3.1 T0: 12 节点的 __init__ 改 Union 接受 (dict / *Config / None),
内部 model_validate() 校验, 保留 self._xxx 实例属性 (向后兼容 5 处测试).

拼写错 (extra="forbid") 立即 ValidationError, 跨进程 model_dump() → dict 链稳定.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from QuantNodes.research.factor_test.config import (
    FactorSetting, TradableSetting,
)


class _NodeBase(BaseModel):
    """12 节点 Config 公共基类
    - extra="forbid": 拼写错立即 ValidationError (新防线)
    """
    model_config = ConfigDict(extra="forbid")


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


class PreprocessNodeConfig(_NodeBase):
    """Node 5: FactorPreprocessNode 配置

    T0-2: 6 隐式默认从节点 __init__ 提升到 Pydantic 字段
    (mad_n=5.0, pct_low=0.025, pct_high=0.975).
    """
    missing: str = Field(default="", description="缺失值处理: ''/ind_avg")
    extreme: str = Field(default="", description="去极值: ''/median/pct_shrink")
    norm: str = Field(default="", description="标准化: ''/zscore/norm")
    mad_n: float = Field(default=5.0, description="median winsorize 倍数 (M5)")
    pct_low: float = Field(default=0.025, description="pct_shrink 下分位 (M5)")
    pct_high: float = Field(default=0.975, description="pct_shrink 上分位 (M5)")
    i18n_name_map: Optional[dict[str, str]] = Field(
        default=None,
        description="自定义行业代码→名称映射 (覆盖全局 INDUSTRY_MAPPING)",
    )


class NeutralizeNodeConfig(_NodeBase):
    """Node 6: FactorNeutralizeNode 配置"""
    industry_neutral: bool = Field(default=False, description="行业中性化")
    risk_neutral: bool = Field(default=False, description="风险因子中性化")
    risk_factors: list = Field(
        default_factory=list, description="风险因子: [(file, key), ...]",
    )


class ICAnalyzerNodeConfig(_NodeBase):
    """Node 7: ICAnalyzerNode 配置"""
    min_group_size: int = Field(
        default=5, description="计算 IC 最少需要的因子值数量",
    )


class GroupAnalyzerNodeConfig(_NodeBase):
    """Node 8: GroupAnalyzerNode 配置"""
    groups: int = Field(default=5, description="分组数")
    factor_direction: int = Field(
        default=1, description="因子方向: 1=越大越好, -1=越小越好",
    )
    floor_mode: str = Field(
        default="group", description="数据不足策略: group=跳过, last=沿用上期",
    )
    hedge: str = Field(
        default="equal", description="对冲基准: HS300/ZZ500/equal/custom",
    )
    hedge_path: Optional[str] = Field(
        default=None, description="自定义对冲基准路径 (hedge=custom 时必填)",
    )


class LongShortNodeConfig(_NodeBase):
    """Node 9: LongShortNode 配置"""
    factor_direction: int = Field(
        default=1, description="因子方向: 1=越大越好, -1=越小越好",
    )


class ScoreNodeConfig(_NodeBase):
    """Node 10: FactorScoreNode 配置

    T0-2: 3 隐式默认补 Pydantic 字段 (n_industries=29, n_size_groups=3, n_quantile_groups=5).
    """
    enabled: bool = Field(default=True, description="是否运行 (False=跳过)")
    n_industries: int = Field(
        default=29, description="行业数 (中信 29 / 申万 30, H15)",
    )
    n_size_groups: int = Field(
        default=3, description="市值分组数 (H15)",
    )
    n_quantile_groups: int = Field(
        default=5, description="因子分位数 (H15)",
    )


class RiskCorrelationNodeConfig(_NodeBase):
    """Node 11: RiskCorrelationNode 配置"""
    factors: str = Field(
        default="all", description="风险因子: 'all' 或 [(file, key), ...]",
    )


class ReportNodeConfig(_NodeBase):
    """Node 12: FactorTestReportNode 配置

    P-1: dir 路径优先级 env QUANTNODES_OUTPUT_DIR > expanduser > default='./output/'.
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
