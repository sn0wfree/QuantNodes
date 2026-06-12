# coding: utf-8
"""Pydantic 配置模型 / Configuration Models"""

from pydantic import BaseModel, Field
from typing import Optional


class FactorSetting(BaseModel):
    """因子配置"""
    name: str = Field(..., description="因子名称")
    factor_dir: str = Field(..., description="因子文件路径")
    factor_key: str = Field(default='', description="H5 key (如因子在 H5 中)")
    format: str = Field(default='h5', description="数据格式: h5/csv/npy/parquet")
    hypothesis: str = Field(default='', description="研究假设 (供 LLM 一致性检查)")
    description: str = Field(default='', description="因子描述 (供 LLM 一致性检查)")
    expression: str = Field(default='', description="代码表达式 (供 LLM 一致性检查)")


class TradableSetting(BaseModel):
    """可交易性配置"""
    no_st: bool = Field(default=True, description="剔除 ST")
    no_suspended: bool = Field(default=True, description="剔除停牌")
    no_up_down_limit: bool = Field(default=False, description="剔除涨跌停")
    min_ipo_days: int = Field(default=360, description="剔除上市不足 N 日的新股")
    trace: Optional[dict] = Field(default=None, description="追踪条件, e.g. {'suspend': (25, 1)}")


class PreprocessSetting(BaseModel):
    """预处理配置"""
    adj_date_beg: int = Field(..., description="起始日期 yyyymmdd")
    adj_date_end: int = Field(..., description="截止日期 yyyymmdd")
    adj_mode: list = Field(default=['M', 'end'], description="调仓模式: [mode, position]")
    sample_index: str = Field(default='all', description="样本池: all/HS300/ZZ500/ZZ800/custom")
    sample_index_customdir: Optional[tuple] = Field(default=None, description="自定义样本池路径")
    sample_industry: str = Field(default='all', description="行业筛选: all/中信行业名")
    tradable: TradableSetting = Field(default_factory=TradableSetting)
    missing: str = Field(default='', description="缺失值处理: ''/ind_avg")
    extreme: str = Field(default='', description="去极值: ''/median/pct_shrink")
    norm: str = Field(default='', description="标准化: ''/zscore/norm")
    industry_neutral: bool = Field(default=False, description="行业中性化")
    risk_neutral: bool = Field(default=False, description="风险因子中性化")
    risk_factors: list = Field(default_factory=list, description="风险因子: [(file, key), ...]")


class ICSetting(BaseModel):
    """IC 分析配置"""
    min_group_size: int = Field(default=5, description="计算 IC 最少需要的因子值数量")


class GroupSetting(BaseModel):
    """分组分析配置"""
    groups: int = Field(default=5, description="分组数")
    factor_direction: int = Field(default=1, description="因子方向: 1=越大越好, -1=越小越好")
    floor_mode: str = Field(default='group', description="数据不足时策略: group=跳过, last=沿用上期")
    hedge: str = Field(default='equal', description="对冲基准: HS300/ZZ500/equal/custom")
    hedge_path: Optional[str] = Field(default=None, description="自定义对冲基准路径")


class LongShortSetting(BaseModel):
    """多空组合配置"""
    factor_direction: int = Field(default=1, description="因子方向: 1=越大越好, -1=越小越好")


class ScoreSetting(BaseModel):
    """市值行业分层打分配置"""
    enabled: bool = Field(default=True, description="是否运行")


class RiskCorrelationSetting(BaseModel):
    """风险因子相关性配置"""
    factors: str = Field(default='all', description="风险因子: 'all' 或 [(file, key), ...]")


class AnalysisSetting(BaseModel):
    """分析配置"""
    ic: ICSetting = Field(default_factory=ICSetting)
    group: GroupSetting = Field(default_factory=GroupSetting)
    longshort: LongShortSetting = Field(default_factory=LongShortSetting)
    score: ScoreSetting = Field(default_factory=ScoreSetting)
    risk_corr: RiskCorrelationSetting = Field(default_factory=RiskCorrelationSetting)


class OutputSetting(BaseModel):
    """输出配置"""
    dir: str = Field(default='./output/', description="输出目录")
    format: list = Field(default=['parquet', 'json'], description="输出格式")


class FeedbackSetting(BaseModel):
    """FactorFeedback 集成配置

    enabled=False 时, 现有行为完全不变 (向后兼容)。
    enabled=True 时, pipeline_runner 自动包装 5 个分析节点返回为 FactorFeedback,
    聚合到 ctx['Feedback'], 并可选择持久化到 output_dir。
    """
    enabled: bool = Field(default=False, description="是否启用 FactorFeedback 自动包装")
    output_dir: Optional[str] = Field(
        default=None,
        description="Parquet/JSON 持久化目录 (None=不持久化, 仅返回 ctx)",
    )
    judge_enabled: bool = Field(default=False, description="是否启用 LLMJudge (hypothesis↔expression 一致性)")
    judge_model: str = Field(default="mock", description="LLMJudge 模型名 (mock/deepseek-v3/...)")
    judge_max_attempts: int = Field(default=3, description="LLMJudge 解析失败最大重试次数")


class QualityGateConfig(BaseModel):
    """QualityGate 集成配置 (Week 3)。

    enabled=False: 不构造 QualityGateNode, 行为不变
    enabled=True:  构造 QualityGateNode, 可在 run_evolution() 中拦截低质量因子
    """
    enabled: bool = Field(default=False, description="是否启用 QualityGateNode")
    zoo_path: Optional[str] = Field(default=None, description="FactorZoo 路径 (None=内存)")


class EvolutionConfig(BaseModel):
    """演化主循环配置 (Week 4)。

    enabled=False: 不演化, run_evolution() 抛错
    enabled=True:  启用多轮演化循环
    """
    enabled: bool = Field(default=False, description="是否启用演化模式")
    max_rounds: int = Field(default=3, description="演化总轮数 (不含 round 0 原始)")
    parents_per_round: int = Field(default=1, description="每轮选几个 parent (crossover 时强制 2)")
    parent_selection_strategy: str = Field(
        default="top_percent_plus_random",
        description="选择策略: best/random/weighted/weighted_inverse/top_percent_plus_random",
    )
    top_percent_threshold: float = Field(default=0.3, description="top_percent_plus_random 阈值")
    metric: str = Field(default="sharpe", description="用于排序/加权的指标")
    pool_dir: Optional[str] = Field(default=None, description="TrajectoryPool 路径 (None=output.dir/trajectory)")
    early_stop_patience: int = Field(default=0, description="连续 N 轮无改善则停 (0=不启用)")


class SingleFactorTestConfig(BaseModel):
    """单因子回测完整配置"""
    factor: FactorSetting
    preprocess: PreprocessSetting
    analysis: AnalysisSetting = Field(default_factory=AnalysisSetting)
    output: OutputSetting = Field(default_factory=OutputSetting)
    feedback: FeedbackSetting = Field(default_factory=FeedbackSetting)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    data_path: str = Field(default='./testdata/test_h5_new/', description="数据根目录")
    load_keys: list = Field(
        # M7: 默认含 tradability filter 必需键 (st/suspend/ud_limit/ipo_days)
        default=['stklist', 'trade_dt', 'cp', 'id_citic1', 'mv_float',
                 'st', 'suspend', 'ud_limit', 'ipo_days'],
        description="需要加载的数据 key 列表"
    )
