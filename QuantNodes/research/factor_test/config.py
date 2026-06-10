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
    sample_index: str = Field(default='all', description="样本池: all/HS300/ZZ500/ZZ800/SZ50/custom")
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
    hedge: str = Field(default='equal', description="对冲基准: HS300/ZZ500/SZ50/equal/custom")
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


class SingleFactorTestConfig(BaseModel):
    """单因子回测完整配置"""
    factor: FactorSetting
    preprocess: PreprocessSetting
    analysis: AnalysisSetting = Field(default_factory=AnalysisSetting)
    output: OutputSetting = Field(default_factory=OutputSetting)
    data_path: str = Field(default='./testdata/test_h5_new/', description="数据根目录")
    load_keys: list = Field(
        default=['stklist', 'trade_dt', 'cp', 'id_citic1', 'mv_float'],
        description="需要加载的数据 key 列表"
    )
