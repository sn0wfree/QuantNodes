from pydantic import BaseModel
from typing import Optional, List


class FactorInfo(BaseModel):
    name: str
    formula: str = ""
    source: str = "manual"
    category: str = "other"
    ic_mean: Optional[float] = None
    ic_std: Optional[float] = None
    icir: Optional[float] = None
    rank_ic_mean: Optional[float] = None
    turnover: Optional[float] = None
    tags: List[str] = []
    description: Optional[str] = None
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class StrategyInfo(BaseModel):
    name: str
    description: str = ""
    category: str = "general"
    tags: List[str] = []
    strategy_yaml: Optional[str] = None
    backtest_result: Optional[dict] = None
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class WikiSearchResult(BaseModel):
    type: str
    name: str
    data: Optional[dict] = None


class WikiStatus(BaseModel):
    factors: int = 0
    strategies: int = 0
    logics: int = 0
    relations: int = 0
