from pydantic import BaseModel
from typing import Optional, List


class FactorAnalyzeRequest(BaseModel):
    expression: str
    universe: str = "hs300"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class FactorAnalyzeResult(BaseModel):
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    turnover: float
    ic_series: List[float] = []
    returns: List[float] = []
    dates: List[int] = []
