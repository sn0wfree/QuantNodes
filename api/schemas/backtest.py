from pydantic import BaseModel
from typing import Optional, List


class BacktestRequest(BaseModel):
    config_yaml: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_cash: Optional[float] = None
    data_path: Optional[str] = None


class BacktestSummary(BaseModel):
    total_return: float = 0
    annual_return: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    win_rate: float = 0
    total_trades: int = 0
    final_cash: float = 0
    total_commission: float = 0
    sortino_ratio: float = 0
    calmar_ratio: float = 0
    profit_factor: float = 0
    avg_trade_pnl: float = 0
    trading_days: int = 0


class BacktestResult(BaseModel):
    id: str
    status: str
    summary: BacktestSummary = BacktestSummary()
    config_info: dict = {}
    warnings: List[str] = []
    errors: List[str] = []
    output_files: dict = {}
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class BacktestTemplate(BaseModel):
    name: str
    description: str
    yaml: str
