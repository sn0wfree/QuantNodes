import { get, post } from './index'

export interface BacktestRequest {
  config_yaml: string
  start_date?: string
  end_date?: string
  initial_cash?: number
  data_path?: string
}

export interface BacktestSummary {
  total_return: number
  annual_return: number
  sharpe_ratio: number
  max_drawdown: number
  win_rate: number
  total_trades: number
  final_cash: number
  total_commission: number
  sortino_ratio: number
  calmar_ratio: number
  profit_factor: number
  avg_trade_pnl: number
  trading_days: number
}

export interface BacktestResult {
  id: string
  status: string
  summary: BacktestSummary
  config_info: Record<string, any>
  warnings: string[]
  errors: string[]
  output_files: Record<string, any>
  created_at?: string
  completed_at?: string
}

export interface BacktestTemplate {
  name: string
  description: string
  yaml: string
}

export const backtestApi = {
  run: (data: BacktestRequest) =>
    post<BacktestResult>('/backtest/run', data),

  getResult: (id: string) =>
    get<BacktestResult>(`/backtest/${id}`),

  getHistory: (params?: { limit?: number; offset?: number }) =>
    get<BacktestResult[]>('/backtest/history', { params }),

  getTemplates: () =>
    get<BacktestTemplate[]>('/backtest/templates'),
}
