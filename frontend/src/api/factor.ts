import { get, post } from './index'

export interface FactorAnalyzeRequest {
  factor_name: string
  expression: string
  universe?: string
  start_date?: string
  end_date?: string
}

export interface FactorAnalyzeResult {
  ic_mean: number
  ic_std: number
  icir: number
  rank_ic_mean: number
  turnover: number
  dates?: string[]
  ic_series?: number[]
  returns?: number[]
}

export const factorApi = {
  analyze: (data: FactorAnalyzeRequest) =>
    post<FactorAnalyzeResult>('/factor/analyze', data),

  getMetrics: (name: string) =>
    get<FactorAnalyzeResult>(`/factor/${name}/metrics`),
}
