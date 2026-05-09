import { get, post, put, del } from './index'

export interface FactorInfo {
  name: string
  formula: string
  source: string
  category: string
  ic_mean?: number
  ic_std?: number
  icir?: number
  rank_ic_mean?: number
  turnover?: number
  tags: string[]
  description?: string
  wiki_page_name?: string
  created_at?: string
  updated_at?: string
}

export interface StrategyInfo {
  name: string
  description: string
  category: string
  tags: string[]
  strategy_yaml?: string
  backtest_result?: Record<string, any>
  wiki_page_name?: string
  created_at?: string
  updated_at?: string
}

export interface WikiSearchParams {
  q: string
  type?: 'factor' | 'strategy' | 'all'
  limit?: number
}

export interface WikiStatus {
  factors: number
  strategies: number
  logics: number
  relations: number
}

export const wikiApi = {
  // Factor endpoints
  getFactors: (params?: { category?: string; source?: string; sort?: string; limit?: number }) =>
    get<FactorInfo[]>('/wiki/factors', { params }),

  getFactor: (name: string) =>
    get<FactorInfo>(`/wiki/factors/${name}`),

  createFactor: (data: Partial<FactorInfo>) =>
    post('/wiki/factors', data),

  updateFactor: (name: string, data: Partial<FactorInfo>) =>
    put(`/wiki/factors/${name}`, data),

  deleteFactor: (name: string) =>
    del(`/wiki/factors/${name}`),

  searchFactors: (q: string, limit?: number) =>
    get<any[]>('/wiki/factors/search', { params: { q, limit } }),

  // Strategy endpoints
  getStrategies: (params?: { category?: string; sort?: string; limit?: number }) =>
    get<StrategyInfo[]>('/wiki/strategies', { params }),

  getStrategy: (name: string) =>
    get<StrategyInfo>(`/wiki/strategies/${name}`),

  createStrategy: (data: Partial<StrategyInfo>) =>
    post('/wiki/strategies', data),

  updateStrategy: (name: string, data: Partial<StrategyInfo>) =>
    put(`/wiki/strategies/${name}`, data),

  deleteStrategy: (name: string) =>
    del(`/wiki/strategies/${name}`),

  searchStrategies: (q: string, limit?: number) =>
    get<any[]>('/wiki/strategies/search', { params: { q, limit } }),

  // General search
  search: (params: WikiSearchParams) =>
    get<any[]>('/wiki/search', { params }),

  // Status
  getStatus: () =>
    get<WikiStatus>('/wiki/status'),
}
