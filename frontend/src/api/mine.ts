import { get, post } from './index'

// ==============================================================================
// Types
// ==============================================================================

export interface MineLogicsConfig {
  source_libs?: string[]
  max_per_lib?: number
  workers?: number
  wiki_path?: string
  live?: boolean
  strict?: boolean
  skip_existing?: boolean
}

export interface MineLogicsStartResponse {
  run_id: string
  status: string
}

export interface MineLogicsStatusResponse {
  run_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped'
  config: MineLogicsConfig
  created_at: number
  started_at: number | null
  completed_at: number | null
  elapsed_seconds: number
  progress: {
    source_libs?: string[]
    current_formula?: string
    done?: number
    total?: number
    n_mined?: number
    n_skipped?: number
    n_failed?: number
  }
}

export interface MineLogicsResultsResponse {
  run_id: string
  status: string
  result: {
    summary: {
      total_attempted: number
      total_mined: number
      total_skipped: number
      total_failed: number
      success_rate: number
    }
    source_breakdown: Record<string, { mined: number; attempted: number }>
    top_factors: Array<{
      formula_id: string
      formula: string
      source_lib: string
      ir: number
      ic_mean: number
      rank_ic: number
      parse_layer: number
      tags: string[]
    }>
    agent_stats: Record<string, {
      call_failures: number
      parse_failures: number
      parse_layer_reached: number
      structured_failures: number
    }>
    warnings: string[]
    wall_clock_s: number
  } | null
}

export interface MineLogicsHistoryResponse {
  runs: Array<{
    run_id: string
    status: string
    config: MineLogicsConfig
    created_at: number
    started_at: number | null
    completed_at: number | null
    elapsed_seconds: number
    progress: Record<string, any>
    error: string | null
  }>
  total: number
}

// ==============================================================================
// WebSocket Event Types
// ==============================================================================

export interface MineLogicsEvent {
  type: 'mining_started' | 'formula_attempted' | 'formula_completed' | 'batch_completed' | 'error' | 'done' | 'heartbeat'
  run_id?: string
  ts?: number
  formula_id?: string
  done?: number
  total?: number
  success?: boolean
  parse_layer?: number
  n_mined?: number
  n_skipped?: number
  n_failed?: number
  wall_clock_s?: number
  message?: string
  source_libs?: string[]
}

// ==============================================================================
// API Client
// ==============================================================================

export const mineLogicsApi = {
  /** 启动批量挖掘 */
  start: (config: MineLogicsConfig = {}) =>
    post<MineLogicsStartResponse>('/mine-logics/start', config),

  /** 查询运行进度 */
  status: (runId: string) =>
    get<MineLogicsStatusResponse>(`/mine-logics/status/${runId}`),

  /** 获取运行结果 */
  results: (runId: string) =>
    get<MineLogicsResultsResponse>(`/mine-logics/results/${runId}`),

  /** 停止运行 */
  stop: (runId: string) =>
    post<{ stopped: boolean; reason?: string }>(`/mine-logics/stop/${runId}`),

  /** 历史运行列表 */
  history: () =>
    get<MineLogicsHistoryResponse>('/mine-logics/history'),
}
