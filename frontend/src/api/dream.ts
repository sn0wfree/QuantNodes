import { get } from './index'

export interface DreamInsight {
  id: string
  title: string
  content: string
  type: string
  category: string
  confidence: number
  created_at: string
  tags: string[]
  insights: string[]
  source: string
}

export interface DreamStats {
  total_insights: number
  by_type: Record<string, number>
  by_category: Record<string, number>
  avg_confidence: number
  recent_trend: { date: string; count: number }[]
  top_tags: { tag: string; count: number }[]
}

export const dreamApi = {
  list: (params?: { limit?: number; type?: string }) =>
    get<DreamInsight[]>('/dreams', { params }),

  getStats: () =>
    get<DreamStats>('/dreams/stats'),

  getInsight: (id: string) =>
    get<DreamInsight>(`/dreams/${id}`),
}
