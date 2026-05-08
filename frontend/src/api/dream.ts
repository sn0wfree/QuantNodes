import { get } from './index'

export interface DreamInsight {
  id: string
  title: string
  content: string
  category: string
  confidence: number
  created_at: string
  tags: string[]
}

export interface DreamStats {
  total_insights: number
  by_category: Record<string, number>
  avg_confidence: number
}

export const dreamApi = {
  list: (params?: { limit?: number; category?: string }) =>
    get<DreamInsight[]>('/dreams', { params }),

  getStats: () =>
    get<DreamStats>('/dreams/stats'),

  getInsight: (id: string) =>
    get<DreamInsight>(`/dreams/${id}`),
}
