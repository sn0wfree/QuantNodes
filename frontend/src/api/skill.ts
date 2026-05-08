import { get, post } from './index'

export interface SkillInfo {
  name: string
  description: string
  type: 'strategy' | 'factor' | 'utility'
  version: string
  enabled: boolean
}

export const skillApi = {
  list: () =>
    get<SkillInfo[]>('/skills'),

  execute: (name: string, params?: Record<string, any>) =>
    post(`/skills/${name}/execute`, params),

  getDetail: (name: string) =>
    get<SkillInfo>(`/skills/${name}`),
}
