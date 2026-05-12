import { defineStore } from 'pinia'
import { ref } from 'vue'
import { get, post, del } from '@/api'

export interface ToolCallEvent {
  id: string
  name: string
  arguments: Record<string, any>
  result?: string
  status: 'running' | 'success' | 'error'
}

export interface SystemMessage {
  id: string
  type: 'compact' | 'info' | 'warning'
  content: string
  timestamp: number
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCallEvent[]
  timestamp: number
}

export interface SessionInfo {
  session_id: string
  message_count: number
  first_message: string
  last_message: string
}

export const useAgentStore = defineStore('agent', () => {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)
  const sessionId = ref<string>('default')
  const sessions = ref<SessionInfo[]>([])
  const currentModel = ref<string>('minimax/minimax-m2.5:free')
  const systemMessages = ref<SystemMessage[]>([])

  const clearMessages = () => {
    messages.value = []
  }

  const loadSessions = async () => {
    try {
      sessions.value = await get<SessionInfo[]>('/chat/sessions')
    } catch {
      sessions.value = []
    }
  }

  const createSession = async (name?: string) => {
    try {
      const result = await post<{ session_id: string }>('/chat/sessions', {
        session_id: name,
      })
      sessionId.value = result.session_id
      messages.value = []
      await loadSessions()
      return result.session_id
    } catch {
      return null
    }
  }

  const switchSession = async (id: string) => {
    sessionId.value = id
    try {
      const history = await get<Array<{ role: string; content: string }>>(
        `/chat/history/${id}`
      )
      messages.value = history.map((m, i) => ({
        id: `${id}-${i}`,
        role: m.role as 'user' | 'assistant',
        content: m.content,
        timestamp: Date.now() - (history.length - i) * 1000,
      }))
    } catch {
      messages.value = []
    }
  }

  const deleteSession = async (id: string) => {
    try {
      await del(`/chat/sessions/${id}`)
      if (sessionId.value === id) {
        sessionId.value = 'default'
        messages.value = []
      }
      await loadSessions()
    } catch {
      // ignore
    }
  }

  return {
    messages,
    isLoading,
    sessionId,
    sessions,
    currentModel,
    systemMessages,
    clearMessages,
    loadSessions,
    createSession,
    switchSession,
    deleteSession,
  }
})
