import { defineStore } from 'pinia'
import { ref } from 'vue'
import { get, post, del } from '@/api'
import { MODEL_REGISTRY } from '@/constants/models'
import type { ModelInfo } from '@/constants/models'

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
  const models = ref<ModelInfo[]>([])
  const modelsLoading = ref(false)
  const modelsLoaded = ref(false)

  // Build/Plan dual mode
  const currentMode = ref<'build' | 'plan'>('build')
  const modeModels = ref<Record<string, { model: string; max_tokens: number }>>({
    build: { model: '', max_tokens: 102400 },
    plan: { model: '', max_tokens: 16000 },
  })
  const defaultMode = ref<string>('build')
  const quality = ref<'high' | 'medium' | 'low'>('high')

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

  const switchMode = (mode: 'build' | 'plan') => {
    currentMode.value = mode
    // Update currentModel to match the mode's model
    const modeConfig = modeModels.value[mode]
    if (modeConfig?.model) {
      currentModel.value = modeConfig.model
    }
  }

  const loadModeModels = async () => {
    try {
      const settings = await get<any>('/settings')
      if (settings?.agent?.mode_models) {
        modeModels.value = settings.agent.mode_models
      }
      if (settings?.agent?.default_mode) {
        defaultMode.value = settings.agent.default_mode
        currentMode.value = settings.agent.default_mode
      }
      // Initialize currentModel from current mode's model
      const activeConfig = modeModels.value[currentMode.value]
      if (activeConfig?.model) {
        currentModel.value = activeConfig.model
      } else if (settings?.agent?.model) {
        currentModel.value = settings.agent.model
      }
    } catch {
      // use defaults
    }
  }

  const fetchModels = async () => {
    if (modelsLoaded.value) return
    modelsLoading.value = true
    try {
      // Fetch from single provider (OpenRouter etc.)
      const data = await get<{ models: ModelInfo[]; source: string }>('/settings/models')
      let allModels = data.models || []

      // Also fetch from configured multi-providers
      try {
        const providerModels = await get<Record<string, Array<{ id: string; name: string }>>>('/settings/providers/models/all')
        for (const [providerName, providerModelsList] of Object.entries(providerModels)) {
          for (const m of providerModelsList) {
            // Avoid duplicates
            if (!allModels.find(existing => existing.id === m.id)) {
              allModels.push({
                id: m.id,
                name: m.name || m.id,
                provider: providerName,
                contextWindow: 0,
                priceIn: 0,
                priceOut: 0,
                tags: [],
              })
            }
          }
        }
      } catch {
        // Multi-provider models fetch is optional
      }

      models.value = allModels
      modelsLoaded.value = true
    } catch {
      models.value = MODEL_REGISTRY
      modelsLoaded.value = true
    } finally {
      modelsLoading.value = false
    }
  }

  return {
    messages,
    isLoading,
    sessionId,
    sessions,
    currentModel,
    systemMessages,
    models,
    modelsLoading,
    modelsLoaded,
    currentMode,
    modeModels,
    defaultMode,
    quality,
    clearMessages,
    loadSessions,
    createSession,
    switchSession,
    deleteSession,
    switchMode,
    loadModeModels,
    fetchModels,
  }
})
