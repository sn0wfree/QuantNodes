import { defineStore } from 'pinia'
import { ref } from 'vue'

export interface ToolCallEvent {
  id: string
  name: string
  arguments: Record<string, any>
  result?: string
  status: 'running' | 'success' | 'error'
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCallEvent[]
  timestamp: number
}

export const useAgentStore = defineStore('agent', () => {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)
  const sessionId = ref<string>('default')

  const clearMessages = () => {
    messages.value = []
    sessionId.value = 'default'
  }

  return {
    messages,
    isLoading,
    sessionId,
    clearMessages,
  }
})
