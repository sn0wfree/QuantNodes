import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ToolCallInfo } from '@/api/agent'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  toolCalls?: ToolCallInfo[]
  timestamp: number
}

export const useAgentStore = defineStore('agent', () => {
  const messages = ref<Message[]>([])
  const isLoading = ref(false)
  const sessionId = ref<string>('')

  const clearMessages = () => {
    messages.value = []
    sessionId.value = ''
  }

  return {
    messages,
    isLoading,
    sessionId,
    clearMessages,
  }
})
