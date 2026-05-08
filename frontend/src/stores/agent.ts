import { defineStore } from 'pinia'
import { ref } from 'vue'
import { agentApi } from '@/api/agent'
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

  const sendMessage = async (content: string) => {
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    }
    messages.value.push(userMessage)

    isLoading.value = true
    try {
      const response = await agentApi.sendMessage({
        content,
        session_id: sessionId.value || `session-${Date.now()}`,
      })

      const assistantMessage: Message = {
        id: response.message_id,
        role: 'assistant',
        content: response.content,
        timestamp: Date.now(),
      }
      messages.value.push(assistantMessage)
    } catch (error) {
      console.error('Failed to send message:', error)
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: 'Failed to get response. Please try again.',
        timestamp: Date.now(),
      }
      messages.value.push(errorMessage)
    } finally {
      isLoading.value = false
    }
  }

  const clearMessages = () => {
    messages.value = []
    sessionId.value = ''
  }

  return {
    messages,
    isLoading,
    sessionId,
    sendMessage,
    clearMessages,
  }
})
