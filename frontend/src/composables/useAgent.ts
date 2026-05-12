import { ref } from 'vue'
import { useWebSocket } from './useWebSocket'
import { useAgentStore } from '@/stores/agent'
import type { ToolCallEvent } from '@/stores/agent'

export function useAgent() {
  const store = useAgentStore()
  const isStreaming = ref(false)
  const streamContent = ref('')
  const currentToolCalls = ref<ToolCallEvent[]>([])

  const wsUrl = '/api/ws/chat'

  const { isConnected, connect, send, disconnect } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      switch (data.type) {
        case 'token':
          streamContent.value += data.content
          break
        case 'chunk':
          streamContent.value += data.content
          break
        case 'tool_call':
          currentToolCalls.value.push({
            id: data.id || `tc-${Date.now()}`,
            name: data.name || data.tool || 'unknown',
            arguments: data.arguments || {},
            status: 'running',
          })
          break
        case 'tool_result':
          const tc = currentToolCalls.value.find(t => t.id === data.id)
          if (tc) {
            tc.status = data.success !== false ? 'success' : 'error'
            tc.result = data.content || data.error || ''
          } else {
            currentToolCalls.value.push({
              id: data.id || `tc-${Date.now()}`,
              name: data.name || data.tool || 'unknown',
              arguments: {},
              result: data.content || data.error || '',
              status: data.success !== false ? 'success' : 'error',
            })
          }
          break
        case 'done':
          isStreaming.value = false
          const content = data.content || streamContent.value
          if (content || currentToolCalls.value.length > 0) {
            store.messages.push({
              id: data.message_id || `msg-${Date.now()}`,
              role: 'assistant',
              content: content,
              toolCalls: currentToolCalls.value.length > 0
                ? [...currentToolCalls.value]
                : undefined,
              timestamp: Date.now(),
            })
            streamContent.value = ''
            currentToolCalls.value = []
          }
          break
        case 'error':
          isStreaming.value = false
          store.messages.push({
            id: `err-${Date.now()}`,
            role: 'assistant',
            content: `Error: ${data.content}`,
            timestamp: Date.now(),
          })
          streamContent.value = ''
          currentToolCalls.value = []
          break
      }
    },
    onDisconnect: () => {
      isStreaming.value = false
    },
  })

  const sendMessage = (content: string, model?: string) => {
    store.messages.push({
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    })

    isStreaming.value = true
    streamContent.value = ''
    currentToolCalls.value = []

    send({
      type: 'message',
      content,
      session_id: store.sessionId,
      model: model || store.currentModel,
    })
  }

  return {
    isConnected,
    isStreaming,
    streamContent,
    currentToolCalls,
    connect,
    sendMessage,
    disconnect,
  }
}
