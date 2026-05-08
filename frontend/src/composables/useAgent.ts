import { ref } from 'vue'
import { useWebSocket } from './useWebSocket'
import { useAgentStore } from '@/stores/agent'

export function useAgent() {
  const store = useAgentStore()
  const isStreaming = ref(false)
  const streamContent = ref('')

  const wsUrl = `${import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'}/ws/chat`

  const { isConnected, connect, send, disconnect } = useWebSocket({
    url: wsUrl,
    onMessage: (data) => {
      switch (data.type) {
        case 'chunk':
          streamContent.value += data.content
          break
        case 'tool_call':
          console.log('Tool call:', data)
          break
        case 'tool_result':
          console.log('Tool result:', data)
          break
        case 'done':
          isStreaming.value = false
          if (streamContent.value) {
            store.messages.push({
              id: data.message_id || `msg-${Date.now()}`,
              role: 'assistant',
              content: streamContent.value,
              timestamp: Date.now(),
            })
            streamContent.value = ''
          }
          break
      }
    },
    onDisconnect: () => {
      isStreaming.value = false
    },
  })

  const sendMessage = (content: string) => {
    store.messages.push({
      id: `user-${Date.now()}`,
      role: 'user',
      content,
      timestamp: Date.now(),
    })

    isStreaming.value = true
    streamContent.value = ''

    send({
      type: 'message',
      content,
      session_id: store.sessionId,
    })
  }

  return {
    isConnected,
    isStreaming,
    streamContent,
    connect,
    sendMessage,
    disconnect,
  }
}
