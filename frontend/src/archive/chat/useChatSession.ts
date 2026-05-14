import { computed } from 'vue'
import { message } from 'ant-design-vue'
import { useAgentStore } from '@/stores/agent'
import dayjs from 'dayjs'

export function useChatSession() {
  const store = useAgentStore()

  const currentSessionLabel = computed(() => {
    if (store.sessionId === 'default') return 'Default Session'
    return store.sessionId
  })

  const createSession = async (name?: string) => {
    return await store.createSession(name)
  }

  const switchSession = async (id: string) => {
    await store.switchSession(id)
  }

  const deleteSession = async (id: string) => {
    await store.deleteSession(id)
  }

  const handleMenuClick = async ({ key }: { key: string }) => {
    if (key === 'new') {
      await createSession()
    } else {
      await switchSession(key)
    }
  }

  const clearHistory = async () => {
    try {
      store.clearMessages()
      message.success('History cleared')
    } catch {
      message.error('Failed to clear history')
    }
  }

  const exportSession = async (format: 'markdown' | 'json' = 'markdown') => {
    try {
      const response = await fetch(`/api/chat/export/${store.sessionId}?format=${format}`)
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `chat-${store.sessionId}-${dayjs().format('YYYY-MM-DD-HHmm')}.${format === 'markdown' ? 'md' : 'json'}`
      a.click()
      URL.revokeObjectURL(url)
      message.success('Session exported')
    } catch {
      message.error('Failed to export session')
    }
  }

  return {
    store,
    currentSessionLabel,
    createSession,
    switchSession,
    deleteSession,
    handleMenuClick,
    clearHistory,
    exportSession,
  }
}
