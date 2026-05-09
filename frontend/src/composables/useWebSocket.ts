import { ref, onUnmounted } from 'vue'

export interface WebSocketOptions {
  url: string
  onMessage?: (data: any) => void
  onError?: (error: Event) => void
  onDisconnect?: () => void
  reconnectInterval?: number
  maxReconnectAttempts?: number
  maxQueueSize?: number
}

export function useWebSocket(options: WebSocketOptions) {
  const {
    url,
    onMessage,
    onError,
    onDisconnect,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    maxQueueSize = 50,
  } = options

  let ws: WebSocket | null = null
  const isConnected = ref(false)
  const reconnectAttempts = ref(0)
  let intentionalClose = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  const pendingMessages: any[] = []

  const connect = () => {
    try {
      intentionalClose = false
      ws = new WebSocket(url)

      ws.onopen = () => {
        isConnected.value = true
        reconnectAttempts.value = 0
        console.log('WebSocket connected')
        flushQueue()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(error)
      }

      ws.onclose = () => {
        isConnected.value = false
        console.log('WebSocket disconnected')
        onDisconnect?.()

        if (!intentionalClose && reconnectAttempts.value < maxReconnectAttempts) {
          reconnectTimer = setTimeout(() => {
            reconnectAttempts.value++
            connect()
          }, reconnectInterval * Math.pow(2, reconnectAttempts.value))
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket:', error)
    }
  }

  const flushQueue = () => {
    while (pendingMessages.length > 0 && ws && ws.readyState === WebSocket.OPEN) {
      const msg = pendingMessages.shift()
      ws.send(JSON.stringify(msg))
    }
  }

  const send = (data: any) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(data))
    } else if (pendingMessages.length < maxQueueSize) {
      pendingMessages.push(data)
    } else {
      console.warn('WebSocket queue full, dropping message')
    }
  }

  const disconnect = () => {
    intentionalClose = true
    pendingMessages.length = 0
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
  }

  onUnmounted(() => {
    disconnect()
  })

  return {
    isConnected,
    connect,
    send,
    disconnect,
  }
}
