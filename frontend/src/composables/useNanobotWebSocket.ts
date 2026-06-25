import { ref, onUnmounted } from 'vue'

/**
 * nanobot WebSocket protocol wrapper.
 *
 * Protocol (per nanobot 0.2.1 ``nanobot/channels/websocket.py``):
 *
 *   1. Client opens HTTP GET ``/webui/bootstrap`` on the gateway URL.
 *      Response: ``{token, ws_path, ws_url, expires_in, model_name, ...}``
 *
 *   2. Client opens ``ws://{host}:{port}/{ws_path}?token={token}``.
 *
 *   3. Server sends events shaped like::
 *
 *        {event: "attached", chat_id, client_id, ...}
 *        {event: "user", chat_id, text}
 *        {event: "message", chat_id, text}                       // streamed tokens
 *        {event: "tool_hint", name, args_preview, ...}
 *        {event: "tool_call", name, args}
 *        {event: "tool_result", name, success, content}
 *        {event: "tool_status", tool, status, ...}
 *        {event: "goal_status", status, ...}
 *        {event: "goal_state", ...}
 *        {event: "error", message}
 *
 *   4. Client sends user messages as plain text frames (legacy) or as
 *      a JSON envelope ``{"type": "message", "content": "..."}``.
 *
 * This composable wraps that flow behind a single ``connect()`` /
 * ``send(text)`` / ``disconnect()`` interface and emits normalized events
 * via ``onEvent``. The ``AgentChat.vue`` view treats ``message`` /
 * ``tool_call`` / ``tool_result`` / ``error`` as the four primary
 * render buckets; the other events are forwarded but not displayed.
 */

export type NanobotEvent =
  | { event: 'attached'; chat_id: string; client_id?: string; [k: string]: any }
  | { event: 'user'; chat_id: string; text: string; [k: string]: any }
  | { event: 'message'; chat_id: string; text: string; [k: string]: any }
  | { event: 'tool_hint'; name: string; [k: string]: any }
  | { event: 'tool_call'; name: string; arguments?: Record<string, any>; [k: string]: any }
  | { event: 'tool_result'; name: string; success: boolean; content?: any; [k: string]: any }
  | { event: 'tool_status'; tool: string; status: string; [k: string]: any }
  | { event: 'goal_status'; status: string; [k: string]: any }
  | { event: 'error'; message: string; [k: string]: any }
  | { event: string; [k: string]: any }

export interface NanobotWebSocketOptions {
  /** Gateway base URL (e.g. ``http://127.0.0.1:18080`` or ``http://localhost:18080``). */
  baseUrl: string
  /** Chat ID — defaults to ``"default"``. */
  chatId?: string
  /** Optional client_id (used for ``allow_from`` ACL on the gateway). */
  clientId?: string
  /** Receive normalized nanobot events. */
  onEvent?: (event: NanobotEvent) => void
  /** Called once the WS handshake succeeds (after token bootstrap). */
  onConnected?: (info: { chatId: string; token: string; modelName?: string }) => void
  /** Called on disconnect / reconnect failure. */
  onError?: (err: unknown) => void
  /** Reconnect delay base in ms (exponential backoff). */
  reconnectInterval?: number
  /** Max reconnect attempts (0 = infinite). */
  maxReconnectAttempts?: number
}

export function useNanobotWebSocket(options: NanobotWebSocketOptions) {
  const {
    baseUrl,
    chatId = 'default',
    clientId = 'quantnodes-webui',
    onEvent,
    onConnected,
    onError,
    reconnectInterval = 1500,
    maxReconnectAttempts = 0,  // infinite by default
  } = options

  const isConnected = ref(false)
  const bootstrapInfo = ref<{ token: string; model_name?: string } | null>(null)
  const reconnectAttempts = ref(0)

  let ws: WebSocket | null = null
  let intentionalClose = false
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  /** GET /webui/bootstrap to get a short-lived WS token.

    v3.0.0: When the gateway binds to 0.0.0.0 (LAN access), it requires
    an ``X-Nanobot-Auth`` header. The browser can't send custom headers
    cross-origin, so we proxy through FastAPI's ``/api/agent/gateway-bootstrap``
    endpoint which adds the token server-side.
    */
  async function fetchBootstrap(): Promise<{ token: string; model_name?: string; ws_path: string }> {
    // v3.0.0: always use FastAPI proxy for bootstrap (handles auth + rewrites ws_url)
    // Works in both Vite dev (proxy → FastAPI → gateway) and production (same origin).
    const resp = await fetch('/api/agent/gateway-bootstrap', {
      headers: clientId ? { 'X-Nanobot-Client': clientId } : {},
    })
    if (!resp.ok) {
      throw new Error(`bootstrap HTTP ${resp.status}: ${await resp.text()}`)
    }
    const data = await resp.json()
    if (!data.token) throw new Error('bootstrap response missing token')
    return data
  }

  /** Convert the gateway HTTP base URL into a ws:// URL with token query. */
  function buildWsUrl(token: string, wsPath: string): string {
    const base = baseUrl.replace(/\/$/, '')
    // base is http://host:port or https://host:port
    const wsBase = base.replace(/^http/, 'ws')
    const path = wsPath.startsWith('/') ? wsPath : `/${wsPath}`
    const params = new URLSearchParams({ token, client_id: clientId })
    return `${wsBase}${path}?${params.toString()}`
  }

  async function connect(): Promise<void> {
    intentionalClose = false
    try {
      const boot = await fetchBootstrap()
      bootstrapInfo.value = { token: boot.token, model_name: boot.model_name }
      const wsUrl = buildWsUrl(boot.token, boot.ws_path)
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        isConnected.value = true
        reconnectAttempts.value = 0
        onConnected?.({
          chatId,
          token: boot.token,
          modelName: boot.model_name,
        })
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data) as NanobotEvent
          onEvent?.(payload)
        } catch (e) {
          // Non-JSON frames are ignored; nanobot only sends JSON anyway.
          if (import.meta.env.DEV) {
            console.warn('useNanobotWebSocket: failed to parse frame', e)
          }
        }
      }

      ws.onerror = (e) => {
        onError?.(e)
      }

      ws.onclose = () => {
        isConnected.value = false
        ws = null
        if (!intentionalClose) {
          if (maxReconnectAttempts === 0 || reconnectAttempts.value < maxReconnectAttempts) {
            reconnectTimer = setTimeout(() => {
              reconnectAttempts.value++
              void connect()
            }, reconnectInterval * Math.pow(1.5, Math.min(reconnectAttempts.value, 6)))
          }
        }
      }
    } catch (e) {
      isConnected.value = false
      onError?.(e)
      if (!intentionalClose && (maxReconnectAttempts === 0 || reconnectAttempts.value < maxReconnectAttempts)) {
        reconnectTimer = setTimeout(() => {
          reconnectAttempts.value++
          void connect()
        }, reconnectInterval * Math.pow(1.5, Math.min(reconnectAttempts.value, 6)))
      }
    }
  }

  /** Send a user message to the agent. */
  function send(text: string): boolean {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      if (import.meta.env.DEV) {
        console.warn('useNanobotWebSocket.send: ws not open')
      }
      return false
    }
    // nanobot accepts either plain text frames or a JSON envelope
    // with ``{"type": "message", "content": "..."}``. We use the
    // envelope for forward-compatibility with typed events.
    ws.send(JSON.stringify({ type: 'message', content: text, chat_id: chatId }))
    return true
  }

  function disconnect(): void {
    intentionalClose = true
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
    isConnected.value = false
  }

  onUnmounted(() => disconnect())

  return {
    isConnected,
    bootstrapInfo,
    reconnectAttempts,
    connect,
    send,
    disconnect,
  }
}
