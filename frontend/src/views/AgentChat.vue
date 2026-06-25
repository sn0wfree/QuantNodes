<template>
  <div class="agent-chat">
    <!-- Loading -->
    <div v-if="loading" class="status-overlay">
      <a-spin size="large" />
      <p>正在连接 nanobot gateway…</p>
    </div>

    <!-- Unavailable -->
    <div v-else-if="!status?.available" class="status-overlay">
      <a-result status="info" title="Agent 功能未启用">
        <template #subTitle>
          <p>{{ status?.hint || 'nanobot-ai 依赖未安装，无法启动 Agent 运行时。' }}</p>
          <a-typography-paragraph>
            <a-typography-text code>pip install 'quantnodes[agent]'</a-typography-text>
            （或 <a-typography-text code>'quantnodes[all]'</a-typography-text> 一键装齐）
          </a-typography-paragraph>
          <p>安装后重启服务即可启用 Agent Chat / WebUI / MCP / 飞书 等功能。</p>
        </template>
      </a-result>
    </div>

    <!-- Error -->
    <div v-else-if="status.state === 'error'" class="status-overlay">
      <a-result status="error" title="Agent 运行时启动失败">
        <template #subTitle>
          <p>{{ status.error }}</p>
          <a-typography-paragraph v-if="status.hint">{{ status.hint }}</a-typography-paragraph>
          <a-button type="primary" @click="restart">重试</a-button>
        </template>
      </a-result>
    </div>

    <!-- Starting -->
    <div v-else-if="status.state === 'starting'" class="status-overlay">
      <a-spin size="large" />
      <p>Agent 运行时启动中（state=starting）…</p>
    </div>

    <!-- Main Chat UI -->
    <template v-else-if="status.state === 'running'">
      <!-- Status Bar -->
      <div class="chat-status-bar">
        <span :class="wsConnected ? 'ws-dot connected' : 'ws-dot disconnected'">
          {{ wsConnected ? '● 已连接' : '⚪ 未连接' }}
        </span>
        <span v-if="modelName" class="model-name">{{ modelName }}</span>
        <span class="spacer" />
        <a-button size="small" @click="showSettings = !showSettings">
          ⚙ Settings
        </a-button>
      </div>

      <div class="chat-body">
        <!-- Left: Sessions -->
        <div class="chat-sidebar">
          <a-button type="primary" block size="small" @click="createSession" class="new-session-btn">
            + 新建会话
          </a-button>
          <div class="session-list">
            <div
              v-for="s in sessions" :key="s.key"
              :class="['session-item', { active: s.key === currentSessionKey }]"
              @click="switchSession(s.key)"
            >
              <div class="session-title">{{ s.title || s.preview?.slice(0, 40) || '(空会话)' }}</div>
              <div class="session-time">{{ formatTime(s.updated_at) }}</div>
            </div>
          </div>
        </div>

        <!-- Center: Messages -->
        <div class="chat-main">
          <div class="messages" ref="messagesEl">
            <div v-if="messages.length === 0" class="empty-state">
              <h2>Welcome</h2>
              <p>发送消息开始与 Agent 对话</p>
            </div>
            <div v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">
              <div class="message-role">{{ msg.role === 'user' ? 'You' : 'Agent' }}</div>
              <div class="message-content" v-html="renderMarkdown(msg.content)" />
              <div v-if="msg.reasoning" class="reasoning">
                <details>
                  <summary>思考过程 ({{ msg.reasoning.length }} chars)</summary>
                  <div class="reasoning-text">{{ msg.reasoning }}</div>
                </details>
              </div>
              <div v-if="msg.toolCalls?.length" class="tool-calls">
                <div v-for="(tc, i) in msg.toolCalls" :key="i" class="tool-call">
                  <span class="tool-icon">🔧</span>
                  <span class="tool-name">{{ tc.name }}</span>
                  <span class="tool-args">{{ summarizeArgs(tc.arguments) }}</span>
                </div>
              </div>
              <div v-if="msg.toolResults?.length" class="tool-results">
                <div v-for="(tr, i) in msg.toolResults" :key="i" class="tool-result"
                     :class="tr.success ? 'success' : 'error'">
                  <span class="tool-icon">{{ tr.success ? '✅' : '❌' }}</span>
                  <span class="tool-name">{{ tr.name }}</span>
                  <span class="tool-content">{{ summarizeContent(tr.content) }}</span>
                </div>
              </div>
            </div>
          </div>
          <div class="chat-input">
            <textarea
              ref="inputEl"
              v-model="inputText"
              @keydown.enter.exact.prevent="sendMessage"
              placeholder="输入消息… (Enter 发送, Shift+Enter 换行)"
              rows="2"
            />
            <button
              class="send-btn"
              :disabled="!inputText.trim() || isStreaming"
              @click="sendMessage"
            >
              {{ isStreaming ? '⏳' : '发送' }}
            </button>
          </div>
        </div>

        <!-- Right: Settings (toggle) -->
        <div v-if="showSettings" class="chat-settings">
          <h4>⚙ Settings</h4>
          <div class="setting-group">
            <label>Model</label>
            <a-input v-model:value="settingsModel" size="small" />
          </div>
          <div class="setting-group">
            <label>Temperature</label>
            <a-input-number v-model:value="settingsTemp" :min="0" :max="2" :step="0.1" size="small" style="width:100%" />
          </div>
          <div class="setting-group">
            <label>Max Tokens</label>
            <a-input-number v-model:value="settingsMaxTokens" :min="256" :max="128000" :step="256" size="small" style="width:100%" />
          </div>
          <a-button type="primary" size="small" block @click="saveSettings">保存设置</a-button>
          <a-button size="small" block @click="loadSettings" style="margin-top:8px">刷新</a-button>
        </div>
      </div>

      <!-- Footer -->
      <div class="chat-footer">
        <span>gateway: {{ gatewayUrl }}</span>
        <span>session: {{ currentSessionKey || '(none)' }}</span>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useNanobotWebSocket, type NanobotEvent } from '@/composables/useNanobotWebSocket'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ html: false, linkify: true, breaks: true })

// ── Config ──────────────────────────────────────────────────
const gatewayUrl = (import.meta.env.VITE_NANOBOT_GATEWAY_URL as string) || 'http://127.0.0.1:18090/'
const gatewayBase = computed(() => gatewayUrl.replace(/\/$/, ''))

// ── Types ───────────────────────────────────────────────────
interface AgentStatus {
  available: boolean
  state: string
  hint?: string
  error?: string
  gateway_host?: string
  gateway_port?: number
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  reasoning?: string
  toolCalls?: { name: string; arguments: any }[]
  toolResults?: { name: string; content: any; success: boolean }[]
  status: 'pending' | 'streaming' | 'done'
}

interface SessionInfo {
  key: string
  title?: string
  preview?: string
  updated_at?: string
  message_count?: number
}

// ── State ───────────────────────────────────────────────────
const status = ref<AgentStatus | null>(null)
const loading = ref(true)
const showSettings = ref(false)
const inputText = ref('')
const messages = ref<ChatMessage[]>([])
const sessions = ref<SessionInfo[]>([])
const currentSessionKey = ref('')
const isStreaming = ref(false)
const modelName = ref('')
const wsConnected = ref(false)

// Settings
const settingsModel = ref('')
const settingsTemp = ref(0.1)
const settingsMaxTokens = ref(8192)

// Refs
const messagesEl = ref<HTMLElement>()
const inputEl = ref<HTMLTextAreaElement>()
let pollHandle: number | null = null

// ── WebSocket ───────────────────────────────────────────────
const {
  isConnected: rawConnected,
  bootstrapInfo,
  connect,
  send,
  disconnect,
} = useNanobotWebSocket({
  baseUrl: gatewayBase.value,
  chatId: computed(() => currentSessionKey.value || 'default'),
  clientId: 'quantnodes-webui',
  onEvent: handleEvent,
  onConnected: (info) => {
    wsConnected.value = true
    modelName.value = info.modelName || ''
  },
  onError: () => { wsConnected.value = false },
})

// ── Event Handler ───────────────────────────────────────────
function handleEvent(ev: NanobotEvent) {
  const last = messages.value[messages.value.length - 1]
  switch (ev.event) {
    case 'message': {
      // Text delta — append to current assistant message or create new
      if (!last || last.role !== 'assistant' || last.status === 'done') {
        messages.value.push({
          id: crypto.randomUUID(),
          role: 'assistant',
          content: ev.text,
          status: 'streaming',
        })
      } else {
        last.content += ev.text
      }
      break
    }
    case 'reasoning_delta': {
      if (last?.role === 'assistant') {
        last.reasoning = (last.reasoning || '') + (ev as any).text
      }
      break
    }
    case 'tool_call': {
      if (last?.role === 'assistant') {
        if (!last.toolCalls) last.toolCalls = []
        last.toolCalls.push({ name: ev.name, arguments: (ev as any).arguments || {} })
      }
      break
    }
    case 'tool_result': {
      if (last?.role === 'assistant') {
        if (!last.toolResults) last.toolResults = []
        last.toolResults.push({
          name: ev.name,
          content: (ev as any).content,
          success: (ev as any).success ?? true,
        })
      }
      break
    }
    case 'turn_end': {
      if (last && last.role === 'assistant') {
        last.status = 'done'
      }
      isStreaming.value = false
      refreshSessions()
      break
    }
    case 'goal_status': {
      if ((ev as any).status === 'running') {
        isStreaming.value = true
      }
      break
    }
    case 'error': {
      messages.value.push({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: `Error: ${(ev as any).message || 'Unknown error'}`,
        status: 'done',
      })
      isStreaming.value = false
      break
    }
    case 'stream_end': {
      // Streaming complete — finalize message
      if (last && last.role === 'assistant' && last.status === 'streaming') {
        last.status = 'done'
      }
      break
    }
  }
  nextTick(() => scrollToBottom())
}

// ── Session Management ──────────────────────────────────────
async function fetchGateway(path: string, opts?: RequestInit): Promise<any> {
  const token = bootstrapInfo.value?.token
  const headers: Record<string, string> = {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(opts?.headers as Record<string, string> || {}),
  }
  const resp = await fetch(`${gatewayBase.value}${path}`, { ...opts, headers })
  if (!resp.ok) throw new Error(`Gateway ${resp.status}: ${await resp.text()}`)
  return resp.json()
}

async function refreshSessions() {
  try {
    const data = await fetchGateway('/api/sessions')
    sessions.value = data.sessions || []
  } catch (e) {
    console.warn('refreshSessions failed:', e)
  }
}

async function switchSession(key: string) {
  if (key === currentSessionKey.value) return
  currentSessionKey.value = key
  messages.value = []
  isStreaming.value = false
  try {
    const data = await fetchGateway(`/api/sessions/${encodeURIComponent(key)}/messages`)
    messages.value = (data.messages || []).map((m: any) => ({
      id: crypto.randomUUID(),
      role: m.role as 'user' | 'assistant',
      content: m.content || '',
      status: 'done' as const,
    }))
    nextTick(() => scrollToBottom())
  } catch (e) {
    console.warn('switchSession failed:', e)
  }
  // Reconnect WS with new chatId
  disconnect()
  await connect()
}

async function createSession() {
  const key = `quantnodes:${Date.now()}`
  currentSessionKey.value = key
  messages.value = []
  isStreaming.value = false
  disconnect()
  await connect()
  refreshSessions()
}

// ── Settings ────────────────────────────────────────────────
async function loadSettings() {
  try {
    const data = await fetchGateway('/api/settings')
    settingsModel.value = data.agent?.model || ''
    settingsTemp.value = data.agent?.temperature ?? 0.1
    settingsMaxTokens.value = data.agent?.max_tokens ?? 8192
  } catch (e) {
    console.warn('loadSettings failed:', e)
  }
}

async function saveSettings() {
  try {
    await fetchGateway('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: settingsModel.value,
        temperature: settingsTemp.value,
        max_tokens: settingsMaxTokens.value,
      }),
    })
  } catch (e) {
    console.warn('saveSettings failed:', e)
  }
}

// ── Send Message ────────────────────────────────────────────
function sendMessage() {
  const text = inputText.value.trim()
  if (!text || isStreaming.value) return
  messages.value.push({
    id: crypto.randomUUID(),
    role: 'user',
    content: text,
    status: 'done',
  })
  send(text)
  inputText.value = ''
  nextTick(() => {
    scrollToBottom()
    inputEl.value?.focus()
  })
}

// ── Status ──────────────────────────────────────────────────
async function fetchStatus() {
  try {
    const resp = await fetch('/api/agent/status')
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    status.value = await resp.json()
  } catch (e) {
    status.value = {
      available: false,
      state: 'error',
      error: e instanceof Error ? e.message : String(e),
      hint: '无法连接到 /api/agent/status — 检查后端服务是否启动。',
    }
  } finally {
    loading.value = false
  }
}

async function restart() {
  loading.value = true
  try {
    await fetch('/api/agent/restart', { method: 'POST' })
  } catch { /* ignore */ }
  await fetchStatus()
}

// ── Helpers ─────────────────────────────────────────────────
function renderMarkdown(text: string): string {
  if (!text) return ''
  return md.render(text)
}

function scrollToBottom() {
  if (messagesEl.value) {
    messagesEl.value.scrollTop = messagesEl.value.scrollHeight
  }
}

function formatTime(iso?: string): string {
  if (!iso) return ''
  const d = new Date(iso)
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`
}

function summarizeArgs(args: any): string {
  if (!args) return '()'
  const s = JSON.stringify(args)
  return s.length > 80 ? s.slice(0, 77) + '...' : s
}

function summarizeContent(content: any): string {
  if (content == null) return '(empty)'
  const s = typeof content === 'string' ? content : JSON.stringify(content)
  return s.length > 120 ? s.slice(0, 117) + '...' : s
}

// ── Lifecycle ───────────────────────────────────────────────
onMounted(async () => {
  await fetchStatus()
  if (status.value?.state === 'running') {
    await connect()
    await refreshSessions()
    await loadSettings()
  }
  // Poll while starting
  pollHandle = window.setInterval(async () => {
    if (status.value?.state === 'starting' || status.value?.state === 'uninitialized') {
      await fetchStatus()
    }
  }, 5000)
})

onUnmounted(() => {
  if (pollHandle !== null) window.clearInterval(pollHandle)
  disconnect()
})
</script>

<style scoped>
.agent-chat {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 64px);
  background: #fff;
  overflow: hidden;
}

/* Status overlay (loading/error/unavailable) */
.status-overlay {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  gap: 16px;
  background: #fafafa;
}

/* Status bar */
.chat-status-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 16px;
  border-bottom: 1px solid #f0f0f0;
  font-size: 13px;
  flex-shrink: 0;
}
.ws-dot { font-weight: 600; }
.ws-dot.connected { color: #52c41a; }
.ws-dot.disconnected { color: #999; }
.model-name { color: #1890ff; font-family: monospace; }
.spacer { flex: 1; }

/* Body layout */
.chat-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* Sidebar */
.chat-sidebar {
  width: 240px;
  border-right: 1px solid #f0f0f0;
  display: flex;
  flex-direction: column;
  padding: 12px;
  gap: 8px;
  overflow: hidden;
}
.new-session-btn { flex-shrink: 0; }
.session-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.session-item {
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s;
}
.session-item:hover { background: #f5f5f5; }
.session-item.active { background: #e6f7ff; border-left: 3px solid #1890ff; }
.session-title { font-size: 13px; line-height: 1.4; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.session-time { font-size: 11px; color: #999; margin-top: 2px; }

/* Main chat area */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

/* Messages */
.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #999;
}
.empty-state h2 { color: #333; margin-bottom: 8px; }

.message { margin-bottom: 16px; }
.message-role {
  font-size: 12px;
  font-weight: 600;
  color: #666;
  margin-bottom: 4px;
}
.user .message-content {
  background: #e6f7ff;
  border-radius: 8px;
  padding: 12px;
}
.assistant .message-content {
  padding: 0;
}

/* Markdown content styles */
.message-content :deep(p) { margin: 0 0 8px; }
.message-content :deep(p:last-child) { margin-bottom: 0; }
.message-content :deep(code) { background: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-size: 13px; }
.message-content :deep(pre) { background: #f6f6f6; padding: 12px; border-radius: 6px; overflow-x: auto; margin: 8px 0; }
.message-content :deep(pre code) { background: none; padding: 0; }
.message-content :deep(ul), .message-content :deep(ol) { padding-left: 20px; margin: 8px 0; }
.message-content :deep(strong) { font-weight: 600; }
.message-content :deep(table) { border-collapse: collapse; margin: 8px 0; }
.message-content :deep(th), .message-content :deep(td) { border: 1px solid #ddd; padding: 6px 10px; font-size: 13px; }

/* Reasoning block */
.reasoning {
  background: #f6f6f6;
  border-left: 3px solid #d9d9d9;
  padding: 8px 12px;
  margin: 6px 0;
  border-radius: 0 6px 6px 0;
}
.reasoning summary { font-size: 13px; color: #666; cursor: pointer; }
.reasoning-text { font-size: 13px; color: #555; margin-top: 6px; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }

/* Tool calls / results */
.tool-calls, .tool-results { margin: 4px 0; }
.tool-call, .tool-result {
  display: flex;
  align-items: baseline;
  gap: 6px;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 13px;
  margin: 3px 0;
}
.tool-call { background: #fff7e6; border-left: 3px solid #faad14; }
.tool-result { background: #f6ffed; border-left: 3px solid #52c41a; }
.tool-result.error { background: #fff2f0; border-left: 3px solid #ff4d4f; }
.tool-icon { font-size: 14px; }
.tool-name { font-weight: 600; font-family: monospace; }
.tool-args, .tool-content { color: #666; font-family: monospace; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* Input */
.chat-input {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid #f0f0f0;
  flex-shrink: 0;
}
.chat-input textarea {
  flex: 1;
  resize: none;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 14px;
  font-family: inherit;
  line-height: 1.5;
}
.chat-input textarea:focus { outline: none; border-color: #1890ff; box-shadow: 0 0 0 2px rgba(24,144,255,0.1); }
.send-btn {
  padding: 8px 24px;
  background: #1890ff;
  color: #fff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  white-space: nowrap;
  align-self: flex-end;
}
.send-btn:hover { background: #40a9ff; }
.send-btn:disabled { background: #d9d9d9; cursor: not-allowed; }

/* Settings panel */
.chat-settings {
  width: 260px;
  border-left: 1px solid #f0f0f0;
  padding: 16px;
  overflow-y: auto;
  flex-shrink: 0;
}
.chat-settings h4 { margin: 0 0 12px; font-size: 14px; }
.setting-group { margin-bottom: 12px; }
.setting-group label { display: block; font-size: 12px; color: #666; margin-bottom: 4px; }

/* Footer */
.chat-footer {
  display: flex;
  gap: 16px;
  padding: 6px 16px;
  border-top: 1px solid #f0f0f0;
  font-size: 12px;
  color: #999;
  flex-shrink: 0;
}
</style>
