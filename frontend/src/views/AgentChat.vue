<template>
  <div class="agent-chat">
    <div v-if="loading" class="status-overlay">
      <a-spin size="large" />
      <p>正在加载 nanobot 运行时状态…</p>
    </div>

    <div v-else-if="!status?.available" class="status-unavailable">
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

    <div v-else-if="status.state === 'starting'" class="status-overlay">
      <a-spin size="large" />
      <p>Agent 运行时启动中（state=starting）…</p>
    </div>

    <div v-else-if="status.state === 'error'" class="status-overlay">
      <a-result status="error" title="Agent 运行时启动失败">
        <template #subTitle>
          <p>{{ status.error }}</p>
          <a-typography-paragraph v-if="status.hint">
            {{ status.hint }}
          </a-typography-paragraph>
          <a-button type="primary" @click="restart">重试</a-button>
        </template>
      </a-result>
    </div>

    <iframe
      v-else-if="status.state === 'running'"
      :src="webuiUrl"
      class="webui-iframe"
      title="nanobot WebUI"
      sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
    />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'

interface AgentStatus {
  available: boolean
  state: 'uninitialized' | 'starting' | 'running' | 'stopping' | 'stopped' | 'error' | 'unavailable'
  hint?: string
  error?: string
  gateway_host?: string
  gateway_port?: number
  workspace?: string
  components?: Record<string, boolean>
}

const status = ref<AgentStatus | null>(null)
const loading = ref(true)

// v3.0.0 Stage 5.3: iframe URL is configurable via Vite env.
// Default points to the in-process nanobot gateway (cfg.gateway.port = 18080).
const webuiUrl = (import.meta.env.VITE_NANOBOT_GATEWAY_URL as string) || 'http://127.0.0.1:18080/'

let pollHandle: number | null = null

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
  } catch {
    /* swallow; status will reflect */
  }
  await fetchStatus()
}

onMounted(() => {
  fetchStatus()
  // Poll every 5s while in transitional states
  pollHandle = window.setInterval(async () => {
    if (status.value?.state === 'starting' || status.value?.state === 'uninitialized') {
      await fetchStatus()
    }
  }, 5000)
})

onUnmounted(() => {
  if (pollHandle !== null) {
    window.clearInterval(pollHandle)
    pollHandle = null
  }
})
</script>

<style scoped>
.agent-chat {
  position: relative;
  width: 100%;
  height: calc(100vh - 64px); /* minus AppHeader */
  background: #fff;
}

.webui-iframe {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}

.status-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  background: #fafafa;
  z-index: 1;
}

.status-unavailable {
  padding: 80px 24px;
}
</style>
