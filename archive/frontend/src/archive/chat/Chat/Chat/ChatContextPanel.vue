<template>
  <div class="context-panel" :class="{ collapsed: !open }">
    <div class="panel-header">
      <span class="panel-title">Context</span>
      <a-button type="text" size="small" @click="$emit('close')">
        <template #icon><close-outlined /></template>
      </a-button>
    </div>

    <div class="panel-body">
      <div class="section session-info">
        <div class="section-title">Session Info</div>
        <div class="info-row">
          <span class="info-label">Model</span>
          <span class="info-value">{{ modelName }}</span>
        </div>
        <div class="info-row">
          <span class="info-label">Messages</span>
          <span class="info-value">{{ messages.length }}</span>
        </div>
        <div class="info-row">
          <span class="info-label">Mode</span>
          <span class="info-value">
            <span :class="modeClass">{{ modeLabel }}</span>
          </span>
        </div>
      </div>

      <div class="section token-usage">
        <div class="section-title">
          Token Usage
          <span class="token-count">{{ totalTokens.toLocaleString() }} / {{ contextWindow }}</span>
        </div>
        <div class="progress-track">
          <div class="progress-fill" :style="{ width: tokenPercent + '%' }" />
        </div>
        <div class="progress-label" :class="{ warning: tokenPercent > 80 }">
          {{ tokenPercent }}% of context window
        </div>
      </div>

      <div class="section context-breakdown">
        <div class="section-title">Context Breakdown</div>
        <div class="breakdown-bar">
          <div
            v-if="userPercent > 0"
            class="breakdown-segment user-seg"
            :style="{ width: userPercent + '%' }"
            :title="`User: ${userPercent}%`"
          />
          <div
            v-if="assistantPercent > 0"
            class="breakdown-segment assistant-seg"
            :style="{ width: assistantPercent + '%' }"
            :title="`Assistant: ${assistantPercent}%`"
          />
          <div
            v-if="systemPercent > 0"
            class="breakdown-segment system-seg"
            :style="{ width: systemPercent + '%' }"
            :title="`System: ${systemPercent}%`"
          />
        </div>
        <div class="breakdown-legend">
          <div class="legend-item">
            <span class="legend-dot user-dot" /> User {{ userPercent }}%
          </div>
          <div class="legend-item">
            <span class="legend-dot assistant-dot" /> Assistant {{ assistantPercent }}%
          </div>
          <div v-if="systemPercent > 0" class="legend-item">
            <span class="legend-dot system-dot" /> System {{ systemPercent }}%
          </div>
        </div>
      </div>

      <div class="section">
        <div class="section-title">Files Changed</div>
        <div class="placeholder-text">
          <file-outlined /> File tracking coming in Phase 2
        </div>
      </div>

      <div class="section">
        <div class="section-title">Git History</div>
        <div class="placeholder-text">
          <code-outlined /> Git integration coming in Phase 2
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAgentStore } from '@/stores/agent'
import { getModelInfo } from '@/constants/models'
import { CloseOutlined, FileOutlined, CodeOutlined } from '@ant-design/icons-vue'

defineProps<{
  open: boolean
}>()

defineEmits<{
  close: []
}>()

const store = useAgentStore()

const messages = computed(() => store.messages)

const modelName = computed(() => {
  const info = getModelInfo(store.currentModel)
  return info?.name || store.currentModel.split('/').pop() || 'Unknown'
})

const contextWindow = computed(() => {
  const info = getModelInfo(store.currentModel)
  if (info?.contextWindow) {
    return formatContextWindow(info.contextWindow)
  }
  return '128K'
})

const modeLabel = computed(() => store.currentMode === 'build' ? 'Build' : 'Plan')
const modeClass = computed(() => store.currentMode === 'build' ? 'mode-build' : 'mode-plan')

const estimateTokens = (text: string) => Math.ceil(text.length / 4)

const userTokens = computed(() => {
  return messages.value
    .filter(m => m.role === 'user')
    .reduce((sum, m) => sum + estimateTokens(m.content), 0)
})

const assistantTokens = computed(() => {
  return messages.value
    .filter(m => m.role === 'assistant')
    .reduce((sum, m) => sum + estimateTokens(m.content), 0)
})

const systemTokens = computed(() => store.systemMessages.length * 200)

const totalTokens = computed(() => userTokens.value + assistantTokens.value + systemTokens.value)

const modelInfo = computed(() => getModelInfo(store.currentModel))
const maxTokens = computed(() => modelInfo.value?.contextWindow || 128000)

const tokenPercent = computed(() => {
  if (maxTokens.value === 0) return 0
  return Math.min(100, Math.round((totalTokens.value / maxTokens.value) * 100))
})

const totalBreakdown = computed(() => userTokens.value + assistantTokens.value + systemTokens.value || 1)

const userPercent = computed(() => Math.round((userTokens.value / totalBreakdown.value) * 100))
const assistantPercent = computed(() => Math.round((assistantTokens.value / totalBreakdown.value) * 100))
const systemPercent = computed(() => Math.round((systemTokens.value / totalBreakdown.value) * 100))

const formatContextWindow = (tokens: number) => {
  if (tokens >= 1000000) return `${(tokens / 1000000).toFixed(0)}M`
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(0)}K`
  return String(tokens)
}
</script>

<style scoped>
.context-panel {
  width: 280px;
  background: var(--chat-bg-secondary, #fafafa);
  border-left: 1px solid var(--chat-border-color, #f0f0f0);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow: hidden;
  transition: width 0.2s ease, opacity 0.2s ease;
}

.context-panel.collapsed {
  width: 0;
  opacity: 0;
  border-left: none;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 36px;
  padding: 0 12px;
  border-bottom: 1px solid var(--chat-border-color, #f0f0f0);
  flex-shrink: 0;
}

.panel-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--chat-text-primary, #333);
}

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.section {
  margin-bottom: 16px;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--chat-text-muted, #999);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 4px 0;
  font-size: 12px;
}

.info-label {
  color: var(--chat-text-secondary, #666);
}

.info-value {
  color: var(--chat-text-primary, #333);
  font-weight: 500;
}

.mode-build {
  color: var(--chat-build-color, #1677ff);
}

.mode-plan {
  color: var(--chat-plan-color, #52c41a);
}

.token-count {
  font-size: 11px;
  font-weight: 400;
  text-transform: none;
  letter-spacing: 0;
}

.progress-track {
  height: 6px;
  background: var(--chat-bg-tertiary, #f0f0f0);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--chat-success, #52c41a), var(--chat-warning, #faad14));
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-label {
  font-size: 11px;
  color: var(--chat-text-muted, #999);
}

.progress-label.warning {
  color: var(--chat-warning, #faad14);
}

.breakdown-bar {
  height: 8px;
  display: flex;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.breakdown-segment {
  transition: width 0.3s ease;
}

.user-seg {
  background: var(--chat-build-color, #1677ff);
}

.assistant-seg {
  background: var(--chat-plan-color, #52c41a);
}

.system-seg {
  background: var(--chat-warning, #faad14);
}

.breakdown-legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.legend-item {
  font-size: 11px;
  color: var(--chat-text-secondary, #666);
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-dot {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  flex-shrink: 0;
}

.user-dot {
  background: var(--chat-build-color, #1677ff);
}

.assistant-dot {
  background: var(--chat-plan-color, #52c41a);
}

.system-dot {
  background: var(--chat-warning, #faad14);
}

.placeholder-text {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
  padding: 8px 0;
  display: flex;
  align-items: center;
  gap: 6px;
}
</style>
