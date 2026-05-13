<template>
  <div class="chat-input-footer">
    <div class="footer-left">
      <div class="agent-indicator" :class="modeClass" @click="$emit('toggleMode')">
        <span class="agent-dot" />
        <span class="agent-name">{{ modeLabel }}</span>
        <span class="agent-icon">{{ modeIcon }}</span>
      </div>
      <span class="separator">|</span>
      <span class="model-name clickable" @click="$emit('openModelSelector')">
        <span class="model-icon">🤖</span>
        {{ modelName || 'Select model' }}
      </span>
    </div>
    <div class="footer-right">
      <a-select
        v-model:value="qualityValue"
        size="small"
        :bordered="false"
        class="quality-select"
        @change="handleQualityChange"
      >
        <a-select-option value="high">High</a-select-option>
        <a-select-option value="medium">Medium</a-select-option>
        <a-select-option value="low">Low</a-select-option>
      </a-select>
      <span class="separator">|</span>
      <span class="token-count">{{ formattedTokens }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  currentMode: 'build' | 'plan'
  modelName?: string
  quality?: string
  tokenCount?: number
}>()

const emit = defineEmits<{
  toggleMode: []
  openModelSelector: []
  qualityChange: [quality: string]
}>()

const modeLabel = computed(() => props.currentMode === 'build' ? 'Build' : 'Plan')
const modeIcon = computed(() => props.currentMode === 'build' ? '⚡' : '📋')
const modeClass = computed(() => `agent-${props.currentMode}`)

const qualityValue = computed({
  get: () => props.quality || 'high',
  set: (val: string) => emit('qualityChange', val),
})

const handleQualityChange = (val: string) => {
  emit('qualityChange', val)
}

const formattedTokens = computed(() => {
  const n = props.tokenCount ?? 0
  if (n >= 1000000) return (n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return (n / 1000).toFixed(1) + 'k'
  return String(n)
})
</script>

<style scoped>
.chat-input-footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 28px;
  padding: 0 12px 2px;
  font-size: 12px;
  color: var(--chat-text-muted, #999);
  flex-shrink: 0;
}

.footer-left,
.footer-right {
  display: flex;
  align-items: center;
  gap: 6px;
}

.agent-indicator {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  padding: 1px 6px;
  border-radius: 4px;
  transition: background 0.15s;
  user-select: none;
}

.agent-indicator:hover {
  background: var(--chat-bg-secondary, #f5f5f5);
}

.agent-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.agent-build .agent-dot {
  background: var(--chat-build-color, #1677ff);
}

.agent-plan .agent-dot {
  background: var(--chat-plan-color, #52c41a);
}

.agent-name {
  font-weight: 600;
  font-size: 12px;
}

.agent-build .agent-name {
  color: var(--chat-build-color, #1677ff);
}

.agent-plan .agent-name {
  color: var(--chat-plan-color, #52c41a);
}

.agent-icon {
  font-size: 11px;
  line-height: 1;
}

.separator {
  color: var(--chat-border-color, #e0e0e0);
}

.model-name {
  display: flex;
  align-items: center;
  gap: 3px;
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.model-name.clickable {
  cursor: pointer;
}

.model-name.clickable:hover {
  color: var(--chat-text-secondary, #666);
}

.model-icon {
  font-size: 11px;
}

.quality-select {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
}

.quality-select :deep(.ant-select-selection-item) {
  font-size: 12px;
}

.token-count {
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
}
</style>
