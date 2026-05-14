<template>
  <div class="chat-header">
    <div class="header-left">
      <span class="session-name">{{ sessionLabel }}</span>
    </div>
    <div class="header-right">
      <a-tooltip title="Compact session">
        <a-button type="text" size="small" :disabled="!hasMessages" @click="$emit('compact')">
          <template #icon><compress-outlined /></template>
        </a-button>
      </a-tooltip>
      <a-tooltip title="Share session">
        <a-button type="text" size="small" :disabled="!hasMessages" @click="$emit('share')">
          <template #icon><share-alt-outlined /></template>
        </a-button>
      </a-tooltip>
      <a-tooltip title="Toggle context panel">
        <a-button type="text" size="small" @click="$emit('togglePanel')">
          <template #icon><right-square-outlined /></template>
        </a-button>
      </a-tooltip>
    </div>
  </div>
</template>

<script setup lang="ts">
import { CompressOutlined, ShareAltOutlined, RightSquareOutlined } from '@ant-design/icons-vue'

defineProps<{
  sessionLabel: string
  hasMessages?: boolean
}>()

defineEmits<{
  compact: []
  share: []
  togglePanel: []
}>()
</script>

<style scoped>
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 36px;
  padding: 0 16px;
  border-bottom: 1px solid var(--chat-border-color, #f0f0f0);
  flex-shrink: 0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.session-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--chat-text-primary, #333);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 2px;
}
</style>
