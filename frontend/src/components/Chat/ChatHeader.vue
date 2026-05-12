<template>
  <div class="chat-header">
    <div class="header-left">
      <span class="status-dot" :class="{ connected: isConnected }"></span>
      <a-dropdown :trigger="['click']">
        <a-button type="text" class="session-dropdown">
          {{ currentSessionLabel }}
          <template #icon><down-outlined /></template>
        </a-button>
        <template #overlay>
          <a-menu @click="onMenuClick">
            <a-menu-item key="new">
              <template #icon><plus-outlined /></template>
              New Chat
            </a-menu-item>
            <a-menu-divider v-if="sessions.length" />
            <a-menu-item
              v-for="s in sessions"
              :key="s.session_id"
              :class="{ 'session-active': s.session_id === activeSessionId }"
            >
              <div class="session-item">
                <span class="session-name">{{ s.session_id }}</span>
                <span class="session-count">{{ s.message_count }} msgs</span>
              </div>
            </a-menu-item>
          </a-menu>
        </template>
      </a-dropdown>
    </div>
    <div class="header-right">
      <a-tooltip title="Switch Model (Ctrl+O)">
        <a-button type="text" size="small" @click="$emit('openModelSelector')">
          <template #icon><control-outlined /></template>
        </a-button>
      </a-tooltip>
      <a-tooltip title="Commands (Ctrl+K)">
        <a-button type="text" size="small" @click="$emit('openCommandPalette')">
          <template #icon><appstore-outlined /></template>
        </a-button>
      </a-tooltip>
      <a-button type="text" size="small" @click="$emit('newChat')">
        <template #icon><plus-outlined /></template>
      </a-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { SessionInfo } from '@/stores/agent'
import { PlusOutlined, DownOutlined, ControlOutlined, AppstoreOutlined } from '@ant-design/icons-vue'

defineProps<{
  isConnected: boolean
  currentSessionLabel: string
  sessions: SessionInfo[]
  activeSessionId: string
}>()

const emit = defineEmits<{
  openModelSelector: []
  openCommandPalette: []
  newChat: []
  menuClick: [info: { key: string }]
}>()

const onMenuClick = (info: { key: string }) => {
  emit('menuClick', info)
}
</script>

<style scoped>
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #d9d9d9;
}

.status-dot.connected {
  background: #52c41a;
}

.session-dropdown {
  font-weight: 500;
  font-size: 15px;
}

.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.session-name {
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.session-count {
  font-size: 12px;
  color: #999;
}

.session-active {
  background: #e6f4ff;
}
</style>
