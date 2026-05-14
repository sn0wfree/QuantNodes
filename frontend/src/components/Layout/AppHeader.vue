<template>
  <a-layout-header class="header">
    <div class="logo">
      <experiment-outlined />
      <span>QuantNodes</span>
    </div>
    <a-menu
      theme="dark"
      mode="horizontal"
      :selectedKeys="selectedKeys"
      @click="handleMenuClick"
    >
      <a-menu-item key="chat">
        <message-outlined />
        <span>Agent Chat</span>
      </a-menu-item>
    </a-menu>
    <div class="header-actions">
      <template v-if="isChatRoute">
        <a-tooltip title="Compact session">
          <a-button type="text" size="small" class="action-btn" @click="$emit('compact')">
            <template #icon><compress-outlined /></template>
          </a-button>
        </a-tooltip>
        <a-tooltip title="Share session">
          <a-button type="text" size="small" class="action-btn" @click="$emit('share')">
            <template #icon><share-alt-outlined /></template>
          </a-button>
        </a-tooltip>
        <a-tooltip title="Toggle context panel">
          <a-button type="text" size="small" class="action-btn" data-testid="toggle-panel" @click="$emit('togglePanel')">
            <template #icon><right-square-outlined /></template>
          </a-button>
        </a-tooltip>
      </template>
    </div>
  </a-layout-header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  ExperimentOutlined,
  MessageOutlined,
  CompressOutlined,
  ShareAltOutlined,
  RightSquareOutlined,
} from '@ant-design/icons-vue'

defineEmits<{
  compact: []
  share: []
  togglePanel: []
}>()

const router = useRouter()
const route = useRoute()

const selectedKeys = computed(() => {
  if (route.path.startsWith('/chat')) return ['chat']
  return []
})

const isChatRoute = computed(() => route.path.startsWith('/chat'))

const handleMenuClick = ({ key }: { key: string }) => {
  router.push(`/${key}`)
}
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: #001529;
  gap: 16px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
  font-size: 18px;
  font-weight: 600;
  margin-right: 24px;
}

.logo :deep(span) {
  color: #fff;
}

.header-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 2px;
}

.action-btn {
  color: rgba(255, 255, 255, 0.65) !important;
}

.action-btn:hover {
  color: #fff !important;
}
</style>