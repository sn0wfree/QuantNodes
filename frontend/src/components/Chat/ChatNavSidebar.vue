<template>
  <div class="chat-nav-sidebar" :class="{ collapsed: isCollapsed }">
    <div class="sidebar-collapsed" v-if="isCollapsed">
      <div class="nav-icons-top">
        <a-tooltip title="Expand sidebar" placement="right">
          <div class="nav-icon" @click="appStore.toggleChatSidebar()">
            <menu-unfold-outlined />
          </div>
        </a-tooltip>
        <a-tooltip v-for="item in navItems" :key="item.path" :title="item.label" placement="right">
          <div class="nav-icon" :class="{ active: isActive(item.path) }" @click="navigate(item.path)">
            <component :is="item.icon" />
          </div>
        </a-tooltip>
      </div>
      <div class="nav-icons-bottom">
        <a-tooltip title="Sessions" placement="right">
          <div class="nav-icon" @click="appStore.toggleChatSidebar()">
            <message-outlined />
          </div>
        </a-tooltip>
      </div>
    </div>

    <div class="sidebar-expanded" v-else>
      <div class="sidebar-header">
        <span class="sidebar-logo">
          <experiment-outlined />
          QuantNodes
        </span>
        <a-button type="text" size="small" @click="appStore.toggleChatSidebar()">
          <template #icon><menu-fold-outlined /></template>
        </a-button>
      </div>

      <a-menu mode="inline" theme="dark" :selectedKeys="selectedKeys" @click="handleMenuClick">
        <a-menu-item v-for="item in navItems" :key="item.path">
          <component :is="item.icon" />
          <span>{{ item.label }}</span>
        </a-menu-item>
      </a-menu>

      <a-divider style="margin: 4px 0; border-color: rgba(255,255,255,0.1)" />

      <div class="session-section">
        <div class="session-header">
          <span>Sessions</span>
          <a-button type="text" size="small" class="new-chat-btn" @click="$emit('newChat')">
            <template #icon><plus-outlined /></template>
          </a-button>
        </div>
        <div class="session-list">
          <div
            v-for="s in sessions"
            :key="s.session_id"
            class="session-item"
            :class="{ active: s.session_id === activeSessionId }"
            @click="switchSession(s.session_id)"
          >
            <span class="session-name" :title="s.session_id">{{ sessionLabel(s) }}</span>
            <span class="session-count">{{ s.message_count }}</span>
          </div>
          <div v-if="!sessions.length" class="session-empty">No sessions yet</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAppStore } from '@/stores/app'
import type { SessionInfo } from '@/stores/agent'
import {
  MenuUnfoldOutlined,
  MenuFoldOutlined,
  ExperimentOutlined,
  HomeOutlined,
  MessageOutlined,
  BookOutlined,
  CodeOutlined,
  LineChartOutlined,
  BarChartOutlined,
  BulbOutlined,
  SettingOutlined,
  PlusOutlined,
} from '@ant-design/icons-vue'

defineProps<{
  sessions: SessionInfo[]
  activeSessionId: string
}>()

const emit = defineEmits<{
  newChat: []
  switchSession: [id: string]
}>()

const router = useRouter()
const route = useRoute()
const appStore = useAppStore()

const isCollapsed = computed(() => appStore.chatSidebarCollapsed)

const navItems = [
  { path: '/', label: 'Dashboard', icon: HomeOutlined },
  { path: '/chat', label: 'Agent Chat', icon: MessageOutlined },
  { path: '/wiki/factors', label: 'Factors', icon: BookOutlined },
  { path: '/wiki/strategies', label: 'Strategies', icon: BookOutlined },
  { path: '/strategy/editor', label: 'Strategy Editor', icon: CodeOutlined },
  { path: '/backtest', label: 'Backtest Center', icon: LineChartOutlined },
  { path: '/factor-analysis', label: 'Factor Analysis', icon: BarChartOutlined },
  { path: '/dream', label: 'Dream Insights', icon: BulbOutlined },
  { path: '/settings', label: 'Settings', icon: SettingOutlined },
]

const selectedKeys = computed(() => {
  const path = route.path
  if (path.startsWith('/wiki/factors')) return ['/wiki/factors']
  if (path.startsWith('/wiki/strategies')) return ['/wiki/strategies']
  if (path.startsWith('/backtest')) return ['/backtest']
  return [path]
})

const isActive = (path: string) => {
  const current = route.path
  if (path === '/') return current === '/'
  return current.startsWith(path)
}

const navigate = (path: string) => {
  router.push(path)
}

const handleMenuClick = ({ key }: { key: string }) => {
  router.push(key)
}

const switchSession = (id: string) => {
  emit('switchSession', id)
}

const sessionLabel = (s: SessionInfo) => {
  if (s.first_message) {
    const text = s.first_message.replace(/[<>]/g, '').slice(0, 40)
    return text + (s.first_message.length > 40 ? '...' : '')
  }
  if (s.session_id === 'default') return 'Default Session'
  return s.session_id
}
</script>

<style scoped>
.chat-nav-sidebar {
  display: flex;
  flex-direction: column;
  background: #001529;
  color: rgba(255, 255, 255, 0.85);
  transition: width 0.25s cubic-bezier(0.4, 0, 0.2, 1);
  overflow: hidden;
  flex-shrink: 0;
}

.chat-nav-sidebar.collapsed {
  width: 48px;
}

.chat-nav-sidebar:not(.collapsed) {
  width: 200px;
}

.sidebar-collapsed {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  height: 100%;
  padding: 8px 0;
}

.nav-icons-top, .nav-icons-bottom {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.nav-icon {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 6px;
  cursor: pointer;
  font-size: 16px;
  color: rgba(255, 255, 255, 0.65);
  transition: all 0.2s;
}

.nav-icon:hover {
  color: #fff;
  background: rgba(255, 255, 255, 0.1);
}

.nav-icon.active {
  color: #fff;
  background: #1677ff;
}

.sidebar-expanded {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  height: 40px;
}

.sidebar-logo {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 15px;
  font-weight: 600;
  color: #fff;
}

.sidebar-expanded :deep(.ant-menu) {
  background: transparent;
}

.sidebar-expanded :deep(.ant-menu-item) {
  margin: 2px 8px;
  border-radius: 6px;
  height: 36px;
  line-height: 36px;
}

.sidebar-expanded :deep(.ant-menu-item:hover) {
  background: rgba(255, 255, 255, 0.1);
}

.sidebar-expanded :deep(.ant-menu-item-selected) {
  background: #1677ff;
}

.session-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  padding: 0 8px;
}

.session-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.45);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.new-chat-btn {
  color: rgba(255, 255, 255, 0.45);
}

.new-chat-btn:hover {
  color: #fff;
}

.session-list {
  flex: 1;
  overflow-y: auto;
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  color: rgba(255, 255, 255, 0.65);
  transition: all 0.2s;
}

.session-item:hover {
  background: rgba(255, 255, 255, 0.1);
  color: #fff;
}

.session-item.active {
  background: rgba(255, 255, 255, 0.15);
  color: #fff;
}

.session-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 120px;
}

.session-count {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.35);
  flex-shrink: 0;
}

.session-empty {
  padding: 8px;
  font-size: 12px;
  color: rgba(255, 255, 255, 0.3);
  text-align: center;
}
</style>
