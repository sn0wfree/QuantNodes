<template>
  <div class="chat-layout">
    <ChatNavSidebar
      :sessions="agentStore.sessions"
      :activeSessionId="agentStore.sessionId"
      @newChat="handleNewChat"
      @switchSession="handleSwitchSession"
    />
    <div class="chat-main">
      <div class="chat-topbar">
        <div class="topbar-left">
          <span class="topbar-logo">
            <experiment-outlined />
          </span>
        </div>
        <div class="topbar-right">
          <a-tooltip title="Command Palette (Ctrl+K)">
            <a-button type="text" size="small" @click="showCommandPalette = true">
              <template #icon><search-outlined /></template>
            </a-button>
          </a-tooltip>
          <a-tooltip title="Settings">
            <a-button type="text" size="small" @click="router.push('/settings')">
              <template #icon><setting-outlined /></template>
            </a-button>
          </a-tooltip>
        </div>
      </div>
      <div class="chat-body">
        <router-view />
      </div>
    </div>
    <CommandPalette :open="showCommandPalette" @close="showCommandPalette = false" />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAgentStore } from '@/stores/agent'
import ChatNavSidebar from '../Chat/ChatNavSidebar.vue'
import CommandPalette from '../Chat/CommandPalette.vue'
import { ExperimentOutlined, SearchOutlined, SettingOutlined } from '@ant-design/icons-vue'

const router = useRouter()
const agentStore = useAgentStore()
const showCommandPalette = ref(false)

const handleNewChat = async () => {
  await agentStore.createSession()
}

const handleSwitchSession = async (id: string) => {
  await agentStore.switchSession(id)
}
</script>

<style scoped>
.chat-layout {
  height: 100vh;
  display: flex;
  background: var(--chat-bg-primary, #ffffff);
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-width: 0;
}

.chat-topbar {
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  border-bottom: 1px solid var(--chat-border-color, #f0f0f0);
  flex-shrink: 0;
}

.topbar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.topbar-logo {
  font-size: 18px;
  color: #1677ff;
}

.topbar-right {
  display: flex;
  align-items: center;
  gap: 4px;
}

.chat-body {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
