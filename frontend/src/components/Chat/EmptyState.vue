<template>
  <div class="empty-state">
    <div class="welcome">
      <div class="welcome-icon">
        <experiment-outlined />
      </div>
      <h2>Welcome to QuantNodes Agent</h2>
      <p class="welcome-desc">Ask me anything about your quant research.</p>

      <div class="suggestions">
        <div class="suggestion" v-for="s in suggestions" :key="s.text" @click="$emit('send', s.text)">
          <component :is="s.icon" class="suggestion-icon" />
          <span>{{ s.text }}</span>
        </div>
      </div>

      <div class="shortcuts">
        <span class="shortcut-item"><kbd>Ctrl+K</kbd> Command palette</span>
        <span class="shortcut-item"><kbd>Ctrl+O</kbd> Switch model</span>
        <span class="shortcut-item"><kbd>Ctrl+N</kbd> New chat</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import {
  ExperimentOutlined,
  BarChartOutlined,
  LineChartOutlined,
  BulbOutlined,
} from '@ant-design/icons-vue'

defineEmits<{
  send: [text: string]
}>()

const suggestions = [
  { text: 'Analyze factor performance for the past year', icon: BarChartOutlined },
  { text: 'Run a backtest on the momentum strategy', icon: LineChartOutlined },
  { text: 'Show me the latest dream insights', icon: BulbOutlined },
]
</script>

<style scoped>
.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
  padding: 40px;
}

.welcome {
  text-align: center;
  max-width: 500px;
}

.welcome-icon {
  font-size: 48px;
  color: #1677ff;
  margin-bottom: 16px;
}

.welcome h2 {
  font-size: 24px;
  font-weight: 600;
  color: var(--chat-text-primary, #333);
  margin: 0 0 8px;
}

.welcome-desc {
  font-size: 15px;
  color: var(--chat-text-secondary, #666);
  margin: 0 0 32px;
}

.suggestions {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 32px;
}

.suggestion {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  border: 1px solid var(--chat-border-color, #e8e8e8);
  border-radius: 8px;
  cursor: pointer;
  text-align: left;
  font-size: 14px;
  color: var(--chat-text-primary, #333);
  transition: all 0.2s;
}

.suggestion:hover {
  border-color: #1677ff;
  background: #f0f5ff;
}

.suggestion-icon {
  color: #1677ff;
  flex-shrink: 0;
}

.shortcuts {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
}

.shortcut-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--chat-text-muted, #999);
}

kbd {
  display: inline-block;
  padding: 1px 5px;
  font-size: 11px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--chat-text-secondary, #666);
  background: var(--chat-bg-primary, #fff);
  border: 1px solid var(--chat-border-color, #d9d9d9);
  border-radius: 3px;
  box-shadow: 0 1px 0 var(--chat-border-color, #d9d9d9);
}
</style>
