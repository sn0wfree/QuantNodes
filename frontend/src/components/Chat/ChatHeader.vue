<template>
  <div class="chat-header">
    <div class="header-left">
      <a-segmented
        v-model:value="currentMode"
        :options="modeOptions"
        size="small"
        @change="handleModeChange"
      />
    </div>
    <div class="header-right">
      <span v-if="tokenCount" class="token-info">
        {{ tokenCount.toLocaleString() }} tokens
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAgentStore } from '@/stores/agent'

const store = useAgentStore()

const currentMode = computed({
  get: () => store.currentMode,
  set: (val) => store.switchMode(val as 'build' | 'plan'),
})

const modeOptions = [
  { label: 'Build', value: 'build' },
  { label: 'Plan', value: 'plan' },
]

defineProps<{
  currentSessionLabel: string
  tokenCount?: number
}>()

const handleModeChange = (val: string) => {
  store.switchMode(val as 'build' | 'plan')
}
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
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.token-info {
  font-size: 12px;
  color: var(--chat-text-muted, #999);
}
</style>
