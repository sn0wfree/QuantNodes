<template>
  <div class="chat-statusbar">
    <span class="status-item">
      <span class="status-label">agent:</span> {{ agentName }}
    </span>
    <span class="separator">·</span>
    <span class="status-item">
      <span class="status-label">model:</span> {{ modelName }}
    </span>
    <span v-if="tokenCount" class="separator">·</span>
    <span v-if="tokenCount" class="status-item">
      <span class="status-label">tokens:</span> {{ tokenCount.toLocaleString() }}
    </span>
    <span v-if="cost" class="separator">·</span>
    <span v-if="cost" class="status-item">
      ${{ cost.toFixed(4) }}
    </span>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  agentName?: string
  modelName?: string
  tokenCount?: number
  cost?: number
}>()
</script>

<style scoped>
.chat-statusbar {
  height: 28px;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 16px;
  border-top: 1px solid var(--chat-border-color, #f0f0f0);
  background: var(--chat-bg-secondary, #fafafa);
  font-size: 12px;
  color: var(--chat-text-muted, #999);
  flex-shrink: 0;
}

.status-item {
  white-space: nowrap;
}

.status-label {
  color: var(--chat-text-muted, #999);
}

.separator {
  color: var(--chat-border-color, #d9d9d9);
}
</style>
