<template>
  <div class="chat-header">
    <div class="header-left">
      <span class="session-title"># {{ currentSessionLabel }}</span>
    </div>
    <div class="header-right">
      <span v-if="tokenCount" class="token-info">
        {{ tokenCount.toLocaleString() }} tokens
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  currentSessionLabel: string
  tokenCount?: number
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
}

.session-title {
  font-size: 14px;
  font-weight: 500;
  color: var(--chat-text-primary, #333);
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
