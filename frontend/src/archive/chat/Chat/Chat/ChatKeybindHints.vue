<template>
  <div class="chat-keybinds" v-if="show">
    <template v-if="isStreaming">
      <span class="hint"><kbd>esc</kbd> interrupt</span>
    </template>
    <template v-else>
      <span class="hint"><kbd>ctrl+k</kbd> commands</span>
      <span class="hint"><kbd>ctrl+n</kbd> new</span>
    </template>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  isStreaming?: boolean
  show?: boolean
}>()
</script>

<style scoped>
.chat-keybinds {
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  font-size: 11px;
  color: var(--chat-text-muted, #999);
  flex-shrink: 0;
}

.hint {
  display: flex;
  align-items: center;
  gap: 4px;
}

kbd {
  display: inline-block;
  padding: 0 4px;
  font-size: 10px;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  color: var(--chat-text-secondary, #666);
  background: var(--chat-bg-primary, #fff);
  border: 1px solid var(--chat-border-color, #d9d9d9);
  border-radius: 3px;
}
</style>
