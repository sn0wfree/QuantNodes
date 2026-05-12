<template>
  <div class="chat-keybinds">
    <template v-if="isStreaming">
      <span class="hint"><kbd>esc</kbd> interrupt</span>
    </template>
    <template v-else>
      <span class="hint"><kbd>ctrl+k</kbd> commands</span>
      <span class="hint"><kbd>ctrl+n</kbd> new</span>
      <span class="hint"><kbd>ctrl+o</kbd> model</span>
      <span class="hint"><kbd>?</kbd> help</span>
    </template>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  isStreaming?: boolean
}>()
</script>

<style scoped>
.chat-keybinds {
  height: 28px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  border-top: 1px solid var(--chat-border-color, #f0f0f0);
  background: var(--chat-bg-secondary, #fafafa);
  font-size: 12px;
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
