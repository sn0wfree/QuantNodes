<template>
  <div class="tool-call-card" :class="{ 'card-running': status === 'running' }">
    <div class="tool-header" @click="expanded = !expanded">
      <right-outlined :class="{ expanded }" class="expand-icon" />
      <tool-outlined class="tool-icon" />
      <span class="tool-name">{{ toolName }}</span>
      <a-tag :color="statusColor" size="small">{{ status }}</a-tag>
      <span class="tool-summary" v-if="summaryText">{{ summaryText }}</span>
    </div>
    <div v-show="expanded" class="tool-detail">
      <div class="section" v-if="hasArgs">
        <div class="section-label">Arguments</div>
        <pre class="code-block">{{ formatJson(arguments) }}</pre>
      </div>
      <div class="section" v-if="result">
        <div class="section-label">Result</div>
        <pre class="code-block">{{ formatResult(result) }}</pre>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { ToolOutlined, RightOutlined } from '@ant-design/icons-vue'

const props = defineProps<{
  toolName: string
  arguments?: Record<string, any>
  result?: Record<string, any> | string
  status: 'running' | 'success' | 'error'
}>()

const expanded = ref(false)

const hasArgs = computed(() => {
  if (!props.arguments) return false
  return Object.keys(props.arguments).length > 0
})

const summaryText = computed(() => {
  if (!props.arguments) return ''
  const args = props.arguments
  if (args.pattern) return args.pattern
  if (args.query) return args.query
  if (args.file_path) return args.file_path
  if (args.command) return args.command
  const values = Object.values(args).filter(v => typeof v === 'string')
  if (values.length > 0) return values.slice(0, 2).join(' ')
  return ''
})

const statusColor = computed(() => {
  switch (props.status) {
    case 'running': return 'processing'
    case 'success': return 'success'
    case 'error': return 'error'
    default: return 'default'
  }
})

const formatJson = (obj: any) => {
  if (!obj) return ''
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

const formatResult = (result: any) => {
  if (typeof result === 'string') return result
  if (result?.output) return result.output
  return formatJson(result)
}
</script>

<style scoped>
.tool-call-card {
  border: 1px solid var(--chat-border-color, #f0f0f0);
  border-radius: 6px;
  background: var(--chat-bg-secondary, #fafafa);
  font-size: 13px;
}

.card-running {
  border-color: #1677ff;
}

.tool-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  cursor: pointer;
  user-select: none;
}

.tool-header:hover {
  background: rgba(0, 0, 0, 0.02);
}

.expand-icon {
  font-size: 10px;
  transition: transform 0.2s;
  color: var(--chat-text-muted, #999);
}

.expand-icon.expanded {
  transform: rotate(90deg);
}

.tool-icon {
  color: #1677ff;
}

.tool-name {
  font-weight: 500;
  color: var(--chat-text-primary, #333);
}

.tool-summary {
  color: var(--chat-text-muted, #999);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 300px;
}

.tool-detail {
  padding: 0 10px 8px;
}

.section {
  margin-bottom: 6px;
}

.section:last-child {
  margin-bottom: 0;
}

.section-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--chat-text-secondary, #666);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}

.code-block {
  background: #f6f8fa;
  padding: 6px 10px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 0;
  font-size: 12px;
  max-height: 150px;
  overflow-y: auto;
  line-height: 1.4;
}
</style>
