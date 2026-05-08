<template>
  <div class="tool-call-card">
    <a-card size="small" :bordered="true">
      <template #title>
        <span class="tool-name">
          <tool-outlined />
          {{ toolName }}
        </span>
        <a-tag :color="statusColor" class="status-tag">{{ status }}</a-tag>
      </template>
      <div class="tool-content">
        <div class="section" v-if="arguments">
          <div class="section-label">Arguments:</div>
          <pre class="code-block">{{ formatJson(arguments) }}</pre>
        </div>
        <div class="section" v-if="result">
          <div class="section-label">Result:</div>
          <pre class="code-block">{{ formatJson(result) }}</pre>
        </div>
      </div>
    </a-card>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { ToolOutlined } from '@ant-design/icons-vue'

const props = defineProps<{
  toolName: string
  arguments?: Record<string, any>
  result?: Record<string, any>
  status: 'running' | 'success' | 'error'
}>()

const statusColor = computed(() => {
  switch (props.status) {
    case 'running':
      return 'processing'
    case 'success':
      return 'success'
    case 'error':
      return 'error'
    default:
      return 'default'
  }
})

const formatJson = (obj: any) => {
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}
</script>

<style scoped>
.tool-call-card {
  margin: 8px 0;
}

.tool-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-weight: 500;
}

.status-tag {
  margin-left: auto;
}

.tool-content {
  font-size: 13px;
}

.section {
  margin-bottom: 8px;
}

.section:last-child {
  margin-bottom: 0;
}

.section-label {
  font-weight: 500;
  margin-bottom: 4px;
  color: #666;
}

.code-block {
  background: #f6f8fa;
  padding: 8px 12px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 0;
  font-size: 12px;
}
</style>
