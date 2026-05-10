<template>
  <div class="tool-call-card">
    <a-card size="small" :bordered="true" :class="{ 'card-running': status === 'running' }">
      <template #title>
        <span class="tool-name">
          <tool-outlined />
          {{ toolName }}
        </span>
        <a-tag :color="statusColor" class="status-tag">{{ status }}</a-tag>
      </template>
      <div class="tool-content">
        <div class="section" v-if="hasArgs">
          <div class="section-label" @click="argsExpanded = !argsExpanded" style="cursor: pointer">
            <right-outlined :class="{ expanded: argsExpanded }" style="font-size: 10px; margin-right: 4px; transition: transform 0.2s" />
            Arguments
          </div>
          <div v-show="argsExpanded">
            <pre class="code-block">{{ formatJson(arguments) }}</pre>
          </div>
        </div>
        <div class="section" v-if="result">
          <div class="section-label" @click="resultExpanded = !resultExpanded" style="cursor: pointer">
            <right-outlined :class="{ expanded: resultExpanded }" style="font-size: 10px; margin-right: 4px; transition: transform 0.2s" />
            Result
          </div>
          <div v-show="resultExpanded">
            <pre class="code-block">{{ formatResult(result) }}</pre>
          </div>
        </div>
      </div>
    </a-card>
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

const argsExpanded = ref(false)
const resultExpanded = ref(false)

const hasArgs = computed(() => {
  if (!props.arguments) return false
  return Object.keys(props.arguments).length > 0
})

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
  margin: 6px 0;
}

.card-running {
  border-color: #1677ff;
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
  display: flex;
  align-items: center;
}

.section-label:hover {
  color: #1677ff;
}

.code-block {
  background: #f6f8fa;
  padding: 8px 12px;
  border-radius: 4px;
  overflow-x: auto;
  margin: 0;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
}
</style>
