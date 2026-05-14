<template>
  <a-modal
    v-model:open="visible"
    title="Tool Permission Required"
    :footer="null"
    :width="480"
    :maskClosable="false"
    :closable="false"
    @cancel="handleDeny"
  >
    <div class="permission-dialog">
      <div class="permission-header">
        <warning-outlined class="permission-icon" />
        <span class="permission-tool">{{ toolName }}</span>
      </div>

      <div class="permission-description">
        The agent wants to execute <strong>{{ toolName }}</strong>.
      </div>

      <div class="permission-args" v-if="hasArgs">
        <div class="args-label" @click="argsExpanded = !argsExpanded">
          <right-outlined :class="{ expanded: argsExpanded }" />
          Arguments
        </div>
        <div v-show="argsExpanded" class="args-content">
          <pre>{{ formatJson(arguments) }}</pre>
        </div>
      </div>

      <div class="permission-actions">
        <a-button type="primary" @click="handleAllow">
          Allow
        </a-button>
        <a-button @click="handleAllowForSession">
          Allow for Session
        </a-button>
        <a-button danger @click="handleDeny">
          Deny
        </a-button>
      </div>
    </div>
  </a-modal>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { WarningOutlined, RightOutlined } from '@ant-design/icons-vue'

const props = defineProps<{
  open: boolean
  toolName: string
  arguments?: Record<string, any>
  requestId: string
}>()

const emit = defineEmits<{
  allow: [requestId: string, remember: boolean]
  deny: [requestId: string]
  close: []
}>()

const visible = computed({
  get: () => props.open,
  set: () => emit('close'),
})

const argsExpanded = ref(false)

const hasArgs = computed(() => {
  if (!props.arguments) return false
  return Object.keys(props.arguments).length > 0
})

const formatJson = (obj: any) => {
  if (!obj) return ''
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

const handleAllow = () => {
  emit('allow', props.requestId, false)
}

const handleAllowForSession = () => {
  emit('allow', props.requestId, true)
}

const handleDeny = () => {
  emit('deny', props.requestId)
}
</script>

<style scoped>
.permission-dialog {
  padding: 0;
}

.permission-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.permission-icon {
  color: #faad14;
  font-size: 20px;
}

.permission-tool {
  font-weight: 600;
  font-size: 16px;
}

.permission-description {
  margin-bottom: 16px;
  color: #666;
}

.permission-args {
  margin-bottom: 16px;
}

.args-label {
  display: flex;
  align-items: center;
  gap: 4px;
  cursor: pointer;
  font-weight: 500;
  color: #666;
  margin-bottom: 8px;
}

.args-label:hover {
  color: #1677ff;
}

.args-content pre {
  background: #f6f8fa;
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 12px;
  max-height: 200px;
  overflow-y: auto;
  margin: 0;
}

.permission-actions {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}
</style>
