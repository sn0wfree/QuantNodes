<template>
  <a-modal
    v-model:open="visible"
    title="Switch Model"
    :footer="null"
    :width="480"
    :maskClosable="true"
    @cancel="emit('close')"
  >
    <div class="model-selector">
      <a-input
        v-model:value="query"
        placeholder="Search models..."
        class="search-input"
        allow-clear
      >
        <template #prefix><search-outlined /></template>
      </a-input>

      <a-spin v-if="store.modelsLoading" class="loading-spinner" />

      <div v-else class="model-list">
        <div v-if="filteredGroups.length === 0" class="no-results">
          No models found
        </div>
        <div v-for="group in filteredGroups" :key="group.provider" class="model-group">
          <div class="group-label">{{ group.provider }}</div>
          <div
            v-for="model in group.models"
            :key="model.id"
            class="model-item"
            :class="{ active: model.id === currentModel }"
            @click="handleSelect(model.id)"
          >
            <div class="model-info">
              <span class="model-name">
                <check-circle-filled v-if="model.id === currentModel" class="current-icon" />
                {{ model.name }}
              </span>
              <span class="model-tags" v-if="model.tags.length">
                <a-tag v-for="tag in model.tags" :key="tag" size="small" :color="getTagColor(tag)">
                  {{ tag }}
                </a-tag>
              </span>
            </div>
            <div class="model-meta">
              <span>context: {{ formatContextWindow(model.contextWindow) }}</span>
              <span>price: {{ formatPrice(model.priceIn) }} in</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </a-modal>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { SearchOutlined, CheckCircleFilled } from '@ant-design/icons-vue'
import { formatPrice, formatContextWindow } from '@/constants/models'
import { useAgentStore } from '@/stores/agent'

const props = defineProps<{
  open: boolean
  currentModel: string
}>()

const emit = defineEmits<{
  select: [modelId: string]
  close: []
}>()

const store = useAgentStore()

const visible = computed({
  get: () => props.open,
  set: () => emit('close'),
})

const query = ref('')

const filteredGroups = computed(() => {
  const models = query.value
    ? store.models.filter(m =>
        m.name.toLowerCase().includes(query.value.toLowerCase()) ||
        m.provider.toLowerCase().includes(query.value.toLowerCase()) ||
        m.id.toLowerCase().includes(query.value.toLowerCase())
      )
    : store.models

  const map = new Map<string, typeof store.models>()
  for (const m of models) {
    if (!map.has(m.provider)) map.set(m.provider, [])
    map.get(m.provider)!.push(m)
  }
  return Array.from(map.entries()).map(([provider, items]) => ({ provider, models: items }))
})

const handleSelect = (modelId: string) => {
  emit('select', modelId)
  emit('close')
}

const getTagColor = (tag: string) => {
  if (tag === 'free') return 'green'
  if (tag === 'tools') return 'blue'
  return 'default'
}

watch(() => props.open, (val) => {
  if (val) {
    query.value = ''
    store.fetchModels()
  }
})
</script>

<style scoped>
.model-selector {
  padding: 0;
}

.search-input {
  margin-bottom: 12px;
}

.loading-spinner {
  display: flex;
  justify-content: center;
  padding: 40px 0;
}

.model-list {
  max-height: 400px;
  overflow-y: auto;
}

.no-results {
  text-align: center;
  color: #999;
  padding: 20px 0;
}

.model-group {
  margin-bottom: 8px;
}

.group-label {
  padding: 4px 0;
  font-size: 12px;
  color: #999;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.model-item {
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s;
  margin-bottom: 2px;
}

.model-item:hover {
  background: #f5f5f5;
}

.model-item.active {
  background: #e6f4ff;
}

.model-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.model-name {
  font-weight: 500;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.current-icon {
  color: #1677ff;
  font-size: 14px;
}

.model-tags {
  display: flex;
  gap: 4px;
}

.model-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #999;
  padding-left: 20px;
}
</style>
