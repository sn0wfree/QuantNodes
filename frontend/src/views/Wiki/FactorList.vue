<template>
  <div class="factor-list">
    <a-page-header title="Factors" sub-title="Manage your quantitative factors">
      <template #extra>
        <a-space>
          <a-input-search
            v-model:value="searchText"
            placeholder="Search factors"
            style="width: 250px"
            @search="handleSearch"
            allowClear
          />
          <a-select
            v-model:value="selectedCategory"
            placeholder="Category"
            style="width: 150px"
            allowClear
            @change="handleCategoryChange"
          >
            <a-select-option value="momentum">Momentum</a-select-option>
            <a-select-option value="value">Value</a-select-option>
            <a-select-option value="quality">Quality</a-select-option>
            <a-select-option value="growth">Growth</a-select-option>
            <a-select-option value="volatility">Volatility</a-select-option>
            <a-select-option value="size">Size</a-select-option>
            <a-select-option value="other">Other</a-select-option>
          </a-select>
          <a-select
            v-model:value="selectedSource"
            placeholder="Source"
            style="width: 150px"
            allowClear
            @change="handleSourceChange"
          >
            <a-select-option value="research_report">Research Report</a-select-option>
            <a-select-option value="auto_research">Auto Research</a-select-option>
            <a-select-option value="manual">Manual</a-select-option>
            <a-select-option value="derived">Derived</a-select-option>
            <a-select-option value="imported">Imported</a-select-option>
          </a-select>
          <a-button type="primary" @click="showCreateModal">
            <template #icon><PlusOutlined /></template>
            New Factor
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-table
      :dataSource="wikiStore.factors"
      :columns="columns"
      :loading="wikiStore.isLoading"
      rowKey="name"
      :pagination="{ pageSize: 20, showSizeChanger: true, showTotal: (total: number) => `Total ${total} factors` }"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'name'">
          <a @click="$router.push(`/wiki/factors/${record.name}`)">{{ record.name }}</a>
        </template>
        <template v-if="column.key === 'category'">
          <a-tag :color="getCategoryColor(record.category)">{{ record.category }}</a-tag>
        </template>
        <template v-if="column.key === 'source'">
          <a-tag>{{ record.source }}</a-tag>
        </template>
        <template v-if="column.key === 'ic_mean'">
          <span :style="{ color: (record.ic_mean || 0) > 0 ? '#52c41a' : '#ff4d4f' }">
            {{ record.ic_mean?.toFixed(4) || '-' }}
          </span>
        </template>
        <template v-if="column.key === 'icir'">
          <span :style="{ color: getIcirColor(record.icir) }">
            {{ record.icir?.toFixed(2) || '-' }}
          </span>
        </template>
        <template v-if="column.key === 'tags'">
          <a-tag v-for="tag in (record.tags || []).slice(0, 3)" :key="tag">{{ tag }}</a-tag>
          <a-tag v-if="(record.tags || []).length > 3">+{{ record.tags.length - 3 }}</a-tag>
        </template>
        <template v-if="column.key === 'action'">
          <a-space>
            <a-button size="small" @click="$router.push(`/wiki/factors/${record.name}`)">View</a-button>
            <a-button size="small" @click="handleAnalyze(record)">Analyze</a-button>
            <a-popconfirm title="Are you sure?" @confirm="handleDelete(record.name)">
              <a-button size="small" danger>Delete</a-button>
            </a-popconfirm>
          </a-space>
        </template>
      </template>
    </a-table>

    <a-modal
      v-model:open="createModalVisible"
      title="Create New Factor"
      @ok="handleCreate"
      :confirmLoading="creating"
    >
      <a-form :model="createForm" layout="vertical">
        <a-form-item label="Name" required>
          <a-input v-model:value="createForm.name" placeholder="factor_name" />
        </a-form-item>
        <a-form-item label="Formula" required>
          <a-textarea v-model:value="createForm.formula" :rows="3" placeholder="e.g., ts_mean(close, 20) / ts_mean(close, 60) - 1" />
        </a-form-item>
        <a-row :gutter="16">
          <a-col :span="12">
            <a-form-item label="Category" required>
              <a-select v-model:value="createForm.category">
                <a-select-option value="momentum">Momentum</a-select-option>
                <a-select-option value="value">Value</a-select-option>
                <a-select-option value="quality">Quality</a-select-option>
                <a-select-option value="growth">Growth</a-select-option>
                <a-select-option value="volatility">Volatility</a-select-option>
                <a-select-option value="size">Size</a-select-option>
                <a-select-option value="other">Other</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
          <a-col :span="12">
            <a-form-item label="Source" required>
              <a-select v-model:value="createForm.source">
                <a-select-option value="research_report">Research Report</a-select-option>
                <a-select-option value="auto_research">Auto Research</a-select-option>
                <a-select-option value="manual">Manual</a-select-option>
                <a-select-option value="derived">Derived</a-select-option>
                <a-select-option value="imported">Imported</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
        </a-row>
        <a-form-item label="Description">
          <a-textarea v-model:value="createForm.description" :rows="2" />
        </a-form-item>
        <a-form-item label="Tags">
          <a-select v-model:value="createForm.tags" mode="tags" placeholder="Add tags" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import { useWikiStore } from '@/stores/wiki'

const router = useRouter()
const wikiStore = useWikiStore()

const searchText = ref('')
const selectedCategory = ref<string | undefined>(undefined)
const selectedSource = ref<string | undefined>(undefined)
const createModalVisible = ref(false)
const creating = ref(false)

const createForm = ref({
  name: '',
  formula: '',
  category: 'momentum' as const,
  source: 'manual' as const,
  description: '',
  tags: [] as string[],
})

const columns = [
  { title: 'Name', key: 'name', dataIndex: 'name', width: 200 },
  { title: 'Category', key: 'category', dataIndex: 'category', width: 120 },
  { title: 'Source', key: 'source', dataIndex: 'source', width: 120 },
  { title: 'Formula', dataIndex: 'formula', ellipsis: true },
  { title: 'IC Mean', key: 'ic_mean', dataIndex: 'ic_mean', width: 100 },
  { title: 'ICIR', key: 'icir', dataIndex: 'icir', width: 80 },
  { title: 'Tags', key: 'tags', width: 200 },
  { title: 'Action', key: 'action', width: 200, fixed: 'right' as const },
]

const getCategoryColor = (category: string) => {
  const colors: Record<string, string> = {
    momentum: 'blue',
    value: 'green',
    quality: 'purple',
    growth: 'orange',
    volatility: 'red',
    size: 'cyan',
  }
  return colors[category] || 'default'
}

const getIcirColor = (icir?: number) => {
  if (!icir) return '#999'
  if (icir > 2) return '#52c41a'
  if (icir > 1) return '#faad14'
  return '#ff4d4f'
}

const handleSearch = async (value: string) => {
  if (value.trim()) {
    await wikiStore.searchFactors(value)
  } else {
    await wikiStore.fetchFactors({
      category: selectedCategory.value,
      source: selectedSource.value,
    })
  }
}

const handleCategoryChange = () => {
  wikiStore.fetchFactors({
    category: selectedCategory.value,
    source: selectedSource.value,
  })
}

const handleSourceChange = () => {
  wikiStore.fetchFactors({
    category: selectedCategory.value,
    source: selectedSource.value,
  })
}

const showCreateModal = () => {
  createForm.value = {
    name: '',
    formula: '',
    category: 'momentum',
    source: 'manual',
    description: '',
    tags: [],
  }
  createModalVisible.value = true
}

const handleCreate = async () => {
  if (!createForm.value.name || !createForm.value.formula) {
    message.error('Please fill in required fields')
    return
  }

  creating.value = true
  try {
    await wikiStore.createFactor(createForm.value)
    message.success('Factor created successfully')
    createModalVisible.value = false
  } catch (error) {
    message.error('Failed to create factor')
  } finally {
    creating.value = false
  }
}

const handleDelete = async (name: string) => {
  try {
    await wikiStore.deleteFactor(name)
    message.success('Factor deleted')
  } catch (error) {
    message.error('Failed to delete factor')
  }
}

const handleAnalyze = (record: any) => {
  router.push({
    path: '/factor-analysis',
    query: { expression: record.formula },
  })
}

onMounted(() => {
  wikiStore.fetchFactors()
})
</script>

<style scoped>
.factor-list {
  padding: 0;
}

:deep(.ant-table-cell) {
  vertical-align: middle;
}
</style>
