<template>
  <div class="strategy-list">
    <a-page-header title="Strategies" sub-title="Manage your trading strategies">
      <template #extra>
        <a-space>
          <a-input-search
            v-model:value="searchText"
            placeholder="Search strategies"
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
            <a-select-option value="mean_reversion">Mean Reversion</a-select-option>
            <a-select-option value="trend_following">Trend Following</a-select-option>
            <a-select-option value="statistical_arbitrage">Statistical Arbitrage</a-select-option>
            <a-select-option value="general">General</a-select-option>
          </a-select>
          <a-button type="primary" @click="showCreateModal">
            <template #icon><PlusOutlined /></template>
            New Strategy
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-table
      :dataSource="wikiStore.strategies"
      :columns="columns"
      :loading="wikiStore.isLoading"
      rowKey="name"
      :pagination="{ pageSize: 20, showSizeChanger: true, showTotal: (total: number) => `Total ${total} strategies` }"
    >
      <template #bodyCell="{ column, record }">
        <template v-if="column.key === 'name'">
          <a @click="$router.push(`/wiki/strategies/${record.name}`)">{{ record.name }}</a>
        </template>
        <template v-if="column.key === 'category'">
          <a-tag :color="getCategoryColor(record.category)">{{ record.category }}</a-tag>
        </template>
        <template v-if="column.key === 'tags'">
          <a-tag v-for="tag in (record.tags || []).slice(0, 3)" :key="tag">{{ tag }}</a-tag>
          <a-tag v-if="(record.tags || []).length > 3">+{{ record.tags.length - 3 }}</a-tag>
        </template>
        <template v-if="column.key === 'backtest'">
          <a-tag v-if="record.backtest_result" color="success">Has Result</a-tag>
          <a-tag v-else>No Result</a-tag>
        </template>
        <template v-if="column.key === 'action'">
          <a-space>
            <a-button size="small" @click="$router.push(`/wiki/strategies/${record.name}`)">View</a-button>
            <a-button size="small" @click="handleBacktest(record)">Backtest</a-button>
            <a-button size="small" @click="handleEdit(record)">Edit</a-button>
          </a-space>
        </template>
      </template>
    </a-table>

    <a-modal
      v-model:open="createModalVisible"
      title="Create New Strategy"
      @ok="handleCreate"
      :confirmLoading="creating"
      width="700px"
    >
      <a-form :model="createForm" layout="vertical">
        <a-form-item label="Name" required>
          <a-input v-model:value="createForm.name" placeholder="strategy_name" />
        </a-form-item>
        <a-form-item label="Description">
          <a-textarea v-model:value="createForm.description" :rows="2" />
        </a-form-item>
        <a-row :gutter="16">
          <a-col :span="12">
            <a-form-item label="Category">
              <a-select v-model:value="createForm.category">
                <a-select-option value="momentum">Momentum</a-select-option>
                <a-select-option value="mean_reversion">Mean Reversion</a-select-option>
                <a-select-option value="trend_following">Trend Following</a-select-option>
                <a-select-option value="statistical_arbitrage">Statistical Arbitrage</a-select-option>
                <a-select-option value="general">General</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
          <a-col :span="12">
            <a-form-item label="Tags">
              <a-select v-model:value="createForm.tags" mode="tags" placeholder="Add tags" />
            </a-form-item>
          </a-col>
        </a-row>
        <a-form-item label="Strategy YAML Configuration">
          <a-textarea
            v-model:value="createForm.strategy_yaml"
            :rows="10"
            placeholder="# Strategy configuration in YAML format"
            style="font-family: monospace"
          />
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
const createModalVisible = ref(false)
const creating = ref(false)

const createForm = ref({
  name: '',
  description: '',
  category: 'general' as const,
  tags: [] as string[],
  strategy_yaml: '',
})

const columns = [
  { title: 'Name', key: 'name', dataIndex: 'name', width: 200 },
  { title: 'Description', dataIndex: 'description', ellipsis: true },
  { title: 'Category', key: 'category', dataIndex: 'category', width: 150 },
  { title: 'Tags', key: 'tags', width: 200 },
  { title: 'Backtest', key: 'backtest', width: 100 },
  { title: 'Action', key: 'action', width: 200, fixed: 'right' as const },
]

const getCategoryColor = (category: string) => {
  const colors: Record<string, string> = {
    momentum: 'blue',
    mean_reversion: 'green',
    trend_following: 'purple',
    statistical_arbitrage: 'orange',
    general: 'default',
  }
  return colors[category] || 'default'
}

const handleSearch = async (value: string) => {
  if (value.trim()) {
    await wikiStore.searchStrategies(value)
  } else {
    await wikiStore.fetchStrategies({
      category: selectedCategory.value,
    })
  }
}

const handleCategoryChange = () => {
  wikiStore.fetchStrategies({
    category: selectedCategory.value,
  })
}

const showCreateModal = () => {
  createForm.value = {
    name: '',
    description: '',
    category: 'general',
    tags: [],
    strategy_yaml: '',
  }
  createModalVisible.value = true
}

const handleCreate = async () => {
  if (!createForm.value.name) {
    message.error('Please enter a strategy name')
    return
  }

  creating.value = true
  try {
    await wikiStore.createStrategy(createForm.value)
    message.success('Strategy created successfully')
    createModalVisible.value = false
  } catch (error) {
    message.error('Failed to create strategy')
  } finally {
    creating.value = false
  }
}

const handleBacktest = (record: any) => {
  router.push({
    path: '/backtest',
    query: { strategy: record.name },
  })
}

const handleEdit = (record: any) => {
  router.push({ path: '/strategy/editor', query: { strategy: record.name } })
}

onMounted(() => {
  wikiStore.fetchStrategies()
})
</script>

<style scoped>
.strategy-list {
  padding: 0;
}

:deep(.ant-table-cell) {
  vertical-align: middle;
}
</style>
