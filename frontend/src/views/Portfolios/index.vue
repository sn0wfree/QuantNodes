<template>
  <div class="portfolios-page">
    <a-row :gutter="[16, 16]">
      <a-col :span="24">
        <a-card title="Portfolios">
          <template #extra>
            <a-button type="primary" @click="showCreateModal = true">
              <template #icon><PlusOutlined /></template>
              Create Portfolio
            </a-button>
          </template>

          <a-table
            :columns="columns"
            :dataSource="portfolios"
            :loading="loading"
            row-key="id"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'performance'">
                <a-tag :color="record.performance >= 0 ? 'green' : 'red'">
                  {{ record.performance >= 0 ? '+' : '' }}{{ record.performance.toFixed(2) }}%
                </a-tag>
              </template>
              <template v-else-if="column.key === 'status'">
                <a-badge :status="record.status === 'active' ? 'success' : 'default'" :text="record.status" />
              </template>
              <template v-else-if="column.key === 'actions'">
                <a-space>
                  <a-button size="small" @click="viewPortfolio(record)">View</a-button>
                  <a-button size="small" type="primary" @click="backtestPortfolio(record)">Backtest</a-button>
                </a-space>
              </template>
            </template>
          </a-table>
        </a-card>
      </a-col>
    </a-row>

    <a-modal
      v-model:open="showCreateModal"
      title="Create Portfolio"
      @ok="createPortfolio"
    >
      <a-form :model="newPortfolio" layout="vertical">
        <a-form-item label="Portfolio Name" required>
          <a-input v-model:value="newPortfolio.name" placeholder="Enter portfolio name" />
        </a-form-item>
        <a-form-item label="Description">
          <a-input v-model:value="newPortfolio.description" type="textarea" placeholder="Portfolio description" />
        </a-form-item>
        <a-form-item label="Initial Capital">
          <a-input-number v-model:value="newPortfolio.capital" :min="0" style="width: 100%" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const columns = [
  { title: 'Name', dataIndex: 'name', key: 'name' },
  { title: 'Strategies', dataIndex: 'strategyCount', key: 'strategyCount' },
  { title: 'Performance', key: 'performance' },
  { title: 'Status', key: 'status' },
  { title: 'Created', dataIndex: 'createdAt', key: 'createdAt' },
  { title: 'Actions', key: 'actions' },
]

const portfolios = ref<any[]>([])
const loading = ref(false)
const showCreateModal = ref(false)

const newPortfolio = ref({
  name: '',
  description: '',
  capital: 1000000,
})

const loadPortfolios = async () => {
  loading.value = true
  try {
    portfolios.value = []
  } catch (error) {
    console.error('Failed to load portfolios:', error)
  } finally {
    loading.value = false
  }
}

const createPortfolio = () => {
  message.success('Portfolio creation feature coming soon')
  showCreateModal.value = false
}

const viewPortfolio = (record: any) => {
  message.info(`View portfolio: ${record.name}`)
}

const backtestPortfolio = (record: any) => {
  message.info(`Backtest portfolio: ${record.name}`)
}

onMounted(() => {
  loadPortfolios()
})
</script>

<style scoped>
.portfolios-page {
  padding: 0;
}
</style>