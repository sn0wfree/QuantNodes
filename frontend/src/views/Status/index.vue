<template>
  <div class="status-page">
    <a-row :gutter="[16, 16]">
      <a-col :span="24">
        <a-card title="System Status">
          <a-descriptions :column="2" bordered>
            <a-descriptions-item label="API Status">
              <a-badge :status="apiStatus" :text="apiStatusText" />
            </a-descriptions-item>
            <a-descriptions-item label="Version">
              {{ version }}
            </a-descriptions-item>
            <a-descriptions-item label="API Health">
              <a-badge :status="healthStatus" :text="healthStatusText" />
            </a-descriptions-item>
            <a-descriptions-item label="Last Checked">
              {{ lastChecked }}
            </a-descriptions-item>
          </a-descriptions>
        </a-card>
      </a-col>
    </a-row>

    <a-row :gutter="[16, 16]" style="margin-top: 16px">
      <a-col :span="12">
        <a-card title="Available Endpoints">
          <a-list :dataSource="endpoints" size="small">
            <template #renderItem="{ item }">
              <a-list-item>
                <a-tag :color="item.method === 'GET' ? 'green' : item.method === 'POST' ? 'blue' : 'orange'">
                  {{ item.method }}
                </a-tag>
                <span style="margin-left: 8px">{{ item.path }}</span>
              </a-list-item>
            </template>
          </a-list>
        </a-card>
      </a-col>
      <a-col :span="12">
        <a-card title="API Capabilities">
          <a-list :dataSource="capabilities" size="small">
            <template #renderItem="{ item }">
              <a-list-item>
                <CheckCircleOutlined style="color: #52c41a; margin-right: 8px" />
                {{ item }}
              </a-list-item>
            </template>
          </a-list>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { CheckCircleOutlined } from '@ant-design/icons-vue'
import { get } from '@/api'

const version = ref('2.5.0')
const apiConnected = ref(false)
const healthStatus = ref<'success' | 'error' | 'processing'>('processing')
const lastChecked = ref(new Date().toLocaleTimeString())

const endpoints = ref([
  { method: 'GET', path: '/api/health' },
  { method: 'GET', path: '/api/stats' },
  { method: 'GET', path: '/api/prompts' },
  { method: 'GET', path: '/api/prompts/strategy/{type}' },
  { method: 'GET', path: '/api/prompts/backtest/{type}' },
  { method: 'GET', path: '/api/prompts/factor/{type}' },
  { method: 'POST', path: '/api/backtest/run' },
  { method: 'POST', path: '/api/factor/analyze' },
  { method: 'POST', path: '/api/code/validate' },
  { method: 'POST', path: '/api/code/execute' },
  { method: 'POST', path: '/api/pipeline/validate' },
  { method: 'POST', path: '/api/strategies' },
  { method: 'GET', path: '/api/strategies' },
  { method: 'GET', path: '/api/wiki/factors' },
  { method: 'GET', path: '/api/wiki/strategies' },
])

const capabilities = ref([
  'Strategy Backtesting',
  'Factor Analysis (IC, Correlation)',
  'Code Validation & Execution',
  'Prompt Library for External Agents',
  'Wiki Knowledge Base Access',
  'Pipeline Validation',
])

const apiStatus = ref<'success' | 'error' | 'processing'>('processing')
const apiStatusText = ref('Checking...')
const healthStatusText = ref('Checking...')

const checkApiStatus = async () => {
  try {
    await get('/health')
    apiConnected.value = true
    apiStatus.value = 'success'
    apiStatusText.value = 'Connected'
    healthStatus.value = 'success'
    healthStatusText.value = 'Healthy'
  } catch {
    apiConnected.value = false
    apiStatus.value = 'error'
    apiStatusText.value = 'Disconnected'
    healthStatus.value = 'error'
    healthStatusText.value = 'Unhealthy'
  }
  lastChecked.value = new Date().toLocaleTimeString()
}

onMounted(() => {
  checkApiStatus()
})
</script>

<style scoped>
.status-page {
  padding: 0;
}
</style>