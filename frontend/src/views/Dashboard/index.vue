<template>
  <div class="dashboard">
    <a-row :gutter="[16, 16]">
      <a-col :span="6">
        <a-card hoverable>
          <a-statistic
            title="Factors"
            :value="stats.factors"
            :value-style="{ color: '#1677ff' }"
          >
            <template #prefix>
              <ExperimentOutlined />
            </template>
          </a-statistic>
        </a-card>
      </a-col>
      <a-col :span="6">
        <a-card hoverable>
          <a-statistic
            title="Strategies"
            :value="stats.strategies"
            :value-style="{ color: '#52c41a' }"
          >
            <template #prefix>
              <BranchesOutlined />
            </template>
          </a-statistic>
        </a-card>
      </a-col>
      <a-col :span="6">
        <a-card hoverable>
          <a-statistic
            title="Backtests"
            :value="stats.backtests"
            :value-style="{ color: '#faad14' }"
          >
            <template #prefix>
              <LineChartOutlined />
            </template>
          </a-statistic>
        </a-card>
      </a-col>
      <a-col :span="6">
        <a-card hoverable>
          <a-statistic
            title="Insights"
            :value="stats.insights"
            :value-style="{ color: '#722ed1' }"
          >
            <template #prefix>
              <BulbOutlined />
            </template>
          </a-statistic>
        </a-card>
      </a-col>
    </a-row>

    <a-row :gutter="[16, 16]" style="margin-top: 16px">
      <a-col :span="16">
        <a-card title="Recent Activity" :loading="loading">
          <a-list :dataSource="recentActivity" item-layout="horizontal" v-if="recentActivity.length">
            <template #renderItem="{ item }">
              <a-list-item>
                <a-list-item-meta>
                  <template #avatar>
                    <a-avatar style="background-color: #1677ff">
                      <ExperimentOutlined v-if="item.type === 'factor'" />
                      <BranchesOutlined v-else />
                    </a-avatar>
                  </template>
                  <template #title>
                    <a @click="$router.push(`/wiki/factors/${item.name}`)">{{ item.name }}</a>
                  </template>
                  <template #description>
                    <a-tag>{{ item.category }}</a-tag>
                    <span style="color: #999; margin-left: 8px">{{ item.updated_at || 'Recently updated' }}</span>
                  </template>
                </a-list-item-meta>
              </a-list-item>
            </template>
          </a-list>
          <a-empty v-else description="No recent activity" />
        </a-card>
      </a-col>
      <a-col :span="8">
        <a-card title="Quick Actions">
          <a-space direction="vertical" style="width: 100%">
            <a-button type="primary" block @click="$router.push('/chat')">
              <template #icon><MessageOutlined /></template>
              Agent Chat
            </a-button>
            <a-button block @click="$router.push('/wiki/factors')">
              <template #icon><ExperimentOutlined /></template>
              Browse Factors
            </a-button>
            <a-button block @click="$router.push('/wiki/strategies')">
              <template #icon><BranchesOutlined /></template>
              Browse Strategies
            </a-button>
            <a-button block @click="$router.push('/backtest')">
              <template #icon><LineChartOutlined /></template>
              Run Backtest
            </a-button>
            <a-button block @click="$router.push('/factor-analysis')">
              <template #icon><BarChartOutlined /></template>
              Factor Analysis
            </a-button>
          </a-space>
        </a-card>

        <a-card title="System Status" style="margin-top: 16px">
          <a-descriptions :column="1" size="small">
            <a-descriptions-item label="API Status">
              <a-badge status="success" text="Connected" />
            </a-descriptions-item>
            <a-descriptions-item label="Agent Status">
              <a-badge :status="agentStatus" :text="agentStatusText" />
            </a-descriptions-item>
          </a-descriptions>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import {
  ExperimentOutlined,
  BranchesOutlined,
  LineChartOutlined,
  BulbOutlined,
  MessageOutlined,
  BarChartOutlined,
} from '@ant-design/icons-vue'
import { get } from '@/api'

const stats = ref({
  factors: 0,
  strategies: 0,
  backtests: 0,
  insights: 0,
})

const recentActivity = ref<any[]>([])
const loading = ref(true)
const agentConnected = ref(false)

const agentStatus = computed(() => agentConnected.value ? 'success' : 'default')
const agentStatusText = computed(() => agentConnected.value ? 'Ready' : 'Initializing')

const fetchStats = async () => {
  try {
    const data = await get<any>('/stats')
    stats.value = data
  } catch (error) {
    console.error('Failed to fetch stats:', error)
  }
}

const fetchActivity = async () => {
  try {
    const data = await get<any[]>('/stats/activity', { params: { limit: 5 } })
    recentActivity.value = data
  } catch (error) {
    console.error('Failed to fetch activity:', error)
  } finally {
    loading.value = false
  }
}

const checkAgentStatus = async () => {
  try {
    await get('/health')
    agentConnected.value = true
  } catch {
    agentConnected.value = false
  }
}

onMounted(() => {
  fetchStats()
  fetchActivity()
  checkAgentStatus()
})
</script>

<style scoped>
.dashboard {
  padding: 0;
}

:deep(.ant-card) {
  border-radius: 8px;
}

:deep(.ant-statistic-title) {
  font-size: 14px;
  color: #666;
}
</style>
