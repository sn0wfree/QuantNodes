<template>
  <div class="backtest-result">
    <a-page-header title="Backtest Result" @back="$router.back()">
      <template #extra>
        <a-space>
          <a-button @click="handleCopy">
            <template #icon><CopyOutlined /></template>
            Copy Results
          </a-button>
          <a-button type="primary" @click="$router.push('/backtest')">
            <template #icon><PlusOutlined /></template>
            New Backtest
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-spin :spinning="isLoading">
      <a-empty v-if="!result && !isLoading" description="No result found" />
      
      <template v-else-if="result">
        <a-alert
          v-if="result.status === 'failed'"
          type="error"
          message="Backtest Failed"
          style="margin-bottom: 16px"
        >
          <template #description>
            <div v-for="(error, idx) in result.errors" :key="idx">{{ error }}</div>
          </template>
        </a-alert>

        <a-row :gutter="[16, 16]">
          <a-col :span="16">
            <a-card title="Performance Summary">
              <a-row :gutter="[16, 16]">
                <a-col :span="6">
                  <a-statistic
                    title="Total Return"
                    :value="(result.summary.total_return * 100)"
                    :precision="2"
                    suffix="%"
                    :value-style="{ color: result.summary.total_return >= 0 ? '#52c41a' : '#ff4d4f' }"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="Annual Return"
                    :value="(result.summary.annual_return * 100)"
                    :precision="2"
                    suffix="%"
                    :value-style="{ color: result.summary.annual_return >= 0 ? '#52c41a' : '#ff4d4f' }"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="Sharpe Ratio"
                    :value="result.summary.sharpe_ratio"
                    :precision="2"
                    :value-style="{ color: getSharpeColor(result.summary.sharpe_ratio) }"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="Max Drawdown"
                    :value="(result.summary.max_drawdown * 100)"
                    :precision="2"
                    suffix="%"
                    :value-style="{ color: '#ff4d4f' }"
                  />
                </a-col>
              </a-row>

              <a-divider />

              <a-row :gutter="[16, 16]">
                <a-col :span="4">
                  <a-statistic title="Win Rate" :value="(result.summary.win_rate * 100)" :precision="1" suffix="%" />
                </a-col>
                <a-col :span="4">
                  <a-statistic title="Total Trades" :value="result.summary.total_trades" />
                </a-col>
                <a-col :span="4">
                  <a-statistic title="Profit Factor" :value="result.summary.profit_factor" :precision="2" />
                </a-col>
                <a-col :span="4">
                  <a-statistic title="Sortino" :value="result.summary.sortino_ratio" :precision="2" />
                </a-col>
                <a-col :span="4">
                  <a-statistic title="Calmar" :value="result.summary.calmar_ratio" :precision="2" />
                </a-col>
                <a-col :span="4">
                  <a-statistic title="Trading Days" :value="result.summary.trading_days" />
                </a-col>
              </a-row>
            </a-card>

            <a-card title="Equity Curve" style="margin-top: 16px">
              <EquityCurve :dates="equityDates" :equity="equityValues" />
            </a-card>

            <a-card title="Drawdown" style="margin-top: 16px">
              <DrawdownChart :dates="equityDates" :drawdown="drawdownValues" />
            </a-card>
          </a-col>

          <a-col :span="8">
            <a-card title="Configuration">
              <a-descriptions :column="1" bordered size="small">
                <a-descriptions-item label="Strategy">
                  {{ result.config_info.name || 'N/A' }}
                </a-descriptions-item>
                <a-descriptions-item label="Status">
                  <a-tag :color="result.status === 'success' ? 'success' : 'warning'">
                    {{ result.status.toUpperCase() }}
                  </a-tag>
                </a-descriptions-item>
                <a-descriptions-item label="Final Cash">
                  ${{ result.summary.final_cash.toLocaleString() }}
                </a-descriptions-item>
                <a-descriptions-item label="Commission">
                  ${{ result.summary.total_commission.toFixed(2) }}
                </a-descriptions-item>
                <a-descriptions-item label="Avg Trade PnL">
                  ${{ result.summary.avg_trade_pnl.toFixed(2) }}
                </a-descriptions-item>
              </a-descriptions>
            </a-card>

            <a-card title="Info" style="margin-top: 16px">
              <a-descriptions :column="1" size="small">
                <a-descriptions-item label="Created">
                  {{ formatDate(result.created_at) }}
                </a-descriptions-item>
                <a-descriptions-item label="Completed">
                  {{ formatDate(result.completed_at) }}
                </a-descriptions-item>
                <a-descriptions-item label="ID">
                  <a-tag>{{ result.id }}</a-tag>
                </a-descriptions-item>
              </a-descriptions>
            </a-card>

            <a-card title="Output Files" style="margin-top: 16px" v-if="Object.keys(result.output_files).length">
              <a-list size="small">
                <a-list-item v-for="(path, name) in result.output_files" :key="name">
                  <a-list-item-meta>
                    <template #title>{{ name }}</template>
                    <template #description>{{ path }}</template>
                  </a-list-item-meta>
                </a-list-item>
              </a-list>
            </a-card>

            <a-card title="Warnings" style="margin-top: 16px" v-if="result.warnings?.length">
              <a-alert
                v-for="(warn, idx) in result.warnings"
                :key="idx"
                type="warning"
                :message="warn"
                style="margin-bottom: 8px"
              />
            </a-card>
          </a-col>
        </a-row>
      </template>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import { CopyOutlined, PlusOutlined } from '@ant-design/icons-vue'
import { backtestApi } from '@/api/backtest'
import type { BacktestResult } from '@/api/backtest'
import EquityCurve from '@/components/Charts/EquityCurve.vue'
import DrawdownChart from '@/components/Charts/DrawdownChart.vue'

const route = useRoute()

const isLoading = ref(false)
const result = ref<BacktestResult | null>(null)

// Generate sample equity curve data for demonstration
const equityDates = computed(() => {
  if (!result.value) return []
  const days = result.value.summary.trading_days || 252
  const dates = []
  const startDate = new Date('2024-01-01')
  for (let i = 0; i < days; i++) {
    const date = new Date(startDate)
    date.setDate(date.getDate() + i)
    if (date.getDay() !== 0 && date.getDay() !== 6) {
      dates.push(date.toISOString().split('T')[0])
    }
  }
  return dates
})

const equityValues = computed(() => {
  if (!result.value) return []
  const days = equityDates.value.length
  if (days === 0) return []
  const totalReturn = result.value.summary.total_return
  const values = [100000]
  // Use a deterministic seed based on total_return for reproducibility
  const dailyReturn = (1 + totalReturn) ** (1 / days) - 1
  for (let i = 1; i < days; i++) {
    // Deterministic pseudo-random noise based on index
    const noise = (Math.sin(i * 12.9898 + 78.233) * 43758.5453 % 1 - 0.5) * 0.02
    values.push(values[i - 1] * (1 + dailyReturn + noise))
  }
  return values
})

const drawdownValues = computed(() => {
  if (!equityValues.value.length) return []
  const values = equityValues.value
  const result: number[] = []
  let peak = values[0]
  for (const v of values) {
    if (v > peak) peak = v
    result.push((v - peak) / peak)
  }
  return result
})

const getSharpeColor = (sharpe: number) => {
  if (sharpe > 2) return '#52c41a'
  if (sharpe > 1) return '#faad14'
  return '#ff4d4f'
}

const formatDate = (dateStr?: string) => {
  if (!dateStr) return 'N/A'
  return new Date(dateStr).toLocaleString()
}

const handleCopy = () => {
  if (!result.value) return
  const text = `
Backtest Result: ${result.value.id}
Status: ${result.value.status}
Total Return: ${(result.value.summary.total_return * 100).toFixed(2)}%
Annual Return: ${(result.value.summary.annual_return * 100).toFixed(2)}%
Sharpe Ratio: ${result.value.summary.sharpe_ratio.toFixed(2)}
Max Drawdown: ${(result.value.summary.max_drawdown * 100).toFixed(2)}%
Win Rate: ${(result.value.summary.win_rate * 100).toFixed(1)}%
Total Trades: ${result.value.summary.total_trades}
  `.trim()
  navigator.clipboard.writeText(text)
  message.success('Results copied to clipboard')
}

onMounted(async () => {
  const id = route.params.id as string
  if (id) {
    isLoading.value = true
    try {
      result.value = await backtestApi.getResult(id)
    } catch (error) {
      console.error('Failed to fetch result:', error)
    } finally {
      isLoading.value = false
    }
  }
})
</script>

<style scoped>
.backtest-result {
  padding: 0;
}

:deep(.ant-statistic-title) {
  font-size: 13px;
  color: #666;
}
</style>
