<template>
  <div class="backtest-center">
    <a-page-header title="Backtest Center" sub-title="Configure and run backtests">
      <template #extra>
        <a-space>
          <a-button @click="showTemplates = true">
            <template #icon><BookOutlined /></template>
            Templates
          </a-button>
          <a-button @click="handleHistory">
            <template #icon><HistoryOutlined /></template>
            History
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-row :gutter="[16, 16]">
      <a-col :span="14">
        <a-card title="Strategy Configuration">
          <MonacoEditor
            ref="editorRef"
            v-model:value="configYaml"
            language="yaml"
            :height="editorHeight"
            :minimap="true"
          />
        </a-card>
      </a-col>

      <a-col :span="10">
        <a-card title="Backtest Parameters">
          <a-form layout="vertical">
            <a-form-item label="Date Range">
              <a-range-picker v-model:value="dateRange" style="width: 100%" />
            </a-form-item>
            <a-form-item label="Initial Cash">
              <a-input-number v-model:value="initialCash" :min="10000" :step="10000" style="width: 100%" />
            </a-form-item>
            <a-form-item label="Data Path (optional)">
              <a-input v-model:value="dataPath" placeholder="path/to/data.csv" />
            </a-form-item>
            <a-form-item>
              <a-button
                type="primary"
                size="large"
                block
                :loading="isRunning"
                :disabled="!configYaml.trim()"
                @click="runBacktest"
              >
                <template #icon><CaretRightOutlined /></template>
                Run Backtest
              </a-button>
            </a-form-item>
          </a-form>
        </a-card>

        <a-card title="Results" style="margin-top: 16px" :loading="isRunning">
          <a-empty v-if="!result && !isRunning" description="Run a backtest to see results" />
          
          <a-alert v-if="result?.status === 'failed'" type="error" message="Backtest Failed" style="margin-bottom: 16px">
            <template #description>
              <div v-for="(error, idx) in result.errors" :key="idx">{{ error }}</div>
            </template>
          </a-alert>

          <a-alert v-if="result?.warnings?.length" type="warning" message="Warnings" style="margin-bottom: 16px">
            <template #description>
              <div v-for="(warn, idx) in result.warnings" :key="idx">{{ warn }}</div>
            </template>
          </a-alert>

          <template v-if="result?.status === 'success' || result?.status === 'warning'">
            <a-descriptions :column="2" bordered size="small">
              <a-descriptions-item label="Status">
                <a-tag :color="result.status === 'success' ? 'success' : 'warning'">
                  {{ result.status.toUpperCase() }}
                </a-tag>
              </a-descriptions-item>
              <a-descriptions-item label="Total Trades">
                {{ result.summary.total_trades }}
              </a-descriptions-item>
              <a-descriptions-item label="Total Return">
                <span :style="{ color: result.summary.total_return >= 0 ? '#52c41a' : '#ff4d4f' }">
                  {{ (result.summary.total_return * 100).toFixed(2) }}%
                </span>
              </a-descriptions-item>
              <a-descriptions-item label="Annual Return">
                <span :style="{ color: result.summary.annual_return >= 0 ? '#52c41a' : '#ff4d4f' }">
                  {{ (result.summary.annual_return * 100).toFixed(2) }}%
                </span>
              </a-descriptions-item>
              <a-descriptions-item label="Sharpe Ratio">
                <span :style="{ color: getSharpeColor(result.summary.sharpe_ratio) }">
                  {{ result.summary.sharpe_ratio.toFixed(2) }}
                </span>
              </a-descriptions-item>
              <a-descriptions-item label="Max Drawdown">
                <span style="color: #ff4d4f">
                  {{ (result.summary.max_drawdown * 100).toFixed(2) }}%
                </span>
              </a-descriptions-item>
              <a-descriptions-item label="Win Rate">
                {{ (result.summary.win_rate * 100).toFixed(2) }}%
              </a-descriptions-item>
              <a-descriptions-item label="Profit Factor">
                {{ result.summary.profit_factor.toFixed(2) }}
              </a-descriptions-item>
              <a-descriptions-item label="Final Cash">
                ${{ result.summary.final_cash.toLocaleString() }}
              </a-descriptions-item>
              <a-descriptions-item label="Commission">
                ${{ result.summary.total_commission.toFixed(2) }}
              </a-descriptions-item>
              <a-descriptions-item label="Sortino Ratio">
                {{ result.summary.sortino_ratio.toFixed(2) }}
              </a-descriptions-item>
              <a-descriptions-item label="Calmar Ratio">
                {{ result.summary.calmar_ratio.toFixed(2) }}
              </a-descriptions-item>
            </a-descriptions>

            <a-button style="margin-top: 16px" @click="handleViewResult">
              <template #icon><LineChartOutlined /></template>
              View Detailed Results
            </a-button>
          </template>
        </a-card>
      </a-col>
    </a-row>

    <a-modal v-model:open="showTemplates" title="Backtest Templates" width="600px">
      <a-list :dataSource="templates" item-layout="horizontal">
        <template #renderItem="{ item }">
          <a-list-item>
            <a-list-item-meta>
              <template #title>{{ item.name }}</template>
              <template #description>{{ item.description }}</template>
            </a-list-item-meta>
            <template #actions>
              <a-button size="small" @click="loadTemplate(item)">Use</a-button>
            </template>
          </a-list-item>
        </template>
      </a-list>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  BookOutlined,
  HistoryOutlined,
  CaretRightOutlined,
  LineChartOutlined,
} from '@ant-design/icons-vue'
import MonacoEditor from '@/components/Editor/MonacoEditor.vue'
import { backtestApi } from '@/api/backtest'
import type { BacktestResult, BacktestTemplate } from '@/api/backtest'

const router = useRouter()
const editorRef = ref<InstanceType<typeof MonacoEditor>>()

const configYaml = ref('')
const dateRange = ref<[any, any] | null>(null)
const initialCash = ref(100000)
const dataPath = ref('')
const isRunning = ref(false)
const result = ref<BacktestResult | null>(null)
const showTemplates = ref(false)
const templates = ref<BacktestTemplate[]>([])

const editorHeight = ref(500)

const getSharpeColor = (sharpe: number) => {
  if (sharpe > 2) return '#52c41a'
  if (sharpe > 1) return '#faad14'
  return '#ff4d4f'
}

const runBacktest = async () => {
  if (!configYaml.value.trim()) {
    message.error('Please enter strategy configuration')
    return
  }

  isRunning.value = true
  result.value = null

  try {
    const response = await backtestApi.run({
      config_yaml: configYaml.value,
      start_date: dateRange.value?.[0]?.format('YYYY-MM-DD'),
      end_date: dateRange.value?.[1]?.format('YYYY-MM-DD'),
      initial_cash: initialCash.value,
      data_path: dataPath.value || undefined,
    })
    
    result.value = response
    
    if (response.status === 'success') {
      message.success('Backtest completed successfully')
    } else if (response.status === 'warning') {
      message.warning('Backtest completed with warnings')
    } else {
      message.error('Backtest failed')
    }
  } catch (error: any) {
    message.error(error.message || 'Failed to run backtest')
  } finally {
    isRunning.value = false
  }
}

const loadTemplate = (template: BacktestTemplate) => {
  configYaml.value = template.yaml
  showTemplates.value = false
  message.info(`Loaded template: ${template.name}`)
}

const handleHistory = () => {
  message.info('Backtest history view coming soon')
}

const handleViewResult = () => {
  if (result.value) {
    router.push(`/backtest/result/${result.value.id}`)
  }
}

const updateEditorHeight = () => {
  editorHeight.value = Math.max(400, window.innerHeight - 300)
}

onMounted(async () => {
  updateEditorHeight()
  window.addEventListener('resize', updateEditorHeight)

  // Load templates
  try {
    templates.value = await backtestApi.getTemplates()
  } catch (error) {
    console.error('Failed to load templates:', error)
  }

  // Load default template
  if (!configYaml.value) {
    configYaml.value = `# Momentum Strategy
# Buy stocks with strong recent momentum

strategy:
  name: momentum_strategy
  universe: hs300
  
signals:
  - name: momentum_20d
    formula: "close / close.shift(20) - 1"
    weight: 0.5
  - name: momentum_60d
    formula: "close / close.shift(60) - 1"
    weight: 0.3
  - name: volume_momentum
    formula: "volume / volume.rolling(20).mean()"
    weight: 0.2
    
portfolio:
  rebalance_days: 5
  max_positions: 20
  position_sizing: equal_weight
  
risk:
  max_drawdown: 0.15
  stop_loss: 0.05
`
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', updateEditorHeight)
})
</script>

<style scoped>
.backtest-center {
  padding: 0;
}

:deep(.ant-descriptions-item-label) {
  font-weight: 500;
}
</style>
