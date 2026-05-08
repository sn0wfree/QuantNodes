<template>
  <div class="factor-analysis">
    <a-page-header title="Factor Analysis" sub-title="Analyze factor performance metrics" />

    <a-card>
      <a-form layout="vertical">
        <a-row :gutter="16">
          <a-col :span="8">
            <a-form-item label="Factor Expression" required>
              <a-input
                v-model:value="form.expression"
                placeholder="e.g., close / close.shift(5) - 1"
                @pressEnter="analyze"
              />
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="Universe">
              <a-select v-model:value="form.universe" style="width: 100%">
                <a-select-option value="hs300">CSI 300</a-select-option>
                <a-select-option value="zz500">CSI 500</a-select-option>
                <a-select-option value="zz1000">CSI 1000</a-select-option>
                <a-select-option value="全A">All A-shares</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="Date Range">
              <a-range-picker v-model:value="form.dateRange" style="width: 100%" />
            </a-form-item>
          </a-col>
        </a-row>
        <a-form-item>
          <a-space>
            <a-button type="primary" :loading="isAnalyzing" @click="analyze">
              <template #icon><BarChartOutlined /></template>
              Analyze Factor
            </a-button>
            <a-button @click="resetForm">Reset</a-button>
          </a-space>
        </a-form-item>
      </a-form>
    </a-card>

    <a-row :gutter="[16, 16]" style="margin-top: 16px" v-if="result">
      <a-col :span="24">
        <a-card title="Performance Metrics">
          <a-row :gutter="[16, 16]">
            <a-col :span="4">
              <a-statistic
                title="IC Mean"
                :value="result.ic_mean"
                :precision="4"
                :value-style="{ color: result.ic_mean > 0 ? '#52c41a' : '#ff4d4f' }"
              />
            </a-col>
            <a-col :span="4">
              <a-statistic
                title="IC Std"
                :value="result.ic_std"
                :precision="4"
              />
            </a-col>
            <a-col :span="4">
              <a-statistic
                title="ICIR"
                :value="result.icir"
                :precision="4"
                :value-style="{ color: result.icir > 2 ? '#52c41a' : result.icir > 1 ? '#faad14' : '#ff4d4f' }"
              />
            </a-col>
            <a-col :span="4">
              <a-statistic
                title="Rank IC"
                :value="result.rank_ic_mean"
                :precision="4"
              />
            </a-col>
            <a-col :span="4">
              <a-statistic
                title="Turnover"
                :value="(result.turnover || 0) * 100"
                :precision="2"
                suffix="%"
                :value-style="{ color: result.turnover < 0.3 ? '#52c41a' : '#faad14' }"
              />
            </a-col>
            <a-col :span="4">
              <a-statistic
                title="Quality Score"
                :value="qualityScore"
                :precision="2"
                :value-style="{ color: qualityScore > 0.7 ? '#52c41a' : '#faad14' }"
              />
            </a-col>
          </a-row>
        </a-card>
      </a-col>

      <a-col :span="12">
        <a-card title="IC Time Series">
          <IcChart :dates="result.dates" :ic-values="result.ic_series" />
        </a-card>
      </a-col>

      <a-col :span="12">
        <a-card title="Daily Returns">
          <ReturnsChart :dates="result.dates" :returns="result.returns" />
        </a-card>
      </a-col>

      <a-col :span="24">
        <a-card title="IC Distribution">
          <IcDistribution :ic-values="result.ic_series" />
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { Dayjs } from 'dayjs'
import { BarChartOutlined } from '@ant-design/icons-vue'
import { post } from '@/api'
import IcChart from '@/components/Charts/IcChart.vue'
import ReturnsChart from '@/components/Charts/ReturnsChart.vue'
import IcDistribution from '@/components/Charts/IcDistribution.vue'

interface AnalyzeResult {
  ic_mean: number
  ic_std: number
  icir: number
  rank_ic_mean: number
  turnover: number
  ic_series: number[]
  returns: number[]
  dates: string[]
}

const form = ref({
  expression: '',
  universe: 'hs300',
  dateRange: null as [Dayjs, Dayjs] | null,
})

const isAnalyzing = ref(false)
const result = ref<AnalyzeResult | null>(null)

const qualityScore = computed(() => {
  if (!result.value) return 0
  const { ic_mean, icir, turnover } = result.value
  const icScore = Math.min(Math.abs(ic_mean) * 20, 1)
  const icirScore = Math.min(icir / 3, 1)
  const turnoverScore = turnover < 0.3 ? 1 : turnover < 0.5 ? 0.7 : 0.4
  return (icScore * 0.4 + icirScore * 0.4 + turnoverScore * 0.2)
})

const analyze = async () => {
  if (!form.value.expression.trim()) return

  isAnalyzing.value = true
  try {
    const data = await post<AnalyzeResult>('/factor/analyze', {
      expression: form.value.expression,
      universe: form.value.universe,
      start_date: form.value.dateRange?.[0]?.format('YYYY-MM-DD'),
      end_date: form.value.dateRange?.[1]?.format('YYYY-MM-DD'),
    })
    result.value = data
  } catch (error) {
    console.error('Analysis failed:', error)
  } finally {
    isAnalyzing.value = false
  }
}

const resetForm = () => {
  form.value = {
    expression: '',
    universe: 'hs300',
    dateRange: null,
  }
  result.value = null
}
</script>

<style scoped>
.factor-analysis {
  padding: 0;
}

:deep(.ant-statistic-title) {
  font-size: 13px;
  color: #666;
}

:deep(.ant-statistic-content) {
  font-size: 24px;
}
</style>
