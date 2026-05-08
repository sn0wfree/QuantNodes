<template>
  <div class="dream-insights">
    <a-page-header title="Dream Insights" sub-title="AI-generated insights from your data">
      <template #extra>
        <a-space>
          <a-select v-model:value="filterType" placeholder="Filter by type" style="width: 180px" allowClear @change="fetchInsights">
            <a-select-option value="factor_insight">Factor Insights</a-select-option>
            <a-select-option value="strategy_insight">Strategy Insights</a-select-option>
            <a-select-option value="market_regime">Market Regime</a-select-option>
            <a-select-option value="risk_alert">Risk Alerts</a-select-option>
          </a-select>
          <a-button type="primary" @click="showGenerateModal">
            <template #icon><PlusOutlined /></template>
            Generate Insight
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-row :gutter="[16, 16]">
      <a-col :span="6">
        <a-card title="Overview" :loading="isLoadingStats">
          <a-row :gutter="[16, 16]">
            <a-col :span="24">
              <a-statistic
                title="Total Insights"
                :value="stats?.total_insights || 0"
                :value-style="{ color: '#1677ff' }"
              >
                <template #prefix><BulbOutlined /></template>
              </a-statistic>
            </a-col>
            <a-col :span="24">
              <a-statistic
                title="Avg Confidence"
                :value="((stats?.avg_confidence || 0) * 100)"
                :precision="1"
                suffix="%"
                :value-style="{ color: (stats?.avg_confidence || 0) > 0.8 ? '#52c41a' : '#faad14' }"
              />
            </a-col>
          </a-row>

          <a-divider />

          <div class="top-tags">
            <div class="section-title">Top Tags</div>
            <a-tag v-for="tag in (stats?.top_tags || []).slice(0, 8)" :key="tag.tag" color="blue">
              {{ tag.tag }} ({{ tag.count }})
            </a-tag>
            <a-empty v-if="!stats?.top_tags?.length" description="No tags" :image-style="{ height: '30px' }" />
          </div>
        </a-card>
      </a-col>

      <a-col :span="18">
        <a-card title="Recent Trend" :loading="isLoadingStats" style="margin-bottom: 16px">
          <TrendChart v-if="stats?.recent_trend?.length" :data="stats.recent_trend" />
          <a-empty v-else description="No trend data" />
        </a-card>

        <a-card title="Distribution" :loading="isLoadingStats">
          <a-row :gutter="16">
            <a-col :span="12">
              <DreamPieChart
                v-if="stats?.by_type && Object.keys(stats.by_type).length"
                :data="stats.by_type"
                title="By Type"
              />
              <a-empty v-else description="No distribution data" />
            </a-col>
            <a-col :span="12">
              <div class="type-list">
                <div v-for="(count, type) in (stats?.by_type || {})" :key="type" class="type-item">
                  <div class="type-label">
                    <span class="type-dot" :style="{ background: getTypeColor(type as string) }"></span>
                    {{ formatType(type as string) }}
                  </div>
                  <div class="type-count">{{ count }}</div>
                </div>
              </div>
            </a-col>
          </a-row>
        </a-card>
      </a-col>
    </a-row>

    <a-card title="Insights" :loading="isLoadingInsights" style="margin-top: 16px">
      <a-list :dataSource="insights" item-layout="vertical" size="large">
        <template #renderItem="{ item }">
          <a-list-item :key="item.id">
            <a-list-item-meta>
              <template #avatar>
                <a-avatar :style="{ background: getTypeColor(item.type) }">
                  <template #icon><BulbOutlined /></template>
                </a-avatar>
              </template>
              <template #title>
                <a @click="viewInsight(item)">{{ item.title }}</a>
              </template>
              <template #description>
                <div class="insight-meta">
                  <a-tag :color="getTypeColor(item.type)">{{ formatType(item.type) }}</a-tag>
                  <a-tag>{{ item.source }}</a-tag>
                  <span class="confidence" :style="{ color: getConfidenceColor(item.confidence) }">
                    Confidence: {{ (item.confidence * 100).toFixed(1) }}%
                  </span>
                  <span class="time">{{ formatTime(item.created_at) }}</span>
                </div>
              </template>
            </a-list-item-meta>
            <div class="insight-content">{{ item.content }}</div>
            <div v-if="item.insights?.length" class="insight-list">
              <div v-for="(insight, idx) in item.insights" :key="idx" class="insight-item">
                <CheckCircleOutlined style="color: #52c41a" />
                {{ insight }}
              </div>
            </div>
            <template #actions>
              <a-tooltip title="View Details">
                <span @click="viewInsight(item)"><EyeOutlined /></span>
              </a-tooltip>
              <a-tooltip title="Copy Content">
                <span @click="copyInsight(item)"><CopyOutlined /></span>
              </a-tooltip>
            </template>
          </a-list-item>
        </template>
      </a-list>
      <a-empty v-if="!insights.length && !isLoadingInsights" description="No insights found" />
    </a-card>

    <a-modal
      v-model:open="generateModalVisible"
      title="Generate New Insight"
      @ok="handleGenerate"
      :confirmLoading="generating"
    >
      <a-form layout="vertical">
        <a-form-item label="Type" required>
          <a-select v-model:value="generateForm.type">
            <a-select-option value="factor_insight">Factor Insight</a-select-option>
            <a-select-option value="strategy_insight">Strategy Insight</a-select-option>
            <a-select-option value="market_regime">Market Regime</a-select-option>
            <a-select-option value="risk_alert">Risk Alert</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="Content" required>
          <a-textarea v-model:value="generateForm.content" :rows="4" />
        </a-form-item>
        <a-form-item label="Source">
          <a-input v-model:value="generateForm.source" />
        </a-form-item>
        <a-form-item label="Confidence">
          <a-slider v-model:value="generateForm.confidence" :min="0" :max="1" :step="0.1" />
        </a-form-item>
        <a-form-item label="Tags">
          <a-select v-model:value="generateForm.tags" mode="tags" placeholder="Add tags" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  BulbOutlined,
  PlusOutlined,
  EyeOutlined,
  CopyOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons-vue'
import { get, post } from '@/api'
import DreamPieChart from '@/components/Charts/DreamPieChart.vue'
import TrendChart from '@/components/Charts/TrendChart.vue'

interface DreamInsight {
  id: string
  title: string
  content: string
  type: string
  category: string
  confidence: number
  created_at: string
  tags: string[]
  insights: string[]
  source: string
}

interface DreamStats {
  total_insights: number
  by_type: Record<string, number>
  by_category: Record<string, number>
  avg_confidence: number
  recent_trend: Array<{ date: string; count: number }>
  top_tags: Array<{ tag: string; count: number }>
}

const isLoadingStats = ref(false)
const isLoadingInsights = ref(false)
const stats = ref<DreamStats | null>(null)
const insights = ref<DreamInsight[]>([])
const filterType = ref<string | undefined>(undefined)
const generateModalVisible = ref(false)
const generating = ref(false)

const generateForm = ref({
  type: 'factor_insight',
  content: '',
  source: '',
  confidence: 0.8,
  tags: [] as string[],
})

const typeColors: Record<string, string> = {
  factor_insight: '#1677ff',
  strategy_insight: '#52c41a',
  market_regime: '#faad14',
  risk_alert: '#ff4d4f',
  wiki_insight: '#722ed1',
}

const getTypeColor = (type: string) => typeColors[type] || '#d9d9d9'

const formatType = (type: string) => {
  return type.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())
}

const getConfidenceColor = (confidence: number) => {
  if (confidence >= 0.8) return '#52c41a'
  if (confidence >= 0.6) return '#faad14'
  return '#ff4d4f'
}

const formatTime = (timestamp: string) => {
  const date = new Date(timestamp)
  return date.toLocaleString()
}

const fetchStats = async () => {
  isLoadingStats.value = true
  try {
    stats.value = await get<DreamStats>('/dreams/stats')
  } catch (error) {
    console.error('Failed to fetch stats:', error)
  } finally {
    isLoadingStats.value = false
  }
}

const fetchInsights = async () => {
  isLoadingInsights.value = true
  try {
    const params: any = { limit: 20 }
    if (filterType.value) {
      params.type = filterType.value
    }
    insights.value = await get<DreamInsight[]>('/dreams', { params })
  } catch (error) {
    console.error('Failed to fetch insights:', error)
  } finally {
    isLoadingInsights.value = false
  }
}

const viewInsight = (insight: DreamInsight) => {
  // Could open a detail modal
  console.log('View insight:', insight)
}

const copyInsight = (insight: DreamInsight) => {
  const text = `${insight.title}\n\n${insight.content}\n\nInsights:\n${insight.insights.map(i => `- ${i}`).join('\n')}`
  navigator.clipboard.writeText(text)
  message.success('Copied to clipboard')
}

const showGenerateModal = () => {
  generateForm.value = {
    type: 'factor_insight',
    content: '',
    source: '',
    confidence: 0.8,
    tags: [],
  }
  generateModalVisible.value = true
}

const handleGenerate = async () => {
  if (!generateForm.value.content.trim()) {
    message.error('Please enter content')
    return
  }

  generating.value = true
  try {
    await post('/dreams/', generateForm.value)
    message.success('Insight generated')
    generateModalVisible.value = false
    await fetchInsights()
    await fetchStats()
  } catch (error) {
    message.error('Failed to generate insight')
  } finally {
    generating.value = false
  }
}

onMounted(() => {
  fetchStats()
  fetchInsights()
})
</script>

<style scoped>
.dream-insights {
  padding: 0;
}

.top-tags {
  margin-top: 8px;
}

.section-title {
  font-weight: 500;
  margin-bottom: 8px;
  color: #666;
  font-size: 13px;
}

.type-list {
  padding: 16px 0;
}

.type-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.type-item:last-child {
  border-bottom: none;
}

.type-label {
  display: flex;
  align-items: center;
  gap: 8px;
}

.type-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}

.type-count {
  font-weight: 600;
  color: #333;
}

.insight-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.confidence {
  font-weight: 500;
}

.time {
  color: #999;
}

.insight-content {
  color: #666;
  margin-top: 8px;
  line-height: 1.6;
}

.insight-list {
  margin-top: 12px;
  padding: 12px;
  background: #f6f8fa;
  border-radius: 6px;
}

.insight-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 8px;
  line-height: 1.5;
}

.insight-item:last-child {
  margin-bottom: 0;
}
</style>
