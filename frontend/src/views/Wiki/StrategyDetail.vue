<template>
  <div class="strategy-detail">
    <a-spin :spinning="wikiStore.isLoading" size="large">
      <template v-if="wikiStore.currentStrategy">
        <a-page-header :title="wikiStore.currentStrategy.name" @back="$router.back()">
          <template #subtitle>
            <a-tag :color="getCategoryColor(wikiStore.currentStrategy.category)">
              {{ wikiStore.currentStrategy.category }}
            </a-tag>
          </template>
          <template #extra>
            <a-space>
              <a-button type="primary" @click="handleBacktest">
                <template #icon><LineChartOutlined /></template>
                Run Backtest
              </a-button>
              <a-button @click="handleEdit">
                <template #icon><EditOutlined /></template>
                Edit
              </a-button>
            </a-space>
          </template>
        </a-page-header>

        <a-row :gutter="[16, 16]">
          <a-col :span="16">
            <a-card title="Description">
              <p v-if="wikiStore.currentStrategy.description">
                {{ wikiStore.currentStrategy.description }}
              </p>
              <a-empty v-else description="No description provided" :image-style="{ height: '40px' }" />
            </a-card>

            <a-card title="Strategy Configuration" style="margin-top: 16px">
              <div v-if="wikiStore.currentStrategy.strategy_yaml" class="yaml-container">
                <pre class="yaml-content">{{ wikiStore.currentStrategy.strategy_yaml }}</pre>
              </div>
              <a-empty v-else description="No configuration" :image-style="{ height: '40px' }" />
            </a-card>

            <a-card title="Backtest Result" style="margin-top: 16px" v-if="wikiStore.currentStrategy.backtest_result">
              <a-descriptions :column="2" bordered>
                <a-descriptions-item label="Total Return">
                  {{ formatPercent(wikiStore.currentStrategy.backtest_result.total_return) }}
                </a-descriptions-item>
                <a-descriptions-item label="Annual Return">
                  {{ formatPercent(wikiStore.currentStrategy.backtest_result.annual_return) }}
                </a-descriptions-item>
                <a-descriptions-item label="Sharpe Ratio">
                  {{ wikiStore.currentStrategy.backtest_result.sharpe_ratio?.toFixed(2) || '-' }}
                </a-descriptions-item>
                <a-descriptions-item label="Max Drawdown">
                  {{ formatPercent(wikiStore.currentStrategy.backtest_result.max_drawdown) }}
                </a-descriptions-item>
              </a-descriptions>
            </a-card>
          </a-col>

          <a-col :span="8">
            <a-card title="Properties">
              <a-descriptions :column="1" bordered size="small">
                <a-descriptions-item label="Name">
                  {{ wikiStore.currentStrategy.name }}
                </a-descriptions-item>
                <a-descriptions-item label="Category">
                  <a-tag :color="getCategoryColor(wikiStore.currentStrategy.category)">
                    {{ wikiStore.currentStrategy.category }}
                  </a-tag>
                </a-descriptions-item>
                <a-descriptions-item label="Wiki Page">
                  <a v-if="wikiStore.currentStrategy.wiki_page_name" href="#" @click.prevent>
                    {{ wikiStore.currentStrategy.wiki_page_name }}
                  </a>
                  <span v-else>-</span>
                </a-descriptions-item>
              </a-descriptions>
            </a-card>

            <a-card title="Tags" style="margin-top: 16px">
              <div v-if="wikiStore.currentStrategy.tags?.length">
                <a-tag v-for="tag in wikiStore.currentStrategy.tags" :key="tag" color="blue">
                  {{ tag }}
                </a-tag>
              </div>
              <a-empty v-else description="No tags" :image-style="{ height: '40px' }" />
            </a-card>
          </a-col>
        </a-row>
      </template>

      <a-empty v-else-if="!wikiStore.isLoading" description="Strategy not found" />
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  LineChartOutlined,
  EditOutlined,
} from '@ant-design/icons-vue'
import { useWikiStore } from '@/stores/wiki'

const route = useRoute()
const router = useRouter()
const wikiStore = useWikiStore()

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

const formatPercent = (value?: number) => {
  if (value === null || value === undefined) return '-'
  return `${(value * 100).toFixed(2)}%`
}

const handleBacktest = () => {
  const strategy = wikiStore.currentStrategy
  if (strategy) {
    router.push({
      path: '/backtest',
      query: { strategy: strategy.name },
    })
  }
}

const handleEdit = () => {
  message.info('Edit functionality coming soon')
}

onMounted(() => {
  const name = route.params.name as string
  if (name) {
    wikiStore.fetchStrategy(name)
  }
})
</script>

<style scoped>
.strategy-detail {
  padding: 0;
}

.yaml-container {
  background: #f6f8fa;
  border-radius: 8px;
  overflow: hidden;
}

.yaml-content {
  padding: 16px;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 13px;
  overflow-x: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}

:deep(.ant-descriptions-item-label) {
  font-weight: 500;
}
</style>
