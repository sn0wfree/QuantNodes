<template>
  <div class="factor-detail">
    <a-spin :spinning="wikiStore.isLoading" size="large">
      <template v-if="wikiStore.currentFactor">
        <a-page-header :title="wikiStore.currentFactor.name" @back="$router.back()">
          <template #subtitle>
            <a-tag :color="getCategoryColor(wikiStore.currentFactor.category)">
              {{ wikiStore.currentFactor.category }}
            </a-tag>
            <a-tag>{{ wikiStore.currentFactor.source }}</a-tag>
          </template>
          <template #extra>
            <a-space>
              <a-button @click="handleAnalyze">
                <template #icon><BarChartOutlined /></template>
                Analyze
              </a-button>
              <a-button @click="handleEdit">
                <template #icon><EditOutlined /></template>
                Edit
              </a-button>
              <a-popconfirm title="Delete this factor?" @confirm="handleDelete">
                <a-button danger>
                  <template #icon><DeleteOutlined /></template>
                  Delete
                </a-button>
              </a-popconfirm>
            </a-space>
          </template>
        </a-page-header>

        <a-row :gutter="[16, 16]">
          <a-col :span="16">
            <a-card title="Formula">
              <div class="formula-container">
                <pre class="formula">{{ wikiStore.currentFactor.formula }}</pre>
                <a-button class="copy-btn" size="small" @click="copyFormula">
                  <template #icon><CopyOutlined /></template>
                </a-button>
              </div>
            </a-card>

            <a-card title="Description" style="margin-top: 16px">
              <p v-if="wikiStore.currentFactor.description">
                {{ wikiStore.currentFactor.description }}
              </p>
              <a-empty v-else description="No description provided" :image-style="{ height: '40px' }" />
            </a-card>

            <a-card title="Performance Metrics" style="margin-top: 16px" v-if="hasMetrics">
              <a-row :gutter="[16, 16]">
                <a-col :span="6">
                  <a-statistic
                    title="IC Mean"
                    :value="wikiStore.currentFactor.ic_mean"
                    :precision="4"
                    :value-style="{ color: (wikiStore.currentFactor.ic_mean || 0) > 0 ? '#52c41a' : '#ff4d4f' }"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="IC Std"
                    :value="wikiStore.currentFactor.ic_std"
                    :precision="4"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="ICIR"
                    :value="wikiStore.currentFactor.icir"
                    :precision="4"
                    :value-style="{ color: getIcirColor(wikiStore.currentFactor.icir) }"
                  />
                </a-col>
                <a-col :span="6">
                  <a-statistic
                    title="Rank IC"
                    :value="wikiStore.currentFactor.rank_ic_mean"
                    :precision="4"
                  />
                </a-col>
              </a-row>
            </a-card>
          </a-col>

          <a-col :span="8">
            <a-card title="Properties">
              <a-descriptions :column="1" bordered size="small">
                <a-descriptions-item label="Name">
                  {{ wikiStore.currentFactor.name }}
                </a-descriptions-item>
                <a-descriptions-item label="Category">
                  <a-tag :color="getCategoryColor(wikiStore.currentFactor.category)">
                    {{ wikiStore.currentFactor.category }}
                  </a-tag>
                </a-descriptions-item>
                <a-descriptions-item label="Source">
                  {{ wikiStore.currentFactor.source }}
                </a-descriptions-item>
                <a-descriptions-item label="Wiki Page">
                  <a v-if="wikiStore.currentFactor.wiki_page_name" href="#" @click.prevent>
                    {{ wikiStore.currentFactor.wiki_page_name }}
                  </a>
                  <span v-else>-</span>
                </a-descriptions-item>
              </a-descriptions>
            </a-card>

            <a-card title="Tags" style="margin-top: 16px">
              <div v-if="wikiStore.currentFactor.tags?.length">
                <a-tag v-for="tag in wikiStore.currentFactor.tags" :key="tag" color="blue">
                  {{ tag }}
                </a-tag>
              </div>
              <a-empty v-else description="No tags" :image-style="{ height: '40px' }" />
            </a-card>

            <a-card title="Related Items" style="margin-top: 16px">
              <a-empty description="No related items" :image-style="{ height: '40px' }" />
            </a-card>
          </a-col>
        </a-row>
      </template>

      <a-empty v-else-if="!wikiStore.isLoading" description="Factor not found" />
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  BarChartOutlined,
  EditOutlined,
  DeleteOutlined,
  CopyOutlined,
} from '@ant-design/icons-vue'
import { useWikiStore } from '@/stores/wiki'

const route = useRoute()
const router = useRouter()
const wikiStore = useWikiStore()

const hasMetrics = computed(() => {
  const factor = wikiStore.currentFactor
  return factor && (
    factor.ic_mean != null ||
    factor.ic_std != null ||
    factor.icir != null ||
    factor.rank_ic_mean != null
  )
})

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

const copyFormula = () => {
  if (wikiStore.currentFactor?.formula) {
    navigator.clipboard.writeText(wikiStore.currentFactor.formula)
    message.success('Formula copied to clipboard')
  }
}

const handleAnalyze = () => {
  const factor = wikiStore.currentFactor
  if (factor) {
    router.push({
      path: '/factor-analysis',
      query: { expression: factor.formula },
    })
  }
}

const handleEdit = () => {
  // TODO: Implement edit modal
  message.info('Edit functionality coming soon')
}

const handleDelete = async () => {
  const name = route.params.name as string
  try {
    await wikiStore.deleteFactor(name)
    message.success('Factor deleted')
    router.push('/wiki/factors')
  } catch (error) {
    message.error('Failed to delete factor')
  }
}

onMounted(() => {
  const name = route.params.name as string
  if (name) {
    wikiStore.fetchFactor(name)
  }
})
</script>

<style scoped>
.factor-detail {
  padding: 0;
}

.formula-container {
  position: relative;
}

.formula {
  background: #f6f8fa;
  padding: 16px;
  border-radius: 8px;
  font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
  font-size: 14px;
  overflow-x: auto;
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}

.copy-btn {
  position: absolute;
  top: 8px;
  right: 8px;
}

:deep(.ant-descriptions-item-label) {
  font-weight: 500;
}
</style>
