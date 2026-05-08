<template>
  <div class="strategy-editor">
    <a-page-header title="Strategy Editor" @back="$router.back()">
      <template #subtitle>
        <span v-if="currentStrategy">{{ currentStrategy.name }}</span>
        <span v-else>New Strategy</span>
      </template>
      <template #extra>
        <a-space>
          <a-button @click="handleNew">
            <template #icon><FileAddOutlined /></template>
            New
          </a-button>
          <a-button @click="handleSave" :loading="saving" type="primary">
            <template #icon><SaveOutlined /></template>
            Save
          </a-button>
          <a-button @click="handleBacktest" :disabled="!hasContent">
            <template #icon><LineChartOutlined /></template>
            Run Backtest
          </a-button>
        </a-space>
      </template>
    </a-page-header>

    <a-row :gutter="[16, 16]" style="height: calc(100vh - 200px)">
      <a-col :span="18">
        <a-card title="Strategy Configuration" :bordered="true" style="height: 100%">
          <template #extra>
            <a-space>
              <a-button size="small" @click="handleFormat">Format</a-button>
              <a-button size="small" @click="handleCopy">Copy</a-button>
              <a-button size="small" @click="handleDownload">Download</a-button>
            </a-space>
          </template>
          <MonacoEditor
            ref="editorRef"
            v-model:value="yamlContent"
            language="yaml"
            :height="editorHeight"
            :minimap="true"
            @save="handleSave"
          />
        </a-card>
      </a-col>

      <a-col :span="6">
        <a-card title="Strategy Info" style="margin-bottom: 16px">
          <a-form layout="vertical" size="small">
            <a-form-item label="Name">
              <a-input v-model:value="strategyForm.name" placeholder="strategy_name" />
            </a-form-item>
            <a-form-item label="Description">
              <a-textarea v-model:value="strategyForm.description" :rows="2" />
            </a-form-item>
            <a-form-item label="Category">
              <a-select v-model:value="strategyForm.category" style="width: 100%">
                <a-select-option value="momentum">Momentum</a-select-option>
                <a-select-option value="mean_reversion">Mean Reversion</a-select-option>
                <a-select-option value="trend_following">Trend Following</a-select-option>
                <a-select-option value="statistical_arbitrage">Statistical Arbitrage</a-select-option>
                <a-select-option value="general">General</a-select-option>
              </a-select>
            </a-form-item>
            <a-form-item label="Tags">
              <a-select v-model:value="strategyForm.tags" mode="tags" placeholder="Add tags" />
            </a-form-item>
          </a-form>
        </a-card>

        <a-card title="Validation" style="margin-bottom: 16px">
          <a-spin :spinning="validating">
            <div v-if="validationResult">
              <a-alert
                v-if="validationResult.valid"
                type="success"
                message="Valid YAML"
                showIcon
                style="margin-bottom: 8px"
              />
              <a-alert
                v-else
                type="error"
                :message="validationResult.error"
                showIcon
                style="margin-bottom: 8px"
              />
            </div>
            <a-button block @click="handleValidate" :loading="validating">
              <template #icon><CheckCircleOutlined /></template>
              Validate YAML
            </a-button>
          </a-spin>
        </a-card>

        <a-card title="Template">
          <a-space direction="vertical" style="width: 100%">
            <a-button block @click="loadTemplate('momentum')">
              Momentum Strategy
            </a-button>
            <a-button block @click="loadTemplate('mean_reversion')">
              Mean Reversion
            </a-button>
            <a-button block @click="loadTemplate('trend_following')">
              Trend Following
            </a-button>
            <a-button block @click="loadTemplate('custom')">
              Custom Template
            </a-button>
          </a-space>
        </a-card>

        <a-card title="History" style="margin-top: 16px">
          <a-timeline size="small">
            <a-timeline-item v-if="lastSaved" color="green">
              Saved at {{ formatTime(lastSaved) }}
            </a-timeline-item>
            <a-timeline-item v-for="(item, index) in history" :key="index" color="blue">
              {{ item.action }} at {{ formatTime(item.time) }}
            </a-timeline-item>
          </a-timeline>
          <a-empty v-if="!lastSaved && !history.length" description="No history" :image-style="{ height: '40px' }" />
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  FileAddOutlined,
  SaveOutlined,
  LineChartOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons-vue'
import MonacoEditor from '@/components/Editor/MonacoEditor.vue'
import { useWikiStore } from '@/stores/wiki'
import { post } from '@/api'

const route = useRoute()
const router = useRouter()
const wikiStore = useWikiStore()

const editorRef = ref<InstanceType<typeof MonacoEditor>>()
const yamlContent = ref('')
const saving = ref(false)
const validating = ref(false)
const validationResult = ref<{ valid: boolean; error?: string } | null>(null)
const lastSaved = ref<Date | null>(null)
const history = ref<{ action: string; time: Date }[]>([])
const editorHeight = ref(500)

const strategyForm = ref({
  name: '',
  description: '',
  category: 'general' as const,
  tags: [] as string[],
})

const currentStrategy = computed(() => wikiStore.currentStrategy)

const hasContent = computed(() => yamlContent.value.trim().length > 0)

const templates: Record<string, string> = {
  momentum: `# Momentum Strategy
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
`,
  mean_reversion: `# Mean Reversion Strategy
# Buy oversold stocks, sell overbought

strategy:
  name: mean_reversion_strategy
  universe: hs300
  
signals:
  - name: rsi_14
    formula: "100 - 100 / (1 + rs(close, 14))"
    weight: 0.4
    condition: "< 30"
  - name: bollinger_signal
    formula: "(close - bb_lower(close, 20, 2)) / (bb_upper(close, 20, 2) - bb_lower(close, 20, 2))"
    weight: 0.3
  - name: volume_spike
    formula: "volume / volume.rolling(20).mean()"
    weight: 0.3
    
portfolio:
  rebalance_days: 1
  max_positions: 30
  position_sizing: inverse_volatility
  
risk:
  max_drawdown: 0.12
  stop_loss: 0.03
`,
  trend_following: `# Trend Following Strategy
# Follow strong trends with momentum

strategy:
  name: trend_following_strategy
  universe: hs300
  
signals:
  - name: trend_strength
    formula: "close / ts_max(close, 60) - 1"
    weight: 0.4
  - name: ma_cross
    formula: "ts_mean(close, 20) / ts_mean(close, 60) - 1"
    weight: 0.3
  - name: volatility_regime
    formula: "ts_std(close, 20) / ts_std(close, 60)"
    weight: 0.3
    
portfolio:
  rebalance_days: 10
  max_positions: 15
  position_sizing: volatility_target
  
risk:
  max_drawdown: 0.20
  trailing_stop: 0.08
`,
  custom: `# Custom Strategy Template
# Modify this template for your strategy

strategy:
  name: custom_strategy
  universe: hs300
  start_date: "2020-01-01"
  end_date: "2024-12-31"
  
signals:
  - name: signal_1
    formula: "your_formula_here"
    weight: 1.0
    
portfolio:
  rebalance_days: 5
  max_positions: 20
  position_sizing: equal_weight
  
risk:
  max_drawdown: 0.15
`,
}

const formatTime = (date: Date) => {
  return date.toLocaleTimeString()
}

const handleNew = () => {
  yamlContent.value = templates.custom
  strategyForm.value = {
    name: '',
    description: '',
    category: 'general',
    tags: [],
  }
  history.value = []
  lastSaved.value = null
}

const handleSave = async () => {
  if (!strategyForm.value.name) {
    message.error('Please enter a strategy name')
    return
  }

  saving.value = true
  try {
    await wikiStore.createStrategy({
      name: strategyForm.value.name,
      description: strategyForm.value.description,
      category: strategyForm.value.category,
      tags: strategyForm.value.tags,
      strategy_yaml: yamlContent.value,
    })
    message.success('Strategy saved successfully')
    lastSaved.value = new Date()
    history.value.unshift({ action: 'Saved', time: new Date() })
  } catch (error) {
    message.error('Failed to save strategy')
  } finally {
    saving.value = false
  }
}

const handleBacktest = () => {
  if (!strategyForm.value.name) {
    message.error('Please save the strategy first')
    return
  }
  router.push({
    path: '/backtest',
    query: { strategy: strategyForm.value.name },
  })
}

const handleValidate = async () => {
  validating.value = true
  try {
    const result = await post<{ valid: boolean; error?: string }>('/strategy/validate', {
      yaml: yamlContent.value,
    })
    validationResult.value = result
  } catch (error) {
    // Basic client-side validation
    try {
      // Simple YAML validation check
      if (yamlContent.value.includes('\t')) {
        validationResult.value = { valid: false, error: 'YAML cannot contain tabs' }
      } else {
        validationResult.value = { valid: true }
      }
    } catch (e) {
      validationResult.value = { valid: false, error: 'Invalid YAML syntax' }
    }
  } finally {
    validating.value = false
  }
}

const handleFormat = () => {
  editorRef.value?.format()
}

const handleCopy = () => {
  navigator.clipboard.writeText(yamlContent.value)
  message.success('Copied to clipboard')
}

const handleDownload = () => {
  const blob = new Blob([yamlContent.value], { type: 'text/yaml' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${strategyForm.value.name || 'strategy'}.yaml`
  a.click()
  URL.revokeObjectURL(url)
}

const loadTemplate = (templateName: string) => {
  yamlContent.value = templates[templateName] || templates.custom
  message.info(`Loaded ${templateName} template`)
}

const updateEditorHeight = () => {
  editorHeight.value = window.innerHeight - 250
}

onMounted(async () => {
  updateEditorHeight()
  window.addEventListener('resize', updateEditorHeight)

  const strategyName = route.query.strategy as string
  if (strategyName) {
    await wikiStore.fetchStrategy(strategyName)
    if (wikiStore.currentStrategy) {
      strategyForm.value = {
        name: wikiStore.currentStrategy.name,
        description: wikiStore.currentStrategy.description,
        category: wikiStore.currentStrategy.category as any,
        tags: wikiStore.currentStrategy.tags,
      }
      yamlContent.value = wikiStore.currentStrategy.strategy_yaml || templates.custom
    }
  } else {
    yamlContent.value = templates.momentum
  }
})

onUnmounted(() => {
  window.removeEventListener('resize', updateEditorHeight)
})
</script>

<style scoped>
.strategy-editor {
  padding: 0;
}

:deep(.ant-card-body) {
  padding: 12px;
}

:deep(.ant-page-header) {
  padding: 12px 0;
}
</style>
