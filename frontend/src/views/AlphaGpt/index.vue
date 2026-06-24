<template>
  <div class="alpha-gpt">
    <a-page-header
      title="Alpha-GPT 自动化因子挖掘"
      sub-title="基于 5 智能体编排 + nanobot Agent 体系的因子发现"
    >
      <template #extra>
        <a-tag color="blue">M6 (v2.7.0+)</a-tag>
        <a-tag color="green">LLM: {{ llmProvider }}</a-tag>
      </template>
    </a-page-header>

    <a-row :gutter="[16, 16]">
      <!-- 左：配置面板 -->
      <a-col :span="8">
        <a-card title="工作流配置" :bordered="false">
          <a-form :model="config" layout="vertical">
            <a-form-item label="研究目标 (Objective)" required>
              <a-textarea
                v-model:value="config.objective"
                :rows="3"
                placeholder="例如：捕捉 A 股反转效应"
              />
            </a-form-item>

            <a-form-item label="数据路径">
              <a-input
                v-model:value="config.dataPath"
                placeholder="留空使用合成数据"
              />
            </a-form-item>

            <a-row :gutter="8">
              <a-col :span="12">
                <a-form-item label="迭代轮次">
                  <a-input-number v-model:value="config.iterations" :min="1" :max="20" />
                </a-form-item>
              </a-col>
              <a-col :span="12">
                <a-form-item label="每轮公式数">
                  <a-input-number v-model:value="config.poolSize" :min="1" :max="50" />
                </a-form-item>
              </a-col>
            </a-row>

            <a-row :gutter="8">
              <a-col :span="12">
                <a-form-item label="最终 top-K">
                  <a-input-number v-model:value="config.topK" :min="1" :max="50" />
                </a-form-item>
              </a-col>
              <a-col :span="12">
                <a-form-item label="LLM Provider">
                  <a-select v-model:value="config.llmProvider">
                    <a-select-option value="mock">mock</a-select-option>
                    <a-select-option value="deepseek">deepseek</a-select-option>
                    <a-select-option value="openai">openai</a-select-option>
                    <a-select-option value="qwen">qwen</a-select-option>
                  </a-select>
                </a-form-item>
              </a-col>
            </a-row>

            <a-form-item>
              <a-checkbox v-model:checked="config.enableBacktest">
                启用 Trading 回测（仅对 top-K 评估）
              </a-checkbox>
            </a-form-item>

            <a-form-item>
              <a-space>
                <a-button
                  type="primary"
                  :loading="running"
                  :disabled="!config.objective"
                  @click="start"
                >
                  启动
                </a-button>
                <a-button :disabled="!running" @click="stop">停止</a-button>
              </a-space>
            </a-form-item>
          </a-form>
        </a-card>

        <a-card title="实时事件" :bordered="false" style="margin-top: 16px">
          <a-timeline>
            <a-timeline-item
              v-for="(evt, idx) in eventTimeline"
              :key="idx"
              :color="eventColor(evt.type)"
            >
              <span class="evt-type">{{ evt.type }}</span>
              <span class="evt-detail" v-if="evt.round">
                round={{ evt.round }}/{{ evt.total_rounds || '?' }}
              </span>
              <span class="evt-detail" v-if="evt.best_ir !== undefined">
                best_ir={{ evt.best_ir?.toFixed(3) }}
              </span>
              <span class="evt-detail" v-if="evt.formulas_evaluated !== undefined">
                formulas={{ evt.formulas_evaluated }}
              </span>
            </a-timeline-item>
          </a-timeline>
        </a-card>
      </a-col>

      <!-- 右：进度 + 结果 -->
      <a-col :span="16">
        <a-row :gutter="[16, 16]">
          <a-col :span="6">
            <MetricCard
              title="状态"
              :value="statusLabel"
              :value-style="{ color: statusColor }"
            />
          </a-col>
          <a-col :span="6">
            <MetricCard title="Session ID" :value="sessionId?.slice(0, 12) || '—'" />
          </a-col>
          <a-col :span="6">
            <MetricCard
              title="公式评估"
              :value="`${summary.successful || 0} / ${summary.total_evaluated || 0}`"
              :value-style="{ color: '#1677ff' }"
            />
          </a-col>
          <a-col :span="6">
            <MetricCard
              title="最佳 IR"
              :value="(summary.best_ir || 0).toFixed(3)"
              :value-style="{ color: '#52c41a' }"
            />
          </a-col>
        </a-row>

        <a-card title="进度" :bordered="false" style="margin-top: 16px">
          <a-progress
            :percent="progressPercent"
            :status="progressStatus"
            :format="(p) => `Round ${currentRound} / ${totalRounds}`"
          />
        </a-card>

        <a-card title="Top-K 公式" :bordered="false" style="margin-top: 16px">
          <a-empty v-if="!finalPool.length" description="工作流完成后将显示结果" />
          <a-table
            v-else
            :columns="resultColumns"
            :data-source="finalPool"
            :pagination="false"
            row-key="formula_id"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'rank'">
                <a-tag :color="rankColor(record.rank)">#{{ record.rank }}</a-tag>
              </template>
              <template v-else-if="column.key === 'formula'">
                <a-typography-text code style="font-size: 12px">
                  {{ record.formula }}
                </a-typography-text>
              </template>
              <template v-else-if="column.key === 'ir'">
                <span :style="{ color: record.ir > 0 ? '#52c41a' : '#ff4d4f', fontWeight: 600 }">
                  {{ record.ir?.toFixed(3) }}
                </span>
              </template>
              <template v-else-if="column.key === 'ic_mean'">
                {{ record.ic_mean?.toFixed(4) }}
              </template>
              <template v-else-if="column.key === 'reason'">
                <a-typography-text type="secondary" style="font-size: 12px">
                  {{ record.selection_reason?.slice(0, 60) }}
                </a-typography-text>
              </template>
            </template>
          </a-table>
        </a-card>

        <a-card title="最佳 IR 演化（按 round）" :bordered="false" style="margin-top: 16px">
          <IcChart
            :dates="irEvolution.dates"
            :ic-values="irEvolution.values"
          />
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted, reactive } from 'vue'
import { message } from 'ant-design-vue'
import MetricCard from '@/components/Charts/MetricCard.vue'
import IcChart from '@/components/Charts/IcChart.vue'
import { useWebSocket } from '@/composables/useWebSocket'

interface AlphaGptEvent {
  type: string
  session_id?: string
  ts?: number
  round?: number
  total_rounds?: number
  best_ir?: number
  formulas_evaluated?: number
  pool?: Array<{
    rank: number
    formula_id: string
    formula: string
    ir: number
    ic_mean: number
    selection_reason?: string
  }>
  summary?: Record<string, any>
  message?: string
}

const config = reactive({
  objective: '',
  dataPath: '',
  iterations: 5,
  poolSize: 10,
  topK: 10,
  llmProvider: 'mock',
  enableBacktest: false,
})

const running = ref(false)
const sessionId = ref<string | null>(null)
const status = ref<string>('idle')  // idle | running | completed | failed | stopped
const eventTimeline = ref<AlphaGptEvent[]>([])
const summary = ref<Record<string, any>>({})
const finalPool = ref<any[]>([])
const allEvaluations = ref<any[]>([])

const irEvolution = computed(() => {
  const rounds = allEvaluations.value
    .filter(e => e.round && e.best_ir !== undefined)
    .sort((a, b) => a.round - b.round)
  return {
    dates: rounds.map(e => `R${e.round}`),
    values: rounds.map(e => e.best_ir || 0),
  }
})

const wsUrl = computed(() => {
  if (!sessionId.value) return ''
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}/api/alpha/alpha-gpt/stream/${sessionId.value}`
})

const { connect, disconnect, isConnected } = useWebSocket({
  url: wsUrl.value,
  onMessage: handleEvent,
  reconnectInterval: 2000,
  maxReconnectAttempts: 3,
})

const llmProvider = computed(() => config.llmProvider)
const statusLabel = computed(() => {
  const map: Record<string, string> = {
    idle: '空闲', running: '运行中', completed: '完成',
    failed: '失败', stopped: '已停止',
  }
  return map[status.value] || status.value
})
const statusColor = computed(() => {
  const map: Record<string, string> = {
    idle: '#999', running: '#1677ff', completed: '#52c41a',
    failed: '#ff4d4f', stopped: '#faad14',
  }
  return map[status.value] || '#999'
})
const currentRound = computed(() => {
  const last = [...eventTimeline.value].reverse().find(e => e.round)
  return last?.round || 0
})
const totalRounds = computed(() => {
  const last = [...eventTimeline.value].reverse().find(e => e.total_rounds)
  return last?.total_rounds || config.iterations
})
const progressPercent = computed(() => {
  if (status.value === 'completed') return 100
  if (status.value === 'idle') return 0
  return Math.min(100, (currentRound.value / totalRounds.value) * 100)
})
const progressStatus = computed(() => {
  if (status.value === 'completed') return 'success'
  if (status.value === 'failed') return 'exception'
  return 'active'
})

const resultColumns = [
  { title: 'Rank', key: 'rank', width: 60 },
  { title: 'Formula', key: 'formula', ellipsis: true },
  { title: 'IR', key: 'ir', width: 80 },
  { title: 'IC Mean', key: 'ic_mean', width: 100 },
  { title: 'Reason', key: 'reason', ellipsis: true },
]

function eventColor(type: string): string {
  const map: Record<string, string> = {
    round_started: 'blue',
    round_completed: 'green',
    formulas_evaluated: 'cyan',
    final_pool_ready: 'gold',
    done: 'green',
    error: 'red',
  }
  return map[type] || 'gray'
}

function rankColor(rank: number): string {
  if (rank === 1) return 'gold'
  if (rank <= 3) return 'orange'
  if (rank <= 10) return 'blue'
  return 'default'
}

function handleEvent(evt: AlphaGptEvent) {
  eventTimeline.value.push(evt)
  if (eventTimeline.value.length > 200) {
    eventTimeline.value = eventTimeline.value.slice(-100)
  }

  switch (evt.type) {
    case 'round_started':
      status.value = 'running'
      break
    case 'round_completed':
      if (evt.formulas_evaluated !== undefined && evt.best_ir !== undefined) {
        allEvaluations.value.push({
          round: evt.round,
          formulas: evt.formulas_evaluated,
          best_ir: evt.best_ir,
        })
      }
      break
    case 'formulas_evaluated':
      // extend allEvaluations
      break
    case 'final_pool_ready':
      finalPool.value = evt.pool || []
      summary.value = evt.summary || {}
      break
    case 'done':
      status.value = 'completed'
      running.value = false
      disconnect()
      message.success('Alpha-GPT workflow completed')
      break
    case 'error':
      status.value = evt.message === 'stopped' ? 'stopped' : 'failed'
      running.value = false
      disconnect()
      if (evt.message && evt.message !== 'stopped') {
        message.error(`Workflow error: ${evt.message}`)
      }
      break
  }
}

async function start() {
  if (!config.objective) {
    message.warning('请填写研究目标')
    return
  }
  running.value = true
  status.value = 'running'
  eventTimeline.value = []
  summary.value = {}
  finalPool.value = []
  allEvaluations.value = []

  try {
    const resp = await fetch('/api/alpha/alpha-gpt/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        objective: config.objective,
        data_path: config.dataPath || null,
        iterations: config.iterations,
        pool_size: config.poolSize,
        top_k: config.topK,
        forward_returns: [1, 5, 20],
        llm_provider: config.llmProvider,
        enable_backtest: config.enableBacktest,
      }),
    })
    const data = await resp.json()
    if (!resp.ok) {
      throw new Error(data.detail || `HTTP ${resp.status}`)
    }
    sessionId.value = data.session_id
    message.info(`工作流已启动: ${data.session_id}`)
    // 重新创建 useWebSocket 因为 url 变化
    setTimeout(() => connect(), 100)
  } catch (e: any) {
    running.value = false
    status.value = 'failed'
    message.error(`启动失败: ${e.message}`)
  }
}

async function stop() {
  if (!sessionId.value) return
  try {
    await fetch(`/api/alpha/alpha-gpt/stop/${sessionId.value}`, { method: 'POST' })
    message.info('已停止')
  } catch (e: any) {
    message.error(`停止失败: ${e.message}`)
  }
}

onUnmounted(() => {
  disconnect()
})
</script>

<style scoped>
.alpha-gpt {
  padding: 16px;
}
.evt-type {
  font-weight: 600;
  margin-right: 8px;
}
.evt-detail {
  margin-left: 8px;
  color: #666;
  font-size: 12px;
}
</style>
