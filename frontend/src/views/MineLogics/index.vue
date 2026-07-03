<template>
  <div class="mine-logics">
    <a-page-header
      title="自动化因子挖掘"
      sub-title="Logic Mining 批量挖掘 + 实时进度 + 结果展示"
    >
      <template #extra>
        <a-tag color="blue">v3.0.3</a-tag>
        <a-tag :color="wsConnected ? 'green' : 'default'">
          WS: {{ wsConnected ? 'Connected' : 'Disconnected' }}
        </a-tag>
      </template>
    </a-page-header>

    <a-row :gutter="[16, 16]">
      <!-- ====== 左：配置面板 ====== -->
      <a-col :span="8">
        <a-card title="挖掘配置" :bordered="false">
          <a-form :model="config" layout="vertical">
            <a-form-item label="来源库">
              <a-checkbox-group v-model:value="config.sourceLibs" :options="sourceLibOptions" />
            </a-form-item>

            <a-row :gutter="8">
              <a-col :span="12">
                <a-form-item label="每库最大条数">
                  <a-input-number v-model:value="config.maxPerLib" :min="1" :max="100" style="width: 100%" />
                </a-form-item>
              </a-col>
              <a-col :span="12">
                <a-form-item label="并发线程数">
                  <a-input-number v-model:value="config.workers" :min="1" :max="16" style="width: 100%" />
                </a-form-item>
              </a-col>
            </a-row>

            <a-form-item label="Wiki 路径">
              <a-input v-model:value="config.wikiPath" placeholder="wiki_auto" />
            </a-form-item>

            <a-row :gutter="8">
              <a-col :span="12">
                <a-form-item label="Live 模式">
                  <a-switch v-model:checked="config.live" />
                  <span style="margin-left: 8px; color: #999">{{ config.live ? '真实 LLM' : '离线' }}</span>
                </a-form-item>
              </a-col>
              <a-col :span="12">
                <a-form-item label="严格模式">
                  <a-switch v-model:checked="config.strict" />
                </a-form-item>
              </a-col>
            </a-row>

            <a-form-item>
              <a-space>
                <a-button
                  type="primary"
                  :loading="isRunning"
                  :disabled="isRunning"
                  @click="handleStart"
                >
                  {{ isRunning ? '挖掘中...' : '开始挖掘' }}
                </a-button>
                <a-button
                  danger
                  :disabled="!isRunning"
                  @click="handleStop"
                >
                  停止
                </a-button>
              </a-space>
            </a-form-item>
          </a-form>

          <!-- 状态摘要 -->
          <a-divider />
          <a-space direction="vertical" :size="4" style="width: 100%">
            <a-row justify="space-between">
              <span>状态</span>
              <a-tag :color="statusColor">{{ statusText }}</a-tag>
            </a-row>
            <a-row justify="space-between" v-if="currentRunId">
              <span>Run ID</span>
              <span style="font-family: monospace; font-size: 12px">{{ currentRunId }}</span>
            </a-row>
            <a-row justify="space-between" v-if="statusData?.elapsed_seconds">
              <span>耗时</span>
              <span>{{ statusData.elapsed_seconds.toFixed(1) }}s</span>
            </a-row>
          </a-space>
        </a-card>
      </a-col>

      <!-- ====== 右：进度面板 ====== -->
      <a-col :span="16">
        <a-card title="挖掘进度" :bordered="false">
          <!-- Progress bar -->
          <a-progress
            :percent="progressPercent"
            :status="progressStatus"
            :stroke-color="progressColor"
          />

          <!-- 统计卡片 -->
          <a-row :gutter="[16, 16]" style="margin-top: 16px">
            <a-col :span="6">
              <a-statistic title="已尝试" :value="progress.done" :suffix="`/ ${progress.total}`" />
            </a-col>
            <a-col :span="6">
              <a-statistic title="已挖掘" :value="progress.nMined" :value-style="{ color: '#52c41a' }" />
            </a-col>
            <a-col :span="6">
              <a-statistic title="已跳过" :value="progress.nSkipped" :value-style="{ color: '#faad14' }" />
            </a-col>
            <a-col :span="6">
              <a-statistic title="失败" :value="progress.nFailed" :value-style="{ color: '#ff4d4f' }" />
            </a-col>
          </a-row>

          <!-- 事件时间线 -->
          <a-divider />
          <div style="max-height: 300px; overflow-y: auto">
            <a-timeline>
              <a-timeline-item
                v-for="(event, idx) in timelineEvents"
                :key="idx"
                :color="timelineItemColor(event.type)"
              >
                <span style="font-size: 12px; color: #999">{{ formatTs(event.ts) }}</span>
                <span style="margin-left: 8px">{{ formatEvent(event) }}</span>
              </a-timeline-item>
            </a-timeline>
            <a-empty v-if="timelineEvents.length === 0" description="暂无事件" :image-style="{ height: '40px' }" />
          </div>
        </a-card>
      </a-col>
    </a-row>

    <!-- ====== 结果面板 ====== -->
    <a-row :gutter="[16, 16]" style="margin-top: 16px" v-if="results">
      <a-col :span="24">
        <a-card title="挖掘结果" :bordered="false">
          <a-tabs>
            <!-- Top Factors 表格 -->
            <a-tab-pane key="factors" tab="Top Factors">
              <a-table
                :columns="factorColumns"
                :data-source="results.result?.top_factors || []"
                :pagination="{ pageSize: 20 }"
                size="small"
                row-key="formula_id"
              >
                <template #bodyCell="{ column, record }">
                  <template v-if="column.key === 'formula'">
                    <span style="font-family: monospace; font-size: 12px">{{ record.formula }}</span>
                  </template>
                  <template v-if="column.key === 'ir'">
                    <span :style="{ color: record.ir > 0 ? '#52c41a' : '#ff4d4f' }">
                      {{ record.ir.toFixed(4) }}
                    </span>
                  </template>
                  <template v-if="column.key === 'tags'">
                    <a-tag v-for="tag in record.tags" :key="tag" size="small">{{ tag }}</a-tag>
                  </template>
                </template>
              </a-table>
            </a-tab-pane>

            <!-- Source Breakdown -->
            <a-tab-pane key="source" tab="来源分析">
              <a-row :gutter="[16, 16]">
                <a-col :span="12">
                  <h4>来源库分布</h4>
                  <a-table
                    :columns="sourceColumns"
                    :data-source="sourceData"
                    :pagination="false"
                    size="small"
                  />
                </a-col>
                <a-col :span="12">
                  <h4>Agent 统计</h4>
                  <a-table
                    :columns="agentColumns"
                    :data-source="agentData"
                    :pagination="false"
                    size="small"
                  />
                </a-col>
              </a-row>
            </a-tab-pane>

            <!-- Warnings -->
            <a-tab-pane key="warnings" tab="警告" v-if="results.result?.warnings?.length">
              <a-alert
                v-for="(w, idx) in results.result.warnings"
                :key="idx"
                :message="w"
                type="warning"
                show-icon
                style="margin-bottom: 8px"
              />
            </a-tab-pane>
          </a-tabs>
        </a-card>
      </a-col>
    </a-row>

    <!-- ====== 历史面板 ====== -->
    <a-row :gutter="[16, 16]" style="margin-top: 16px">
      <a-col :span="24">
        <a-card title="历史运行" :bordered="false">
          <a-table
            :columns="historyColumns"
            :data-source="historyRuns"
            :pagination="{ pageSize: 10 }"
            size="small"
            row-key="run_id"
            :loading="historyLoading"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'status'">
                <a-tag :color="historyStatusColor(record.status)">{{ record.status }}</a-tag>
              </template>
              <template v-if="column.key === 'created_at'">
                {{ formatDate(record.created_at) }}
              </template>
              <template v-if="column.key === 'elapsed'">
                {{ record.elapsed_seconds ? record.elapsed_seconds.toFixed(1) + 's' : '-' }}
              </template>
              <template v-if="column.key === 'progress'">
                <span v-if="record.progress?.n_mined !== undefined">
                  mined={{ record.progress.n_mined }} failed={{ record.progress.n_failed }}
                </span>
                <span v-else-if="record.error" style="color: #ff4d4f">{{ record.error }}</span>
                <span v-else>-</span>
              </template>
              <template v-if="column.key === 'action'">
                <a-button type="link" size="small" @click="viewHistoryResults(record.run_id)">
                  查看
                </a-button>
              </template>
            </template>
          </a-table>
        </a-card>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, reactive } from 'vue'
import { message } from 'ant-design-vue'
import { mineLogicsApi } from '@/api/mine'
import type { MineLogicsEvent, MineLogicsStatusResponse, MineLogicsResultsResponse, MineLogicsHistoryResponse } from '@/api/mine'

// ==============================================================================
// Config
// ==============================================================================

const config = reactive({
  sourceLibs: ['alpha101', 'alpha158', 'alpha191'],
  maxPerLib: 10,
  workers: 4,
  wikiPath: 'wiki_auto',
  live: false,
  strict: false,
})

const sourceLibOptions = [
  { label: 'alpha101', value: 'alpha101' },
  { label: 'alpha158', value: 'alpha158' },
  { label: 'alpha191', value: 'alpha191' },
]

// ==============================================================================
// State
// ==============================================================================

const currentRunId = ref<string | null>(null)
const isRunning = ref(false)
const statusData = ref<MineLogicsStatusResponse | null>(null)
const results = ref<MineLogicsResultsResponse | null>(null)
const timelineEvents = ref<MineLogicsEvent[]>([])
const historyRuns = ref<MineLogicsHistoryResponse['runs']>([])
const historyLoading = ref(false)
const wsConnected = ref(false)

const progress = reactive({
  done: 0,
  total: 0,
  nMined: 0,
  nSkipped: 0,
  nFailed: 0,
})

let ws: WebSocket | null = null
let pollTimer: number | null = null

// ==============================================================================
// Computed
// ==============================================================================

const statusText = computed(() => {
  if (!statusData.value) return 'idle'
  return statusData.value.status
})

const statusColor = computed(() => {
  const s = statusData.value?.status
  if (s === 'running') return 'processing'
  if (s === 'completed') return 'success'
  if (s === 'failed') return 'error'
  if (s === 'stopped') return 'warning'
  return 'default'
})

const progressPercent = computed(() => {
  if (progress.total === 0) return 0
  return Math.round((progress.done / progress.total) * 100)
})

const progressStatus = computed(() => {
  if (!isRunning.value) return 'normal'
  return 'active'
})

const progressColor = computed(() => {
  if (progress.nFailed > 0) return '#ff4d4f'
  return '#1677ff'
})

// ==============================================================================
// Table columns
// ==============================================================================

const factorColumns = [
  { title: 'Rank', key: 'rank', width: 60, customRender: ({ index }: any) => index + 1 },
  { title: 'Formula ID', dataIndex: 'formula_id', key: 'formula_id', width: 150 },
  { title: 'Source', dataIndex: 'source_lib', key: 'source_lib', width: 80 },
  { title: 'Formula', key: 'formula', ellipsis: true },
  { title: 'IR', key: 'ir', width: 80, sorter: (a: any, b: any) => a.ir - b.ir },
  { title: 'IC Mean', dataIndex: 'ic_mean', key: 'ic_mean', width: 80,
    customRender: ({ text }: any) => text?.toFixed(4) ?? '-' },
  { title: 'Parse Layer', dataIndex: 'parse_layer', key: 'parse_layer', width: 100 },
  { title: 'Tags', key: 'tags', width: 200 },
]

const sourceColumns = [
  { title: 'Source Lib', dataIndex: 'source_lib', key: 'source_lib' },
  { title: 'Attempted', dataIndex: 'attempted', key: 'attempted' },
  { title: 'Mined', dataIndex: 'mined', key: 'mined' },
]

const agentColumns = [
  { title: 'Agent', dataIndex: 'agent', key: 'agent' },
  { title: 'Call Failures', dataIndex: 'callFailures', key: 'callFailures' },
  { title: 'Parse Failures', dataIndex: 'parseFailures', key: 'parseFailures' },
  { title: 'Structured Failures', dataIndex: 'structuredFailures', key: 'structuredFailures' },
]

const historyColumns = [
  { title: 'Run ID', dataIndex: 'run_id', key: 'run_id', width: 180 },
  { title: 'Status', key: 'status', width: 100 },
  { title: 'Created', key: 'created_at', width: 180 },
  { title: 'Duration', key: 'elapsed', width: 100 },
  { title: 'Progress', key: 'progress', width: 200 },
  { title: '操作', key: 'action', width: 80 },
]

// ==============================================================================
// Computed for tables
// ==============================================================================

const sourceData = computed(() => {
  if (!results.value?.result?.source_breakdown) return []
  return Object.entries(results.value.result.source_breakdown).map(([lib, counts]) => ({
    source_lib: lib,
    attempted: counts.attempted,
    mined: counts.mined,
  }))
})

const agentData = computed(() => {
  if (!results.value?.result?.agent_stats) return []
  return Object.entries(results.value.result.agent_stats).map(([agent, stats]) => ({
    agent,
    callFailures: stats.call_failures,
    parseFailures: stats.parse_failures,
    structuredFailures: stats.structured_failures,
  }))
})

// ==============================================================================
// Actions
// ==============================================================================

async function handleStart() {
  if (config.sourceLibs.length === 0) {
    message.warning('请至少选择一个来源库')
    return
  }
  try {
    const resp = await mineLogicsApi.start({
      source_libs: config.sourceLibs,
      max_per_lib: config.maxPerLib,
      workers: config.workers,
      wiki_path: config.wikiPath,
      live: config.live,
      strict: config.strict,
    })
    currentRunId.value = resp.run_id
    isRunning.value = true
    statusData.value = null
    results.value = null
    timelineEvents.value = []
    progress.done = 0
    progress.total = 0
    progress.nMined = 0
    progress.nSkipped = 0
    progress.nFailed = 0

    // 连接 WebSocket
    connectWs(resp.run_id)

    // 开始轮询 status (作为 WS 的 fallback)
    startPolling(resp.run_id)

    message.success(`挖掘已启动: ${resp.run_id}`)
  } catch (err: any) {
    message.error(`启动失败: ${err.message}`)
  }
}

async function handleStop() {
  if (!currentRunId.value) return
  try {
    await mineLogicsApi.stop(currentRunId.value)
    message.info('已发送停止信号')
  } catch (err: any) {
    message.error(`停止失败: ${err.message}`)
  }
}

// ==============================================================================
// WebSocket
// ==============================================================================

function connectWs(runId: string) {
  disconnectWs()
  const host = window.location.host
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${protocol}//${host}/api/mine-logics/stream/${runId}`

  ws = new WebSocket(url)
  wsConnected.value = true

  ws.onmessage = (evt) => {
    try {
      const event: MineLogicsEvent = JSON.parse(evt.data)
      handleWsEvent(event)
    } catch {
      // ignore parse errors
    }
  }

  ws.onerror = () => {
    wsConnected.value = false
  }

  ws.onclose = () => {
    wsConnected.value = false
  }
}

function disconnectWs() {
  if (ws) {
    ws.close()
    ws = null
  }
  wsConnected.value = false
}

function handleWsEvent(event: MineLogicsEvent) {
  // 忽略 heartbeat
  if (event.type === 'heartbeat') return

  // 添加到时间线
  timelineEvents.value.push(event)

  switch (event.type) {
    case 'mining_started':
      progress.total = 0
      progress.done = 0
      break

    case 'formula_attempted':
      progress.done = event.done || 0
      progress.total = event.total || 0
      break

    case 'batch_completed':
      progress.nMined = event.n_mined || 0
      progress.nSkipped = event.n_skipped || 0
      progress.nFailed = event.n_failed || 0
      isRunning.value = false
      // 加载结果
      if (currentRunId.value) {
        loadResults(currentRunId.value)
      }
      break

    case 'error':
      isRunning.value = false
      message.error(event.message || '挖掘出错')
      break

    case 'done':
      isRunning.value = false
      stopPolling()
      loadHistory()
      break
  }
}

// ==============================================================================
// Polling (fallback)
// ==============================================================================

function startPolling(runId: string) {
  stopPolling()
  pollTimer = window.setInterval(async () => {
    try {
      const status = await mineLogicsApi.status(runId)
      statusData.value = status
      if (status.status !== 'running' && status.status !== 'pending') {
        isRunning.value = false
        stopPolling()
        if (status.status === 'completed') {
          loadResults(runId)
        }
      }
    } catch {
      // ignore
    }
  }, 3000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

// ==============================================================================
// Data loading
// ==============================================================================

async function loadResults(runId: string) {
  try {
    results.value = await mineLogicsApi.results(runId)
  } catch (err: any) {
    console.error('Failed to load results:', err)
  }
}

async function loadHistory() {
  historyLoading.value = true
  try {
    const resp = await mineLogicsApi.history()
    historyRuns.value = resp.runs
  } catch (err: any) {
    console.error('Failed to load history:', err)
  } finally {
    historyLoading.value = false
  }
}

async function viewHistoryResults(runId: string) {
  currentRunId.value = runId
  await loadResults(runId)
  message.info(`已加载 ${runId} 的结果`)
}

// ==============================================================================
// Formatting helpers
// ==============================================================================

function formatTs(ts?: number): string {
  if (!ts) return ''
  return new Date(ts * 1000).toLocaleTimeString()
}

function formatDate(ts?: number): string {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString()
}

function formatEvent(event: MineLogicsEvent): string {
  switch (event.type) {
    case 'mining_started':
      return `开始挖掘 (${event.source_libs?.join(', ')})`
    case 'formula_attempted':
      return `挖掘: ${event.formula_id}`
    case 'formula_completed':
      return `完成: ${event.formula_id} (${event.success ? '成功' : '失败'})`
    case 'batch_completed':
      return `批量完成: mined=${event.n_mined} skipped=${event.n_skipped} failed=${event.n_failed}`
    case 'error':
      return `错误: ${event.message}`
    case 'done':
      return '会话结束'
    default:
      return JSON.stringify(event)
  }
}

function timelineItemColor(type: string): string {
  switch (type) {
    case 'mining_started': return 'blue'
    case 'formula_attempted': return 'gray'
    case 'formula_completed': return 'green'
    case 'batch_completed': return 'green'
    case 'error': return 'red'
    case 'done': return 'green'
    default: return 'gray'
  }
}

function historyStatusColor(status: string): string {
  switch (status) {
    case 'running': return 'processing'
    case 'completed': return 'success'
    case 'failed': return 'error'
    case 'stopped': return 'warning'
    default: return 'default'
  }
}

// ==============================================================================
// Lifecycle
// ==============================================================================

onMounted(() => {
  loadHistory()
})

onUnmounted(() => {
  disconnectWs()
  stopPolling()
})
</script>

<style scoped>
.mine-logics {
  padding: 0;
}
</style>
