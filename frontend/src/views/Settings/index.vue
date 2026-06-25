<template>
  <div class="settings">
    <a-page-header title="Settings" sub-title="Configure your QuantNodes workspace">
      <template #extra>
        <a-space>
          <a-button @click="handleExport">
            <template #icon><DownloadOutlined /></template>
            Export
          </a-button>
          <a-button @click="showImportModal = true">
            <template #icon><UploadOutlined /></template>
            Import
          </a-button>
          <a-popconfirm title="Reset all settings to defaults?" @confirm="handleReset">
            <a-button danger>
              <template #icon><ReloadOutlined /></template>
              Reset
            </a-button>
          </a-popconfirm>
        </a-space>
      </template>
    </a-page-header>

    <a-tabs v-model:activeKey="activeTab">
      <a-tab-pane key="appearance" tab="Appearance">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="Theme">
              <a-radio-group v-model:value="localSettings.appearance.theme" @change="handleThemeChange">
                <a-radio-button value="light">
                  Light
                </a-radio-button>
                <a-radio-button value="dark">
                  Dark
                </a-radio-button>
                <a-radio-button value="system">
                  <template #icon><DesktopOutlined /></template>
                  System
                </a-radio-button>
              </a-radio-group>
            </a-form-item>
            <a-form-item label="Language">
              <a-select v-model:value="localSettings.appearance.language" style="width: 200px">
                <a-select-option value="en">English</a-select-option>
                <a-select-option value="zh">Chinese (中文)</a-select-option>
              </a-select>
            </a-form-item>
            <a-form-item label="Compact Mode">
              <a-switch v-model:checked="localSettings.appearance.compact_mode" />
            </a-form-item>
            <a-form-item label="Sidebar Collapsed by Default">
              <a-switch v-model:checked="localSettings.appearance.sidebar_collapsed" />
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="api" tab="API Connection">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="API Base URL">
              <a-input v-model:value="localSettings.api.base_url" placeholder="http://localhost:19380" />
            </a-form-item>
            <a-form-item label="WebSocket URL">
              <a-input v-model:value="localSettings.api.ws_url" placeholder="ws://localhost:19380" />
            </a-form-item>
            <a-form-item label="Request Timeout (ms)">
              <a-input-number v-model:value="localSettings.api.timeout" :min="5000" :max="120000" :step="1000" />
            </a-form-item>
            <a-form-item>
              <a-button type="primary" @click="testConnection">
                <template #icon><ApiOutlined /></template>
                Test Connection
              </a-button>
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="agent" tab="Agent">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="LLM Provider">
              <a-select v-model:value="localSettings.agent.provider" style="width: 200px">
                <a-select-option value="openai">OpenAI</a-select-option>
                <a-select-option value="anthropic">Anthropic</a-select-option>
                <a-select-option value="azure">Azure OpenAI</a-select-option>
                <a-select-option value="local">Local (Ollama)</a-select-option>
                <a-select-option value="custom">Custom (OpenAI-Compatible)</a-select-option>
              </a-select>
            </a-form-item>
            <a-form-item label="Default Mode">
              <a-radio-group v-model:value="localSettings.agent.default_mode">
                <a-radio-button value="build">Build</a-radio-button>
                <a-radio-button value="plan">Plan</a-radio-button>
              </a-radio-group>
              <div class="field-help">Which mode is active by default when opening chat</div>
            </a-form-item>
            <a-divider>Build Mode</a-divider>
            <a-form-item label="Build Model">
              <a-input v-model:value="localSettings.agent.mode_models.build.model" placeholder="e.g. deepseek-chat, gpt-4o" />
              <div class="field-help">Model used for code writing and implementation tasks</div>
            </a-form-item>
            <a-form-item label="Build Max Tokens">
              <a-input-number v-model:value="localSettings.agent.mode_models.build.max_tokens" :min="1024" :max="1000000" :step="1024" style="width: 200px" />
            </a-form-item>
            <a-divider>Plan Mode</a-divider>
            <a-form-item label="Plan Model">
              <a-input v-model:value="localSettings.agent.mode_models.plan.model" placeholder="e.g. deepseek-reasoner, claude-3.5-sonnet" />
              <div class="field-help">Model used for analysis, reasoning, and planning tasks</div>
            </a-form-item>
            <a-form-item label="Plan Max Tokens">
              <a-input-number v-model:value="localSettings.agent.mode_models.plan.max_tokens" :min="1024" :max="1000000" :step="1024" style="width: 200px" />
            </a-form-item>
            <a-divider>Fallback (Legacy)</a-divider>
            <a-form-item label="Fallback Model">
              <a-input v-model:value="localSettings.agent.model" placeholder="gpt-4" />
              <div class="field-help">Used when mode_models is not configured</div>
            </a-form-item>
            <a-form-item label="API Key">
              <a-input-password v-model:value="localSettings.agent.api_key" placeholder="sk-..." />
              <div class="field-help">Your API key is stored locally and never sent to our servers.</div>
            </a-form-item>
            <a-form-item label="API Base URL (optional)">
              <a-input v-model:value="localSettings.agent.api_base" placeholder="https://api.openai.com/v1" />
            </a-form-item>
            <a-form-item label="Max Iterations">
              <a-slider v-model:value="localSettings.agent.max_iterations" :min="1" :max="20" :marks="{ 1: '1', 5: '5', 10: '10', 20: '20' }" />
            </a-form-item>
            <a-form-item label="Temperature">
              <a-slider v-model:value="localSettings.agent.temperature" :min="0" :max="2" :step="0.1" :marks="{ 0: '0', 0.7: '0.7', 1: '1', 2: '2' }" />
            </a-form-item>
            <a-form-item label="LLM Request Timeout (seconds)">
              <a-input-number v-model:value="localSettings.agent.llm_timeout" :min="5" :max="300" :step="5" />
            </a-form-item>
            <a-form-item label="Max Retries">
              <a-input-number v-model:value="localSettings.agent.llm_max_retries" :min="0" :max="5" />
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="providers" tab="Providers">
        <a-card>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <span style="font-size: 14px; color: #666;">
              Configure multiple LLM providers for model routing and fallback.
            </span>
            <a-button type="primary" @click="showAddProvider">
              <template #icon><PlusOutlined /></template>
              Add Provider
            </a-button>
          </div>

          <div v-if="Object.keys(providers).length === 0" style="text-align: center; padding: 40px 0; color: #999;">
            No providers configured. Add a provider to enable multi-model routing.
          </div>

          <div v-for="(config, name) in providers" :key="name" class="provider-card">
            <div class="provider-header">
              <div class="provider-title">
                <span class="provider-name">{{ name }}</span>
                <a-tag v-if="name === localSettings.agent.provider" color="blue">Default</a-tag>
                <a-tag v-if="config.priority === 1" color="green">Priority {{ config.priority }}</a-tag>
                <a-tag v-else>Priority {{ config.priority || 1 }}</a-tag>
              </div>
              <a-space>
                <a-button size="small" @click="testProvider(name)" :loading="testingProvider === name">
                  Test
                </a-button>
                <a-button size="small" @click="editProvider(name, config)">Edit</a-button>
                <a-popconfirm title="Delete this provider?" @confirm="deleteProvider(name)">
                  <a-button size="small" danger>Delete</a-button>
                </a-popconfirm>
              </a-space>
            </div>
            <div class="provider-details">
              <div><span class="detail-label">Base URL:</span> {{ config.api_base || '(not set)' }}</div>
              <div>
                <span class="detail-label">API Key:</span>
                {{ config.api_key ? maskKey(config.api_key) : '(not set)' }}
              </div>
              <div>
                <span class="detail-label">Models:</span>
                {{ (config.models || []).join(', ') || '(not configured)' }}
              </div>
              <div v-if="config.extra_headers && Object.keys(config.extra_headers).length">
                <span class="detail-label">Extra Headers:</span>
                {{ JSON.stringify(config.extra_headers) }}
              </div>
            </div>
            <div v-if="providerTestResults[name]" class="provider-test-result" :class="providerTestResults[name].ok ? 'test-ok' : 'test-fail'">
              {{ providerTestResults[name].ok ? `Connected (${providerTestResults[name].model_count} models)` : providerTestResults[name].error }}
            </div>
          </div>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="editor" tab="Editor">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="Font Size">
              <a-input-number v-model:value="localSettings.editor.font_size" :min="10" :max="24" />
            </a-form-item>
            <a-form-item label="Tab Size">
              <a-radio-group v-model:value="localSettings.editor.tab_size">
                <a-radio-button :value="2">2</a-radio-button>
                <a-radio-button :value="4">4</a-radio-button>
              </a-radio-group>
            </a-form-item>
            <a-form-item label="Word Wrap">
              <a-switch v-model:checked="localSettings.editor.word_wrap" />
            </a-form-item>
            <a-form-item label="Minimap">
              <a-switch v-model:checked="localSettings.editor.minimap" />
            </a-form-item>
            <a-form-item label="Auto Save">
              <a-switch v-model:checked="localSettings.editor.auto_save" />
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="backtest" tab="Backtest">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="Default Initial Cash">
              <a-input-number v-model:value="localSettings.backtest.default_initial_cash" :min="10000" :step="10000" style="width: 200px" />
            </a-form-item>
            <a-form-item label="Default Commission Rate">
              <a-input-number v-model:value="localSettings.backtest.default_commission" :min="0" :max="0.01" :step="0.0001" :precision="4" style="width: 200px" />
            </a-form-item>
            <a-form-item label="Auto Save Results">
              <a-switch v-model:checked="localSettings.backtest.auto_save_results" />
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>

      <a-tab-pane key="notifications" tab="Notifications">
        <a-card>
          <a-form layout="vertical">
            <a-form-item label="Enable Notifications">
              <a-switch v-model:checked="localSettings.notifications.enabled" />
            </a-form-item>
            <a-form-item label="Sound">
              <a-switch v-model:checked="localSettings.notifications.sound" :disabled="!localSettings.notifications.enabled" />
            </a-form-item>
            <a-form-item label="Desktop Notifications">
              <a-switch v-model:checked="localSettings.notifications.desktop" :disabled="!localSettings.notifications.enabled" />
            </a-form-item>
          </a-form>
        </a-card>
      </a-tab-pane>
    </a-tabs>

    <div class="settings-actions">
      <a-button type="primary" size="large" @click="handleSave" :loading="saving">
        <template #icon><SaveOutlined /></template>
        Save Settings
      </a-button>
    </div>

    <a-modal v-model:open="showImportModal" title="Import Settings" @ok="handleImport">
      <a-textarea v-model:value="importJson" :rows="10" placeholder="Paste settings JSON here..." />
    </a-modal>

    <!-- Add/Edit Provider Modal -->
    <a-modal
      v-model:open="showProviderModal"
      :title="editingProviderName ? 'Edit Provider' : 'Add Provider'"
      @ok="handleSaveProvider"
      :width="560"
    >
      <a-form layout="vertical">
        <a-form-item label="Provider Name">
          <a-input
            v-model:value="providerForm.name"
            :disabled="!!editingProviderName"
            placeholder="e.g. deepseek, dashscope, openrouter"
          />
          <div class="field-help">Unique identifier for this provider</div>
        </a-form-item>
        <a-form-item label="Base URL">
          <a-input v-model:value="providerForm.api_base" placeholder="https://api.deepseek.com/v1" />
        </a-form-item>
        <a-form-item label="API Key">
          <a-input-password v-model:value="providerForm.api_key" placeholder="sk-..." />
        </a-form-item>
        <a-form-item label="Models (comma-separated)">
          <a-textarea v-model:value="providerForm.modelsStr" :rows="2" placeholder="deepseek-chat, deepseek-reasoner" />
          <div class="field-help">List of model IDs available from this provider</div>
        </a-form-item>
        <a-form-item label="Priority (lower = preferred)">
          <a-input-number v-model:value="providerForm.priority" :min="1" :max="10" />
        </a-form-item>
        <a-form-item label="Extra Headers (JSON, optional)">
          <a-input v-model:value="providerForm.extraHeadersStr" placeholder='{"X-OpenRouter-Title": "QuantNodes"}' />
        </a-form-item>
        <a-form-item label="Preset">
          <a-select v-model:value="providerPreset" placeholder="Select a preset..." @change="applyPreset" allow-clear>
            <a-select-option value="deepseek">DeepSeek</a-select-option>
            <a-select-option value="dashscope">DashScope (阿里百炼)</a-select-option>
            <a-select-option value="siliconflow">SiliconFlow (硅基流动)</a-select-option>
            <a-select-option value="openrouter">OpenRouter</a-select-option>
            <a-select-option value="zhipu">智谱 GLM</a-select-option>
            <a-select-option value="moonshot">月之暗面 (Kimi)</a-select-option>
            <a-select-option value="ollama">Ollama (Local)</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, watch } from 'vue'
import { message } from 'ant-design-vue'
import {
  DownloadOutlined,
  UploadOutlined,
  ReloadOutlined,
  SaveOutlined,
  DesktopOutlined,
  ApiOutlined,
  PlusOutlined,
} from '@ant-design/icons-vue'
import { useAppStore } from '@/stores/app'
import { get, put, post, del } from '@/api'

const appStore = useAppStore()

const activeTab = ref('appearance')
const saving = ref(false)
const showImportModal = ref(false)
const importJson = ref('')

// Provider management
const providers = ref<Record<string, any>>({})
const showProviderModal = ref(false)
const editingProviderName = ref<string | null>(null)
const providerPreset = ref<string | null>(null)
const testingProvider = ref<string | null>(null)
const providerTestResults = ref<Record<string, { ok: boolean; error?: string; model_count?: number }>>({})

const providerForm = reactive({
  name: '',
  api_base: '',
  api_key: '',
  modelsStr: '',
  priority: 1,
  extraHeadersStr: '',
})

const PROVIDER_PRESETS: Record<string, { api_base: string; models: string; priority: number; extraHeaders?: string }> = {
  deepseek: { api_base: 'https://api.deepseek.com/v1', models: 'deepseek-chat, deepseek-reasoner', priority: 1 },
  dashscope: { api_base: 'https://dashscope.aliyuncs.com/compatible-mode/v1', models: 'deepseek-v4-pro, qwen3.6-plus, qwen3.6-flash', priority: 1 },
  siliconflow: { api_base: 'https://api.siliconflow.cn/v1', models: 'Pro/deepseek-ai/DeepSeek-R1, Qwen/Qwen2.5-72B-Instruct', priority: 2 },
  openrouter: { api_base: 'https://openrouter.ai/api/v1', models: '', priority: 3, extraHeaders: '{"X-OpenRouter-Title": "QuantNodes"}' },
  zhipu: { api_base: 'https://open.bigmodel.cn/api/paas/v4/', models: 'glm-4, glm-4-plus', priority: 1 },
  moonshot: { api_base: 'https://api.moonshot.cn/v1', models: 'moonshot-v1-128k, kimi-k2.5', priority: 1 },
  ollama: { api_base: 'http://localhost:11434/v1', models: '', priority: 1 },
}

const applyPreset = () => {
  if (!providerPreset.value) return
  const p = PROVIDER_PRESETS[providerPreset.value]
  if (!p) return
  providerForm.name = providerPreset.value
  providerForm.api_base = p.api_base
  providerForm.modelsStr = p.models
  providerForm.priority = p.priority
  providerForm.extraHeadersStr = p.extraHeaders || ''
}

const showAddProvider = () => {
  editingProviderName.value = null
  providerPreset.value = null
  providerForm.name = ''
  providerForm.api_base = ''
  providerForm.api_key = ''
  providerForm.modelsStr = ''
  providerForm.priority = 1
  providerForm.extraHeadersStr = ''
  showProviderModal.value = true
}

const editProvider = (name: string, config: any) => {
  editingProviderName.value = name
  providerPreset.value = null
  providerForm.name = name
  providerForm.api_base = config.api_base || ''
  providerForm.api_key = config.api_key || ''
  providerForm.modelsStr = (config.models || []).join(', ')
  providerForm.priority = config.priority || 1
  providerForm.extraHeadersStr = config.extra_headers ? JSON.stringify(config.extra_headers) : ''
  showProviderModal.value = true
}

const handleSaveProvider = async () => {
  const name = providerForm.name.trim()
  if (!name) {
    message.error('Provider name is required')
    return
  }
  const models = providerForm.modelsStr.split(',').map(s => s.trim()).filter(Boolean)
  let extraHeaders = {}
  if (providerForm.extraHeadersStr.trim()) {
    try {
      extraHeaders = JSON.parse(providerForm.extraHeadersStr)
    } catch {
      message.error('Invalid JSON in extra headers')
      return
    }
  }
  const config = {
    api_base: providerForm.api_base,
    api_key: providerForm.api_key,
    models,
    priority: providerForm.priority,
    extra_headers: extraHeaders,
  }
  try {
    if (editingProviderName.value) {
      await put(`/settings/providers/${editingProviderName.value}`, { config })
    } else {
      await post('/settings/providers', { name, config })
    }
    message.success(editingProviderName.value ? 'Provider updated' : 'Provider added')
    showProviderModal.value = false
    loadProviders()
  } catch {
    message.error('Failed to save provider')
  }
}

const deleteProvider = async (name: string) => {
  try {
    await del(`/settings/providers/${name}`)
    message.success('Provider deleted')
    loadProviders()
  } catch {
    message.error('Failed to delete provider')
  }
}

const testProvider = async (name: string) => {
  testingProvider.value = name
  try {
    const result = await post<any>(`/settings/providers/${name}/test`)
    providerTestResults.value[name] = result
  } catch {
    providerTestResults.value[name] = { ok: false, error: 'Request failed' }
  } finally {
    testingProvider.value = null
  }
}

const maskKey = (key: string) => {
  if (!key || key.length <= 8) return '****'
  return key.slice(0, 4) + '*'.repeat(key.length - 8) + key.slice(-4)
}

const loadProviders = async () => {
  try {
    const data = await get<Record<string, any>>('/settings/providers')
    providers.value = data || {}
  } catch {
    providers.value = {}
  }
}

const localSettings = reactive({
  appearance: {
    theme: 'light' as string,
    language: 'en',
    sidebar_collapsed: false,
    compact_mode: false,
  },
  api: {
    base_url: 'http://localhost:19380',
    ws_url: 'ws://localhost:19380',
    timeout: 30000,
  },
  agent: {
    provider: 'openai',
    model: 'gpt-4',
    api_key: '',
    api_base: '',
    max_iterations: 5,
    temperature: 0.7,
    max_tokens: 102400,
    llm_timeout: 60,
    llm_max_retries: 3,
    default_mode: 'build',
    mode_models: {
      build: { model: '', max_tokens: 102400 },
      plan: { model: '', max_tokens: 16000 },
    },
  },
  editor: {
    font_size: 14,
    tab_size: 2,
    word_wrap: true,
    minimap: true,
    auto_save: true,
  },
  backtest: {
    default_initial_cash: 100000,
    default_commission: 0.001,
    auto_save_results: true,
  },
  notifications: {
    enabled: true,
    sound: true,
    desktop: false,
  },
})

const handleThemeChange = () => {
  const theme = localSettings.appearance.theme
  if (theme === 'system') {
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    appStore.setTheme(prefersDark ? 'dark' : 'light')
  } else {
    appStore.setTheme(theme as 'light' | 'dark')
  }
}

const testConnection = async () => {
  try {
    await get('/health')
    message.success('Connection successful')
  } catch {
    message.error('Connection failed')
  }
}

const handleSave = async () => {
  saving.value = true
  try {
    await put('/settings', { settings: localSettings })
    message.success('Settings saved')
  } catch {
    message.error('Failed to save settings')
  } finally {
    saving.value = false
  }
}

const handleExport = async () => {
  try {
    const data = await get<string>('/settings/export')
    const blob = new Blob([data], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'quantnodes-settings.json'
    a.click()
    URL.revokeObjectURL(url)
    message.success('Settings exported')
  } catch {
    message.error('Failed to export settings')
  }
}

const handleImport = async () => {
  try {
    await post('/settings/import', { json_data: importJson.value })
    message.success('Settings imported')
    showImportModal.value = false
    loadSettings()
  } catch {
    message.error('Failed to import settings')
  }
}

const handleReset = async () => {
  try {
    const settings = await post<any>('/settings/reset')
    Object.assign(localSettings, settings)
    message.success('Settings reset to defaults')
  } catch {
    message.error('Failed to reset settings')
  }
}

const loadSettings = async () => {
  try {
    const settings = await get<any>('/settings')
    if (settings) {
      Object.assign(localSettings, settings)
      appStore.setTheme(localSettings.appearance.theme as 'light' | 'dark')
    }
  } catch {
    console.log('Using default settings')
  }
}

watch(() => localSettings.appearance.theme, handleThemeChange)

onMounted(() => {
  loadSettings()
  loadProviders()
})
</script>

<style scoped>
.settings {
  padding: 0;
  max-width: 900px;
}

.settings-actions {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.field-help {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

:deep(.ant-tabs-tab) {
  font-size: 14px;
}

.provider-card {
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
  margin-bottom: 12px;
  background: #fafafa;
}

.provider-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.provider-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.provider-name {
  font-weight: 600;
  font-size: 15px;
}

.provider-details {
  font-size: 13px;
  color: #666;
  line-height: 1.8;
}

.detail-label {
  font-weight: 500;
  color: #333;
}

.provider-test-result {
  margin-top: 8px;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
}

.test-ok {
  background: #f6ffed;
  color: #52c41a;
  border: 1px solid #b7eb8f;
}

.test-fail {
  background: #fff2f0;
  color: #ff4d4f;
  border: 1px solid #ffccc7;
}
</style>
