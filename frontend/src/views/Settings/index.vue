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
              <a-input v-model:value="localSettings.api.base_url" placeholder="http://localhost:8000" />
            </a-form-item>
            <a-form-item label="WebSocket URL">
              <a-input v-model:value="localSettings.api.ws_url" placeholder="ws://localhost:8000" />
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
            <a-form-item label="Model">
              <a-input v-model:value="localSettings.agent.model" placeholder="gpt-4" />
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
} from '@ant-design/icons-vue'
import { useAppStore } from '@/stores/app'
import { get, put, post } from '@/api'

const appStore = useAppStore()

const activeTab = ref('appearance')
const saving = ref(false)
const showImportModal = ref(false)
const importJson = ref('')

const localSettings = reactive({
  appearance: {
    theme: 'light' as string,
    language: 'en',
    sidebar_collapsed: false,
    compact_mode: false,
  },
  api: {
    base_url: 'http://localhost:8000',
    ws_url: 'ws://localhost:8000',
    timeout: 30000,
  },
  agent: {
    provider: 'openai',
    model: 'gpt-4',
    api_key: '',
    api_base: '',
    max_iterations: 5,
    temperature: 0.7,
    llm_timeout: 60,
    llm_max_retries: 3,
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
</style>
