<template>
  <a-layout-sider class="sidebar" :collapsed="false">
    <a-menu
      :selectedKeys="selectedKeys"
      v-model:openKeys="openKeys"
      theme="dark"
      mode="inline"
      @click="handleMenuClick"
    >
      <a-menu-item key="/">
        <home-outlined />
        <span>Dashboard</span>
      </a-menu-item>
      <a-menu-item key="/agent-chat" v-if="agentEnabled">
        <message-outlined />
        <span>Agent Chat</span>
      </a-menu-item>
      <a-sub-menu key="wiki">
        <template #icon>
          <book-outlined />
        </template>
        <template #title>Wiki</template>
        <a-menu-item key="/wiki/factors">Factors</a-menu-item>
        <a-menu-item key="/wiki/strategies">Strategies</a-menu-item>
      </a-sub-menu>
      <a-menu-item key="/strategy/editor">
        <code-outlined />
        <span>Strategy Editor</span>
      </a-menu-item>
      <a-menu-item key="/backtest">
        <line-chart-outlined />
        <span>Backtest Center</span>
      </a-menu-item>
      <a-menu-item key="/factor-analysis">
        <bar-chart-outlined />
        <span>Factor Analysis</span>
      </a-menu-item>
      <a-menu-item key="/dream">
        <bulb-outlined />
        <span>Dream Insights</span>
      </a-menu-item>
      <a-menu-item key="/settings">
        <setting-outlined />
        <span>Settings</span>
      </a-menu-item>
    </a-menu>
  </a-layout-sider>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
  HomeOutlined,
  MessageOutlined,
  BookOutlined,
  LineChartOutlined,
  BarChartOutlined,
  BulbOutlined,
  SettingOutlined,
  CodeOutlined,
} from '@ant-design/icons-vue'

const router = useRouter()
const route = useRoute()

// v3.0.0 Stage 5.3: 隐藏 Agent Chat 入口（仅在 VITE_AGENT_ENABLED=true 时显示）。
// 这是个 build-time flag：让纯量化库部署（未装 [agent] extra）默认不显示该入口，
// 同时保留 agent 安装的部署里 sidebar 多一项。
const agentEnabled = import.meta.env.VITE_AGENT_ENABLED !== 'false'

const openKeys = ref<string[]>([])

const selectedKeys = computed(() => [route.path])

watch(
  () => route.path,
  (path) => {
    if (path.startsWith('/wiki') && !openKeys.value.includes('wiki')) {
      openKeys.value = ['wiki']
    }
  },
  { immediate: true }
)

const handleMenuClick = ({ key }: { key: string }) => {
  router.push(key)
}
</script>

<style scoped>
.sidebar {
  background: #001529;
}
</style>
