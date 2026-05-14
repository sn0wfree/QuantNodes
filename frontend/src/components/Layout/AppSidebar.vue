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
      <a-menu-item key="/chat">
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
