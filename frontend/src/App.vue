<template>
  <a-config-provider :theme="themeConfig">
    <router-view />
  </a-config-provider>
</template>

<script setup lang="ts">
import { computed, watch, onMounted } from 'vue'
import { theme } from 'ant-design-vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()

const themeConfig = computed(() => ({
  token: {
    colorPrimary: '#1677ff',
    borderRadius: 6,
  },
  algorithm: appStore.isDarkMode ? theme.darkAlgorithm : undefined,
}))

watch(
  () => appStore.theme,
  (newTheme) => {
    document.documentElement.setAttribute('data-theme', newTheme)
    if (newTheme === 'dark') {
      document.body.classList.add('dark')
    } else {
      document.body.classList.remove('dark')
    }
  },
  { immediate: true }
)

onMounted(() => {
  const savedTheme = localStorage.getItem('quantnodes-theme') as 'light' | 'dark' | null
  if (savedTheme) {
    appStore.setTheme(savedTheme)
  }
})
</script>

<style>
#app {
  height: 100vh;
}

/* Dark mode styles */
body.dark {
  background-color: #141414;
  color: rgba(255, 255, 255, 0.85);
}

body.dark .ant-layout {
  background: #141414;
}

body.dark .ant-layout-header {
  background: #1f1f1f;
}

body.dark .ant-layout-sider {
  background: #1f1f1f;
}

body.dark .ant-card {
  background: #1f1f1f;
  border-color: #303030;
}

body.dark .ant-table {
  background: #1f1f1f;
}

body.dark .ant-modal-content {
  background: #1f1f1f;
}
</style>
