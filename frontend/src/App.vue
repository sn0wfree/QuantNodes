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

/* Chat layout CSS variables - light mode (default) */
:root {
  --chat-sidebar-width: 200px;
  --chat-sidebar-collapsed: 48px;
  --chat-topbar-height: 40px;
  --chat-header-height: 36px;
  --chat-statusbar-height: 28px;
  --chat-keybinds-height: 28px;
  --chat-border-color: #f0f0f0;
  --chat-bg-primary: #ffffff;
  --chat-bg-secondary: #fafafa;
  --chat-text-primary: #333333;
  --chat-text-secondary: #666666;
  --chat-text-muted: #999999;
}

/* Chat layout CSS variables - dark mode */
body.dark {
  --chat-border-color: #303030;
  --chat-bg-primary: #1a1a1a;
  --chat-bg-secondary: #141414;
  --chat-text-primary: #e8e8e8;
  --chat-text-secondary: #aaaaaa;
  --chat-text-muted: #666666;
}
</style>
