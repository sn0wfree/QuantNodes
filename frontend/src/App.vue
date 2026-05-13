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
  /* Layout */
  --chat-sidebar-width: 200px;
  --chat-sidebar-collapsed: 48px;
  --chat-topbar-height: 40px;
  --chat-header-height: 36px;
  --chat-footer-height: 28px;
  --chat-statusbar-height: 24px;
  --chat-keybinds-height: 24px;

  /* Surface colors */
  --chat-bg-primary: #ffffff;
  --chat-bg-secondary: #fafafa;
  --chat-bg-tertiary: #f0f0f0;
  --chat-bg-hover: #f5f5f5;
  --chat-bg-active: #e8e8e8;

  /* Text colors */
  --chat-text-primary: #333333;
  --chat-text-secondary: #666666;
  --chat-text-muted: #999999;
  --chat-text-inverse: #ffffff;

  /* Border colors */
  --chat-border-color: #f0f0f0;
  --chat-border-strong: #d9d9d9;
  --chat-border-active: #1677ff;

  /* Agent colors */
  --chat-build-color: #1677ff;
  --chat-build-bg: #e6f4ff;
  --chat-plan-color: #52c41a;
  --chat-plan-bg: #f6ffed;

  /* Status colors */
  --chat-error: #ff4d4f;
  --chat-warning: #faad14;
  --chat-success: #52c41a;
  --chat-info: #1677ff;

  /* Shadows */
  --chat-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.06);
  --chat-shadow-md: 0 2px 8px rgba(0, 0, 0, 0.08);
}

/* Chat layout CSS variables - dark mode */
body.dark {
  /* Surface colors */
  --chat-bg-primary: #1a1a1a;
  --chat-bg-secondary: #141414;
  --chat-bg-tertiary: #1f1f1f;
  --chat-bg-hover: #262626;
  --chat-bg-active: #303030;

  /* Text colors */
  --chat-text-primary: #e8e8e8;
  --chat-text-secondary: #aaaaaa;
  --chat-text-muted: #666666;
  --chat-text-inverse: #141414;

  /* Border colors */
  --chat-border-color: #303030;
  --chat-border-strong: #434343;
  --chat-border-active: #1677ff;

  /* Agent colors */
  --chat-build-color: #4096ff;
  --chat-build-bg: #111d2c;
  --chat-plan-color: #73d13d;
  --chat-plan-bg: #162312;

  /* Status colors */
  --chat-error: #ff7875;
  --chat-warning: #ffd666;
  --chat-success: #73d13d;
  --chat-info: #4096ff;

  /* Shadows */
  --chat-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
  --chat-shadow-md: 0 2px 8px rgba(0, 0, 0, 0.3);
}
</style>
