<template>
  <a-layout class="app-layout">
    <AppHeader @togglePanel="appStore.toggleContextPanel()" />
    <a-layout>
      <AppSidebar />
      <a-layout-content class="content">
        <router-view />
      </a-layout-content>
    </a-layout>
  </a-layout>
</template>

<script setup lang="ts">
import AppHeader from './AppHeader.vue'
import AppSidebar from './AppSidebar.vue'
import { useAppStore } from '@/stores/app'

const appStore = useAppStore()
</script>

<style scoped>
.app-layout {
  height: 100vh;
}

.content {
  flex: 1;
  min-width: 0;
  margin: 0;
  padding: 0;
  overflow-y: auto;
  background: transparent;
}
</style>