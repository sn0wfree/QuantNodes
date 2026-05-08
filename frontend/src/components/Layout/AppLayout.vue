<template>
  <a-layout class="app-layout">
    <AppHeader />
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
</script>

<style scoped>
.app-layout {
  height: 100vh;
}

.content {
  margin: 16px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  overflow-y: auto;
}

:global(body.dark) .content {
  background: #1f1f1f;
}
</style>
