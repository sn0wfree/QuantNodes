<template>
  <a-layout-header class="header">
    <div class="logo">
      <experiment-outlined />
      <span>QuantNodes</span>
    </div>
    <a-menu
      theme="dark"
      mode="horizontal"
      :selected-keys="selectedKeys"
      @click="handleMenuClick"
    >
      <a-menu-item key="chat">
        <message-outlined />
        <span>Agent Chat</span>
      </a-menu-item>
    </a-menu>
  </a-layout-header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ExperimentOutlined, MessageOutlined } from '@ant-design/icons-vue'

const router = useRouter()
const route = useRoute()

const selectedKeys = computed(() => {
  if (route.path.startsWith('/chat')) return ['chat']
  return []
})

const handleMenuClick = ({ key }: { key: string }) => {
  router.push(`/${key}`)
}
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  padding: 0 24px;
  background: #001529;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #fff;
  font-size: 18px;
  font-weight: 600;
  margin-right: 48px;
}

.logo :deep(span) {
  color: #fff;
}
</style>
