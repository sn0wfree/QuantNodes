<template>
  <div class="metric-card">
    <a-statistic :title="title" :value="value" :suffix="suffix" :value-style="{ color }">
      <template #prefix>
        <component :is="icon" v-if="icon" />
      </template>
    </a-statistic>
  </div>
</template>

<script setup lang="ts">
import type { Component } from 'vue'

defineProps<{
  title: string
  value: number
  suffix?: string
  color?: string
  icon?: Component
}>()
</script>

<style scoped>
.metric-card {
  padding: 16px;
  background: #fafafa;
  border-radius: 8px;
  text-align: center;
}
</style>
