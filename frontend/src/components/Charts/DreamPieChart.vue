<template>
  <div class="dream-chart">
    <v-chart :option="option" autoresize style="height: 300px" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
} from 'echarts/components'

use([CanvasRenderer, PieChart, BarChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

const props = defineProps<{
  data: Record<string, number>
  title?: string
}>()

const option = computed(() => {
  const colors = {
    factor_insight: '#1677ff',
    strategy_insight: '#52c41a',
    market_regime: '#faad14',
    risk_alert: '#ff4d4f',
    wiki_insight: '#722ed1',
  }

  const pieData = Object.entries(props.data).map(([name, value]) => ({
    name: name.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase()),
    value,
    itemStyle: { color: colors[name as keyof typeof colors] || '#d9d9d9' },
  }))

  return {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)',
    },
    legend: {
      orient: 'vertical',
      right: 10,
      top: 'center',
    },
    series: [
      {
        name: props.title || 'Insights',
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['40%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 10,
          borderColor: '#fff',
          borderWidth: 2,
        },
        label: {
          show: false,
          position: 'center',
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 16,
            fontWeight: 'bold',
          },
        },
        labelLine: {
          show: false,
        },
        data: pieData,
      },
    ],
  }
})
</script>

<style scoped>
.dream-chart {
  width: 100%;
}
</style>
