<template>
  <div class="returns-chart">
    <v-chart :option="option" autoresize style="height: 300px" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
} from 'echarts/components'

use([CanvasRenderer, BarChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

const props = defineProps<{
  dates: string[]
  returns: number[]
  title?: string
}>()

const option = computed(() => ({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow',
    },
  },
  legend: {
    data: ['Returns'],
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    data: props.dates,
    axisLabel: {
      formatter: (value: string) => {
        const parts = value.split('-')
        return `${parts[1]}/${parts[2]}`
      },
    },
  },
  yAxis: {
    type: 'value',
    name: 'Return',
    axisLabel: {
      formatter: (value: number) => `${(value * 100).toFixed(1)}%`,
    },
  },
  series: [
    {
      name: 'Returns',
      type: 'bar',
      data: props.returns.map((r) => ({
        value: r,
        itemStyle: {
          color: r >= 0 ? '#52c41a' : '#ff4d4f',
        },
      })),
    },
  ],
}))
</script>

<style scoped>
.returns-chart {
  width: 100%;
}
</style>
