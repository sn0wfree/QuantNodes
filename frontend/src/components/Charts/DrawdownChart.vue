<template>
  <div class="drawdown-chart">
    <v-chart :option="option" autoresize style="height: 200px" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  DataZoomComponent,
} from 'echarts/components'

use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, GridComponent, DataZoomComponent])

const props = defineProps<{
  dates: string[]
  drawdown: number[]
}>()

const option = computed(() => ({
  tooltip: {
    trigger: 'axis',
    formatter: (params: any) => {
      const data = params[0]
      return `${data.name}<br/>Drawdown: ${(data.value * 100).toFixed(2)}%`
    },
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '15%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
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
    name: 'Drawdown',
    axisLabel: {
      formatter: (value: number) => `${(value * 100).toFixed(0)}%`,
    },
  },
  dataZoom: [
    {
      type: 'inside',
      start: 0,
      end: 100,
    },
  ],
  series: [
    {
      name: 'Drawdown',
      type: 'line',
      smooth: true,
      data: props.drawdown,
      lineStyle: {
        color: '#ff4d4f',
        width: 2,
      },
      itemStyle: {
        color: '#ff4d4f',
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(255, 77, 79, 0.4)' },
            { offset: 1, color: 'rgba(255, 77, 79, 0.05)' },
          ],
        },
      },
    },
  ],
}))
</script>

<style scoped>
.drawdown-chart {
  width: 100%;
}
</style>
