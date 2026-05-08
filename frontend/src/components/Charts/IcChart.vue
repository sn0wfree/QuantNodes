<template>
  <div class="ic-chart">
    <v-chart :option="option" autoresize style="height: 300px" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, BarChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
} from 'echarts/components'

use([
  CanvasRenderer,
  LineChart,
  BarChart,
  TitleComponent,
  TooltipComponent,
  GridComponent,
  LegendComponent,
  DataZoomComponent,
])

const props = defineProps<{
  dates: string[] | number[]
  icValues: number[]
  title?: string
}>()

const option = computed(() => ({
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'cross',
    },
  },
  legend: {
    data: ['IC'],
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
      formatter: (value: number) => {
        const date = new Date(value)
        return `${date.getMonth() + 1}/${date.getDate()}`
      },
    },
  },
  yAxis: {
    type: 'value',
    name: 'IC',
    axisLabel: {
      formatter: (value: number) => value.toFixed(4),
    },
  },
  dataZoom: [
    {
      type: 'inside',
      start: 0,
      end: 100,
    },
    {
      type: 'slider',
      start: 0,
      end: 100,
    },
  ],
  series: [
    {
      name: 'IC',
      type: 'line',
      smooth: true,
      data: props.icValues,
      lineStyle: {
        color: '#1677ff',
        width: 2,
      },
      itemStyle: {
        color: '#1677ff',
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(22, 119, 255, 0.3)' },
            { offset: 1, color: 'rgba(22, 119, 255, 0.05)' },
          ],
        },
      },
    },
  ],
}))
</script>

<style scoped>
.ic-chart {
  width: 100%;
}
</style>
