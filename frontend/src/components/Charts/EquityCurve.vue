<template>
  <div class="equity-curve">
    <v-chart :option="option" autoresize style="height: 400px" />
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
  LegendComponent,
} from 'echarts/components'

use([CanvasRenderer, LineChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

const props = defineProps<{
  dates: string[]
  equity: number[]
  benchmark?: number[]
}>()

const option = computed(() => ({
  tooltip: {
    trigger: 'axis',
  },
  legend: {
    data: props.benchmark ? ['Strategy', 'Benchmark'] : ['Strategy'],
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '3%',
    containLabel: true,
  },
  xAxis: {
    type: 'category',
    boundaryGap: false,
    data: props.dates,
  },
  yAxis: {
    type: 'value',
  },
  series: [
    {
      name: 'Strategy',
      type: 'line',
      smooth: true,
      data: props.equity,
    },
    ...(props.benchmark
      ? [
          {
            name: 'Benchmark',
            type: 'line',
            smooth: true,
            data: props.benchmark,
            lineStyle: { type: 'dashed' },
          },
        ]
      : []),
  ],
}))
</script>
