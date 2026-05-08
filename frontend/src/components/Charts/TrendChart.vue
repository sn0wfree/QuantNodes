<template>
  <div class="trend-chart">
    <v-chart :option="option" autoresize style="height: 200px" />
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
} from 'echarts/components'

use([CanvasRenderer, LineChart, BarChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

const props = defineProps<{
  data: Array<{ date: string; count: number }>
}>()

const option = computed(() => {
  const dates = props.data.map((d) => d.date.split('-').slice(1).join('/'))
  const counts = props.data.map((d) => d.count)

  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: {
        fontSize: 11,
      },
    },
    yAxis: {
      type: 'value',
      name: 'Count',
      axisLabel: {
        fontSize: 11,
      },
    },
    series: [
      {
        name: 'Insights',
        type: 'bar',
        data: counts,
        itemStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: '#1677ff' },
              { offset: 1, color: '#69b1ff' },
            ],
          },
          borderRadius: [4, 4, 0, 0],
        },
      },
    ],
  }
})
</script>

<style scoped>
.trend-chart {
  width: 100%;
}
</style>
