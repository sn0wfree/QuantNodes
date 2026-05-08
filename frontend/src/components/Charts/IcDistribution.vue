<template>
  <div class="ic-distribution">
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
  icValues: number[]
}>()

const option = computed(() => {
  if (!props.icValues || props.icValues.length === 0) {
    return {
      title: { text: 'No IC data', left: 'center', top: 'center' },
    }
  }

  const bins = 20
  const arr = props.icValues
  let min = arr[0]
  let max = arr[0]
  let sum = 0
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < min) min = arr[i]
    if (arr[i] > max) max = arr[i]
    sum += arr[i]
  }
  
  if (min === max) {
    return {
      title: { text: `All values = ${min.toFixed(4)}`, left: 'center', top: 'center' },
    }
  }

  const binWidth = (max - min) / bins
  
  const histogram: number[] = new Array(bins).fill(0)
  const binLabels: string[] = []
  
  for (let i = 0; i < bins; i++) {
    const binStart = min + i * binWidth
    const binEnd = binStart + binWidth
    binLabels.push(`${binStart.toFixed(3)}-${binEnd.toFixed(3)}`)
    
    for (const value of arr) {
      if (value >= binStart && value < binEnd) {
        histogram[i]++
      }
    }
  }
  
  const mean = sum / arr.length
  const positiveRatio = arr.filter(v => v > 0).length / arr.length

  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: any) => {
        const data = params[0]
        return `IC Range: ${data.name}<br/>Count: ${data.value}`
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
      data: binLabels,
      axisLabel: {
        rotate: 45,
        fontSize: 10,
      },
    },
    yAxis: {
      type: 'value',
      name: 'Count',
    },
    series: [
      {
        name: 'Frequency',
        type: 'bar',
        data: histogram.map((count, i) => {
          const binStart = min + i * binWidth
          return {
            value: count,
            itemStyle: {
              color: binStart >= 0 ? '#52c41a' : '#ff4d4f',
            },
          }
        }),
        markLine: {
          data: [
            {
              xAxis: binLabels.findIndex((_, i) => {
                const binStart = min + i * binWidth
                return binStart <= mean && mean < binStart + binWidth
              }),
              label: {
                formatter: `Mean: ${mean.toFixed(4)}`,
              },
              lineStyle: {
                color: '#1677ff',
                type: 'dashed',
              },
            },
          ],
        },
      },
    ],
    graphic: [
      {
        type: 'text',
        left: 'center',
        top: 10,
        style: {
          text: `Positive Ratio: ${(positiveRatio * 100).toFixed(1)}%`,
          fontSize: 14,
          fill: '#666',
        },
      },
    ],
  }
})
</script>

<style scoped>
.ic-distribution {
  width: 100%;
}
</style>
