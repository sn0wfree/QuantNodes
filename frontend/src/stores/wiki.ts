import { defineStore } from 'pinia'
import { ref } from 'vue'
import { wikiApi } from '@/api/wiki'
import type { FactorInfo, StrategyInfo } from '@/api/wiki'

export const useWikiStore = defineStore('wiki', () => {
  const factors = ref<FactorInfo[]>([])
  const strategies = ref<StrategyInfo[]>([])
  const currentFactor = ref<FactorInfo | null>(null)
  const currentStrategy = ref<StrategyInfo | null>(null)
  const isLoading = ref(false)
  const searchQuery = ref('')
  const selectedCategory = ref<string | undefined>(undefined)

  const fetchFactors = async (params?: { category?: string; source?: string; sort?: string; limit?: number }) => {
    isLoading.value = true
    try {
      const data = await wikiApi.getFactors(params)
      factors.value = data
    } catch (error) {
      console.error('Failed to fetch factors:', error)
      factors.value = []
    } finally {
      isLoading.value = false
    }
  }

  const fetchFactor = async (name: string) => {
    isLoading.value = true
    try {
      const data = await wikiApi.getFactor(name)
      currentFactor.value = data
    } catch (error) {
      console.error('Failed to fetch factor:', error)
      currentFactor.value = null
    } finally {
      isLoading.value = false
    }
  }

  const createFactor = async (factor: Partial<FactorInfo>) => {
    try {
      const data = await wikiApi.createFactor(factor)
      await fetchFactors()
      return data
    } catch (error) {
      console.error('Failed to create factor:', error)
      throw error
    }
  }

  const updateFactor = async (name: string, factor: Partial<FactorInfo>) => {
    try {
      const data = await wikiApi.updateFactor(name, factor)
      await fetchFactors()
      return data
    } catch (error) {
      console.error('Failed to update factor:', error)
      throw error
    }
  }

  const deleteFactor = async (name: string) => {
    try {
      await wikiApi.deleteFactor(name)
      await fetchFactors()
    } catch (error) {
      console.error('Failed to delete factor:', error)
      throw error
    }
  }

  const fetchStrategies = async (params?: { category?: string; sort?: string; limit?: number }) => {
    isLoading.value = true
    try {
      const data = await wikiApi.getStrategies(params)
      strategies.value = data
    } catch (error) {
      console.error('Failed to fetch strategies:', error)
      strategies.value = []
    } finally {
      isLoading.value = false
    }
  }

  const fetchStrategy = async (name: string) => {
    isLoading.value = true
    try {
      const data = await wikiApi.getStrategy(name)
      currentStrategy.value = data
    } catch (error) {
      console.error('Failed to fetch strategy:', error)
      currentStrategy.value = null
    } finally {
      isLoading.value = false
    }
  }

  const createStrategy = async (strategy: Partial<StrategyInfo>) => {
    try {
      const data = await wikiApi.createStrategy(strategy)
      await fetchStrategies()
      return data
    } catch (error) {
      console.error('Failed to create strategy:', error)
      throw error
    }
  }

  const updateStrategy = async (name: string, strategy: Partial<StrategyInfo>) => {
    try {
      const data = await wikiApi.updateStrategy(name, strategy)
      await fetchStrategies()
      return data
    } catch (error) {
      console.error('Failed to update strategy:', error)
      throw error
    }
  }

  const deleteStrategy = async (name: string) => {
    try {
      await wikiApi.deleteStrategy(name)
      await fetchStrategies()
    } catch (error) {
      console.error('Failed to delete strategy:', error)
      throw error
    }
  }

  const searchFactors = async (query: string) => {
    isLoading.value = true
    try {
      const data = await wikiApi.search({ q: query, type: 'factor' })
      factors.value = data.map((r: any) => ({
        name: r.page_name?.split('/').pop() || r.name || '',
        formula: r.formula || '',
        source: r.source || r.page_type || '',
        category: r.category || 'other',
        tags: r.tags || [],
        description: r.description || r.content || '',
        ic_mean: r.ic_mean,
        ic_std: r.ic_std,
        icir: r.icir,
        rank_ic_mean: r.rank_ic_mean,
      }))
    } catch (error) {
      console.error('Failed to search factors:', error)
    } finally {
      isLoading.value = false
    }
  }

  const searchStrategies = async (query: string) => {
    isLoading.value = true
    try {
      const data = await wikiApi.search({ q: query, type: 'strategy' })
      strategies.value = data.map((r: any) => ({
        name: r.page_name?.split('/').pop() || r.name || '',
        description: r.description || r.content || '',
        category: r.category || 'general',
        tags: r.tags || [],
        strategy_yaml: r.strategy_yaml || '',
      }))
    } catch (error) {
      console.error('Failed to search strategies:', error)
    } finally {
      isLoading.value = false
    }
  }

  return {
    factors,
    strategies,
    currentFactor,
    currentStrategy,
    isLoading,
    searchQuery,
    selectedCategory,
    fetchFactors,
    fetchFactor,
    createFactor,
    updateFactor,
    deleteFactor,
    fetchStrategies,
    fetchStrategy,
    createStrategy,
    updateStrategy,
    deleteStrategy,
    searchFactors,
    searchStrategies,
  }
})
