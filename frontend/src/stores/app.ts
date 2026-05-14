import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'

const CONTEXT_WIDTH_KEY = 'quantnodes-context-panel-width'
const TOOLS_WIDTH_KEY = 'quantnodes-tools-panel-width'

function loadPanelWidth(key: string, defaultWidth: number): number {
  try {
    const saved = localStorage.getItem(key)
    if (saved) {
      const n = parseInt(saved, 10)
      if (!isNaN(n) && n > 0) return n
    }
  } catch { /* ignore */ }
  return defaultWidth
}

export const useAppStore = defineStore('app', () => {
  const sidebarCollapsed = ref(false)
  const chatSidebarCollapsed = ref(true)
  const contextPanelOpen = ref(false)
  const toolsPanelOpen = ref(false)
  const contextPanelWidth = ref(loadPanelWidth(CONTEXT_WIDTH_KEY, 240))
  const toolsPanelWidth = ref(loadPanelWidth(TOOLS_WIDTH_KEY, 260))
  const theme = ref<'light' | 'dark'>(
    (localStorage.getItem('quantnodes-theme') as 'light' | 'dark') || 'light'
  )

  const isDarkMode = computed(() => theme.value === 'dark')

  watch(theme, (newTheme) => {
    localStorage.setItem('quantnodes-theme', newTheme)
    document.documentElement.setAttribute('data-theme', newTheme)
  }, { immediate: true })

  watch(contextPanelWidth, (val) => {
    localStorage.setItem(CONTEXT_WIDTH_KEY, String(Math.round(val)))
  })

  watch(toolsPanelWidth, (val) => {
    localStorage.setItem(TOOLS_WIDTH_KEY, String(Math.round(val)))
  })

  const toggleSidebar = () => {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  const toggleChatSidebar = () => {
    chatSidebarCollapsed.value = !chatSidebarCollapsed.value
  }

  const toggleContextPanel = () => {
    contextPanelOpen.value = !contextPanelOpen.value
    if (contextPanelOpen.value) {
      toolsPanelOpen.value = true
    } else {
      toolsPanelOpen.value = false
    }
  }

  const toggleToolsPanel = () => {
    toolsPanelOpen.value = !toolsPanelOpen.value
  }

  const setTheme = (newTheme: 'light' | 'dark') => {
    theme.value = newTheme
    localStorage.setItem('quantnodes-theme', newTheme)
  }

  return {
    sidebarCollapsed,
    chatSidebarCollapsed,
    contextPanelOpen,
    toolsPanelOpen,
    contextPanelWidth,
    toolsPanelWidth,
    theme,
    isDarkMode,
    toggleSidebar,
    toggleChatSidebar,
    toggleContextPanel,
    toggleToolsPanel,
    setTheme,
  }
})
